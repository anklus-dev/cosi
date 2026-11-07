"""
offline_ingest.py

Sprint 1: Storage and Ingestion Pipeline
Ref: cosi_design_doc.md, Section 4 - Subsystem A (Storage) & Subsystem B (Ingestion & Chunker)

Responsibilities:
    1. Initialize the SQLite database (`documents` and `chunks` tables).
    2. Walk the local notes directory (./notes) and detect new/modified markdown files
       via a hash comparison against the `documents` table (Incremental Check).
       Notes deleted from disk are removed from the index.
    3. Chunk changed files into paragraph-sized blocks (150-500 chars) on double-newlines.
    4. Run each chunk through the NLTK PorterStemmer to build `stemmed_tokens` (Chapter 2:
       Set Theory / equivalence classes -- see local_search_engine.py for a reference
       BM25 tokenizer/stemmer implementation).
    5. POST each chunk to the Ollama embeddings endpoint (nomic-embed-text, URL from
       config.py / COSI_OLLAMA_URL) and store the resulting 768-dim float vector as a BLOB.

Run this script manually (or via the WebUI's "Refresh Index" button, per the
"Prompted Refresh" mitigation strategy in cosi_project_management.pptx) whenever notes change.

Graceful fallback: if Ollama is unreachable, ingestion still runs -- chunks are stored
with stemmed_tokens but a NULL embedding_vector, so keyword (BM25) search still works
even if vector search doesn't. This mirrors the fallback strategy in the design doc.
Ollama is checked once per run; if it is down, no per-chunk requests are attempted, so a
refresh never stalls on one timeout per chunk. Files indexed without embeddings are
re-embedded automatically on the next refresh where Ollama is available.

Paths: `documents.file_path` is stored RELATIVE to the notes directory (e.g. "kyoto.md").
Absolute paths differ between the host (/home/...) and the container (/app/notes/...), so
storing them would make a shared data/cosi.db re-index everything as "new" on each side
and leave duplicate rows behind. Rows from older indexes that used absolute paths are
detected as stale and replaced on the next refresh.
"""

import hashlib
import json
import os
import re
import sqlite3
import struct
import urllib.error
import urllib.request

from nltk.stem import PorterStemmer

import time

from config import (EMBED_BATCH_SIZE, EMBED_BATCH_URL, EMBED_MODEL, EMBED_URL, NOTES_DIR_OVERRIDE,
                    OLLAMA_PING_TIMEOUT, OLLAMA_TIMEOUT, OLLAMA_URL, TAGS_URL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "cosi.db")
NOTES_DIR = os.path.abspath(NOTES_DIR_OVERRIDE) if NOTES_DIR_OVERRIDE else os.path.join(BASE_DIR, "notes")

# Kept for backwards compatibility with anything that imported the old constant.
OLLAMA_EMBED_URL = EMBED_URL

MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 500
CHUNK_MODE = "sentence"   # or "hard" for the original fixed-width cut
SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!;:])\s+")

_stemmer = PorterStemmer()


def init_db(db_path=DB_PATH):
    """
    Create `documents` and `chunks` tables if they do not already exist.
    Schema per cosi_design_doc.md Section 4, Subsystem A.
    Returns an open sqlite3.Connection.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            last_modified REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            stemmed_tokens TEXT NOT NULL,
            embedding_vector BLOB,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
        """
    )
    conn.commit()
    return conn


def hash_file(file_path):
    """Return a SHA256 hash of a file's contents, used to detect edits."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _scan_notes(notes_dir=NOTES_DIR):
    """
    Walk notes_dir and return (markdown, unsupported):
        markdown     {relative_path: absolute_path} for every .md file
        unsupported  sorted relative paths of every other (non-hidden) file
    Relative paths use "/" so the index is identical on Linux, macOS and Windows.
    """
    markdown = {}
    unsupported = []
    if not os.path.isdir(notes_dir):
        return markdown, unsupported

    for root, dirs, files in os.walk(notes_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # skip .git, .obsidian, ...
        for name in files:
            if name.startswith("."):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, notes_dir).replace(os.sep, "/")
            if name.lower().endswith(".md"):
                markdown[rel] = path
            else:
                unsupported.append(rel)

    return markdown, sorted(unsupported)


def find_changed_files(conn, notes_dir=NOTES_DIR):
    """
    Compare markdown files on disk against the `documents` table.
    Returns a list of absolute file paths that are new or have changed since last ingest.
    """
    known_hashes = dict(conn.execute("SELECT file_path, file_hash FROM documents").fetchall())
    markdown, _unsupported = _scan_notes(notes_dir)
    return [path for rel, path in markdown.items() if known_hashes.get(rel) != hash_file(path)]


def _documents_missing_embeddings(conn):
    """Relative paths of documents with at least one chunk stored without an embedding."""
    rows = conn.execute(
        """
        SELECT DISTINCT documents.file_path
        FROM documents JOIN chunks ON chunks.document_id = documents.id
        WHERE chunks.embedding_vector IS NULL
        """
    ).fetchall()
    return {row[0] for row in rows}


def _split_long_paragraph(paragraph, max_chars=MAX_CHUNK_CHARS):
    """
    Split a paragraph longer than max_chars into pieces that end on sentence
    boundaries (. ? ! ; :), packing as many whole sentences as fit. A single
    sentence longer than max_chars is split at the last space before the limit,
    so no word is ever cut in half.
    """
    sentences = [s for s in SENTENCE_BOUNDARY.split(paragraph) if s]
    pieces = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # one enormous sentence: fall back to word boundary
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(raw_text, mode=None):
    """
    Split raw markdown text on double-newlines into paragraph-sized chunks,
    keeping each chunk between MIN_CHUNK_CHARS and MAX_CHUNK_CHARS to avoid
    concept dilution (see cosi_design_doc.md Section 6: Semantic Dilution risk).

    - Short adjacent paragraphs are merged together until they clear MIN_CHUNK_CHARS.
    - mode "sentence" (default): paragraphs longer than MAX_CHUNK_CHARS are split
      on sentence boundaries, so no chunk ends mid-sentence or mid-word.
    - mode "hard": the original behaviour -- cut every MAX_CHUNK_CHARS characters.
      Kept so eval/run_eval.py can measure the difference.
    """
    mode = mode or CHUNK_MODE
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    if mode == "hard":
        return _chunk_hard(paragraphs)

    chunks = []
    buffer = ""
    for para in paragraphs:
        para = re.sub(r"[ \t]*\n[ \t]*", " ", para) if len(para) > MAX_CHUNK_CHARS else para
        pieces = _split_long_paragraph(para) if len(para) > MAX_CHUNK_CHARS else [para]
        for piece in pieces:
            # A leftover that is already a reasonable chunk is emitted on its own rather than
            # pushing the next piece over the limit. A tiny leftover (a heading, a one-line
            # passage) stays attached so it never becomes a context-free chunk by itself; that
            # can exceed MAX_CHUNK_CHARS by less than MIN_CHUNK_CHARS.
            if buffer and len(buffer) + 2 + len(piece) > MAX_CHUNK_CHARS and len(buffer) >= MIN_CHUNK_CHARS // 2:
                chunks.append(buffer)
                buffer = ""
            buffer = f"{buffer}\n\n{piece}" if buffer else piece
            if len(buffer) >= MIN_CHUNK_CHARS:
                chunks.append(buffer)
                buffer = ""

    if buffer:
        chunks.append(buffer)

    return chunks


def _chunk_hard(paragraphs):
    """Original chunker: merge short paragraphs, hard-cut long ones every MAX_CHUNK_CHARS."""
    chunks = []
    buffer = ""
    for para in paragraphs:
        buffer = f"{buffer}\n\n{para}".strip() if buffer else para

        while len(buffer) > MAX_CHUNK_CHARS:
            chunks.append(buffer[:MAX_CHUNK_CHARS].strip())
            buffer = buffer[MAX_CHUNK_CHARS:].strip()

        if len(buffer) >= MIN_CHUNK_CHARS:
            chunks.append(buffer)
            buffer = ""

    if buffer:
        chunks.append(buffer)

    return chunks


def stem_chunk(raw_text):
    """
    Lowercase, strip punctuation, tokenize, and stem a chunk's text.
    Returns a space-separated string of stems for storage in `chunks.stemmed_tokens`.
    Mirrors local_search_engine.py's _tokenize_and_stem().
    """
    text = raw_text.lower()
    words = re.findall(r"\b\w+\b", text)
    stems = [_stemmer.stem(w) for w in words]
    return " ".join(stems)


def ollama_available():
    """
    Fast check (OLLAMA_PING_TIMEOUT seconds) that Ollama is reachable and the
    embedding model is pulled. Returns (ok, reason).
    """
    try:
        with urllib.request.urlopen(TAGS_URL, timeout=OLLAMA_PING_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as e:  # URLError/timeouts are OSErrors; bad JSON is ValueError
        return False, f"Ollama unreachable at {OLLAMA_URL} ({e})"

    names = {m.get("name", "") for m in payload.get("models", [])}
    if not any(n == EMBED_MODEL or n.startswith(EMBED_MODEL + ":") for n in names):
        return False, f"model '{EMBED_MODEL}' is not pulled (run: ollama pull {EMBED_MODEL})"
    return True, f"Ollama reachable at {OLLAMA_URL}"


# Set once we learn the Ollama server is too old for /api/embed (added in Ollama 0.3).
_legacy_endpoint_only = False


def _post_json(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _legacy_embedding(text, model=EMBED_MODEL):
    """One text through the old /api/embeddings endpoint. Returns a vector or None."""
    try:
        body = _post_json(EMBED_URL, {"model": model, "prompt": text}, OLLAMA_TIMEOUT)
        return body.get("embedding") or None
    except (OSError, ValueError) as e:  # URLError, HTTPError, timeouts, bad JSON
        print(f"  [warn] Ollama embedding failed at {EMBED_URL} ({e}); storing chunk without embedding.")
        return None


def _batch_request(texts, model):
    """One /api/embed call for a list of texts. Returns the list of vectors (raises on failure)."""
    # A batch takes longer than a single text, especially on a CPU, so scale the timeout.
    body = _post_json(EMBED_BATCH_URL, {"model": model, "input": list(texts)},
                      OLLAMA_TIMEOUT + 2 * len(texts))
    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError(f"expected {len(texts)} embeddings, got {type(vectors).__name__}")
    return vectors


def get_embeddings(texts, model=EMBED_MODEL):
    """
    Embed a list of texts with as few Ollama requests as possible.

    Sends the whole list to /api/embed in one request (callers keep lists to about
    EMBED_BATCH_SIZE). Returns a list the same length as `texts`, holding a vector
    or None for each text, so callers can apply the keyword-only fallback per chunk.

    Fallbacks: an Ollama too old for /api/embed (HTTP 404) switches to the legacy
    one-text-per-request endpoint for the rest of the run; a batch that Ollama
    rejects for another reason is retried one text at a time; an unreachable
    Ollama returns all None straight away.
    """
    global _legacy_endpoint_only
    texts = list(texts)
    if not texts:
        return []
    if not _legacy_endpoint_only:
        try:
            return _batch_request(texts, model)
        except urllib.error.HTTPError as e:
            if e.code == 404 and "model" not in _error_text(e):
                print("  [info] this Ollama has no /api/embed; using the older one-at-a-time endpoint.")
                _legacy_endpoint_only = True
            elif len(texts) > 1:
                print(f"  [warn] batch of {len(texts)} rejected by Ollama ({e}); retrying one at a time.")
                return [get_embeddings([t], model)[0] for t in texts]
            else:
                print(f"  [warn] Ollama embedding failed at {EMBED_BATCH_URL} ({e}); storing chunk without embedding.")
                return [None]
        except (OSError, ValueError) as e:  # unreachable, timeout, malformed reply
            print(f"  [warn] Ollama embedding failed at {EMBED_BATCH_URL} ({e}); storing chunk(s) without embeddings.")
            return [None] * len(texts)
    return [_legacy_embedding(t, model) for t in texts]


def _error_text(http_error):
    try:
        return http_error.read().decode("utf-8", "replace").lower()
    except Exception:
        return ""


def get_embedding(text, model=EMBED_MODEL):
    """
    Embed one text (used for search queries). Same endpoint as indexing, so a
    query and the passages it is compared with are always embedded the same way.
    Returns the vector, or None if Ollama is unreachable.
    """
    return get_embeddings([text], model)[0]


def _vector_to_blob(vector):
    """Pack a list of floats into a compact binary BLOB for SQLite storage."""
    if vector is None:
        return None
    return struct.pack(f"{len(vector)}f", *vector)


def blob_to_vector(blob):
    """Unpack a BLOB back into a list of floats. Returns None for NULL blobs."""
    if blob is None:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _store_document(conn, known, rel, file_hash, last_modified, chunks, vectors):
    """
    Upsert one note's document row, replace its chunks, and commit.
    (Simplest correct approach: re-chunk the whole file on any change.)
    """
    if rel in known:
        document_id = known[rel][0]
        conn.execute(
            "UPDATE documents SET file_hash = ?, last_modified = ? WHERE id = ?",
            (file_hash, last_modified, document_id),
        )
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    else:
        cur = conn.execute(
            "INSERT INTO documents (file_path, file_hash, last_modified) VALUES (?, ?, ?)",
            (rel, file_hash, last_modified),
        )
        document_id = cur.lastrowid
    conn.executemany(
        """
        INSERT INTO chunks (document_id, raw_text, stemmed_tokens, embedding_vector)
        VALUES (?, ?, ?, ?)
        """,
        [(document_id, chunk, stem_chunk(chunk), _vector_to_blob(vector))
         for chunk, vector in zip(chunks, vectors)],
    )
    conn.commit()


def ingest(notes_dir=NOTES_DIR, db_path=DB_PATH):
    """
    Main entry point: init DB, find changed files, chunk + stem + embed each,
    remove notes that were deleted, and write results into SQLite.

    Returns a stats dict (consumed by app.py's /api/refresh):
        files_indexed, chunks_indexed, files_removed, chunks_without_embeddings,
        embeddings_enabled, unsupported_files, message, and (if embeddings are off) warning.
    """
    conn = init_db(db_path)
    try:
        embeddings_enabled, ollama_reason = ollama_available()
        if not embeddings_enabled:
            print(f"[warn] {ollama_reason}; indexing keyword-only.")

        markdown, unsupported = _scan_notes(notes_dir)
        known = {row[0]: (row[1], row[2]) for row in
                 conn.execute("SELECT file_path, id, file_hash FROM documents").fetchall()}

        # 1. Drop documents whose file no longer exists (includes old absolute-path rows).
        stale = [rel for rel in known if rel not in markdown]
        for rel in stale:
            doc_id = known[rel][0]
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()  # so a per-file rollback below can't undo these deletions

        # 2. Decide what to (re-)index: new/edited files, plus files that were indexed
        #    without embeddings last time if Ollama is available now.
        retry = _documents_missing_embeddings(conn) if embeddings_enabled else set()
        to_index = []
        unreadable = []  # files that couldn't be opened (broken links, permissions, deleted mid-scan)
        for rel, path in sorted(markdown.items()):
            try:
                file_hash = hash_file(path)
            except OSError as e:
                print(f"  [warn] skipping {rel}: {e}")
                unreadable.append(rel)
                continue
            if rel not in known or known[rel][1] != file_hash or rel in retry:
                to_index.append((rel, path, file_hash))

        # 3. Read and chunk every file first (fast: no Ollama involved), so embedding
        #    can be done in full batches across files. Vaults are mostly short notes,
        #    so batching only within a file would barely help.
        pending = []  # (rel, file_hash, last_modified, chunks)
        for rel, path, file_hash in to_index:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()
                last_modified = os.path.getmtime(path)
            except OSError as e:  # vanished or became unreadable since the scan
                print(f"  [warn] skipping {rel}: {e}")
                unreadable.append(rel)
                continue
            pending.append((rel, file_hash, last_modified, chunk_text(raw_text)))

        grand_total = sum(len(chunks) for *_, chunks in pending)
        if pending:
            print(f"Indexing {len(pending)} file(s), {grand_total} chunk(s)"
                  + (f", {EMBED_BATCH_SIZE} per embedding request" if embeddings_enabled else "") + " ...")

        # 4. Embed in batches and save file by file. A window of files is embedded
        #    together, then written and committed, so progress survives interruption.
        total_chunks = 0
        missing_vectors = 0
        files_indexed = 0
        started = time.perf_counter()
        i = 0
        while i < len(pending):
            window = [pending[i]]
            i += 1
            while i < len(pending) and sum(len(p[3]) for p in window) < EMBED_BATCH_SIZE:
                window.append(pending[i])
                i += 1
            texts = [chunk for *_, chunks in window for chunk in chunks]
            vectors = [None] * len(texts)
            if embeddings_enabled:
                vectors = []
                for start in range(0, len(texts), EMBED_BATCH_SIZE):
                    vectors.extend(get_embeddings(texts[start:start + EMBED_BATCH_SIZE]))
                if texts and all(v is None for v in vectors):
                    # Either Ollama went away mid-run, or it rejected these particular
                    # passages. Only stop embedding for the rest of the refresh if it's gone.
                    still_up, reason = ollama_available()
                    if not still_up:
                        embeddings_enabled = False
                        ollama_reason = f"embedding stopped during ingest: {reason}"
            vector_iter = iter(vectors)
            for rel, file_hash, last_modified, chunks in window:
                file_vectors = [next(vector_iter) for _ in chunks]
                _store_document(conn, known, rel, file_hash, last_modified, chunks, file_vectors)
                files_indexed += 1
                total_chunks += len(chunks)
                missing_vectors += sum(1 for v in file_vectors if v is None)
            rate = total_chunks / max(time.perf_counter() - started, 1e-6)
            print(f"  {total_chunks}/{grand_total} chunks ({rate:.0f}/s)")

        conn.commit()
    finally:
        conn.close()

    if files_indexed or stale:
        message = (f"Indexed {files_indexed} file(s), {total_chunks} chunk(s); "
                   f"removed {len(stale)} deleted file(s).")
    else:
        message = "No new or modified notes. Index is up to date."
    if unreadable:
        message += f" Skipped {len(unreadable)} unreadable file(s)."
    print(message)

    stats = {
        "files_indexed": files_indexed,
        "chunks_indexed": total_chunks,
        "files_removed": len(stale),
        "chunks_without_embeddings": missing_vectors,
        "embeddings_enabled": embeddings_enabled,
        "unsupported_files": unsupported,
        "unreadable_files": sorted(unreadable),
        "message": message,
    }
    if not embeddings_enabled:
        stats["warning"] = f"Indexed without embeddings: {ollama_reason}"
    return stats


if __name__ == "__main__":
    ingest()
