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

from config import EMBED_MODEL, EMBED_URL, OLLAMA_PING_TIMEOUT, OLLAMA_TIMEOUT, OLLAMA_URL, TAGS_URL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "cosi.db")
NOTES_DIR = os.path.join(BASE_DIR, "notes")

# Kept for backwards compatibility with anything that imported the old constant.
OLLAMA_EMBED_URL = EMBED_URL

MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 500

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


def chunk_text(raw_text):
    """
    Split raw markdown text on double-newlines into paragraph-sized chunks,
    keeping each chunk between MIN_CHUNK_CHARS and MAX_CHUNK_CHARS to avoid
    concept dilution (see cosi_design_doc.md Section 6: Semantic Dilution risk).

    - Short adjacent paragraphs are merged together until they clear MIN_CHUNK_CHARS.
    - Paragraphs longer than MAX_CHUNK_CHARS are hard-split on that boundary.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

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


def get_embedding(text, model=EMBED_MODEL, url=EMBED_URL):
    """
    POST text to the Ollama embeddings endpoint and return the 768-float vector.
    Returns None (instead of raising) if Ollama is unreachable or answers with
    something unexpected, so callers can apply the graceful keyword-only fallback.
    """
    data = {"model": model, "prompt": text}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("embedding") or None
    except (OSError, ValueError) as e:  # URLError, HTTPError, timeouts, bad JSON
        print(f"  [warn] Ollama embedding failed at {url} ({e}); storing chunk without embedding.")
        return None


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

        # 2. Decide what to (re-)index: new/edited files, plus files that were indexed
        #    without embeddings last time if Ollama is available now.
        retry = _documents_missing_embeddings(conn) if embeddings_enabled else set()
        to_index = []
        for rel, path in sorted(markdown.items()):
            file_hash = hash_file(path)
            if rel not in known or known[rel][1] != file_hash or rel in retry:
                to_index.append((rel, path, file_hash))

        total_chunks = 0
        missing_vectors = 0
        for rel, path, file_hash in to_index:
            print(f"Indexing {rel} ...")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            last_modified = os.path.getmtime(path)

            # Upsert the document row, then wipe out any of its old chunks
            # (simplest correct approach: re-chunk the whole file on any change).
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

            for chunk in chunk_text(raw_text):
                vector = None
                if embeddings_enabled:
                    vector = get_embedding(chunk)
                    if vector is None:
                        # Ollama went away mid-run: stop trying for the rest of this refresh.
                        embeddings_enabled = False
                        ollama_reason = "embedding request failed during ingest"
                if vector is None:
                    missing_vectors += 1
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, raw_text, stemmed_tokens, embedding_vector)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, chunk, stem_chunk(chunk), _vector_to_blob(vector)),
                )
                total_chunks += 1

            conn.commit()

        conn.commit()
    finally:
        conn.close()

    if to_index or stale:
        message = (f"Indexed {len(to_index)} file(s), {total_chunks} chunk(s); "
                   f"removed {len(stale)} deleted file(s).")
    else:
        message = "No new or modified notes. Index is up to date."
    print(message)

    stats = {
        "files_indexed": len(to_index),
        "chunks_indexed": total_chunks,
        "files_removed": len(stale),
        "chunks_without_embeddings": missing_vectors,
        "embeddings_enabled": embeddings_enabled,
        "unsupported_files": unsupported,
        "message": message,
    }
    if not embeddings_enabled:
        stats["warning"] = f"Indexed without embeddings: {ollama_reason}"
    return stats


if __name__ == "__main__":
    ingest()
