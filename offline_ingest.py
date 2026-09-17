"""
offline_ingest.py

Sprint 1: Storage and Ingestion Pipeline
Ref: cosi_design_doc.md, Section 4 - Subsystem A (Storage) & Subsystem B (Ingestion & Chunker)

Responsibilities:
    1. Initialize the SQLite database (`documents` and `chunks` tables).
    2. Walk the local notes directory (./notes) and detect new/modified markdown files
       via a hash comparison against the `documents` table (Incremental Check).
    3. Chunk changed files into paragraph-sized blocks (150-500 chars) on double-newlines.
    4. Run each chunk through the NLTK PorterStemmer to build `stemmed_tokens` (Chapter 2:
       Set Theory / equivalence classes -- see local_search_engine.py for a reference
       BM25 tokenizer/stemmer implementation).
    5. POST each chunk to the local Ollama embeddings endpoint (nomic-embed-text) and store
       the resulting 768-dim float vector as a BLOB.

Run this script manually (or via the WebUI's "Refresh Index" button, per the
"Prompted Refresh" mitigation strategy in cosi_project_management.pptx) whenever notes change.

Graceful fallback: if Ollama is unreachable, ingestion still runs -- chunks are stored
with stemmed_tokens but a NULL embedding_vector, so keyword (BM25) search still works
even if vector search doesn't. This mirrors the fallback strategy in the design doc.
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

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "cosi.db")
NOTES_DIR = os.path.join(os.path.dirname(__file__), "notes")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

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


def find_changed_files(conn, notes_dir=NOTES_DIR):
    """
    Compare markdown files on disk against the `documents` table.
    Returns a list of absolute file paths that are new or have changed since last ingest.
    """
    changed = []
    if not os.path.isdir(notes_dir):
        return changed

    cur = conn.execute("SELECT file_path, file_hash FROM documents")
    known_hashes = {row[0]: row[1] for row in cur.fetchall()}

    for root, _dirs, files in os.walk(notes_dir):
        for name in files:
            if not name.lower().endswith(".md"):
                continue
            path = os.path.join(root, name)
            current_hash = hash_file(path)
            if known_hashes.get(path) != current_hash:
                changed.append(path)

    return changed


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


def get_embedding(text, model=EMBED_MODEL, url=OLLAMA_EMBED_URL):
    """
    POST text to the local Ollama embeddings endpoint and return the 768-float vector.
    Returns None (instead of raising) if Ollama is unreachable, so callers can apply
    the graceful keyword-only fallback.
    """
    data = {"model": model, "prompt": text}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("embedding")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"  [warn] Ollama unreachable ({e}); storing chunk without embedding.")
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
    and write results into SQLite. Prints a short summary as it runs.
    """
    conn = init_db(db_path)
    changed_files = find_changed_files(conn, notes_dir)

    if not changed_files:
        print("No new or modified notes. Index is up to date.")
        conn.close()
        return

    total_chunks = 0
    for file_path in changed_files:
        print(f"Indexing {file_path} ...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        file_hash = hash_file(file_path)
        last_modified = os.path.getmtime(file_path)

        # Upsert the document row, then wipe out any of its old chunks
        # (simplest correct approach: re-chunk the whole file on any change).
        cur = conn.execute(
            "SELECT id FROM documents WHERE file_path = ?", (file_path,)
        )
        row = cur.fetchone()
        if row:
            document_id = row[0]
            conn.execute(
                "UPDATE documents SET file_hash = ?, last_modified = ? WHERE id = ?",
                (file_hash, last_modified, document_id),
            )
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        else:
            cur = conn.execute(
                "INSERT INTO documents (file_path, file_hash, last_modified) VALUES (?, ?, ?)",
                (file_path, file_hash, last_modified),
            )
            document_id = cur.lastrowid

        chunks = chunk_text(raw_text)
        for chunk in chunks:
            stemmed = stem_chunk(chunk)
            vector = get_embedding(chunk)
            conn.execute(
                """
                INSERT INTO chunks (document_id, raw_text, stemmed_tokens, embedding_vector)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, chunk, stemmed, _vector_to_blob(vector)),
            )
            total_chunks += 1

        conn.commit()

    print(f"Done. Indexed {len(changed_files)} file(s), {total_chunks} chunk(s) total.")
    conn.close()


if __name__ == "__main__":
    ingest()
