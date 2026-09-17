"""
cosi_engine.py

Sprint 2: Core Search Mathematics
Ref: cosi_design_doc.md, Section 4 - Subsystem C (Retrieval & Hybrid Ranking Engine)

Responsibilities:
    1. Keyword Branch: tokenize/stem the query, look up `stemmed_tokens` overlaps in
       SQLite, and score with Okapi BM25 (TF within chunk x IDF across all chunks).
       Ported from local_search_engine.py's in-memory reference implementation to
       read from SQLite instead.
    2. Vector Branch: embed the query via Ollama (offline_ingest.get_embedding),
       load cached chunk vectors from SQLite, and compute Cosine Similarity with NumPy.
    3. Score Blender: normalize both score arrays to [0, 1] and combine:
           Final Score = (0.4 * BM25) + (0.6 * Cosine Similarity)
    4. Return the top-N ranked chunks (source file, raw text, and component scores)
       for consumption by app.py.

Graceful fallback: if no chunk has a stored embedding (Ollama was offline during
ingest) or the query itself can't be embedded (Ollama offline right now), the
vector branch is skipped and results are ranked on BM25 alone.
"""

import math
import re
import sqlite3

import numpy as np

from offline_ingest import DB_PATH, blob_to_vector, get_embedding, stem_chunk
from nltk.stem import PorterStemmer

BM25_WEIGHT = 0.4
COSINE_WEIGHT = 0.6

# BM25 tuning parameters (matching local_search_engine.py's defaults)
BM25_B = 0.75
BM25_K1 = 1.5

_stemmer = PorterStemmer()


def tokenize_and_stem_query(query):
    """Lowercase, strip punctuation, tokenize, and stem the raw query string."""
    text = query.lower()
    words = re.findall(r"\b\w+\b", text)
    return [_stemmer.stem(w) for w in words]


def bm25_search(query_tokens, db_path=DB_PATH):
    """
    Query the `chunks` table for stemmed-token overlaps and compute BM25 scores.
    Returns {chunk_id: bm25_score}.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, stemmed_tokens FROM chunks").fetchall()
    conn.close()

    if not rows:
        return {}

    # Build an inverted index + doc lengths on the fly, same shape as
    # local_search_engine.py's LocalSearchEngine, just sourced from SQLite.
    index = {}          # {stem: {chunk_id: term_frequency}}
    doc_lengths = {}     # {chunk_id: token_count}

    for chunk_id, stemmed_tokens in rows:
        tokens = stemmed_tokens.split()
        doc_lengths[chunk_id] = len(tokens)
        for token in tokens:
            index.setdefault(token, {})
            index[token][chunk_id] = index[token].get(chunk_id, 0) + 1

    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)
    N = len(doc_lengths)

    scores = {}
    for token in query_tokens:
        if token not in index:
            continue

        df = len(index[token])
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for chunk_id, tf in index[token].items():
            doc_len = doc_lengths[chunk_id]
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / avg_doc_length))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (idf * (numerator / denominator))

    return scores


def vector_search(query_vector, db_path=DB_PATH):
    """
    Load cached chunk embedding BLOBs from SQLite, compute cosine similarity
    against `query_vector` for each, and return {chunk_id: cosine_score}.
    Chunks with no stored embedding (NULL blob) are skipped.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, embedding_vector FROM chunks").fetchall()
    conn.close()

    q = np.array(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return {}

    scores = {}
    for chunk_id, blob in rows:
        vector = blob_to_vector(blob)
        if vector is None:
            continue
        d = np.array(vector, dtype=np.float32)
        d_norm = np.linalg.norm(d)
        if d_norm == 0:
            continue
        scores[chunk_id] = float(np.dot(q, d) / (q_norm * d_norm))

    return scores


def normalize_scores(scores_dict):
    """Min-max normalize a {id: score} dict to the 0.0-1.0 range."""
    if not scores_dict:
        return {}

    values = list(scores_dict.values())
    lo, hi = min(values), max(values)

    if hi == lo:
        # All scores identical (e.g. a single match) -- treat as fully relevant.
        return {k: 1.0 for k in scores_dict}

    return {k: (v - lo) / (hi - lo) for k, v in scores_dict.items()}


def blend_scores(bm25_scores, cosine_scores):
    """
    Combine normalized BM25 and cosine score dicts into a single ranked list
    using Final Score = (0.4 * BM25) + (0.6 * Cosine Similarity).
    Returns a list of (chunk_id, final_score) sorted descending.
    """
    bm25_norm = normalize_scores(bm25_scores)
    cosine_norm = normalize_scores(cosine_scores)

    all_ids = set(bm25_norm) | set(cosine_norm)
    blended = {}
    for chunk_id in all_ids:
        b = bm25_norm.get(chunk_id, 0.0)
        c = cosine_norm.get(chunk_id, 0.0)
        blended[chunk_id] = (BM25_WEIGHT * b) + (COSINE_WEIGHT * c)

    return sorted(blended.items(), key=lambda x: x[1], reverse=True)


def _fetch_chunk_records(chunk_ids, db_path=DB_PATH):
    """Fetch raw_text + source file_path for a list of chunk ids, preserving order."""
    if not chunk_ids:
        return []

    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT chunks.id, chunks.raw_text, documents.file_path
        FROM chunks JOIN documents ON chunks.document_id = documents.id
        WHERE chunks.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    conn.close()

    by_id = {row[0]: {"id": row[0], "raw_text": row[1], "file_path": row[2]} for row in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


def search(query, top_n=5, db_path=DB_PATH):
    """
    Main entry point used by app.py: run the keyword + vector branches,
    blend scores, and return the top_n ranked chunk records.

    Falls back to keyword-only (BM25) results if Ollama is unreachable,
    per the "Graceful Fallback" mitigation in cosi_design_doc.md Section 6.
    """
    query_tokens = tokenize_and_stem_query(query)
    bm25_scores = bm25_search(query_tokens, db_path=db_path)

    query_vector = get_embedding(query)
    cosine_scores = vector_search(query_vector, db_path=db_path) if query_vector else {}

    ranked = blend_scores(bm25_scores, cosine_scores)[:top_n]
    ranked_ids = [chunk_id for chunk_id, _score in ranked]
    score_by_id = dict(ranked)

    records = _fetch_chunk_records(ranked_ids, db_path=db_path)
    for record in records:
        record["score"] = round(score_by_id[record["id"]], 4)

    return records


if __name__ == "__main__":
    # Simple CLI test harness per Sprint 2 task list.
    user_query = input("cosi search> ")
    results = search(user_query)
    if not results:
        print("No results.")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']}] {r['file_path']}")
        print(f"   {r['raw_text'][:100]}...")
