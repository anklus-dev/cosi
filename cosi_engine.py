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
           Final Score = (0.2 * BM25) + (0.8 * Cosine Similarity)
    4. Return the ranked chunks (source file, raw text, blended score, and the
       BM25 / cosine component scores) for consumption by app.py.

Graceful fallback (cosi_design_doc.md Section 6): Ollama is checked with a fast
ping (check_ollama) at startup and then at most every OLLAMA_STATUS_TTL seconds.
While it is down, the query is never sent for embedding -- so searches don't
stall waiting on a dead connection -- and results are ranked on BM25 alone.
The same fallback applies if no chunk has a stored embedding (Ollama was
offline during ingest).

Performance note: rebuilding the BM25 inverted index and re-reading every chunk's
embedding from SQLite on every single query doesn't scale past a small corpus.
_get_corpus() caches both structures in memory, keyed off the database file's
mtime, and only rebuilds when the DB has actually changed (i.e. after an ingest).
"""

import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request

import numpy as np
from nltk.stem import PorterStemmer

from config import EMBED_MODEL, OLLAMA_PING_TIMEOUT, OLLAMA_STATUS_TTL, OLLAMA_URL, TAGS_URL
from offline_ingest import DB_PATH, blob_to_vector, get_embedding

BM25_WEIGHT = 0.2
COSINE_WEIGHT = 0.8

# BM25 tuning parameters (matching local_search_engine.py's defaults)
BM25_B = 0.75
BM25_K1 = 1.5

# Relative-threshold ranking: rather than always returning a fixed top_n
# regardless of relevance, drop results whose blended score falls below
# RELATIVE_THRESHOLD_RATIO of the top result's score. This mirrors the
# design doc's similarity relation R (x R y if cosine similarity exceeds a
# threshold tau) applied to the *blended* score instead of raw cosine alone.
RELATIVE_THRESHOLD_RATIO = 0.5
MAX_RESULTS_CAP = 20

_stemmer = PorterStemmer()

# {db_path: {"mtime": float, "index": ..., "doc_lengths": ..., "avg_doc_length": ...,
#            "N": ..., "embedding_ids": [...], "embedding_matrix": np.ndarray | None}}
_corpus_cache = {}

# Cached result of the last Ollama health ping.
_ollama_status = {"ok": False, "reason": "not checked yet", "checked_at": None}


# ---------------------------------------------------------------------------
# Ollama health check (Graceful Fallback)
# ---------------------------------------------------------------------------

def _ping_ollama():
    """
    One fast request to Ollama's /api/tags. Confirms the server is reachable
    AND that the embedding model has been pulled. Returns (ok, reason).
    """
    try:
        with urllib.request.urlopen(TAGS_URL, timeout=OLLAMA_PING_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as e:  # URLError/timeouts are OSErrors; bad JSON is ValueError
        return False, f"Ollama unreachable at {OLLAMA_URL} ({e})"

    names = {m.get("name", "") for m in payload.get("models", [])}
    if not any(n == EMBED_MODEL or n.startswith(EMBED_MODEL + ":") for n in names):
        return False, f"model '{EMBED_MODEL}' is not pulled (run: ollama pull {EMBED_MODEL})"

    return True, f"Ollama reachable at {OLLAMA_URL} with '{EMBED_MODEL}'"


def check_ollama(force=False):
    """
    Return the cached Ollama status, re-pinging if forced or if the cached
    result is older than OLLAMA_STATUS_TTL. Lets search recover automatically
    once Ollama comes back up, without pinging on every keystroke.
    """
    checked_at = _ollama_status["checked_at"]
    stale = checked_at is None or (time.monotonic() - checked_at) > OLLAMA_STATUS_TTL
    if force or stale:
        ok, reason = _ping_ollama()
        _ollama_status.update(ok=ok, reason=reason, checked_at=time.monotonic())
    return {"ok": _ollama_status["ok"], "reason": _ollama_status["reason"]}


def _mark_ollama_down(reason):
    """Record a failure seen mid-search so later queries skip Ollama until the next re-check."""
    _ollama_status.update(ok=False, reason=reason, checked_at=time.monotonic())


# ---------------------------------------------------------------------------
# Corpus loading & caching
# ---------------------------------------------------------------------------

def tokenize_and_stem_query(query):
    """Lowercase, strip punctuation, tokenize, and stem the raw query string."""
    text = query.lower()
    words = re.findall(r"\b\w+\b", text)
    return [_stemmer.stem(w) for w in words]


def _empty_corpus():
    return {
        "index": {},
        "doc_lengths": {},
        "avg_doc_length": 0.0,
        "N": 0,
        "embedding_ids": [],
        "embedding_matrix": None,
    }


def _build_corpus(db_path):
    """
    Read every chunk once and build both the BM25 inverted index and a
    stacked NumPy matrix of embeddings, so a single DB read serves both
    search branches.

    A fresh container (or a deleted data/ folder) has no database or no
    `chunks` table yet; that returns an empty corpus instead of crashing,
    so the UI still loads and the user can click Refresh Index.
    """
    if not os.path.exists(db_path):
        return _empty_corpus()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, stemmed_tokens, embedding_vector FROM chunks").fetchall()
    except sqlite3.OperationalError:
        return _empty_corpus()
    finally:
        conn.close()

    index = {}          # {stem: {chunk_id: term_frequency}}
    doc_lengths = {}     # {chunk_id: token_count}
    embedding_ids = []
    embedding_rows = []

    for chunk_id, stemmed_tokens, blob in rows:
        tokens = (stemmed_tokens or "").split()
        doc_lengths[chunk_id] = len(tokens)
        for token in tokens:
            index.setdefault(token, {})
            index[token][chunk_id] = index[token].get(chunk_id, 0) + 1

        vector = blob_to_vector(blob)
        if vector is not None:
            embedding_ids.append(chunk_id)
            embedding_rows.append(vector)

    avg_doc_length = (sum(doc_lengths.values()) / len(doc_lengths)) if doc_lengths else 0.0
    embedding_matrix = np.array(embedding_rows, dtype=np.float32) if embedding_rows else None

    return {
        "index": index,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "N": len(doc_lengths),
        "embedding_ids": embedding_ids,
        "embedding_matrix": embedding_matrix,
    }


def _get_corpus(db_path):
    """
    Return the cached corpus structures for db_path, rebuilding only if the
    database file's mtime has changed since the last build (i.e. ingest ran).
    """
    try:
        mtime = os.path.getmtime(db_path)
    except OSError:
        mtime = None

    cached = _corpus_cache.get(db_path)
    if cached is not None and cached["mtime"] == mtime:
        return cached

    corpus = _build_corpus(db_path)
    corpus["mtime"] = mtime
    _corpus_cache[db_path] = corpus
    return corpus


def corpus_stats(db_path=DB_PATH):
    """Small summary for the /api/status endpoint."""
    corpus = _get_corpus(db_path)
    return {"chunks": corpus["N"], "chunks_with_embeddings": len(corpus["embedding_ids"])}


# ---------------------------------------------------------------------------
# Scoring branches
# ---------------------------------------------------------------------------

def bm25_search(query_tokens, db_path=DB_PATH):
    """
    Score every chunk against query_tokens using Okapi BM25, reading the
    inverted index from the (mtime-invalidated) in-memory cache.
    Returns {chunk_id: bm25_score}.
    """
    corpus = _get_corpus(db_path)
    index = corpus["index"]
    doc_lengths = corpus["doc_lengths"]
    avg_doc_length = corpus["avg_doc_length"]
    N = corpus["N"]

    if N == 0 or avg_doc_length == 0:
        return {}

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
    Compute cosine similarity between query_vector and every cached chunk
    embedding using a single vectorized NumPy operation against the cached
    embedding matrix, instead of a per-chunk Python loop.
    Returns {chunk_id: cosine_score}.
    """
    corpus = _get_corpus(db_path)
    embedding_matrix = corpus["embedding_matrix"]
    embedding_ids = corpus["embedding_ids"]

    if embedding_matrix is None or query_vector is None:
        return {}

    q = np.array(query_vector, dtype=np.float32)
    if q.shape[0] != embedding_matrix.shape[1]:
        # e.g. the model was changed since the index was built -- vectors aren't comparable.
        return {}

    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return {}

    d_norms = np.linalg.norm(embedding_matrix, axis=1)
    valid = d_norms != 0
    similarities = np.zeros(len(embedding_ids), dtype=np.float32)
    similarities[valid] = (embedding_matrix[valid] @ q) / (d_norms[valid] * q_norm)

    return {chunk_id: float(sim) for chunk_id, sim in zip(embedding_ids, similarities)}


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
    using Final Score = (0.2 * BM25) + (0.8 * Cosine Similarity).
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


def _apply_relative_threshold(ranked, ratio=RELATIVE_THRESHOLD_RATIO, hard_cap=MAX_RESULTS_CAP):
    """
    Given a descending-sorted [(chunk_id, score), ...] list, keep only
    results within `ratio` of the top score, up to `hard_cap` results.
    This is the tau-style cutoff from the design doc's similarity relation R,
    applied to the blended score rather than raw cosine similarity alone.
    """
    if not ranked:
        return ranked

    top_score = ranked[0][1]
    if top_score <= 0:
        # Nothing meaningfully matched; don't let a 0.5 * 0 threshold pass everything.
        return ranked[:hard_cap]

    cutoff = top_score * ratio
    filtered = [(cid, score) for cid, score in ranked if score >= cutoff]
    return filtered[:hard_cap]


def _fetch_chunk_records(chunk_ids, db_path=DB_PATH):
    """Fetch raw_text + source file_path for a list of chunk ids, preserving order."""
    if not chunk_ids:
        return []

    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"""
            SELECT chunks.id, chunks.raw_text, documents.file_path
            FROM chunks JOIN documents ON chunks.document_id = documents.id
            WHERE chunks.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()

    by_id = {row[0]: {"id": row[0], "raw_text": row[1], "file_path": row[2]} for row in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search(query, top_n=MAX_RESULTS_CAP, db_path=DB_PATH,
           relative_threshold=RELATIVE_THRESHOLD_RATIO):
    """
    Main entry point used by app.py: run the keyword + vector branches,
    blend scores, apply the relative-threshold cutoff, and return the
    ranked chunk records.

    Each record contains:
        id, raw_text, file_path
        score          blended final score (0.2 * bm25_score + 0.8 * cosine_score)
        bm25_score     normalized [0, 1] keyword component
        cosine_score   normalized [0, 1] vector component
        bm25_raw       raw Okapi BM25 score (0.0 if no keyword overlap)
        cosine_raw     raw cosine similarity (None in keyword-only mode)
        mode           "hybrid" or "keyword" (keyword = Ollama fallback)

    Falls back to keyword-only (BM25) results if Ollama is unreachable,
    per the "Graceful Fallback" mitigation in cosi_design_doc.md Section 6.
    """
    query_tokens = tokenize_and_stem_query(query)
    bm25_scores = bm25_search(query_tokens, db_path=db_path)

    query_vector = None
    if check_ollama()["ok"]:
        query_vector = get_embedding(query)
        if not query_vector:
            _mark_ollama_down("embedding request failed during search")
            query_vector = None

    cosine_scores = vector_search(query_vector, db_path=db_path) if query_vector else {}
    mode = "hybrid" if cosine_scores else "keyword"

    ranked = blend_scores(bm25_scores, cosine_scores)
    ranked = _apply_relative_threshold(ranked, ratio=relative_threshold, hard_cap=top_n)

    ranked_ids = [chunk_id for chunk_id, _score in ranked]
    score_by_id = dict(ranked)
    bm25_norm = normalize_scores(bm25_scores)
    cosine_norm = normalize_scores(cosine_scores)

    records = _fetch_chunk_records(ranked_ids, db_path=db_path)
    for record in records:
        cid = record["id"]
        record["score"] = round(score_by_id[cid], 4)
        record["bm25_score"] = round(bm25_norm.get(cid, 0.0), 4)
        record["cosine_score"] = round(cosine_norm.get(cid, 0.0), 4)
        record["bm25_raw"] = round(bm25_scores.get(cid, 0.0), 4)
        record["cosine_raw"] = round(cosine_scores[cid], 4) if cid in cosine_scores else None
        record["mode"] = mode

    return records


if __name__ == "__main__":
    # Simple CLI test harness per Sprint 2 task list.
    status = check_ollama(force=True)
    print(f"[cosi] {'hybrid' if status['ok'] else 'keyword-only'} mode: {status['reason']}")
    user_query = input("cosi search> ")
    results = search(user_query)
    if not results:
        print("No results.")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']}  bm25={r['bm25_score']}  cos={r['cosine_score']}] {r['file_path']}")
        print(f"   {r['raw_text'][:100]}...")
