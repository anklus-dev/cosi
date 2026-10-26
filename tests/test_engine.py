"""
tests/test_engine.py

Unit tests for the search math in cosi_engine.py and the chunker in offline_ingest.py.
Each test builds a tiny throwaway SQLite index with hand-picked text and vectors, so the
expected answer is known in advance. No Ollama needed.

Run from the project root (inside `nix develop`):
    python -m pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cosi_engine  # noqa: E402
import offline_ingest  # noqa: E402


def make_index(tmp_path, chunks):
    """chunks: list of (file, text, vector-or-None). Returns the db path."""
    db_path = str(tmp_path / "test.db")
    conn = offline_ingest.init_db(db_path)
    doc_ids = {}
    for file, text, vector in chunks:
        if file not in doc_ids:
            doc_ids[file] = conn.execute(
                "INSERT INTO documents (file_path, file_hash, last_modified) VALUES (?, 'h', 0)", (file,)
            ).lastrowid
        conn.execute(
            "INSERT INTO chunks (document_id, raw_text, stemmed_tokens, embedding_vector) VALUES (?, ?, ?, ?)",
            (doc_ids[file], text, offline_ingest.stem_chunk(text), offline_ingest._vector_to_blob(vector)),
        )
    conn.commit()
    conn.close()
    cosi_engine._corpus_cache.clear()
    return db_path


@pytest.fixture
def ollama_down(monkeypatch):
    monkeypatch.setattr(cosi_engine, "_ping_ollama", lambda: (False, "test: offline"))
    cosi_engine._ollama_status.update(ok=False, reason="test", checked_at=None)


# ---------------------------------------------------------------------------
# Keyword branch (BM25) and stemming
# ---------------------------------------------------------------------------

def test_bm25_search_returns_expected_top_match(tmp_path):
    db = make_index(tmp_path, [
        ("a.md", "The tram climbs the steep hill above the river.", None),
        ("b.md", "Hawker centers serve chicken rice and laksa.", None),
        ("c.md", "Cherry blossoms bloom in early April.", None),
    ])
    scores = cosi_engine.bm25_search(cosi_engine.tokenize_and_stem_query("chicken rice"), db_path=db)
    assert max(scores, key=scores.get) == 2  # chunk ids start at 1


def test_stemming_puts_word_forms_in_one_equivalence_class():
    stems = {offline_ingest._stemmer.stem(w) for w in ["connecting", "connection", "connects", "connected"]}
    assert stems == {"connect"}


def test_bm25_rewards_rarer_terms(tmp_path):
    # "temple" appears in one chunk, "city" in all three: the rare word should matter more (IDF).
    db = make_index(tmp_path, [
        ("a.md", "The city has a famous temple.", None),
        ("b.md", "The city has a busy market.", None),
        ("c.md", "The city has a long river.", None),
    ])
    rare = cosi_engine.bm25_search(["templ"], db_path=db)
    common = cosi_engine.bm25_search(["citi"], db_path=db)
    assert rare[1] > common[1]


# ---------------------------------------------------------------------------
# Vector branch (cosine similarity)
# ---------------------------------------------------------------------------

def test_vector_search_cosine_similarity(tmp_path):
    db = make_index(tmp_path, [
        ("a.md", "same direction", [1.0, 0.0, 0.0]),
        ("b.md", "orthogonal", [0.0, 1.0, 0.0]),
        ("c.md", "opposite", [-1.0, 0.0, 0.0]),
        ("d.md", "scaled copy", [5.0, 0.0, 0.0]),
    ])
    sims = cosi_engine.vector_search([2.0, 0.0, 0.0], db_path=db)
    assert sims[1] == pytest.approx(1.0)
    assert sims[2] == pytest.approx(0.0)
    assert sims[3] == pytest.approx(-1.0)
    assert sims[4] == pytest.approx(1.0)  # cosine ignores length, only the angle matters


def test_vector_search_ignores_mismatched_dimensions(tmp_path):
    db = make_index(tmp_path, [("a.md", "x", [1.0, 0.0, 0.0])])
    assert cosi_engine.vector_search([1.0, 0.0], db_path=db) == {}


# ---------------------------------------------------------------------------
# Normalization, blending and the relative threshold
# ---------------------------------------------------------------------------

def test_normalize_scores_min_max():
    assert cosi_engine.normalize_scores({1: 2.0, 2: 4.0, 3: 6.0}) == {1: 0.0, 2: 0.5, 3: 1.0}
    assert cosi_engine.normalize_scores({1: 3.0}) == {1: 1.0}
    assert cosi_engine.normalize_scores({}) == {}


def test_blend_scores_weights():
    # Chunk 1 is the best keyword match, chunk 2 the best vector match.
    ranked = dict(cosi_engine.blend_scores({1: 10.0, 2: 0.0}, {1: 0.0, 2: 0.9}))
    assert ranked[1] == pytest.approx(cosi_engine.BM25_WEIGHT)    # 0.4 * 1 + 0.6 * 0
    assert ranked[2] == pytest.approx(cosi_engine.COSINE_WEIGHT)  # 0.4 * 0 + 0.6 * 1
    assert cosi_engine.BM25_WEIGHT + cosi_engine.COSINE_WEIGHT == pytest.approx(1.0)


def test_relative_threshold_drops_weak_results():
    ranked = [(1, 0.9), (2, 0.5), (3, 0.44), (4, 0.1)]
    kept = cosi_engine._apply_relative_threshold(ranked, ratio=0.5)
    assert [cid for cid, _ in kept] == [1, 2]  # 0.44 < 0.5 * 0.9


# ---------------------------------------------------------------------------
# End-to-end search and graceful fallback
# ---------------------------------------------------------------------------

def test_search_falls_back_to_keyword_mode_without_ollama(tmp_path, ollama_down):
    db = make_index(tmp_path, [
        ("kyoto.md", "Fushimi Inari has thousands of torii gates.", [1.0, 0.0]),
        ("lisbon.md", "The number 28 tram climbs narrow streets.", [0.0, 1.0]),
    ])
    results = cosi_engine.search("torii gates", db_path=db)
    assert results[0]["file_path"] == "kyoto.md"
    assert results[0]["mode"] == "keyword"
    assert results[0]["cosine_raw"] is None


def test_search_uses_both_branches_when_ollama_is_up(tmp_path, monkeypatch):
    db = make_index(tmp_path, [
        ("kyoto.md", "Fushimi Inari has thousands of torii gates.", [1.0, 0.0]),
        ("reykjavik.md", "Winter brings the northern lights.", [0.0, 1.0]),
    ])
    monkeypatch.setattr(cosi_engine, "_ping_ollama", lambda: (True, "test: up"))
    cosi_engine._ollama_status.update(checked_at=None)
    monkeypatch.setattr(cosi_engine, "get_embedding", lambda text: [0.1, 1.0])  # "means" aurora
    results = cosi_engine.search("aurora borealis", db_path=db)  # no shared words with either chunk
    assert results[0]["file_path"] == "reykjavik.md"
    assert results[0]["mode"] == "hybrid"


def test_search_on_missing_index_returns_nothing(tmp_path, ollama_down):
    assert cosi_engine.search("anything", db_path=str(tmp_path / "missing.db")) == []


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

LONG_PARAGRAPH = " ".join(f"This is sentence number {i}, which says something about topic {i}." for i in range(40))


def test_sentence_chunker_never_cuts_mid_sentence():
    chunks = offline_ingest.chunk_text(LONG_PARAGRAPH, mode="sentence")
    assert len(chunks) > 1
    assert all(c.rstrip().endswith(".") for c in chunks)
    assert all(len(c) <= offline_ingest.MAX_CHUNK_CHARS for c in chunks)


def test_hard_chunker_cuts_at_fixed_width():
    chunks = offline_ingest.chunk_text(LONG_PARAGRAPH, mode="hard")
    assert len(chunks[0]) == offline_ingest.MAX_CHUNK_CHARS
    assert not chunks[0].endswith(".")  # the problem the sentence chunker fixes


def test_chunker_merges_short_paragraphs_and_keeps_headings_attached():
    text = "# Title\n\nShort one.\n\nAnother short one.\n\n" + "A longer paragraph. " * 12
    chunks = offline_ingest.chunk_text(text, mode="sentence")
    assert chunks[0].startswith("# Title")
    assert all(len(c) >= 75 for c in chunks)
