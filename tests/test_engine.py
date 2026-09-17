"""
tests/test_engine.py

Sprint 2 task: "Create a command-line interface test script to input terms
and verify ranking." This is the pytest-style counterpart to cosi_engine.py's
__main__ CLI block -- use it to sanity-check BM25 scoring, cosine similarity,
and the blended ranker against small known inputs.

Currently placeholder stubs; fill in once cosi_engine.py functions are implemented.
"""

# import cosi_engine


def test_bm25_search_returns_expected_top_match():
    # TODO: build a tiny in-memory/fixture DB with known term frequencies,
    #       assert bm25_search() ranks the obviously-relevant doc first.
    pass


def test_vector_search_cosine_similarity():
    # TODO: feed hand-crafted vectors (e.g. identical vector -> similarity ~1.0,
    #       orthogonal vectors -> similarity ~0.0) and check vector_search() math.
    pass


def test_blend_scores_weights():
    # TODO: verify Final Score = (0.4 * BM25) + (0.6 * Cosine) with normalized inputs.
    pass
