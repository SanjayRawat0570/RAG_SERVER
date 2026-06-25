"""Tests for F13: Semantic Search."""
from __future__ import annotations

import pytest

from app.rag.query import process_query
from app.rag.search import BM25, tokenize, extract_highlight, reciprocal_rank_fusion
from app.rag.search.semantic import semantic_search
from app.rag.search.hybrid import hybrid_search
from app.rag.vectorstore import InMemoryVectorStore, VectorRecord, reset_stores, get_store


# ── Fixtures ───────────────────────────────────────────────────────────────────

CORPUS = [
    ("doc-profit",   "Our quarterly profit was $2.5 million after tax deductions."),
    ("doc-revenue",  "Total revenue reached $10 million driven by product sales."),
    ("doc-expenses", "Operating expenses rose 15 percent due to headcount growth."),
    ("doc-ceo",      "The CEO gave a keynote speech at the annual conference."),
    ("doc-office",   "The new office location is downtown near the train station."),
]


@pytest.fixture(autouse=True)
def _clean():
    reset_stores()
    yield
    reset_stores()


def _populated_store(name: str = "corpus", embed_model: str = "local-hash",
                     dim: int = 256) -> InMemoryVectorStore:
    from app.rag.embeddings import embed_texts
    store = get_store(name, dim)
    texts = [text for _, text in CORPUS]
    vectors = embed_texts(texts, embed_model, dim)
    records = [
        VectorRecord(
            id=doc_id,
            vector=vec,
            metadata={"text": text, "doc_id": doc_id},
        )
        for (doc_id, text), vec in zip(CORPUS, vectors)
    ]
    store.upsert(records)
    return store


# ── BM25 unit tests ────────────────────────────────────────────────────────────

def test_f13_bm25_ranks_relevant_first():
    corpus = [tokenize(text) for _, text in CORPUS]
    bm25 = BM25(corpus)
    scores = bm25.scores(tokenize("profit quarterly"))
    best = scores.index(max(scores))
    assert CORPUS[best][0] == "doc-profit"


def test_f13_bm25_zero_score_for_missing_term():
    corpus = [tokenize("apple banana cherry")]
    bm25 = BM25(corpus)
    assert bm25.scores(tokenize("zxqw"))[0] == 0.0


def test_f13_bm25_empty_corpus():
    bm25 = BM25([])
    assert bm25.scores(tokenize("profit")) == []


def test_f13_bm25_idf_penalizes_common_terms():
    corpus = [["the", "cat"], ["the", "dog"], ["a", "fish"]]
    bm25 = BM25(corpus)
    # "the" appears in 2/3 docs, so lower IDF than "fish" (1/3 docs)
    assert bm25.idf.get("the", 0) < bm25.idf.get("fish", 0)


# ── Query processor ────────────────────────────────────────────────────────────

def test_f13_process_query_question_intent():
    result = process_query("What is our profit?")
    assert result["intent"] == "question"


def test_f13_process_query_keyword_intent():
    result = process_query("profit margin 2024")
    assert result["intent"] == "keyword"


def test_f13_process_query_extracts_entities():
    result = process_query("Revenue from Apple Inc in 2024")
    assert "Apple" in result["entities"]["capitalized"]
    assert "2024" in result["entities"]["numbers"]


def test_f13_process_query_keyword_extraction():
    result = process_query("What is the total revenue for the year?")
    assert "revenue" in result["keywords"]
    assert "the" not in result["keywords"]  # stopword removed


def test_f13_process_query_synonym_expansion():
    result = process_query("profit", synonyms={"profit": ["net income", "earnings"]})
    # expansion contains whole phrases as returned by the synonym map
    assert "net income" in result["expansion"] or "earnings" in result["expansion"]


def test_f13_process_query_expanded_query_includes_synonyms():
    result = process_query("profit", synonyms={"profit": ["income"]})
    assert "income" in result["expanded_query"]


# ── Highlight extraction ───────────────────────────────────────────────────────

def test_f13_highlight_returns_relevant_sentence():
    text = "The CEO gave a speech. Our profit grew by 20 percent this quarter. The office is closed."
    highlight = extract_highlight(text, "What is the profit?")
    assert "profit" in highlight.lower()


def test_f13_highlight_max_length():
    text = "word " * 200
    highlight = extract_highlight(text, "word", max_len=100)
    assert len(highlight) <= 100


def test_f13_highlight_empty_text():
    assert extract_highlight("", "query") == ""


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_f13_rrf_merges_two_lists():
    dense  = [{"id": "a", "score": 0.9, "metadata": {}},
              {"id": "b", "score": 0.8, "metadata": {}},
              {"id": "c", "score": 0.7, "metadata": {}}]
    sparse = [{"id": "b", "score": 5.2, "metadata": {}},
              {"id": "d", "score": 4.1, "metadata": {}},
              {"id": "a", "score": 3.0, "metadata": {}}]
    merged = reciprocal_rank_fusion(dense, sparse)
    ids = [h["id"] for h in merged]
    # "a" and "b" appear in both lists → should rank high
    assert ids[0] in ("a", "b")
    assert "c" in ids
    assert "d" in ids


def test_f13_rrf_scores_sorted_descending():
    dense  = [{"id": str(i), "score": float(i), "metadata": {}} for i in range(5)]
    sparse = [{"id": str(i), "score": float(i), "metadata": {}} for i in range(5)]
    merged = reciprocal_rank_fusion(dense, sparse)
    scores = [h["score"] for h in merged]
    assert scores == sorted(scores, reverse=True)


def test_f13_rrf_empty_lists():
    merged = reciprocal_rank_fusion([], [])
    assert merged == []


def test_f13_rrf_single_list():
    hits = [{"id": "x", "score": 1.0, "metadata": {}},
            {"id": "y", "score": 0.5, "metadata": {}}]
    merged = reciprocal_rank_fusion(hits)
    assert [h["id"] for h in merged] == ["x", "y"]


def test_f13_rrf_weights():
    dense  = [{"id": "a", "score": 0.9, "metadata": {}}]
    sparse = [{"id": "b", "score": 9.9, "metadata": {}}]
    # Give dense 2x weight → "a" should beat "b"
    merged = reciprocal_rank_fusion(dense, sparse, weights=[2.0, 1.0])
    assert merged[0]["id"] == "a"


# ── Semantic search ────────────────────────────────────────────────────────────

def test_f13_semantic_search_returns_hits():
    _populated_store()
    result = semantic_search("What is our profit?", store_name="corpus", top_k=3)
    assert result["total"] == 3
    assert len(result["hits"]) == 3


def test_f13_semantic_search_hits_have_required_fields():
    _populated_store()
    result = semantic_search("profit", store_name="corpus", top_k=2)
    for h in result["hits"]:
        assert "id" in h
        assert "score" in h
        assert "metadata" in h
        assert "highlight" in h


def test_f13_semantic_search_includes_query_analysis():
    _populated_store()
    result = semantic_search("What is profit?", store_name="corpus")
    qa = result["query_analysis"]
    assert "intent" in qa
    assert "keywords" in qa
    assert "expanded_query" in qa


def test_f13_semantic_search_metadata_filter():
    from app.rag.embeddings import embed_texts
    store = get_store("filter-store", 256)
    vecs = embed_texts([t for _, t in CORPUS], "local-hash", 256)
    for (doc_id, text), vec in zip(CORPUS, vecs):
        group = "finance" if doc_id in ("doc-profit", "doc-revenue", "doc-expenses") else "other"
        store.upsert([VectorRecord(id=doc_id, vector=vec,
                                   metadata={"text": text, "group": group})])
    result = semantic_search("profit", store_name="filter-store", top_k=5,
                             filters={"group": "finance"})
    assert all(h["metadata"]["group"] == "finance" for h in result["hits"])


def test_f13_semantic_search_unknown_store():
    result = semantic_search("profit", store_name="no-such-store")
    assert result["total"] == 0
    assert "error" in result


def test_f13_semantic_search_with_rerank():
    _populated_store()
    result = semantic_search("profit", store_name="corpus", top_k=3,
                             rerank_method="cross_encoder")
    assert result["total"] <= 3
    for h in result["hits"]:
        assert "score" in h


def test_f13_semantic_search_scores_sorted():
    _populated_store()
    result = semantic_search("profit revenue expenses", store_name="corpus", top_k=5)
    scores = [h["score"] for h in result["hits"]]
    assert scores == sorted(scores, reverse=True)


# ── Hybrid search ──────────────────────────────────────────────────────────────

def test_f13_hybrid_search_returns_hits():
    _populated_store()
    result = hybrid_search("profit quarterly", store_name="corpus", top_k=3)
    assert len(result["hits"]) <= 3


def test_f13_hybrid_search_reports_dense_sparse_counts():
    _populated_store()
    result = hybrid_search("profit", store_name="corpus", top_k=5)
    assert "dense_count" in result
    assert "sparse_count" in result


def test_f13_hybrid_no_duplicate_ids():
    _populated_store()
    result = hybrid_search("revenue profit", store_name="corpus", top_k=5)
    ids = [h["id"] for h in result["hits"]]
    assert len(ids) == len(set(ids))


# ── API tests ─────────────────────────────────────────────────────────────────
# Each API test creates its own store so the autouse `_clean` fixture (which
# runs reset_stores before every function) doesn't invalidate shared state.

def _make_api_store(name: str = "api-corpus") -> None:
    """Populate a named store for API tests."""
    from app.rag.embeddings import embed_texts
    store = get_store(name, 256)
    vecs = embed_texts([t for _, t in CORPUS], "local-hash", 256)
    for (doc_id, text), vec in zip(CORPUS, vecs):
        store.upsert([VectorRecord(id=doc_id, vector=vec,
                                   metadata={"text": text, "doc_id": doc_id})])


@pytest.mark.asyncio
async def test_f13_api_strategies():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/search/strategies",
                          headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "semantic" in data["search_modes"]
    assert "keyword" in data["search_modes"]
    assert "hybrid" in data["search_modes"]
    assert "cross_encoder" in data["rerank_strategies"]


@pytest.mark.asyncio
async def test_f13_api_semantic():
    _make_api_store()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/semantic", json={
            "query": "What is our profit?",
            "store": "api-corpus",
            "top_k": 3,
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert all("highlight" in h for h in data["hits"])


@pytest.mark.asyncio
async def test_f13_api_semantic_with_rerank():
    _make_api_store()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/semantic", json={
            "query": "profit",
            "store": "api-corpus",
            "top_k": 3,
            "rerank": "cross_encoder",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    assert resp.json()["total"] <= 3


@pytest.mark.asyncio
async def test_f13_api_semantic_bad_rerank():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/semantic", json={
            "query": "profit", "store": "api-corpus", "rerank": "no_such_method",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_f13_api_keyword():
    _make_api_store()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/keyword", json={
            "query": "profit quarterly",
            "store": "api-corpus",
            "top_k": 3,
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["hits"], list)
    if data["hits"]:
        assert data["hits"][0]["id"] == "doc-profit"


@pytest.mark.asyncio
async def test_f13_api_hybrid():
    _make_api_store()
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/hybrid", json={
            "query": "profit revenue",
            "store": "api-corpus",
            "top_k": 3,
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "dense_count" in data
    assert "sparse_count" in data
    assert len(data["hits"]) <= 3


@pytest.mark.asyncio
async def test_f13_api_rerank():
    from fastapi.testclient import TestClient
    from app.main import app
    hits = [
        {"id": f"d{i}", "score": float(i) / 10,
         "metadata": {"text": f"Document {i} about profit and revenue"}}
        for i in range(4)
    ]
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/rerank", json={
            "query":  "profit",
            "hits":   hits,
            "method": "cross_encoder",
            "top_n":  3,
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["method"] == "cross_encoder"


@pytest.mark.asyncio
async def test_f13_api_rerank_mmr():
    from fastapi.testclient import TestClient
    from app.main import app
    hits = [
        {"id": f"d{i}", "score": 0.9,
         "metadata": {"text": "profit and revenue growth"}}
        for i in range(5)
    ]
    with TestClient(app) as client:
        resp = client.post("/api/v1/search/rerank", json={
            "query":  "profit",
            "hits":   hits,
            "method": "mmr",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    assert resp.json()["total"] == 5
