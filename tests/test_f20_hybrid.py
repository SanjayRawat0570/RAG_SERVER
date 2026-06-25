"""Tests for F20: Hybrid Search — smart weighting, profiles, explain, compare."""
from __future__ import annotations

import pytest

from app.rag.embeddings import embed_texts
from app.rag.search.hybrid import hybrid_search, reciprocal_rank_fusion
from app.rag.search.weights import (
    PROFILES, auto_weights, classify_query, get_profile,
    normalize_to_unit, WeightProfile, DEFAULT_PROFILE,
)
from app.rag.vectorstore import VectorRecord, get_store, reset_stores


# ── Fixtures ───────────────────────────────────────────────────────────────────

CORPUS = [
    ("d1", "Revenue was $10 million in fiscal year 2024."),
    ("d2", "The company's income grew by 15 percent year over year."),
    ("d3", "Operating expenses increased due to higher headcount costs."),
    ("d4", "def calculate_revenue(sales, returns): return sales - returns"),
    ("d5", "Strategic overview: long-term growth and competitive positioning."),
]


@pytest.fixture(autouse=True)
def _store():
    reset_stores()
    store = get_store("hybrid-test", 256)
    vecs  = embed_texts([t for _, t in CORPUS], "local-hash", 256)
    for (doc_id, text), vec in zip(CORPUS, vecs):
        store.upsert([VectorRecord(id=doc_id, vector=vec,
                                   metadata={"text": text, "doc_id": doc_id})])
    yield
    reset_stores()


# ── WeightProfile model ────────────────────────────────────────────────────────

def test_f20_profiles_all_present():
    required = {"balanced", "semantic", "keyword", "technical", "conceptual", "equal"}
    assert required.issubset(PROFILES.keys())


def test_f20_profile_alphas_sum_to_one():
    for name, p in PROFILES.items():
        total = round(p.semantic_alpha + p.keyword_alpha, 6)
        assert total == 1.0, f"Profile '{name}' alphas don't sum to 1.0"


def test_f20_balanced_profile_weights():
    p = PROFILES["balanced"]
    assert p.semantic_alpha == 0.6
    assert p.keyword_alpha  == 0.4


def test_f20_keyword_profile_favours_keyword():
    p = PROFILES["keyword"]
    assert p.keyword_alpha > p.semantic_alpha


def test_f20_semantic_profile_favours_semantic():
    p = PROFILES["semantic"]
    assert p.semantic_alpha > p.keyword_alpha


def test_f20_technical_profile_favours_keyword():
    p = PROFILES["technical"]
    assert p.keyword_alpha > p.semantic_alpha


def test_f20_get_profile_known():
    p = get_profile("balanced")
    assert isinstance(p, WeightProfile)
    assert p.name == "balanced"


def test_f20_get_profile_unknown_falls_back():
    p = get_profile("nonexistent")
    assert p.name == DEFAULT_PROFILE


# ── normalize_to_unit ──────────────────────────────────────────────────────────

def test_f20_normalize_to_unit_basic():
    s, k = normalize_to_unit(3.0, 7.0)
    assert abs(s - 0.3) < 1e-9
    assert abs(k - 0.7) < 1e-9


def test_f20_normalize_to_unit_zero_input():
    s, k = normalize_to_unit(0.0, 0.0)
    assert s == 0.5 and k == 0.5


def test_f20_normalize_to_unit_sums_to_one():
    s, k = normalize_to_unit(2.0, 8.0)
    assert abs(s + k - 1.0) < 1e-9


# ── Query classifier ───────────────────────────────────────────────────────────

def test_f20_classify_technical_query():
    q = "def calculate_revenue(sales, returns)"
    assert classify_query(q) == "technical"


def test_f20_classify_conceptual_query():
    q = "Explain the fundamental principles of microservice architecture"
    assert classify_query(q) == "conceptual"


def test_f20_classify_analytical_query():
    q = "Compare revenue versus expenses between 2022 and 2024"
    qtype = classify_query(q)
    assert qtype in ("analytical", "factual", "general")


def test_f20_classify_factual_query():
    q = "What is the total revenue for 2024?"
    assert classify_query(q) in ("factual", "general")


def test_f20_classify_general_fallback():
    q = "revenue"
    assert classify_query(q) == "general"


def test_f20_auto_weights_returns_profile_and_type():
    profile, qtype = auto_weights("revenue growth")
    assert isinstance(profile, WeightProfile)
    assert isinstance(qtype, str)
    assert qtype in ("technical", "conceptual", "analytical", "factual", "general")


def test_f20_auto_weights_technical_uses_keyword_heavy_profile():
    profile, qtype = auto_weights("def calculate_revenue(sales)")
    assert qtype == "technical"
    assert profile.keyword_alpha >= profile.semantic_alpha


def test_f20_auto_weights_conceptual_uses_semantic_heavy_profile():
    profile, qtype = auto_weights("explain the conceptual framework for understanding revenue")
    assert profile.semantic_alpha >= profile.keyword_alpha


# ── RRF still works ────────────────────────────────────────────────────────────

def test_f20_rrf_merges_two_lists():
    a = [{"id": "x", "score": 0.9, "metadata": {}},
         {"id": "y", "score": 0.8, "metadata": {}}]
    b = [{"id": "y", "score": 0.7, "metadata": {}},
         {"id": "z", "score": 0.6, "metadata": {}}]
    merged = reciprocal_rank_fusion(a, b)
    ids = [h["id"] for h in merged]
    assert "x" in ids and "y" in ids and "z" in ids


def test_f20_rrf_boosts_documents_appearing_in_both():
    a = [{"id": "shared", "score": 0.5, "metadata": {}},
         {"id": "only-a",  "score": 0.4, "metadata": {}}]
    b = [{"id": "shared", "score": 0.5, "metadata": {}},
         {"id": "only-b",  "score": 0.4, "metadata": {}}]
    merged = reciprocal_rank_fusion(a, b)
    top = merged[0]["id"]
    assert top == "shared"


def test_f20_rrf_respects_weights():
    # If dense_weight >> sparse_weight, dense-only docs should rank higher.
    a = [{"id": "dense-top", "score": 0.9, "metadata": {}}]
    b = [{"id": "sparse-top", "score": 0.9, "metadata": {}}]
    merged_dense_heavy = reciprocal_rank_fusion(a, b, weights=[10.0, 1.0])
    assert merged_dense_heavy[0]["id"] == "dense-top"


# ── hybrid_search: weight_profile ────────────────────────────────────────────

def test_f20_hybrid_weight_profile_balanced():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3,
                           weight_profile="balanced")
    assert result["weights"]["profile"] == "balanced"
    assert abs(result["weights"]["dense"]  - 0.6) < 0.01
    assert abs(result["weights"]["sparse"] - 0.4) < 0.01


def test_f20_hybrid_weight_profile_keyword():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3,
                           weight_profile="keyword")
    assert result["weights"]["sparse"] > result["weights"]["dense"]


def test_f20_hybrid_weight_profile_semantic():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3,
                           weight_profile="semantic")
    assert result["weights"]["dense"] > result["weights"]["sparse"]


def test_f20_hybrid_auto_weight_returns_query_type():
    result = hybrid_search("def calculate_revenue()", store_name="hybrid-test",
                           namespace="default", top_k=3, auto_weight=True)
    assert "query_type" in result
    assert result["query_type"] == "technical"


def test_f20_hybrid_auto_weight_profile_name_in_weights():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3, auto_weight=True)
    assert result["weights"]["profile"] in PROFILES


def test_f20_hybrid_returns_weights_always():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3)
    assert "weights" in result
    assert "dense"   in result["weights"]
    assert "sparse"  in result["weights"]
    assert "profile" in result["weights"]


# ── explain mode ───────────────────────────────────────────────────────────────

def test_f20_hybrid_explain_adds_explain_field():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3, explain=True)
    for hit in result["hits"]:
        assert "explain" in hit
        assert "source" in hit["explain"]
        assert hit["explain"]["source"] in ("semantic", "keyword", "both")


def test_f20_hybrid_explain_has_weight_info():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3, explain=True,
                           weight_profile="balanced")
    for hit in result["hits"]:
        e = hit["explain"]
        assert "dense_weight"  in e
        assert "sparse_weight" in e
        assert e["weight_profile"] == "balanced"


def test_f20_hybrid_no_explain_field_by_default():
    result = hybrid_search("revenue", store_name="hybrid-test",
                           namespace="default", top_k=3, explain=False)
    for hit in result["hits"]:
        assert "explain" not in hit


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f20_api_list_profiles():
    with _client() as c:
        resp = c.get("/api/v1/search/profiles", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "profiles"  in data
    assert "balanced"  in data["profiles"]
    assert "semantic"  in data["profiles"]
    assert "keyword"   in data["profiles"]
    assert "technical" in data["profiles"]
    assert "default"   in data


def test_f20_api_classify():
    with _client() as c:
        resp = c.post("/api/v1/search/classify",
                      json={"query": "def calculate_revenue(sales)"},
                      headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_type"] == "technical"
    assert "recommended_profile" in data
    assert "weights"             in data


def test_f20_api_classify_missing_query():
    with _client() as c:
        resp = c.post("/api/v1/search/classify", json={}, headers=AUTH)
    assert resp.status_code == 422


def test_f20_api_hybrid_with_profile():
    with _client() as c:
        resp = c.post("/api/v1/search/hybrid", json={
            "query":          "revenue",
            "store":          "hybrid-test",
            "namespace":      "default",
            "top_k":          3,
            "weight_profile": "balanced",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["weights"]["profile"] == "balanced"


def test_f20_api_hybrid_auto_weight():
    with _client() as c:
        resp = c.post("/api/v1/search/hybrid", json={
            "query":       "revenue",
            "store":       "hybrid-test",
            "namespace":   "default",
            "top_k":       3,
            "auto_weight": True,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["weights"]["profile"] in PROFILES


def test_f20_api_hybrid_explain():
    with _client() as c:
        resp = c.post("/api/v1/search/hybrid", json={
            "query":     "revenue",
            "store":     "hybrid-test",
            "namespace": "default",
            "top_k":     3,
            "explain":   True,
        }, headers=AUTH)
    assert resp.status_code == 200
    for hit in resp.json()["hits"]:
        assert "explain" in hit
        assert "source"  in hit["explain"]


def test_f20_api_compare_profiles():
    with _client() as c:
        resp = c.post("/api/v1/search/compare", json={
            "query":     "revenue",
            "store":     "hybrid-test",
            "namespace": "default",
            "top_k":     3,
            "profiles":  ["balanced", "semantic", "keyword"],
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "results"  in data
    assert "overlap"  in data
    assert "summary"  in data
    assert "balanced" in data["results"]
    assert "semantic" in data["results"]
    assert "keyword"  in data["results"]


def test_f20_api_compare_invalid_profile():
    with _client() as c:
        resp = c.post("/api/v1/search/compare", json={
            "query":    "revenue",
            "store":    "hybrid-test",
            "profiles": ["balanced", "nonexistent"],
        }, headers=AUTH)
    assert resp.status_code == 422


def test_f20_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/search/profiles")
    assert resp.status_code == 401
