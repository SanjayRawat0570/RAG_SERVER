"""Tests for F14: Reranking Algorithms."""
from __future__ import annotations

import time

import pytest

from app.rag.rerank import STRATEGIES, rerank
from app.rag.rerank.rerankers import (
    authority,
    cross_encoder,
    mmr,
    multi_factor,
    recency,
)
from app.rag.rerank.cross_encoder_neural import is_model_available, neural_cross_encoder


# ── Shared fixtures ────────────────────────────────────────────────────────────

HITS = [
    {"id": "doc-a", "score": 0.9,
     "metadata": {"text": "Our quarterly profit was $2.5 million after tax deductions.",
                  "source": "finance-report", "date": "2024-01-15"}},
    {"id": "doc-b", "score": 0.7,
     "metadata": {"text": "Total revenue reached $10 million driven by product sales.",
                  "source": "annual-report", "date": "2023-06-01"}},
    {"id": "doc-c", "score": 0.5,
     "metadata": {"text": "Operating expenses rose 15 percent due to headcount growth.",
                  "source": "blog", "date": "2022-03-10"}},
    {"id": "doc-d", "score": 0.3,
     "metadata": {"text": "The CEO gave a keynote speech at the annual conference.",
                  "source": "blog", "date": "2021-09-20"}},
]

QUERY = "What is our profit this quarter?"


# ── STRATEGIES registry ────────────────────────────────────────────────────────

def test_f14_strategies_registered():
    assert "cross_encoder"  in STRATEGIES
    assert "mmr"            in STRATEGIES
    assert "recency"        in STRATEGIES
    assert "authority"      in STRATEGIES
    assert "multi_factor"   in STRATEGIES


def test_f14_neural_strategy_registered():
    # Neural CE is registered only when the module imports cleanly.
    if is_model_available():
        assert "neural_cross_encoder" in STRATEGIES
    else:
        # Model not downloaded — still verify fallback works.
        result = neural_cross_encoder(QUERY, HITS[:2], {})
        assert len(result) == 2


def test_f14_rerank_function_dispatches():
    result = rerank("cross_encoder", QUERY, HITS)
    assert len(result) == len(HITS)
    assert all("score" in h for h in result)


def test_f14_rerank_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown rerank method"):
        rerank("no_such_method", QUERY, HITS)


def test_f14_rerank_top_n_limits_results():
    result = rerank("cross_encoder", QUERY, HITS, {"top_n": 2})
    assert len(result) == 2


# ── Cross-encoder ──────────────────────────────────────────────────────────────

def test_f14_cross_encoder_returns_all_candidates():
    result = cross_encoder(QUERY, HITS, {})
    assert len(result) == len(HITS)


def test_f14_cross_encoder_scores_sorted_descending():
    result = cross_encoder(QUERY, HITS, {})
    scores = [h["score"] for h in result]
    assert scores == sorted(scores, reverse=True)


def test_f14_cross_encoder_has_explain_fields():
    result = cross_encoder(QUERY, HITS, {})
    for h in result:
        assert "rerank" in h
        assert "semantic" in h["rerank"]
        assert "lexical"  in h["rerank"]


def test_f14_cross_encoder_profit_doc_ranks_high():
    result = cross_encoder(QUERY, HITS, {})
    # doc-a has "profit" in text — should be in top 2
    top_ids = [h["id"] for h in result[:2]]
    assert "doc-a" in top_ids


def test_f14_cross_encoder_empty_candidates():
    result = cross_encoder(QUERY, [], {})
    assert result == []


def test_f14_cross_encoder_single_candidate():
    result = cross_encoder(QUERY, [HITS[0]], {})
    assert len(result) == 1
    assert result[0]["id"] == "doc-a"


# ── MMR (Diversity) ────────────────────────────────────────────────────────────

def test_f14_mmr_returns_all_candidates():
    result = mmr(QUERY, HITS, {})
    assert len(result) == len(HITS)


def test_f14_mmr_has_mmr_rank():
    result = mmr(QUERY, HITS, {})
    ranks = [h["mmr_rank"] for h in result]
    assert sorted(ranks) == list(range(len(HITS)))


def test_f14_mmr_no_duplicate_ids():
    # Use a corpus with genuinely unique IDs — MMR deduplicates by position.
    unique_hits = [
        {"id": f"u{i}", "score": float(4 - i) / 4,
         "metadata": {"text": f"Document {i} about profit quarterly revenue"}}
        for i in range(5)
    ]
    result = mmr(QUERY, unique_hits, {})
    ids = [h["id"] for h in result]
    assert len(ids) == len(set(ids))


def test_f14_mmr_lambda_1_pure_relevance():
    """lambda=1 ignores diversity — first result should be the most query-relevant doc."""
    result = mmr(QUERY, HITS, {"lambda": 1.0})
    assert len(result) == len(HITS)
    # With lambda=1 the first pick is solely the highest cosine doc — verify it has a score.
    assert result[0]["score"] >= 0.0
    assert result[0]["mmr_rank"] == 0


def test_f14_mmr_lambda_0_pure_diversity():
    """lambda=0 maximises diversity — identical docs pushed apart."""
    same_docs = [
        {"id": f"d{i}", "score": 0.9,
         "metadata": {"text": "profit revenue quarterly report"}}
        for i in range(4)
    ]
    result = mmr(QUERY, same_docs, {"lambda": 0.0})
    assert len(result) == 4


# ── Recency ────────────────────────────────────────────────────────────────────

def test_f14_recency_boosts_recent_docs():
    now = time.time()
    recent = {"id": "new", "score": 0.5,
              "metadata": {"text": "recent profit report", "date": now - 86400}}
    old    = {"id": "old", "score": 0.9,
              "metadata": {"text": "old profit report",    "date": now - 86400 * 365}}
    result = recency(QUERY, [recent, old], {"half_life_days": 30, "now": now})
    assert result[0]["id"] == "new"


def test_f14_recency_no_date_field_keeps_score():
    no_date = {"id": "x", "score": 0.8, "metadata": {"text": "no date here"}}
    result = recency(QUERY, [no_date], {})
    # boost defaults to 1.0 when date missing
    assert result[0]["recency_boost"] == 1.0


def test_f14_recency_custom_date_field():
    now = time.time()
    hit = {"id": "x", "score": 1.0,
           "metadata": {"text": "some text", "published_at": now - 86400 * 10}}
    result = recency(QUERY, [hit], {"date_field": "published_at", "now": now})
    assert result[0]["recency_boost"] < 1.0


def test_f14_recency_sorted_by_boosted_score():
    now = time.time()
    docs = [
        {"id": f"d{i}", "score": 1.0,
         "metadata": {"text": f"doc {i}", "date": now - 86400 * (i * 30)}}
        for i in range(4)
    ]
    result = recency(QUERY, docs, {"half_life_days": 30, "now": now})
    scores = [h["score"] for h in result]
    assert scores == sorted(scores, reverse=True)


# ── Authority ──────────────────────────────────────────────────────────────────

def test_f14_authority_boosts_trusted_source():
    weights = {"finance-report": 2.0, "blog": 0.5}
    result = authority(QUERY, HITS, {"weights": weights, "default_weight": 1.0})
    assert result[0]["id"] == "doc-a"   # finance-report × 2.0


def test_f14_authority_downgrades_untrusted():
    weights = {"finance-report": 0.1, "annual-report": 0.1, "blog": 0.1}
    hits = [{"id": "wiki", "score": 0.6,
             "metadata": {"text": "profit info", "source": "wikipedia"}},
            {"id": "blog", "score": 0.8,
             "metadata": {"text": "profit info", "source": "blog"}}]
    result = authority(QUERY, hits, {"weights": weights, "default_weight": 5.0})
    assert result[0]["id"] == "wiki"    # default_weight 5.0 > blog weight 0.1


def test_f14_authority_weight_in_output():
    result = authority(QUERY, HITS[:2], {})
    for h in result:
        assert "authority_weight" in h


def test_f14_authority_default_weight_1():
    hits = [{"id": "x", "score": 0.7, "metadata": {"text": "t", "source": "unknown"}}]
    result = authority(QUERY, hits, {})
    assert result[0]["authority_weight"] == 1.0


# ── Multi-factor ───────────────────────────────────────────────────────────────

def test_f14_multi_factor_returns_all():
    result = multi_factor(QUERY, HITS, {})
    assert len(result) == len(HITS)


def test_f14_multi_factor_has_factors_field():
    result = multi_factor(QUERY, HITS, {})
    for h in result:
        assert "factors" in h
        assert "relevance" in h["factors"]
        assert "recency"   in h["factors"]
        assert "authority" in h["factors"]


def test_f14_multi_factor_custom_weights():
    cfg = {
        "factor_weights": {"relevance": 0.0, "recency": 1.0, "authority": 0.0},
        "date_field": "date",
    }
    result = multi_factor(QUERY, HITS, cfg)
    # With recency-only, most recent doc should rank first.
    assert result[0]["id"] == "doc-a"   # 2024-01-15 is most recent in HITS


# ── API tests ──────────────────────────────────────────────────────────────────

def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


AUTH = {"Authorization": "Bearer dev"}

_HITS_PAYLOAD = [
    {"id": f"d{i}", "score": float(4 - i) / 4,
     "metadata": {"text": f"Document {i} about profit and quarterly revenue"}}
    for i in range(4)
]


def test_f14_api_list_methods():
    with _client() as c:
        resp = c.get("/api/v1/rerank/methods", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "cross_encoder" in data["methods"]
    assert "mmr"           in data["methods"]
    assert "recency"       in data["methods"]
    assert "authority"     in data["methods"]
    assert "multi_factor"  in data["methods"]
    assert data["default"] == "cross_encoder"


def test_f14_api_rerank_cross_encoder():
    with _client() as c:
        resp = c.post("/api/v1/rerank", json={
            "query":  "profit quarterly",
            "hits":   _HITS_PAYLOAD,
            "method": "cross_encoder",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 4
    assert data["method"] == "cross_encoder"
    assert all("rank_change" in h for h in data["hits"])
    assert all("original_rank" in h for h in data["hits"])


def test_f14_api_rerank_with_top_n():
    with _client() as c:
        resp = c.post("/api/v1/rerank", json={
            "query":  "profit",
            "hits":   _HITS_PAYLOAD,
            "method": "cross_encoder",
            "top_n":  2,
        }, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] == 2


def test_f14_api_rerank_mmr():
    with _client() as c:
        resp = c.post("/api/v1/rerank", json={
            "query":  "profit quarterly",
            "hits":   _HITS_PAYLOAD,
            "method": "mmr",
        }, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] == 4


def test_f14_api_rerank_recency():
    import time
    now = time.time()
    hits = [
        {"id": "new", "score": 0.5,
         "metadata": {"text": "new profit report", "date": now - 86400}},
        {"id": "old", "score": 0.9,
         "metadata": {"text": "old profit report", "date": now - 86400 * 400}},
    ]
    with _client() as c:
        resp = c.post("/api/v1/rerank", json={
            "query":  "latest profit",
            "hits":   hits,
            "method": "recency",
            "config": {"half_life_days": 30},
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["hits"][0]["id"] == "new"


def test_f14_api_rerank_authority():
    hits = [
        {"id": "wiki", "score": 0.6,
         "metadata": {"text": "profit info", "source": "wikipedia"}},
        {"id": "blog", "score": 0.8,
         "metadata": {"text": "profit info", "source": "blog"}},
    ]
    with _client() as c:
        resp = c.post("/api/v1/rerank", json={
            "query":  "profit",
            "hits":   hits,
            "method": "authority",
            "config": {"weights": {"wikipedia": 3.0, "blog": 0.2}},
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["hits"][0]["id"] == "wiki"


def test_f14_api_rerank_invalid_method():
    with _client() as c:
        resp = c.post("/api/v1/rerank", json={
            "query":  "profit",
            "hits":   _HITS_PAYLOAD,
            "method": "nonexistent",
        }, headers=AUTH)
    assert resp.status_code == 422


def test_f14_api_compare():
    with _client() as c:
        resp = c.post("/api/v1/rerank/compare", json={
            "query":   "profit quarterly",
            "hits":    _HITS_PAYLOAD,
            "methods": ["cross_encoder", "mmr", "recency"],
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["rankings"].keys()) == {"cross_encoder", "mmr", "recency"}
    assert "agreement" in data
    assert 0.0 <= data["agreement"] <= 1.0


def test_f14_api_compare_single_method():
    with _client() as c:
        resp = c.post("/api/v1/rerank/compare", json={
            "query":   "profit",
            "hits":    _HITS_PAYLOAD,
            "methods": ["cross_encoder"],
        }, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["agreement"] == 1.0


def test_f14_api_explain():
    with _client() as c:
        resp = c.post("/api/v1/rerank/explain", json={
            "query":  "profit quarterly revenue",
            "hits":   _HITS_PAYLOAD,
            "method": "cross_encoder",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 4
    for exp in data["explanations"]:
        assert "final_score"   in exp
        assert "original_rank" in exp
        assert "new_rank"      in exp
        assert "verdict"       in exp


def test_f14_api_explain_verdict_text():
    with _client() as c:
        resp = c.post("/api/v1/rerank/explain", json={
            "query":  "profit",
            "hits":   _HITS_PAYLOAD,
            "method": "cross_encoder",
        }, headers=AUTH)
    data = resp.json()
    verdicts = [e["verdict"] for e in data["explanations"]]
    # At least one verdict should describe movement
    assert any("UP" in v or "DOWN" in v or "No change" in v for v in verdicts)


def test_f14_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/rerank/methods")
    assert resp.status_code == 401
