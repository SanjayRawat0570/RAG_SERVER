"""Tests for F15: Context Augmentation."""
from __future__ import annotations

import pytest

from app.rag.context import (
    build_context,
    build_prompt,
    estimate_complexity,
    group_by_source,
    list_templates,
    organize_context,
)
from app.rag.models import estimate_tokens
from app.rag.vectorstore import VectorRecord, get_store, reset_stores


# ── Shared data ────────────────────────────────────────────────────────────────

HITS = [
    {"id": "h1", "score": 0.95,
     "metadata": {"text": "Annual revenue was $1M in 2024 showing strong growth.",
                  "source": "finance-report", "date": "2024-01-15"}},
    {"id": "h2", "score": 0.87,
     "metadata": {"text": "Operating expenses rose 15 percent due to headcount growth.",
                  "source": "budget-doc", "date": "2023-06-01"}},
    {"id": "h3", "score": 0.72,
     "metadata": {"text": "The CEO outlined a 5-year strategy at the annual summit.",
                  "source": "board-minutes", "date": "2022-03-10"}},
    {"id": "h4", "score": 0.60,
     "metadata": {"text": "Capital expenditure budget was set at $200K for facilities.",
                  "source": "budget-doc", "date": "2022-01-01"}},
]


# ── estimate_tokens ────────────────────────────────────────────────────────────

def test_f15_estimate_tokens_basic():
    assert estimate_tokens("hello world") > 0


def test_f15_estimate_tokens_empty():
    assert estimate_tokens("") == 1  # max(1, ...)


def test_f15_estimate_tokens_proportional():
    short = estimate_tokens("hi")
    long  = estimate_tokens("word " * 100)
    assert long > short


# ── Complexity estimator ───────────────────────────────────────────────────────

def test_f15_complexity_simple():
    result = estimate_complexity("What is revenue?")
    assert result["level"] == "simple"
    assert result["recommended_chunks"] <= 3
    assert result["recommended_tokens"] <= 1000


def test_f15_complexity_moderate():
    result = estimate_complexity("Why did expenses increase this year?")
    assert result["level"] in ("moderate", "complex")
    assert result["recommended_chunks"] >= 3


def test_f15_complexity_complex_temporal():
    result = estimate_complexity("How did revenue trend over the past 5 years and why did it change?")
    assert result["level"] == "complex"
    assert result["recommended_chunks"] >= 6
    assert result["recommended_tokens"] >= 2000


def test_f15_complexity_complex_comparative():
    # "Compare...versus...between" fires 1 signal (comparative) → moderate/complex.
    result = estimate_complexity("Compare revenue versus expenses between 2022 and 2024")
    assert result["level"] in ("moderate", "complex")
    assert "comparative" in result["signals"]


def test_f15_complexity_has_required_fields():
    result = estimate_complexity("profit margin")
    for key in ("level", "score", "signals", "recommended_chunks", "recommended_tokens"):
        assert key in result


def test_f15_complexity_signals_causal():
    result = estimate_complexity("Explain why the profit margin decreased")
    assert "causal" in result["signals"]


# ── build_context (builder.py) ────────────────────────────────────────────────

def test_f15_build_context_includes_all_within_budget():
    result = build_context(HITS, config={"max_context_tokens": 5000})
    assert len(result["included"]) == len(HITS)
    assert result["dropped"] == []


def test_f15_build_context_respects_token_budget():
    # Very small budget — should drop some chunks.
    result = build_context(HITS, config={"max_context_tokens": 20})
    # At least one chunk included, but not all.
    assert len(result["included"]) >= 1
    assert result["token_estimate"] <= 25  # some slack for rounding


def test_f15_build_context_citations_match_included():
    result = build_context(HITS)
    assert len(result["citations"]) == len(result["included"])


def test_f15_build_context_returns_context_text():
    result = build_context(HITS[:2])
    assert len(result["context"]) > 0
    assert "revenue" in result["context"].lower()


def test_f15_build_context_empty_hits():
    result = build_context([])
    assert result["context"] == ""
    assert result["included"] == []


def test_f15_build_context_always_includes_first():
    tiny_hit = {"id": "x", "score": 1.0,
                "metadata": {"text": "tiny"}}
    result = build_context([tiny_hit], config={"max_context_tokens": 1})
    assert "x" in result["included"]


# ── organize_context ───────────────────────────────────────────────────────────

def _add_context_text(hits):
    """Fake selector output — add context_text from metadata text."""
    out = []
    for h in hits:
        new = dict(h)
        new["context_text"] = h["metadata"].get("text", "")
        new["token_count"]  = estimate_tokens(new["context_text"])
        out.append(new)
    return out


def test_f15_organize_relevance():
    chunks = _add_context_text(HITS)
    result = organize_context(chunks, strategy="relevance")
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_f15_organize_source_groups():
    chunks = _add_context_text(HITS)
    result = organize_context(chunks, strategy="source")
    # budget-doc has 2 chunks (h2, h4) — they should be adjacent.
    ids = [r["id"] for r in result]
    pos_h2 = ids.index("h2")
    pos_h4 = ids.index("h4")
    assert abs(pos_h2 - pos_h4) == 1


def test_f15_organize_chronological():
    chunks = _add_context_text(HITS)
    result = organize_context(chunks, strategy="chronological", date_field="date")
    # h1 (2024) should be first.
    assert result[0]["id"] == "h1"


def test_f15_organize_diversity_alternates():
    chunks = _add_context_text(HITS)
    result = organize_context(chunks, strategy="diversity", source_field="source")
    sources = [r["metadata"]["source"] for r in result]
    # No two consecutive chunks should be from the same source when possible.
    consecutive_same = sum(
        1 for i in range(len(sources) - 1) if sources[i] == sources[i + 1]
    )
    assert consecutive_same == 0  # budget-doc has 2 chunks but they get split


def test_f15_organize_adds_order_field():
    chunks = _add_context_text(HITS)
    result = organize_context(chunks)
    for i, c in enumerate(result):
        assert c["order"] == i


def test_f15_organize_empty():
    assert organize_context([]) == []


# ── group_by_source ────────────────────────────────────────────────────────────

def test_f15_group_by_source():
    chunks = _add_context_text(HITS)
    groups = group_by_source(chunks)
    assert "finance-report" in groups
    assert "budget-doc" in groups
    assert len(groups["budget-doc"]) == 2


# ── build_prompt ───────────────────────────────────────────────────────────────

def test_f15_build_prompt_contains_query():
    result = build_prompt("What is revenue?", "Revenue was $1M.")
    assert "What is revenue?" in result["prompt"]


def test_f15_build_prompt_contains_context():
    result = build_prompt("query", "Revenue was $1M.")
    assert "Revenue was $1M" in result["prompt"]


def test_f15_build_prompt_has_messages():
    result = build_prompt("query", "context")
    assert "messages" in result
    roles = [m["role"] for m in result["messages"]]
    assert "system" in roles
    assert "user" in roles


def test_f15_build_prompt_chain_of_thought():
    result = build_prompt("query", "context", config={"chain_of_thought": True})
    assert "step" in result["prompt"].lower() or "think" in result["prompt"].lower()


def test_f15_build_prompt_template_names():
    for name in ("default", "qa", "summarize", "extract", "chain_of_thought"):
        result = build_prompt("query", "context", config={"template_name": name})
        assert "system" in result
        assert len(result["system"]) > 10


def test_f15_build_prompt_custom_system():
    result = build_prompt("q", "ctx", config={"system": "Custom system prompt."})
    assert result["system"] == "Custom system prompt."


def test_f15_build_prompt_empty_context():
    result = build_prompt("What is profit?", "")
    assert "What is profit?" in result["prompt"]


# ── list_templates ─────────────────────────────────────────────────────────────

def test_f15_list_templates_returns_dict():
    templates = list_templates()
    assert isinstance(templates, dict)
    assert "default" in templates
    assert "qa" in templates
    assert "summarize" in templates


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}

_HITS_PAYLOAD = [h.copy() for h in HITS]


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f15_api_templates():
    with _client() as c:
        resp = c.get("/api/v1/context/templates", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "default" in data["templates"]
    assert "qa"      in data["templates"]
    assert data["default"] == "default"


def test_f15_api_estimate_simple():
    with _client() as c:
        resp = c.post("/api/v1/context/estimate",
                      json={"query": "What is revenue?"}, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["level"] == "simple"
    assert data["recommended_chunks"] <= 3


def test_f15_api_estimate_complex():
    with _client() as c:
        resp = c.post("/api/v1/context/estimate",
                      json={"query": "How did revenue trend over 5 years and why did it change?"},
                      headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["level"] == "complex"


def test_f15_api_estimate_missing_query():
    with _client() as c:
        resp = c.post("/api/v1/context/estimate", json={}, headers=AUTH)
    assert resp.status_code == 422


def test_f15_api_select():
    with _client() as c:
        resp = c.post("/api/v1/context/select", json={
            "hits":       _HITS_PAYLOAD,
            "max_tokens": 5000,
            "organize":   "relevance",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == len(HITS)
    assert data["total_tokens"] > 0


def test_f15_api_select_small_budget():
    with _client() as c:
        resp = c.post("/api/v1/context/select", json={
            "hits":       _HITS_PAYLOAD,
            "max_tokens": 20,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] >= 1
    assert data["total_chunks"] < len(HITS)


def test_f15_api_select_by_source_group():
    with _client() as c:
        resp = c.post("/api/v1/context/select", json={
            "hits":      _HITS_PAYLOAD,
            "organize":  "source",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "by_source" in data
    assert "budget-doc" in data["by_source"]


def test_f15_api_assemble():
    with _client() as c:
        resp = c.post("/api/v1/context/assemble", json={
            "query":    "What is our revenue?",
            "hits":     _HITS_PAYLOAD,
            "template": "qa",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt" in data
    assert "What is our revenue?" in data["prompt"]["prompt"]
    assert len(data["context_chunks"]) >= 1
    assert "complexity" in data


def test_f15_api_assemble_chain_of_thought():
    with _client() as c:
        resp = c.post("/api/v1/context/assemble", json={
            "query":           "Why did expenses increase?",
            "hits":            _HITS_PAYLOAD,
            "chain_of_thought": True,
        }, headers=AUTH)
    assert resp.status_code == 200
    prompt_text = resp.json()["prompt"]["prompt"].lower()
    assert "step" in prompt_text or "think" in prompt_text


def test_f15_api_assemble_sources():
    with _client() as c:
        resp = c.post("/api/v1/context/assemble", json={
            "query": "revenue and expenses",
            "hits":  _HITS_PAYLOAD,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sources"]) >= 1


def test_f15_api_build_store_not_found():
    with _client() as c:
        resp = c.post("/api/v1/context/build", json={
            "query": "What is revenue?",
            "store": "nonexistent-store",
        }, headers=AUTH)
    assert resp.status_code == 404


@pytest.fixture(autouse=True)
def _clean():
    reset_stores()
    yield
    reset_stores()


def test_f15_api_build_with_store():
    from app.rag.embeddings import embed_texts
    store = get_store("ctx-store", 256)
    texts = [h["metadata"]["text"] for h in HITS]
    vecs  = embed_texts(texts, "local-hash", 256)
    for hit, vec in zip(HITS, vecs):
        store.upsert([VectorRecord(id=hit["id"], vector=vec, metadata=hit["metadata"])])

    with _client() as c:
        resp = c.post("/api/v1/context/build", json={
            "query": "What is revenue?",
            "store": "ctx-store",
            "top_k": 4,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt" in data
    assert "complexity" in data
    assert data["store"] == "ctx-store"


def test_f15_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/context/templates")
    assert resp.status_code == 401
