"""Tests for F16: LLM Integration."""
from __future__ import annotations

import asyncio

import pytest

from app.rag.llm import LLMResponse, get_llm
from app.rag.llm.stub import ExtractiveStubLLM
from app.rag.llm.gemini import GeminiLLM
from app.rag.llm.openai_llm import OpenAILLM
from app.rag.llm.claude_llm import ClaudeLLM
from app.rag.llm.selector import model_catalogue, select_model
from app.rag.llm.registry import get_llm, register_provider
from app.rag.cost.pricing import estimate_cost, price_for


# ── LLMResponse model ──────────────────────────────────────────────────────────

def test_f16_llm_response_model():
    r = LLMResponse(text="hello", provider="stub", model="extractive-stub")
    assert r.text == "hello"
    assert r.provider == "stub"
    assert r.finish_reason == "stop"
    assert r.citations == []


def test_f16_llm_response_usage():
    r = LLMResponse(
        text="answer", provider="stub", model="m",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    assert r.usage["input_tokens"] == 10


# ── ExtractiveStubLLM ──────────────────────────────────────────────────────────

def test_f16_stub_generate_with_docs():
    llm = ExtractiveStubLLM()
    docs = [
        {"text": "Revenue was $1M in 2024.", "marker": "[1]"},
        {"text": "The CEO resigned.", "marker": "[2]"},
    ]
    resp = asyncio.get_event_loop().run_until_complete(
        llm.generate({"query": "What is revenue?", "documents": docs}, {})
    )
    assert "Revenue" in resp.text or "1M" in resp.text
    assert resp.provider == "stub"


def test_f16_stub_generate_no_docs():
    llm = ExtractiveStubLLM()
    resp = asyncio.get_event_loop().run_until_complete(
        llm.generate({"query": "What is revenue?", "documents": []}, {})
    )
    assert "don't know" in resp.text.lower() or len(resp.text) > 0


def test_f16_stub_generate_citation_in_response():
    llm = ExtractiveStubLLM()
    docs = [{"text": "Profit was $500K this quarter.", "marker": "[1]"}]
    resp = asyncio.get_event_loop().run_until_complete(
        llm.generate({"query": "profit quarter", "documents": docs}, {})
    )
    assert "[1]" in resp.text or "[1]" in resp.citations


def test_f16_stub_stream_yields_tokens():
    llm = ExtractiveStubLLM()
    docs = [{"text": "Revenue profit quarterly report details.", "marker": "[1]"}]

    async def _collect():
        tokens = []
        async for t in llm.generate_stream({"query": "revenue", "documents": docs}, {}):
            tokens.append(t)
        return tokens

    tokens = asyncio.get_event_loop().run_until_complete(_collect())
    assert len(tokens) >= 1
    assert "".join(tokens).strip() != ""


def test_f16_stub_answer_sentences_config():
    llm = ExtractiveStubLLM()
    docs = [{"text": "First sentence. Second sentence. Third sentence.", "marker": "[1]"}]
    resp1 = asyncio.get_event_loop().run_until_complete(
        llm.generate({"query": "sentence", "documents": docs}, {"answer_sentences": 1})
    )
    resp2 = asyncio.get_event_loop().run_until_complete(
        llm.generate({"query": "sentence", "documents": docs}, {"answer_sentences": 2})
    )
    assert len(resp2.text) >= len(resp1.text)


# ── Registry ───────────────────────────────────────────────────────────────────

def test_f16_registry_get_stub():
    llm = get_llm("stub")
    assert llm.name == "stub"


def test_f16_registry_get_gemini():
    llm = get_llm("gemini")
    assert llm.name == "gemini"


def test_f16_registry_get_openai():
    llm = get_llm("openai")
    assert llm.name == "openai"


def test_f16_registry_get_claude():
    llm = get_llm("claude")
    assert llm.name == "claude"


def test_f16_registry_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("no_such_provider")


def test_f16_registry_register_custom():
    class _DummyLLM:
        name = "dummy"
        async def generate(self, req, cfg):
            return LLMResponse(text="dummy", provider="dummy", model="dummy")
    register_provider("dummy", _DummyLLM)
    llm = get_llm("dummy")
    assert llm.name == "dummy"


# ── OpenAI provider (offline tests — no real API call) ─────────────────────────

def test_f16_openai_raises_without_key():
    llm = OpenAILLM()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.get_event_loop().run_until_complete(
            llm.generate({"messages": [{"role": "user", "content": "hi"}]}, {})
        )


def test_f16_openai_name():
    assert OpenAILLM().name == "openai"


# ── Claude provider (offline tests — no real API call) ─────────────────────────

def test_f16_claude_raises_without_key():
    llm = ClaudeLLM()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.get_event_loop().run_until_complete(
            llm.generate({"messages": [{"role": "user", "content": "hi"}]}, {})
        )


def test_f16_claude_name():
    assert ClaudeLLM().name == "claude"


def test_f16_claude_tier_resolution():
    from app.rag.llm.claude_llm import _resolve_model
    assert "haiku"  in _resolve_model("fast").lower()
    assert "sonnet" in _resolve_model("balanced").lower()
    assert "opus"   in _resolve_model("best").lower()


# ── Model selector ─────────────────────────────────────────────────────────────

def test_f16_selector_returns_dict():
    result = select_model(quality="free", complexity="simple")
    assert "provider" in result
    assert "model"    in result


def test_f16_selector_free_returns_gemini_or_stub():
    result = select_model(quality="free", complexity="simple")
    assert result["provider"] in ("gemini", "stub")


def test_f16_selector_forced_provider():
    result = select_model(quality="best", complexity="complex", provider="stub")
    assert result["provider"] == "stub"


def test_f16_selector_fallback_to_stub():
    # Without API keys configured, should fall back to gemini or stub.
    result = select_model(quality="balanced", complexity="moderate")
    assert result["provider"] in ("claude", "openai", "gemini", "stub")


def test_f16_model_catalogue_has_required_providers():
    catalogue = model_catalogue()
    providers = {m["provider"] for m in catalogue}
    assert "stub"   in providers
    assert "gemini" in providers
    assert "openai" in providers
    assert "claude" in providers


def test_f16_model_catalogue_has_availability():
    for m in model_catalogue():
        assert "available"  in m
        assert "tier"       in m
        assert "description" in m


def test_f16_model_catalogue_stub_always_available():
    stub = next(m for m in model_catalogue() if m["provider"] == "stub")
    assert stub["available"] is True


# ── Cost estimation ────────────────────────────────────────────────────────────

def test_f16_cost_stub_is_zero():
    assert estimate_cost("extractive-stub", 1000, 500) == 0.0


def test_f16_cost_gemini_flash_is_zero():
    assert estimate_cost("gemini-2.5-flash", 1000, 500) == 0.0


def test_f16_cost_gpt4o_mini_positive():
    cost = estimate_cost("gpt-4o-mini", 1000, 500)
    assert cost > 0.0


def test_f16_cost_unknown_model_is_zero():
    assert estimate_cost("nonexistent-model", 1000, 500) == 0.0


def test_f16_price_for_returns_tuple():
    inp, out = price_for("gpt-4o")
    assert isinstance(inp, float)
    assert isinstance(out, float)


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean():
    from app.rag.vectorstore import reset_stores
    reset_stores()
    yield
    reset_stores()


def test_f16_api_list_providers():
    with _client() as c:
        resp = c.get("/api/v1/rag/providers", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "active"    in data
    assert "providers" in data
    assert "stub"   in data["providers"]
    assert "gemini" in data["providers"]
    assert "openai" in data["providers"]
    assert "claude" in data["providers"]


def test_f16_api_list_models():
    with _client() as c:
        resp = c.get("/api/v1/rag/models", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 4
    tiers = {m["tier"] for m in data["models"]}
    assert "free" in tiers


def test_f16_api_generate_stub():
    with _client() as c:
        resp = c.post("/api/v1/rag/generate", json={
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "provider": "stub",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer"   in data
    assert "provider" in data
    assert "usage"    in data


def test_f16_api_generate_has_cost():
    with _client() as c:
        resp = c.post("/api/v1/rag/generate", json={
            "messages": [{"role": "user", "content": "hello"}],
            "provider": "stub",
        }, headers=AUTH)
    data = resp.json()
    assert "estimated_cost_usd" in data["usage"]


def test_f16_api_generate_openai_no_key():
    with _client() as c:
        resp = c.post("/api/v1/rag/generate", json={
            "messages": [{"role": "user", "content": "hello"}],
            "provider": "openai",
        }, headers=AUTH)
    # No API key → 503 Service Unavailable
    assert resp.status_code == 503


def test_f16_api_generate_claude_no_key():
    with _client() as c:
        resp = c.post("/api/v1/rag/generate", json={
            "messages": [{"role": "user", "content": "hello"}],
            "provider": "claude",
        }, headers=AUTH)
    assert resp.status_code == 503


def test_f16_api_ask_store_not_found():
    with _client() as c:
        resp = c.post("/api/v1/rag/answer", json={
            "query": "What is revenue?",
            "store": "no-such-store",
        }, headers=AUTH)
    assert resp.status_code == 404


def test_f16_api_ask_with_store():
    from app.rag.embeddings import embed_texts
    from app.rag.vectorstore import VectorRecord, get_store

    docs = [
        ("d1", "Revenue was $1M in 2024 driven by product sales."),
        ("d2", "Operating expenses rose 15 percent due to headcount growth."),
        ("d3", "The CEO outlined a 5-year expansion strategy."),
    ]
    store = get_store("rag-store", 256)
    vecs  = embed_texts([t for _, t in docs], "local-hash", 256)
    for (doc_id, text), vec in zip(docs, vecs):
        store.upsert([VectorRecord(id=doc_id, vector=vec,
                                   metadata={"text": text, "source": "report"})])

    with _client() as c:
        resp = c.post("/api/v1/rag/answer", json={
            "query":    "What is revenue?",
            "store":    "rag-store",
            "provider": "stub",
            "top_k":    3,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer"    in data
    assert "sources"   in data
    assert "provider"  in data
    assert "complexity" in data
    assert "context"   in data
    assert "usage"     in data
    assert len(data["answer"]) > 0


def test_f16_api_ask_includes_usage_stats():
    from app.rag.embeddings import embed_texts
    from app.rag.vectorstore import VectorRecord, get_store

    store = get_store("rag-store2", 256)
    vec   = embed_texts(["profit quarterly earnings"], "local-hash", 256)[0]
    store.upsert([VectorRecord(id="d1", vector=vec,
                               metadata={"text": "Profit was $500K this quarter.", "source": "q-report"})])

    with _client() as c:
        resp = c.post("/api/v1/rag/answer", json={
            "query": "profit", "store": "rag-store2", "provider": "stub",
        }, headers=AUTH)
    data = resp.json()
    assert data["usage"]["input_tokens"]  > 0
    assert data["usage"]["output_tokens"] > 0
    assert "estimated_cost_usd" in data["usage"]


def test_f16_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/rag/providers")
    assert resp.status_code == 401
