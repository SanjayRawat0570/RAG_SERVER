"""Semantic answer cache on /rag/ask and /rag/stream/ask (P2)."""
from __future__ import annotations

import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.rag.cache import answers
from app.rag.cache.semantic import SemanticCache, bump_kb_version, kb_version

AUTH = {"Authorization": "Bearer dev"}
DOC = "# Parking\n\nThe parking garage opens at 7am. Each employee gets one parking badge."
Q = "When does the parking garage open?"


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _tenant(client, text: str = DOC) -> str:
    tenant = f"sc-{uuid.uuid4().hex[:6]}"
    assert client.post("/api/v1/rag/index", headers=AUTH,
                       json={"tenant": tenant, "filename": "parking.md", "text": text}).status_code == 200
    return tenant


def _ask(client, tenant: str, question: str = Q, **extra) -> dict:
    return client.post("/api/v1/rag/ask", headers=AUTH, json={"tenant": tenant, "question": question, **extra}).json()


def test_repeat_is_served_from_the_semantic_cache_with_its_own_citations(client):
    tenant = _tenant(client)
    first, second = _ask(client, tenant), _ask(client, tenant)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True and second["cache_type"] == "semantic"
    for key in ("answer", "citations", "cited_markers", "confidence"):
        assert second[key] == first[key]


def test_other_tenants_never_see_the_entry(client):
    a, b = _tenant(client), _tenant(client)
    _ask(client, a)
    assert _ask(client, b)["cache_hit"] is False


def test_changed_documents_invalidate_cached_answers(client):
    tenant = _tenant(client)
    assert "7am" in _ask(client, tenant)["answer"]
    client.post("/api/v1/rag/index", headers=AUTH, json={
        "tenant": tenant, "filename": "parking.md", "text": DOC.replace("7am", "6am")})
    after = _ask(client, tenant)
    assert after["cache_hit"] is False and "6am" in after["answer"]


def test_fallback_answers_are_not_cached(client, monkeypatch):
    tenant = _tenant(client)
    # Gemini without a key: the gateway refuses and the pipeline falls back to extractive.
    monkeypatch.setattr(settings, "gemini_api_key", "")
    first = _ask(client, tenant, provider="gemini")
    assert first["provider"] == "extractive-fallback"
    assert _ask(client, tenant, provider="gemini")["cache_hit"] is False


def test_agentic_mode_and_the_switch(client, monkeypatch):
    tenant = _tenant(client)
    # (A repeat may still hit the exact LLM-response cache; it must not be a semantic hit.)
    _ask(client, tenant, mode="agentic")
    assert _ask(client, tenant, mode="agentic").get("cache_type") != "semantic"
    monkeypatch.setattr(settings, "semantic_cache", False)
    _ask(client, tenant)
    assert _ask(client, tenant).get("cache_type") != "semantic"


def test_stream_uses_the_same_cache(client):
    tenant = _tenant(client)

    def stream() -> list[dict]:
        with client.stream("POST", "/api/v1/rag/stream/ask", headers=AUTH,
                           json={"tenant": tenant, "question": Q}) as r:
            body = "".join(r.iter_text())
        return [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]

    first = next(e for e in stream() if e["event"] == "complete")
    events = stream()
    second = next(e for e in events if e["event"] == "complete")
    assert first.get("cache_hit") in (None, False) and second["cache_hit"] is True
    assert "".join(e["token"] for e in events if e["event"] == "token").strip() == first["answer"].strip()
    assert any(e["event"] == "sources" for e in events)
    # ...and /rag/ask reuses what the stream cached.
    assert _ask(client, tenant)["cache_hit"] is True


@pytest.mark.parametrize("a,b", [
    ("What does error E-4012 mean?", "What does error E-4021 mean?"),
    ("What is the rate limit of the Orders API v2?", "How does the rate limit of the Orders API v2 compare with v1?"),
    ("What was the meal allowance under the 2023 policy?", "What is the current meal allowance?"),
    ("Which team owns Project Atlas?", "Who leads the team that owns Project Atlas?"),
    ("How many leave days can be carried over?", "By when must carried-over leave days be used?"),
])
def test_specifics_tell_near_misses_apart(a, b):
    assert answers.specifics(a) != answers.specifics(b)


@pytest.mark.parametrize("a,b", [
    ("What does error code E-4012 mean?", "What is the meaning of E-4012?"),
    ("Who is the CEO?", "Who is the chief executive officer of the company?"),
    ("How many days of annual leave do I get?", "What is the yearly vacation entitlement in days?"),
])
def test_specifics_keep_paraphrases_together(a, b):
    assert answers.specifics(a) == answers.specifics(b)


def test_entries_expire_and_are_bounded(monkeypatch):
    cache = SemanticCache(f"t-{uuid.uuid4().hex[:6]}", 2, threshold=0.9, ttl=60, max_entries=3, tenant="ttl-t")
    for i in range(5):
        cache.put(f"q{i}", [1.0, 0.001 * i], {"n": i})
    assert cache.store.count("default") == 3
    assert cache.lookup([1.0, 0.0]) is not None
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 120)
    assert cache.lookup([1.0, 0.0]) is None


def test_kb_version_is_per_tenant():
    before_a, before_b = kb_version("kv-a"), kb_version("kv-b")
    bump_kb_version("kv-a/team")
    assert kb_version("kv-a") == before_a + 1 and kb_version("kv-b") == before_b
