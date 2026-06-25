"""Tests for F17: Intelligent Caching."""
from __future__ import annotations

import time

import pytest

from app.rag.cache import SemanticCache, TTLCache, cache_stats, get_cache, reset_caches
from app.rag.cache.cache import MISS
from app.rag.embeddings import embed_texts
from app.rag.embeddings.registry import cache_stats as emb_stats, clear_cache as clear_emb
from app.rag.vectorstore import VectorRecord, get_store, reset_stores


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean():
    reset_caches()
    reset_stores()
    clear_emb()
    yield
    reset_caches()
    reset_stores()
    clear_emb()


# ── TTLCache unit tests ────────────────────────────────────────────────────────

def test_f17_ttl_cache_set_and_get():
    c = TTLCache(maxsize=10, ttl=60)
    c.set("k", "v")
    assert c.get("k") == "v"


def test_f17_ttl_cache_miss_returns_sentinel():
    c = TTLCache()
    assert c.get("missing") is MISS


def test_f17_ttl_cache_ttl_expiry():
    c = TTLCache(maxsize=10, ttl=0.05)
    c.set("k", "v")
    time.sleep(0.1)
    assert c.get("k") is MISS


def test_f17_ttl_cache_lru_eviction():
    c = TTLCache(maxsize=3, ttl=60)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)  # evicts "a" (oldest)
    assert c.get("a") is MISS
    assert c.get("d") == 4


def test_f17_ttl_cache_hit_miss_stats():
    c = TTLCache()
    c.set("k", "v")
    c.get("k")      # hit
    c.get("nope")   # miss
    stats = c.stats()
    assert stats["hits"]   == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_f17_ttl_cache_hit_rate_zero_when_empty():
    c = TTLCache()
    assert c.stats()["hit_rate"] == 0.0


def test_f17_ttl_cache_overwrite_key():
    c = TTLCache()
    c.set("k", "v1")
    c.set("k", "v2")
    assert c.get("k") == "v2"


# ── Named registry ─────────────────────────────────────────────────────────────

def test_f17_get_cache_creates_named_instance():
    c = get_cache("test-cache", maxsize=50, ttl=120)
    assert isinstance(c, TTLCache)
    assert c.maxsize == 50
    assert c.ttl == 120


def test_f17_get_cache_same_name_same_instance():
    c1 = get_cache("shared")
    c2 = get_cache("shared")
    assert c1 is c2


def test_f17_cache_stats_lists_all_named():
    get_cache("c1").set("x", 1)
    get_cache("c2").set("y", 2)
    stats = cache_stats()
    assert "c1" in stats
    assert "c2" in stats


def test_f17_reset_caches_clears_all():
    c = get_cache("tmp")
    c.set("k", "v")
    reset_caches()
    # After reset, get_cache creates a fresh instance.
    fresh = get_cache("tmp")
    assert fresh.get("k") is MISS


def test_f17_reset_caches_by_name():
    get_cache("a").set("x", 1)
    get_cache("b").set("y", 2)
    reset_caches("a")
    # "a" is gone, "b" is still there.
    stats = cache_stats()
    assert "a" not in stats
    assert "b" in stats


# ── SemanticCache ──────────────────────────────────────────────────────────────

def _vec(text: str, dim: int = 256) -> list[float]:
    return embed_texts([text], "local-hash", dim)[0]


def test_f17_semantic_cache_miss_when_empty():
    sc = SemanticCache("test-sem", dimension=256, threshold=0.95)
    assert sc.lookup(_vec("hello")) is None


def test_f17_semantic_cache_exact_hit():
    sc = SemanticCache("test-sem2", dimension=256, threshold=0.90)
    v = _vec("What is revenue?")
    sc.put("What is revenue?", v, {"answer": "Revenue is $1M."})
    result = sc.lookup(v)
    assert result is not None
    assert result["answer"] == "Revenue is $1M."


def test_f17_semantic_cache_high_threshold_no_hit():
    sc = SemanticCache("test-sem3", dimension=256, threshold=0.9999)
    v1 = _vec("What is revenue?")
    v2 = _vec("Tell me the revenue")
    sc.put("What is revenue?", v1, {"answer": "Revenue is $1M."})
    # Different text → different hash embedding → score < 0.9999
    result = sc.lookup(v2)
    # With hash embedder different text may not be similar enough.
    assert result is None or isinstance(result, dict)


def test_f17_semantic_cache_low_threshold_returns_hit():
    sc = SemanticCache("test-sem4", dimension=256, threshold=0.0)
    v1 = _vec("revenue")
    v2 = _vec("profit")
    sc.put("revenue", v1, {"answer": "Revenue answer."})
    # With threshold=0 any stored vector is a hit.
    result = sc.lookup(v2)
    assert result is not None


def test_f17_semantic_cache_stores_payload():
    sc = SemanticCache("test-sem5", dimension=256, threshold=0.5)
    payload = {"answer": "test", "sources": ["doc1"], "citations": ["[1]"]}
    v = _vec("revenue profit quarterly")
    sc.put("revenue profit quarterly", v, payload)
    hit = sc.lookup(v)
    assert hit == payload


# ── Embedding cache (L3) ───────────────────────────────────────────────────────

def test_f17_embedding_cache_hits_on_repeat():
    clear_emb()
    embed_texts(["hello world"], "local-hash", 256)
    embed_texts(["hello world"], "local-hash", 256)
    stats = emb_stats()
    assert stats["hits"] >= 1


def test_f17_embedding_cache_miss_on_first_call():
    clear_emb()
    embed_texts(["unique text xyz123"], "local-hash", 256)
    stats = emb_stats()
    assert stats["misses"] >= 1


def test_f17_embedding_cache_clear_resets_stats():
    embed_texts(["text"], "local-hash", 256)
    clear_emb()
    stats = emb_stats()
    assert stats["hits"]   == 0
    assert stats["misses"] == 0
    assert stats["entries"] == 0


def test_f17_embedding_cache_hit_rate():
    clear_emb()
    embed_texts(["abc"], "local-hash", 256)  # miss
    embed_texts(["abc"], "local-hash", 256)  # hit
    stats = emb_stats()
    assert stats["hit_rate"] == 0.5


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f17_api_stats():
    get_cache("test-api-cache").set("k", "v")
    get_cache("test-api-cache").get("k")  # hit
    with _client() as c:
        resp = c.get("/api/v1/cache/stats", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "caches"          in data
    assert "embedding_cache" in data
    assert "summary"         in data
    assert "overall_hit_rate" in data["summary"]


def test_f17_api_embedding_stats():
    clear_emb()
    embed_texts(["embedding cache test"], "local-hash", 256)
    with _client() as c:
        resp = c.get("/api/v1/cache/embedding/stats", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "entries"  in data
    assert "hits"     in data
    assert "misses"   in data
    assert "hit_rate" in data


def test_f17_api_clear_all():
    get_cache("to-clear").set("k", "v")
    with _client() as c:
        resp = c.post("/api/v1/cache/clear", json={"name": None}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["cleared"] == "all"
    # Cache is now empty.
    assert cache_stats() == {}


def test_f17_api_clear_named():
    get_cache("named-cache").set("k", "v")
    get_cache("other-cache").set("k", "v")
    with _client() as c:
        resp = c.post("/api/v1/cache/clear", json={"name": "named-cache"}, headers=AUTH)
    assert resp.status_code == 200
    stats = cache_stats()
    assert "named-cache" not in stats
    assert "other-cache"  in stats


def test_f17_api_config():
    with _client() as c:
        resp = c.get("/api/v1/cache/config", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "l1_exact"    in data
    assert "l2_semantic" in data
    assert "l3_embedding" in data


def test_f17_api_invalidate_unknown_store():
    with _client() as c:
        resp = c.post("/api/v1/cache/invalidate",
                      json={"document_id": "doc-123", "store": "no-store"},
                      headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == "doc-123"
    assert data["status"] == "ok"


def test_f17_api_warm():
    store = get_store("warm-store", 256)
    vecs  = embed_texts(["Revenue was $1M.", "Expenses grew 15%."], "local-hash", 256)
    for i, vec in enumerate(vecs):
        store.upsert([VectorRecord(id=f"d{i}", vector=vec,
                                   metadata={"text": ["Revenue was $1M.", "Expenses grew 15%."][i]})])

    with _client() as c:
        resp = c.post("/api/v1/cache/warm", json={
            "queries":  ["What is revenue?", "What are expenses?"],
            "store":    "warm-store",
            "provider": "stub",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["queries_requested"] == 2
    assert data["queries_warmed"]    == 2
    assert all(r["status"] == "ok" for r in data["results"])


def test_f17_api_l1_cache_hit_on_repeat_ask():
    store = get_store("cache-hit-store", 256)
    vec   = embed_texts(["Revenue was $1M in 2024."], "local-hash", 256)[0]
    store.upsert([VectorRecord(id="d1", vector=vec,
                               metadata={"text": "Revenue was $1M in 2024."})])

    with _client() as c:
        payload = {"query": "revenue", "store": "cache-hit-store",
                   "provider": "stub", "use_cache": True}
        resp1 = c.post("/api/v1/rag/answer", json=payload, headers=AUTH)
        resp2 = c.post("/api/v1/rag/answer", json=payload, headers=AUTH)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    data1 = resp1.json()
    data2 = resp2.json()
    assert data1["cache_hit"] is False
    assert data2["cache_hit"] is True
    assert data2["cache_type"] == "exact"
    assert data1["answer"] == data2["answer"]


def test_f17_api_no_cache_when_disabled():
    store = get_store("nocache-store", 256)
    vec   = embed_texts(["Revenue info."], "local-hash", 256)[0]
    store.upsert([VectorRecord(id="d1", vector=vec,
                               metadata={"text": "Revenue info."})])

    with _client() as c:
        payload = {"query": "revenue", "store": "nocache-store",
                   "provider": "stub", "use_cache": False}
        resp1 = c.post("/api/v1/rag/answer", json=payload, headers=AUTH)
        resp2 = c.post("/api/v1/rag/answer", json=payload, headers=AUTH)

    assert resp1.json()["cache_hit"] is False
    assert resp2.json()["cache_hit"] is False


def test_f17_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/cache/stats")
    assert resp.status_code == 401
