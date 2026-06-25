"""Tests for F11: Multi-Model Embedding System."""
from __future__ import annotations

import math
import pytest

from app.rag.embeddings import (
    cache_stats,
    clear_cache,
    embed_texts,
    get_embedder,
    model_info,
    select_model,
)
from app.rag.embeddings.base import HashEmbedder
from app.rag.embeddings.registry import _FACTORIES


# ── HashEmbedder unit tests ────────────────────────────────────────────────────

def test_f11_hash_embedder_dimension():
    e = HashEmbedder(dimension=128)
    vecs = e.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 128


def test_f11_hash_embedder_l2_normalized():
    e = HashEmbedder(dimension=64)
    vec = e.embed(["test sentence"])[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5


def test_f11_hash_embedder_deterministic():
    e = HashEmbedder(dimension=64)
    a = e.embed(["same text"])[0]
    b = e.embed(["same text"])[0]
    assert a == b


def test_f11_hash_embedder_different_texts_differ():
    e = HashEmbedder(dimension=256)
    a = e.embed(["revenue"])[0]
    b = e.embed(["completely unrelated"])[0]
    assert a != b


def test_f11_hash_embedder_batch():
    e = HashEmbedder(dimension=64)
    vecs = e.embed(["alpha", "beta", "gamma"])
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 64


def test_f11_hash_embedder_empty_text():
    e = HashEmbedder(dimension=64)
    vecs = e.embed([""])
    assert len(vecs) == 1
    # Empty text gives zero vector; norm guard → all zeros
    assert all(x == 0.0 for x in vecs[0])


# ── Registry & get_embedder ────────────────────────────────────────────────────

def test_f11_local_hash_always_registered():
    assert "local-hash" in _FACTORIES


def test_f11_sentence_transformer_registered():
    # sentence-transformers 5.6.0 is installed in this env
    assert "all-MiniLM-L6-v2" in _FACTORIES
    assert "all-mpnet-base-v2" in _FACTORIES


def test_f11_get_embedder_returns_hash():
    e = get_embedder("local-hash", 128)
    assert isinstance(e, HashEmbedder)
    assert e.dimension == 128


def test_f11_get_embedder_unknown_raises():
    with pytest.raises(ValueError, match="Unknown embedding model"):
        get_embedder("does-not-exist")


def test_f11_get_embedder_cached():
    e1 = get_embedder("local-hash", 64)
    e2 = get_embedder("local-hash", 64)
    assert e1 is e2


# ── embed_texts + cache ────────────────────────────────────────────────────────

def test_f11_embed_texts_returns_vectors():
    clear_cache()
    vecs = embed_texts(["hello", "world"], "local-hash", 64)
    assert len(vecs) == 2
    assert all(len(v) == 64 for v in vecs)


def test_f11_embed_texts_cache_hit():
    clear_cache()
    embed_texts(["cached text"], "local-hash", 64)
    embed_texts(["cached text"], "local-hash", 64)
    stats = cache_stats()
    assert stats["hits"] >= 1


def test_f11_embed_texts_cache_miss_then_hit():
    clear_cache()
    embed_texts(["unique text abc"], "local-hash", 64)
    before = cache_stats()["hits"]
    embed_texts(["unique text abc"], "local-hash", 64)
    after = cache_stats()["hits"]
    assert after == before + 1


def test_f11_cache_stats_structure():
    clear_cache()
    embed_texts(["x"], "local-hash", 64)
    stats = cache_stats()
    assert "entries" in stats
    assert "hits"    in stats
    assert "misses"  in stats
    assert "hit_rate" in stats
    assert 0.0 <= stats["hit_rate"] <= 1.0


def test_f11_clear_cache_resets_stats():
    embed_texts(["something"], "local-hash", 64)
    clear_cache()
    stats = cache_stats()
    assert stats["entries"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_f11_embed_texts_default_model():
    vecs = embed_texts(["default model"])
    assert len(vecs) == 1
    assert len(vecs[0]) > 0


# ── Sentence-Transformers adapter ─────────────────────────────────────────────

def test_f11_sentence_transformer_embed():
    from app.rag.embeddings.sentence_tf import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vecs = e.embed(["revenue", "income"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384


def test_f11_sentence_transformer_normalized():
    from app.rag.embeddings.sentence_tf import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    vec = e.embed(["normalization test"])[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_f11_sentence_transformer_similar_texts_close():
    from app.rag.embeddings.sentence_tf import SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    a = e.embed(["The quarterly revenue increased significantly"])[0]
    b = e.embed(["Sales grew substantially this quarter"])[0]
    c = e.embed(["The cat sat on the mat reading a book"])[0]
    sim_ab = sum(x * y for x, y in zip(a, b))
    sim_ac = sum(x * y for x, y in zip(a, c))
    # Similar texts should be closer than dissimilar texts
    assert sim_ab > sim_ac


def test_f11_sentence_transformer_via_registry():
    clear_cache()
    vecs = embed_texts(["test via registry"], "all-MiniLM-L6-v2")
    assert len(vecs) == 1
    assert len(vecs[0]) == 384


def test_f11_sentence_transformer_batch():
    vecs = embed_texts(
        ["alpha", "beta", "gamma", "delta"],
        "all-MiniLM-L6-v2",
    )
    assert len(vecs) == 4
    assert all(len(v) == 384 for v in vecs)


# ── Dynamic model selector ────────────────────────────────────────────────────

def test_f11_select_code_prefers_mpnet():
    model, dim = select_model(content_type="code", quality="free")
    assert model == "all-mpnet-base-v2"
    assert dim == 768


def test_f11_select_free_returns_local():
    model, dim = select_model(quality="free")
    assert model in ("all-MiniLM-L6-v2", "local-hash")
    assert dim <= 384


def test_f11_select_cheap_returns_local():
    model, dim = select_model(quality="cheap")
    assert model in ("all-MiniLM-L6-v2", "local-hash")


def test_f11_select_non_english_multilingual():
    model, dim = select_model(language="fr", quality="free")
    # Should try multilingual model or fall back
    assert model in (
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
        "local-hash",
    )


def test_f11_select_english_not_multilingual():
    model, _ = select_model(language="en", quality="free")
    assert "multilingual" not in model


def test_f11_select_model_always_returns_something():
    # Should never raise, even with unusual inputs
    for ct in (None, "code", "text", "image"):
        for lang in (None, "en", "fr", "zh"):
            for q in ("best", "balanced", "cheap", "free"):
                name, dim = select_model(content_type=ct, language=lang, quality=q)
                assert isinstance(name, str)
                assert isinstance(dim, int)
                assert dim > 0


# ── model_info catalogue ──────────────────────────────────────────────────────

def test_f11_model_info_contains_expected_models():
    info = model_info()
    for expected in ("local-hash", "all-MiniLM-L6-v2", "text-embedding-3-small"):
        assert expected in info


def test_f11_model_info_structure():
    info = model_info()
    for name, meta in info.items():
        assert "provider" in meta
        assert "dimension" in meta
        assert "available" in meta
        assert "local" in meta
        assert "multilingual" in meta


def test_f11_local_hash_always_available():
    info = model_info()
    assert info["local-hash"]["available"] is True


def test_f11_sentence_tf_available_when_installed():
    info = model_info()
    assert info["all-MiniLM-L6-v2"]["available"] is True


# ── API tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f11_api_list_models():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/models",
                          headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "local-hash" in data["models"]
    assert "available" in data


@pytest.mark.asyncio
async def test_f11_api_embed_local_hash():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/embed",
            json={"texts": ["hello", "world"], "model": "local-hash", "dimension": 64},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["dimension"] == 64
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2


@pytest.mark.asyncio
async def test_f11_api_embed_sentence_tf():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/embed",
            json={"texts": ["Revenue grew by 20 percent"], "model": "all-MiniLM-L6-v2"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["dimension"] == 384
    assert len(data["embeddings"][0]) == 384


@pytest.mark.asyncio
async def test_f11_api_embed_no_vectors():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/embed",
            json={"texts": ["test"], "model": "local-hash", "include_vectors": False},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    assert "embeddings" not in resp.json()


@pytest.mark.asyncio
async def test_f11_api_embed_unknown_model():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/embed",
            json={"texts": ["test"], "model": "no-such-model"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_f11_api_similarity():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/similarity",
            json={
                "text_a": "revenue",
                "text_b": "income",
                "model":  "all-MiniLM-L6-v2",
            },
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert -1.0 <= data["score"] <= 1.0
    assert "interpretation" in data


@pytest.mark.asyncio
async def test_f11_api_similarity_identical():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/similarity",
            json={
                "text_a": "identical text",
                "text_b": "identical text",
                "model":  "local-hash",
            },
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    assert resp.json()["score"] > 0.99


@pytest.mark.asyncio
async def test_f11_api_select_model():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/select",
            json={"quality": "free"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "model" in data
    assert "dimension" in data
    assert data["local"] is True


@pytest.mark.asyncio
async def test_f11_api_select_code():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/select",
            json={"content_type": "code", "quality": "free"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "all-mpnet-base-v2"


@pytest.mark.asyncio
async def test_f11_api_select_auto_detects_code():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/embeddings/select",
            json={
                "quality": "free",
                "sample_text": "def calculate_revenue(sales):\n    return sales",
            },
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["signals_used"]["content_type"] == "code"


@pytest.mark.asyncio
async def test_f11_api_cache_stats():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/cache",
                          headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "entries" in data
    assert "hit_rate" in data


@pytest.mark.asyncio
async def test_f11_api_clear_cache():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.delete("/api/v1/embeddings/cache",
                             headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    assert resp.json()["cleared"] is True
