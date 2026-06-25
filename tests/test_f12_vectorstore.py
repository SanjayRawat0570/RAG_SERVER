"""Tests for F12: Vector Database Operations."""
from __future__ import annotations

import math
import pytest

from app.rag.vectorstore import (
    InMemoryVectorStore,
    HNSWVectorStore,
    SearchHit,
    VectorRecord,
    get_store,
    index_recommendation,
    list_namespaces,
    list_stores,
    reset_stores,
    store_stats,
    suggest_index_type,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_stores():
    """Isolate each test by clearing the store registry."""
    reset_stores()
    yield
    reset_stores()


def _flat_store(dim: int = 4) -> InMemoryVectorStore:
    return InMemoryVectorStore(name="test", dimension=dim)


def _records(n: int = 5, dim: int = 4) -> list[VectorRecord]:
    import random
    rng = random.Random(42)
    records = []
    for i in range(n):
        vec = [rng.uniform(-1, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / norm for x in vec] if norm > 0 else vec
        records.append(VectorRecord(id=f"rec-{i}", vector=vec, metadata={"index": i}))
    return records


# ── InMemoryVectorStore — upsert ───────────────────────────────────────────────

def test_f12_upsert_inserts_new():
    store = _flat_store()
    recs = _records(3)
    n = store.upsert(recs)
    assert n == 3
    assert store.count() == 3


def test_f12_upsert_updates_existing():
    store = _flat_store()
    store.upsert([VectorRecord(id="a", vector=[1, 0, 0, 0], metadata={"v": 1})])
    store.upsert([VectorRecord(id="a", vector=[0, 1, 0, 0], metadata={"v": 2})])
    assert store.count() == 1  # same id → update, no duplicate


def test_f12_upsert_returns_count():
    store = _flat_store()
    n = store.upsert(_records(7))
    assert n == 7


def test_f12_upsert_dimension_mismatch_raises():
    store = _flat_store(dim=4)
    with pytest.raises(ValueError, match="dim"):
        store.upsert([VectorRecord(id="x", vector=[1.0, 0.0])])


def test_f12_upsert_namespaces_isolated():
    store = _flat_store()
    store.upsert(_records(3), namespace="tenant-a")
    store.upsert(_records(2), namespace="tenant-b")
    assert store.count("tenant-a") == 3
    assert store.count("tenant-b") == 2
    assert store.count() == 5


# ── InMemoryVectorStore — search ───────────────────────────────────────────────

def test_f12_search_returns_top_k():
    store = _flat_store()
    store.upsert(_records(10))
    hits = store.search([1, 0, 0, 0], top_k=3)
    assert len(hits) == 3


def test_f12_search_empty_store():
    store = _flat_store()
    hits = store.search([1, 0, 0, 0])
    assert hits == []


def test_f12_search_scores_sorted():
    store = _flat_store()
    store.upsert(_records(10))
    hits = store.search([1, 0, 0, 0], top_k=5)
    for i in range(len(hits) - 1):
        assert hits[i].score >= hits[i + 1].score


def test_f12_search_identical_vector_scores_one():
    store = _flat_store()
    store.upsert([VectorRecord(id="exact", vector=[1, 0, 0, 0], metadata={})])
    hits = store.search([1, 0, 0, 0], top_k=1)
    assert len(hits) == 1
    assert abs(hits[0].score - 1.0) < 1e-5


def test_f12_search_metadata_filter():
    store = _flat_store()
    recs = _records(10)
    for i, r in enumerate(recs):
        r.metadata["group"] = "a" if i < 5 else "b"
    store.upsert(recs)
    hits = store.search([1, 0, 0, 0], top_k=10, metadata_filter={"group": "a"})
    assert all(h.metadata["group"] == "a" for h in hits)
    assert len(hits) <= 5


def test_f12_search_namespace_isolation():
    store = _flat_store()
    store.upsert([VectorRecord(id="a1", vector=[1, 0, 0, 0], metadata={})], namespace="ns-a")
    store.upsert([VectorRecord(id="b1", vector=[1, 0, 0, 0], metadata={})], namespace="ns-b")
    hits_a = store.search([1, 0, 0, 0], namespace="ns-a")
    assert all(h.id == "a1" for h in hits_a)


def test_f12_search_returns_metadata():
    store = _flat_store()
    store.upsert([VectorRecord(id="doc1", vector=[1, 0, 0, 0],
                               metadata={"title": "Revenue Report", "year": 2024})])
    hits = store.search([1, 0, 0, 0], top_k=1)
    assert hits[0].metadata["title"] == "Revenue Report"
    assert hits[0].metadata["year"] == 2024


# ── InMemoryVectorStore — delete ───────────────────────────────────────────────

def test_f12_delete_removes_records():
    store = _flat_store()
    store.upsert(_records(5))
    removed = store.delete(["rec-0", "rec-1"])
    assert removed == 2
    assert store.count() == 3


def test_f12_delete_nonexistent_returns_zero():
    store = _flat_store()
    store.upsert(_records(3))
    removed = store.delete(["no-such-id"])
    assert removed == 0


def test_f12_deleted_records_not_returned_in_search():
    store = _flat_store()
    store.upsert([VectorRecord(id="keep", vector=[1, 0, 0, 0], metadata={}),
                  VectorRecord(id="drop", vector=[1, 0, 0, 0], metadata={})])
    store.delete(["drop"])
    hits = store.search([1, 0, 0, 0], top_k=10)
    ids = [h.id for h in hits]
    assert "drop" not in ids
    assert "keep" in ids


# ── InMemoryVectorStore — stats ────────────────────────────────────────────────

def test_f12_stats_structure():
    store = _flat_store()
    store.upsert(_records(3), namespace="ns1")
    stats = store.stats()
    assert stats["name"] == "test"
    assert stats["dimension"] == 4
    assert "index" in stats
    assert stats["total"] == 3
    assert "ns1" in stats["namespaces"]


# ── HNSWVectorStore ────────────────────────────────────────────────────────────

def test_f12_hnsw_same_interface():
    store = HNSWVectorStore(name="hnsw-test", dimension=4)
    store.upsert(_records(5))
    hits = store.search([1, 0, 0, 0], top_k=3)
    assert len(hits) == 3
    assert all(isinstance(h, SearchHit) for h in hits)


def test_f12_hnsw_upsert_updates():
    store = HNSWVectorStore(name="hnsw-test", dimension=4)
    store.upsert([VectorRecord(id="a", vector=[1, 0, 0, 0], metadata={"v": 1})])
    store.upsert([VectorRecord(id="a", vector=[0, 1, 0, 0], metadata={"v": 2})])
    assert store.count() == 1


def test_f12_hnsw_stats_reports_index():
    store = HNSWVectorStore(name="hnsw-test", dimension=4)
    assert "hnsw" in store.stats()["index"]


def test_f12_hnsw_delete():
    store = HNSWVectorStore(name="hnsw-test", dimension=4)
    store.upsert(_records(5))
    store.delete(["rec-0"])
    assert store.count() == 4


# ── Index type selection ───────────────────────────────────────────────────────

def test_f12_suggest_flat_small():
    assert suggest_index_type(100) == "flat"
    assert suggest_index_type(9_999) == "flat"


def test_f12_suggest_hnsw_medium():
    assert suggest_index_type(10_000) == "hnsw"
    assert suggest_index_type(500_000) == "hnsw"


def test_f12_suggest_ivf_large():
    assert suggest_index_type(1_000_000) == "ivf"
    assert suggest_index_type(5_000_000) == "ivf"


def test_f12_index_recommendation_structure():
    rec = index_recommendation(50_000)
    assert rec["suggested"] == "hnsw"
    assert "reason" in rec
    assert "recall" in rec
    assert "thresholds" in rec
    assert rec["thresholds"]["flat"] == 10_000


def test_f12_index_recommendation_flat():
    rec = index_recommendation(1_000)
    assert rec["suggested"] == "flat"


def test_f12_index_recommendation_ivf():
    rec = index_recommendation(2_000_000)
    assert rec["suggested"] == "ivf"


# ── Registry ───────────────────────────────────────────────────────────────────

def test_f12_registry_get_store():
    store = get_store("s1", dimension=4, backend="memory")
    assert store is not None
    assert store.dimension == 4


def test_f12_registry_singleton():
    s1 = get_store("singleton", dimension=4)
    s2 = get_store("singleton", dimension=4)
    assert s1 is s2


def test_f12_registry_dimension_conflict_raises():
    get_store("conflict", dimension=4)
    with pytest.raises(ValueError, match="dim"):
        get_store("conflict", dimension=8)


def test_f12_registry_hnsw_backend():
    store = get_store("hnsw-s", dimension=4, backend="hnsw")
    assert isinstance(store, HNSWVectorStore)


def test_f12_list_namespaces():
    s = get_store("ns-test", dimension=4)
    s.upsert(_records(2), namespace="a")
    s.upsert(_records(3), namespace="b")
    ns = list_namespaces("ns-test")
    names = [n["namespace"] for n in ns]
    assert "a" in names
    assert "b" in names


def test_f12_list_stores():
    get_store("ls1", dimension=4)
    get_store("ls2", dimension=4)
    stores = list_stores()
    names = [s["name"] for s in stores]
    assert "ls1" in names
    assert "ls2" in names


# ── API tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f12_api_upsert():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/vectors/upsert", json={
            "records":   [{"id": "v1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"doc": "a"}}],
            "store":     "api-test",
            "namespace": "default",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    assert resp.json()["upserted"] == 1


@pytest.mark.asyncio
async def test_f12_api_search():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        # First upsert some vectors
        client.post("/api/v1/vectors/upsert", json={
            "records": [{"id": f"s{i}", "vector": [float(i == 0), float(i == 1), 0.0, 0.0], "metadata": {}}
                        for i in range(4)],
            "store": "search-test",
        }, headers={"Authorization": "Bearer dev"})
        resp = client.post("/api/v1/vectors/search", json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "top_k":  2,
            "store":  "search-test",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] <= 2
    assert "hits" in data


@pytest.mark.asyncio
async def test_f12_api_delete():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        client.post("/api/v1/vectors/upsert", json={
            "records": [{"id": "del-1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {}}],
            "store": "del-test",
        }, headers={"Authorization": "Bearer dev"})
        resp = client.request("DELETE", "/api/v1/vectors/records", json={
            "ids":   ["del-1"],
            "store": "del-test",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 1


@pytest.mark.asyncio
async def test_f12_api_list_stores():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/vectors/stores",
                          headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "stores" in data
    assert "total_vectors" in data


@pytest.mark.asyncio
async def test_f12_api_namespaces():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        client.post("/api/v1/vectors/upsert", json={
            "records":   [{"id": "n1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {}}],
            "store":     "ns-api",
            "namespace": "tenant-x",
        }, headers={"Authorization": "Bearer dev"})
        resp = client.get("/api/v1/vectors/stores/ns-api/ns",
                          headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    ns_names = [n["namespace"] for n in data["namespaces"]]
    assert "tenant-x" in ns_names


@pytest.mark.asyncio
async def test_f12_api_index_suggest():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/vectors/index/suggest",
                           json={"vector_count": 50_000},
                           headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["suggested"] == "hnsw"
    assert "reason" in data


@pytest.mark.asyncio
async def test_f12_api_maintenance():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        # Create store first
        client.post("/api/v1/vectors/upsert", json={
            "records": [{"id": "m1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {}}],
            "store":   "maint-test",
        }, headers={"Authorization": "Bearer dev"})
        resp = client.post("/api/v1/vectors/maintenance/maint-test",
                           headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "actions" in data
    assert "suggested_index" in data
    assert "generated_at" in data


@pytest.mark.asyncio
async def test_f12_api_upsert_chunks():
    from fastapi.testclient import TestClient
    from app.main import app
    chunks = [
        {"chunk_id": f"c{i}", "text": f"Revenue grew by {i * 10} percent.", "metadata": {}}
        for i in range(3)
    ]
    with TestClient(app) as client:
        resp = client.post("/api/v1/vectors/upsert-chunks", json={
            "chunks":    chunks,
            "model":     "local-hash",
            "store":     "chunk-test",
            "namespace": "default",
        }, headers={"Authorization": "Bearer dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["upserted"] == 3
    assert data["model"] == "local-hash"
