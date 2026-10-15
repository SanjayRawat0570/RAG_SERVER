"""Queries are embedded with the store's own model (P2: 256 vs 384 dimension 500s).

The test suite pins the 256-d hash embedder, which is exactly why the bug
(legacy endpoints always embedding queries with the 256-d default) never showed
here while production uses 384-d bge. These tests make a 384-d model active.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.rag.embeddings import registry
from app.rag.embeddings.base import HashEmbedder
from app.rag.vectorstore import VectorRecord, get_store, reset_stores

AUTH = {"Authorization": "Bearer dev"}
STORE = "kb384"
TEXTS = {"parking": "# Parking\n\nThe parking garage opens at 7am.",
         "vpn": "# VPN\n\nConnect to the VPN with the corporate client."}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(registry._FACTORIES, "fake-384", lambda dim: HashEmbedder(dimension=dim, name="fake-384"))
    monkeypatch.setitem(registry._MODEL_DIMS, "fake-384", 384)
    monkeypatch.setattr(registry, "active_embedding", lambda: ("fake-384", 384))
    reset_stores(STORE)
    vecs = registry.embed_texts(list(TEXTS.values()), "fake-384", 384)
    get_store(STORE, 384).upsert([
        VectorRecord(id=k, vector=v, metadata={"text": t, "source": f"{k}.md", "document_id": k})
        for (k, t), v in zip(TEXTS.items(), vecs)], namespace="dim-t")
    from app.main import app

    with TestClient(app) as c:
        yield c
    reset_stores(STORE)


@pytest.mark.parametrize("path", ["/api/v1/search/hybrid", "/api/v1/search/semantic"])
def test_search_on_a_384d_store_uses_its_model(client, path):
    r = client.post(path, headers=AUTH, json={"query": "parking garage", "store": STORE, "namespace": "dim-t"})
    assert r.status_code == 200, r.text
    assert r.json()["hits"][0]["id"] == "parking"


def test_rag_answer_on_a_384d_store(client):
    r = client.post("/api/v1/rag/answer", headers=AUTH, json={
        "query": "When does the parking garage open?", "store": STORE, "namespace": "dim-t", "provider": "stub"})
    assert r.status_code == 200, r.text
    assert "7am" in r.json()["answer"]


def test_wrong_explicit_model_is_a_422_not_a_500(client):
    r = client.post("/api/v1/search/hybrid", headers=AUTH, json={
        "query": "parking", "store": STORE, "namespace": "dim-t", "embed_model": "local-hash"})
    assert r.status_code == 422 and "384" in r.json()["detail"]


def test_upsert_chunks_matches_the_existing_store(client):
    r = client.post("/api/v1/vectors/upsert-chunks", headers=AUTH, json={
        "chunks": [{"text": "The canteen opens at noon.", "chunk_id": "canteen"}],
        "store": STORE, "namespace": "dim-t"})
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "fake-384"


def test_model_for_store():
    assert registry.model_for_store(256) == "local-hash"
    with pytest.raises(ValueError):
        registry.model_for_store(384, "local-hash")
