"""Tests for F11 (embeddings) and F12 (vector store + retrieval)."""
import math

from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef
from app.rag.embeddings import embed_texts, get_embedder
from app.rag.embeddings.registry import clear_cache
from app.rag.vectorstore import VectorRecord, get_store, reset_stores


# --------------------------------------------------------------------------- F11
def test_f11_embedding_is_deterministic_and_normalized():
    a1 = embed_texts(["encryption and security"])[0]
    a2 = embed_texts(["encryption and security"])[0]
    assert a1 == a2  # deterministic
    assert math.isclose(sum(x * x for x in a1), 1.0, rel_tol=1e-5)  # L2-normalized
    assert len(a1) == get_embedder().dimension


def test_f11_similar_text_scores_higher():
    base = embed_texts(["data is encrypted at rest using aes"])[0]
    near = embed_texts(["our data encryption uses aes at rest"])[0]
    far = embed_texts(["the cat sat on the warm windowsill"])[0]
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    assert dot(base, near) > dot(base, far)


def test_f11_cache_reuses_vectors():
    clear_cache()
    v1 = embed_texts(["cache me"])[0]
    v2 = embed_texts(["cache me"])[0]
    assert v1 is v2  # second call returns the cached object


# --------------------------------------------------------------------------- F12
def test_f12_upsert_update_and_search():
    reset_stores("t")
    store = get_store("t", dimension=3)
    store.upsert([
        VectorRecord(id="a", vector=[1, 0, 0], metadata={"tag": "x"}),
        VectorRecord(id="b", vector=[0, 1, 0], metadata={"tag": "y"}),
    ])
    assert store.count() == 2
    hits = store.search([1, 0, 0], top_k=1)
    assert hits[0].id == "a" and math.isclose(hits[0].score, 1.0, rel_tol=1e-5)

    # Upsert with an existing id updates rather than duplicates.
    store.upsert([VectorRecord(id="a", vector=[0, 0, 1], metadata={"tag": "z"})])
    assert store.count() == 2
    assert store.search([0, 0, 1], top_k=1)[0].metadata["tag"] == "z"


def test_f12_metadata_filter_and_namespaces():
    reset_stores("t2")
    store = get_store("t2", dimension=3)
    store.upsert([VectorRecord(id="a", vector=[1, 0, 0], metadata={"lang": "en"})], namespace="acme")
    store.upsert([VectorRecord(id="b", vector=[1, 0, 0], metadata={"lang": "fr"})], namespace="acme")
    store.upsert([VectorRecord(id="c", vector=[1, 0, 0], metadata={"lang": "en"})], namespace="globex")

    # Namespace isolation.
    assert store.count("acme") == 2 and store.count("globex") == 1
    # Metadata pre-filter.
    hits = store.search([1, 0, 0], top_k=5, namespace="acme", metadata_filter={"lang": "fr"})
    assert [h.id for h in hits] == ["b"]


def test_f12_delete():
    reset_stores("t3")
    store = get_store("t3", dimension=2)
    store.upsert([VectorRecord(id="a", vector=[1, 0]), VectorRecord(id="b", vector=[0, 1])])
    assert store.delete(["a"]) == 1
    assert store.count() == 1 and store.search([1, 0], top_k=5)[0].id == "b"


# ----------------------------------------------------------- end-to-end pipeline
async def test_f11_f12_index_then_search_node_flow():
    reset_stores("kbtest")
    clear_cache()
    index_wf = WorkflowDef(
        name="index",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "ingest", "type": "ingest",
             "config": {"text": "$.inputs.body", "filename": "k.md", "document_id": "k1"}},
            {"id": "chunk", "type": "chunk",
             "config": {"strategy": "structure", "chunk_size": 120, "size_unit": "chars"}},
            {"id": "embed", "type": "embed", "config": {"dimension": 256}},
            {"id": "upsert", "type": "upsert",
             "config": {"store": "kbtest", "namespace": "acme", "dimension": 256}},
            {"id": "out", "type": "output", "config": {"value": "$.upsert"}},
        ],
        edges=[
            {"source": "in", "target": "ingest"},
            {"source": "ingest", "target": "chunk"},
            {"source": "chunk", "target": "embed"},
            {"source": "embed", "target": "upsert"},
            {"source": "upsert", "target": "out"},
        ],
    )
    body = ("# Billing\nInvoices are issued monthly, payment due in 30 days.\n\n"
            "# Security\nData is encrypted at rest and in transit with AES-256.")
    idx = await WorkflowExecutor(index_wf).run({"body": body})
    assert idx.status == "success"
    assert idx.outputs["out"]["upserted"] >= 2

    search_wf = WorkflowDef(
        name="search",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "search", "type": "vector_search",
             "config": {"store": "kbtest", "namespace": "acme",
                        "query": "$.inputs.q", "dimension": 256, "top_k": 1}},
            {"id": "out", "type": "output", "config": {"value": "$.search"}},
        ],
        edges=[{"source": "in", "target": "search"}, {"source": "search", "target": "out"}],
    )
    res = await WorkflowExecutor(search_wf).run({"q": "how is data encrypted at rest?"})
    hits = res.outputs["out"]
    assert len(hits) == 1
    # The encryption query should retrieve the Security section, not Billing.
    assert hits[0]["metadata"]["heading"] == "Security"
