"""Tests for F14 (reranking) and F15 (context augmentation + prompt)."""
import time

from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef
from app.rag.context import build_context, build_prompt
from app.rag.rerank import rerank
from app.rag.vectorstore import reset_stores


def _hit(hid, text, **meta):
    return {"id": hid, "score": 0.0, "metadata": {"text": text, **meta}}


# --------------------------------------------------------------------------- F14
def test_f14_cross_encoder_orders_by_relevance():
    cands = [
        _hit("a", "data is encrypted at rest using aes-256"),
        _hit("b", "invoices are issued monthly"),
        _hit("c", "support is available around the clock"),
    ]
    ranked = rerank("cross_encoder", "how is data encrypted", cands)
    assert ranked[0]["id"] == "a"
    assert "semantic" in ranked[0]["rerank"]


def test_f14_mmr_demotes_near_duplicates():
    cands = [
        _hit("a", "data is encrypted at rest using aes"),
        _hit("a2", "data is encrypted at rest using aes"),  # near-duplicate of a
        _hit("b", "support team is reachable on the help portal"),
    ]
    ranked = rerank("mmr", "data encryption", cands, {"lambda": 0.5})
    # The diverse doc should not be pushed to last behind both duplicates.
    assert ranked[-1]["id"] in {"a", "a2"}


def test_f14_recency_decay_prefers_newer():
    now = time.time()
    cands = [
        {"id": "old", "score": 1.0, "metadata": {"date": "2000-01-01"}},
        {"id": "new", "score": 1.0, "metadata": {"date": "2024-01-01"}},
    ]
    ranked = rerank("recency", "q", cands, {"half_life_days": 365, "now": now})
    assert ranked[0]["id"] == "new"


def test_f14_authority_weights_sources():
    cands = [
        {"id": "blog", "score": 1.0, "metadata": {"source": "blog"}},
        {"id": "doc", "score": 1.0, "metadata": {"source": "official"}},
    ]
    ranked = rerank("authority", "q", cands, {"weights": {"official": 5.0, "blog": 1.0}})
    assert ranked[0]["id"] == "doc"


def test_f14_multi_factor_blends_signals():
    cands = [
        _hit("a", "data encryption aes", source="official"),
        _hit("b", "data encryption aes", source="blog"),
    ]
    # Identical text -> equal relevance; authority weighting must break the tie.
    ranked = rerank("multi_factor", "data encryption", cands, {
        "factor_weights": {"relevance": 0.0, "recency": 0.0, "authority": 1.0},
        "weights": {"official": 3.0, "blog": 1.0},
    })
    assert ranked[0]["id"] == "a"
    assert "factors" in ranked[0]


# --------------------------------------------------------------------------- F15
def test_f15_context_budget_and_citations():
    hits = [
        _hit("c1", "alpha " * 50, heading="A", source="kb"),
        _hit("c2", "bravo " * 50, heading="B", source="kb"),
        _hit("c3", "charlie " * 50, heading="C", source="kb"),
    ]
    ctx = build_context(hits, {"max_context_tokens": 60})
    assert len(ctx["included"]) >= 1
    assert ctx["dropped"]  # budget forced some out
    assert ctx["token_estimate"] <= 60 + 60  # at most one over-budget include
    assert ctx["citations"][0]["marker"] == "[1]"


def test_f15_prompt_has_messages_and_cot():
    p = build_prompt("What is X?", "[1] some context", {"chain_of_thought": True})
    assert p["messages"][0]["role"] == "system"
    assert p["messages"][1]["role"] == "user"
    assert "step by step" in p["prompt"].lower()
    assert "What is X?" in p["prompt"]


# ----------------------------------------------------------- end-to-end pipeline
async def test_f14_f15_retrieve_rerank_augment_flow():
    reset_stores("ra")
    index = WorkflowDef(
        name="idx",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "ingest", "type": "ingest",
             "config": {"text": "$.inputs.body", "filename": "k.md", "document_id": "k1"}},
            {"id": "chunk", "type": "chunk",
             "config": {"strategy": "structure", "chunk_size": 200, "size_unit": "chars"}},
            {"id": "embed", "type": "embed", "config": {"dimension": 256}},
            {"id": "upsert", "type": "upsert",
             "config": {"store": "ra", "namespace": "acme", "dimension": 256}},
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
    body = ("# Billing\nInvoices are issued monthly.\n\n"
            "# Security\nData is encrypted at rest and in transit using AES-256.")
    await WorkflowExecutor(index).run({"body": body})

    pipe = WorkflowDef(
        name="rag",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "qp", "type": "query_process", "config": {"query": "$.inputs.q"}},
            {"id": "dense", "type": "vector_search",
             "config": {"store": "ra", "namespace": "acme", "query": "$.qp.normalized",
                        "dimension": 256, "top_k": 5}},
            {"id": "sparse", "type": "keyword_search",
             "config": {"store": "ra", "namespace": "acme", "query": "$.qp.expanded_query",
                        "dimension": 256, "top_k": 5}},
            {"id": "fuse", "type": "merge", "config": {"strategy": "rrf", "top_n": 5}},
            {"id": "rerank", "type": "rerank",
             "config": {"method": "cross_encoder", "query": "$.qp.normalized", "top_n": 2}},
            {"id": "augment", "type": "augment",
             "config": {"query": "$.qp.normalized", "max_context_tokens": 300}},
            {"id": "out", "type": "output", "config": {"value": "$.augment"}},
        ],
        edges=[
            {"source": "in", "target": "qp"},
            {"source": "qp", "target": "dense"},
            {"source": "qp", "target": "sparse"},
            {"source": "dense", "target": "fuse"},
            {"source": "sparse", "target": "fuse"},
            {"source": "fuse", "target": "rerank"},
            {"source": "rerank", "target": "augment"},
            {"source": "augment", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(pipe).run({"q": "How is data encrypted with AES?"})
    assert res.status == "success"
    aug = res.outputs["out"]
    assert aug["messages"][0]["role"] == "system"
    assert "encrypted" in aug["context"].lower()
    assert aug["citations"][0]["marker"] == "[1]"
    # The top reranked/cited chunk should be the Security section.
    assert aug["citations"][0]["heading"] == "Security"
