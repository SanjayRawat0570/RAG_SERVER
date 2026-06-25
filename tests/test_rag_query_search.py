"""Tests for F13: query processing, BM25 keyword search, hybrid (RRF) fusion."""
from app.engine.executor import WorkflowExecutor
from app.engine.merging import merge
from app.models.workflow import WorkflowDef
from app.rag.query import process_query
from app.rag.search import BM25, tokenize
from app.rag.vectorstore import reset_stores


# --------------------------------------------------------------- query (F13)
def test_f13_intent_detection():
    assert process_query("How is data encrypted?")["intent"] == "question"
    assert process_query("list all invoices")["intent"] == "command"
    assert process_query("aes encryption keys")["intent"] == "keyword"


def test_f13_entities_and_expansion():
    qp = process_query("Show AES details from 2024-01-01 for 30 days",
                       synonyms={"aes": ["encryption"]})
    assert "2024-01-01" in qp["entities"]["dates"]
    assert "30" in qp["entities"]["numbers"]
    assert "encryption" in qp["expansion"]
    assert "the" not in qp["keywords"]  # stopwords removed


# ----------------------------------------------------------------- BM25 (F13)
def test_f13_bm25_ranks_exact_terms():
    corpus = [
        tokenize("data is encrypted at rest using aes"),
        tokenize("invoices are issued monthly"),
        tokenize("support is available 24/7"),
    ]
    bm25 = BM25(corpus)
    scores = bm25.scores(tokenize("aes encryption"))
    assert scores[0] > scores[1] and scores[0] > scores[2]


# ------------------------------------------------------------------ RRF (F4/F13)
def test_f13_rrf_fuses_rankings():
    dense = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    sparse = [{"id": "b"}, {"id": "a"}, {"id": "d"}]
    fused = merge("rrf", [dense, sparse], {"top_n": 3})
    # 'a' (ranks 0 & 1) and 'b' (ranks 1 & 0) outrank singly-listed docs.
    assert {fused[0]["id"], fused[1]["id"]} == {"a", "b"}
    assert fused[0]["score"] >= fused[2]["score"]


# ------------------------------------------------------- hybrid end-to-end
async def _index(store, namespace):
    wf = WorkflowDef(
        name="idx",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "ingest", "type": "ingest",
             "config": {"text": "$.inputs.body", "filename": "k.md", "document_id": "k1"}},
            {"id": "chunk", "type": "chunk",
             "config": {"strategy": "structure", "chunk_size": 200, "size_unit": "chars"}},
            {"id": "embed", "type": "embed", "config": {"dimension": 256}},
            {"id": "upsert", "type": "upsert",
             "config": {"store": store, "namespace": namespace, "dimension": 256}},
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
    body = ("# Billing\nInvoices are issued monthly and payment is due in 30 days.\n\n"
            "# Security\nData is encrypted at rest and in transit using AES-256.\n\n"
            "# Support\nContact support 24/7 via the help portal.")
    return await WorkflowExecutor(wf).run({"body": body})


async def test_f13_hybrid_search_pipeline():
    reset_stores("hyb")
    await _index("hyb", "acme")

    wf = WorkflowDef(
        name="hybrid",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "qp", "type": "query_process", "config": {"query": "$.inputs.q"}},
            {"id": "dense", "type": "vector_search",
             "config": {"store": "hyb", "namespace": "acme",
                        "query": "$.qp.normalized", "dimension": 256, "top_k": 5}},
            {"id": "sparse", "type": "keyword_search",
             "config": {"store": "hyb", "namespace": "acme",
                        "query": "$.qp.expanded_query", "dimension": 256, "top_k": 5}},
            {"id": "fuse", "type": "merge", "config": {"strategy": "rrf", "top_n": 3}},
            {"id": "out", "type": "output",
             "config": {"value": {"intent": "$.qp.intent", "results": "$.fuse"}}},
        ],
        edges=[
            {"source": "in", "target": "qp"},
            {"source": "qp", "target": "dense"},
            {"source": "qp", "target": "sparse"},
            {"source": "dense", "target": "fuse"},
            {"source": "sparse", "target": "fuse"},
            {"source": "fuse", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({"q": "How is data encrypted with AES?"})
    assert res.status == "success"
    out = res.outputs["out"]
    assert out["intent"] == "question"
    # Hybrid fusion should surface the Security section at the top.
    assert out["results"][0]["metadata"]["heading"] == "Security"
