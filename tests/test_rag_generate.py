"""Tests for F16 LLM generation (offline stub + provider wiring)."""
import pytest

from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef
from app.rag.llm import get_llm
from app.rag.llm.gemini import GeminiLLM
from app.rag.llm.stub import ExtractiveStubLLM
from app.rag.vectorstore import reset_stores


# --------------------------------------------------------------------------- F16
async def test_f16_stub_extracts_relevant_answer():
    llm = ExtractiveStubLLM()
    request = {
        "query": "how is data encrypted",
        "documents": [
            {"marker": "[1]", "text": "Data is encrypted at rest using AES-256. It is safe."},
            {"marker": "[2]", "text": "Invoices are issued monthly."},
        ],
    }
    resp = await llm.generate(request, {"answer_sentences": 1})
    assert "encrypted" in resp.text.lower()
    assert resp.citations == ["[1]"]
    assert resp.provider == "stub"


async def test_f16_stub_says_dont_know_without_overlap():
    llm = ExtractiveStubLLM()
    resp = await llm.generate(
        {"query": "weather on mars", "documents": [{"marker": "[1]", "text": "Invoices monthly."}]},
        {},
    )
    assert "don't know" in resp.text.lower()
    assert resp.citations == []


def test_f16_registry_resolves_providers():
    assert isinstance(get_llm("stub"), ExtractiveStubLLM)
    assert isinstance(get_llm("gemini"), GeminiLLM)
    with pytest.raises(ValueError):
        get_llm("nope")


async def test_f16_gemini_requires_key():
    # No GEMINI_API_KEY in the test env -> a clear error (no network call made).
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await GeminiLLM().generate({"messages": [{"role": "user", "content": "hi"}]}, {})


# ----------------------------------------------------------- full RAG end-to-end
async def test_f16_full_rag_pipeline_offline():
    reset_stores("full")
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
             "config": {"store": "full", "namespace": "acme", "dimension": 256}},
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

    rag = WorkflowDef(
        name="rag",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "qp", "type": "query_process", "config": {"query": "$.inputs.q"}},
            {"id": "dense", "type": "vector_search",
             "config": {"store": "full", "namespace": "acme", "query": "$.qp.normalized",
                        "dimension": 256, "top_k": 5}},
            {"id": "sparse", "type": "keyword_search",
             "config": {"store": "full", "namespace": "acme", "query": "$.qp.expanded_query",
                        "dimension": 256, "top_k": 5}},
            {"id": "fuse", "type": "merge", "config": {"strategy": "rrf", "top_n": 5}},
            {"id": "rerank", "type": "rerank",
             "config": {"method": "cross_encoder", "query": "$.qp.normalized", "top_n": 2}},
            {"id": "augment", "type": "augment",
             "config": {"query": "$.qp.normalized", "max_context_tokens": 300}},
            {"id": "generate", "type": "generate", "config": {"provider": "stub"}},
            {"id": "out", "type": "output",
             "config": {"value": {"answer": "$.generate.answer",
                                  "citations": "$.generate.citations",
                                  "provider": "$.generate.provider"}}},
        ],
        edges=[
            {"source": "in", "target": "qp"},
            {"source": "qp", "target": "dense"},
            {"source": "qp", "target": "sparse"},
            {"source": "dense", "target": "fuse"},
            {"source": "sparse", "target": "fuse"},
            {"source": "fuse", "target": "rerank"},
            {"source": "rerank", "target": "augment"},
            {"source": "augment", "target": "generate"},
            {"source": "generate", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(rag).run({"q": "How is data encrypted with AES?"})
    assert res.status == "success"
    out = res.outputs["out"]
    assert out["provider"] == "stub"
    assert "encrypted" in out["answer"].lower()
    assert out["citations"] == ["[1]"]
