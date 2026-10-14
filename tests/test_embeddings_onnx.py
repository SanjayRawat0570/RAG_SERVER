"""Local semantic embeddings (fastembed/ONNX) and the active-model selection.

Tests that need the real bge-small model run only when it is already downloaded
(``python -m app.rag.embeddings.onnx_embed download``); the selection and
fallback logic is tested with fakes and always runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.rag.embeddings import registry
from app.rag.embeddings.registry import active_embedding, default_model, embed_texts


def _model_downloaded() -> bool:
    return any(Path(settings.models_dir).glob("*bge-small-en-v1.5*"))


needs_model = pytest.mark.skipif(not _model_downloaded(), reason="bge-small model not downloaded")


@pytest.fixture
def selection(monkeypatch):
    """Choose the configured model for one test; restores the cached selection."""
    def choose(name: str):
        monkeypatch.setattr(settings, "embedding_model", name)
        active_embedding.cache_clear()
        return active_embedding()
    yield choose
    active_embedding.cache_clear()


def test_hash_is_the_default_and_nodes_follow_the_dimension(selection):
    assert selection("local-hash") == ("local-hash", 256)
    assert default_model(256) == "local-hash"
    assert default_model(None) == "local-hash"


def test_unloadable_model_falls_back_to_hash_with_a_warning(selection, monkeypatch, caplog):
    class Broken:
        def embed(self, texts):
            raise RuntimeError("model files missing and download disabled")

    monkeypatch.setitem(registry._FACTORIES, "bge-small-en-v1.5", lambda dim: Broken())
    registry._embedders.pop(("bge-small-en-v1.5", 384), None)
    with caplog.at_level("WARNING"):
        assert selection("bge-small-en-v1.5") == ("local-hash", 256)
    assert "unavailable" in caplog.text
    registry._embedders.pop(("bge-small-en-v1.5", 384), None)


def test_dimension_mismatch_keeps_the_hash_model(selection, monkeypatch):
    class Fake384:
        def embed(self, texts):
            return [[0.0] * 384 for _ in texts]

    monkeypatch.setitem(registry._FACTORIES, "bge-small-en-v1.5", lambda dim: Fake384())
    registry._embedders.pop(("bge-small-en-v1.5", 384), None)
    assert selection("bge-small-en-v1.5") == ("bge-small-en-v1.5", 384)
    assert default_model(384) == "bge-small-en-v1.5"
    # A workflow that explicitly asks for 256-dim vectors keeps getting hash vectors.
    assert default_model(256) == "local-hash"
    registry._embedders.pop(("bge-small-en-v1.5", 384), None)


def test_queries_use_the_query_encoder_and_are_cached_separately(monkeypatch):
    calls = []

    class QueryAware:
        def embed(self, texts):
            calls.append(("passage", texts))
            return [[1.0, 0.0] for _ in texts]

        def embed_queries(self, texts):
            calls.append(("query", texts))
            return [[0.0, 1.0] for _ in texts]

    monkeypatch.setitem(registry._FACTORIES, "qa-fake", lambda dim: QueryAware())
    registry.clear_cache()
    assert embed_texts(["x"], "qa-fake", 2) == [[1.0, 0.0]]
    assert embed_texts(["x"], "qa-fake", 2, kind="query") == [[0.0, 1.0]]
    assert embed_texts(["x"], "qa-fake", 2, kind="query") == [[0.0, 1.0]]   # cached
    assert calls == [("passage", ["x"]), ("query", ["x"])]
    registry._embedders.pop(("qa-fake", 2), None)


@needs_model
def test_bge_small_understands_paraphrase():
    q = np.array(embed_texts(["What is the minimum password length?"], "bge-small-en-v1.5", 384, kind="query")[0])
    good, bad = (np.array(v) for v in embed_texts(
        ["Passwords must be at least 14 characters long.", "The Mobile team is led by Tom Becker."],
        "bge-small-en-v1.5", 384))
    assert len(q) == 384
    assert q @ good > q @ bad + 0.2


@needs_model
async def test_semantic_index_finds_a_paraphrased_question(selection):
    """End to end on a 384-dim store: the paraphrase is retrieved first."""
    import uuid

    from app.api import pipelines
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef

    assert selection("bge-small-en-v1.5") == ("bge-small-en-v1.5", 384)
    tenant, store = f"onnx-{uuid.uuid4().hex[:6]}", f"kb-onnx-{uuid.uuid4().hex[:6]}"
    index = WorkflowDef(name="i", nodes=[
        {"id": "in", "type": "input"},
        {"id": "ingest", "type": "ingest", "config": {"text": "$.inputs.text", "filename": "s.md", "format": "text"}},
        {"id": "chunk", "type": "chunk", "config": {"strategy": "structure"}},
        {"id": "embed", "type": "embed", "config": {"dimension": 384}},
        {"id": "up", "type": "upsert", "config": {"store": store, "namespace": tenant, "dimension": 384}},
    ], edges=[{"source": "in", "target": "ingest"}, {"source": "ingest", "target": "chunk"},
              {"source": "chunk", "target": "embed"}, {"source": "embed", "target": "up"}])
    await WorkflowExecutor(index).run({"text": "# Access\n\nPasswords must be at least 14 characters long.\n\n"
                                               "# Teams\n\nThe Mobile team is led by Tom Becker."})
    search = WorkflowDef(name="s", nodes=[
        {"id": "in", "type": "input"},
        {"id": "dense", "type": "vector_search", "config": {"store": store, "namespace": tenant, "dimension": 384,
                                                           "query": "$.inputs.q", "top_k": 2}},
    ], edges=[{"source": "in", "target": "dense"}])
    result = await WorkflowExecutor(search).run({"q": "What is the minimum password length?"})
    hits = next(r.output for r in result.results if r.node_id == "dense")
    assert "14 characters" in hits[0]["metadata"]["text"]
    assert pipelines.STORE == "kb"   # the shared knowledge base store is untouched by this test
