"""Neural ONNX cross-encoder reranker behind RERANKER (P2)."""
from __future__ import annotations

import pytest

from app.config import settings
from app.rag.rerank import onnx_cross_encoder as onnx
from app.rag.rerank import rerank

CANDIDATES = [
    {"id": "office", "score": 0.9, "metadata": {"text": "The office opens at 8am on weekdays."}},
    {"id": "pw", "score": 0.1, "metadata": {"text": "Passwords must be at least 14 characters long."}},
]


@pytest.fixture
def fresh_model(monkeypatch):
    monkeypatch.setattr(onnx, "_model", None)
    monkeypatch.setattr(onnx, "_failed", False)


def test_model_ranks_the_answer_first(fresh_model):
    if not onnx.available():
        pytest.skip("cross-encoder model not downloaded (python -m app.rag.rerank.onnx_cross_encoder download)")
    ranked = rerank("onnx_cross_encoder", "What is the minimum password length?", CANDIDATES, {"top_n": 2})
    assert [c["id"] for c in ranked] == ["pw", "office"]
    assert ranked[0]["score"] > 0.5 > ranked[1]["score"]
    assert ranked[0]["rerank"]["retrieval_score"] == 0.1


def test_missing_model_falls_back_to_the_heuristic(fresh_model, monkeypatch, caplog):
    monkeypatch.setattr(settings, "reranker_model", "no-such/model")
    monkeypatch.setattr(settings, "allow_model_download", False)
    ranked = rerank("onnx_cross_encoder", "minimum password length", CANDIDATES, {})
    assert len(ranked) == 2 and all("semantic" in c["rerank"] for c in ranked)  # the heuristic ran
    assert "using the heuristic cross_encoder" in caplog.text


@pytest.mark.parametrize("flag,method", [("heuristic", "cross_encoder"), ("onnx", "onnx_cross_encoder")])
def test_flag_selects_the_pipeline_reranker(monkeypatch, flag, method):
    from app.api.pipelines import build_ask_workflow

    monkeypatch.setattr(settings, "reranker", flag)
    monkeypatch.setattr(settings, "rerank_pool", 7)
    nodes = {n.id: n for n in build_ask_workflow("stub").nodes}
    assert nodes["rerank"].config["method"] == method
    assert nodes["dense"].config["top_k"] == nodes["fuse"].config["top_n"] == 7
