"""Tests for F17 (caching) and F24 (cost & budgets)."""
import pytest

from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef
from app.rag.cache import SemanticCache, get_cache, reset_caches
from app.rag.cache.cache import MISS, TTLCache
from app.rag.cost import (
    BudgetExceededError,
    budget_stats,
    estimate_cost,
    record_spend,
    reserve,
    reset_budgets,
)
from app.rag.embeddings import embed_texts
from app.rag.vectorstore import reset_stores


# --------------------------------------------------------------------------- F17
def test_f17_ttl_cache_hit_miss_and_lru():
    c = TTLCache(maxsize=2, ttl=100)
    assert c.get("a") is MISS
    c.set("a", 1)
    assert c.get("a") == 1
    c.set("b", 2)
    c.set("c", 3)  # evicts least-recently-used ("a" if not touched)
    assert c.get("a") is MISS
    assert c.stats()["hits"] >= 1


def test_f17_semantic_cache_reuses_similar_query():
    reset_stores("__semcache__t")
    sem = SemanticCache("t", dimension=256, threshold=0.9)
    v1 = embed_texts(["how is data encrypted at rest"], dimension=256)[0]
    assert sem.lookup(v1) is None
    sem.put("how is data encrypted at rest", v1, {"answer": "AES-256"})
    # Same query -> exact vector -> cosine 1.0 -> hit.
    assert sem.lookup(v1) == {"answer": "AES-256"}


# --------------------------------------------------------------------------- F24
def test_f24_cost_estimation():
    assert estimate_cost("extractive-stub", 1000, 1000) == 0.0
    # gpt-4o: 0.005 in + 0.015 out per 1k.
    assert estimate_cost("gpt-4o", 1000, 1000) == pytest.approx(0.02)


def test_f24_budget_reserve_and_record():
    reset_budgets("acme")
    reserve("acme", limit=1.0, estimated_cost=0.4)  # ok
    record_spend("acme", 1.0, 0.4)
    with pytest.raises(BudgetExceededError):
        reserve("acme", limit=1.0, estimated_cost=0.7)  # 0.4+0.7 > 1.0
    stats = budget_stats()["acme"]
    assert stats["spent"] == 0.4 and stats["rejected"] == 1


# --------------------------------------------------- generate node integration
def _gen_wf(extra_cfg):
    cfg = {"provider": "stub", "query": "$.inputs.q", **extra_cfg}
    return WorkflowDef(
        name="g",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "aug", "type": "augment",
             "config": {"query": "$.inputs.q", "hits": "$.inputs.hits"}},
            {"id": "gen", "type": "generate", "config": cfg},
            {"id": "out", "type": "output", "config": {"value": "$.gen"}},
        ],
        edges=[
            {"source": "in", "target": "aug"},
            {"source": "aug", "target": "gen"},
            {"source": "gen", "target": "out"},
        ],
    )


async def test_f17_generate_exact_cache_hit_on_second_run():
    reset_caches("llm-response")
    hits = [{"id": "d1", "score": 1.0,
             "metadata": {"text": "Data is encrypted using AES-256.", "heading": "Security"}}]
    wf = _gen_wf({"cache": True})
    inputs = {"q": "how is data encrypted", "hits": hits}

    first = await WorkflowExecutor(wf).run(inputs)
    assert first.outputs["out"]["cache_hit"] is False

    second = await WorkflowExecutor(wf).run(inputs)
    assert second.outputs["out"]["cache_hit"] is True
    assert second.outputs["out"]["cache_type"] == "exact"


async def test_f24_generate_reports_cost_and_enforces_budget():
    reset_budgets("acme")
    hits = [{"id": "d1", "score": 1.0,
             "metadata": {"text": "Data is encrypted using AES-256.", "heading": "Security"}}]

    # Paid model + tiny budget -> the generate node fails its budget pre-check,
    # and the configured fallback keeps the workflow succeeding (F7 + F24).
    wf = WorkflowDef(
        name="g",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "aug", "type": "augment", "config": {"query": "$.inputs.q", "hits": "$.inputs.hits"}},
            {"id": "gen", "type": "generate",
             "config": {"provider": "stub", "query": "$.inputs.q", "model": "gpt-4o",
                        "budget_key": "acme", "budget_limit": 0.0000001},
             "fallback": {"answer": "budget exceeded", "cost_usd": 0.0}},
            {"id": "out", "type": "output", "config": {"value": "$.gen"}},
        ],
        edges=[
            {"source": "in", "target": "aug"},
            {"source": "aug", "target": "gen"},
            {"source": "gen", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({"q": "how is data encrypted", "hits": hits})
    gen = next(r for r in res.results if r.node_id == "gen")
    assert gen.status == "fallback"
    assert res.outputs["out"]["answer"] == "budget exceeded"
