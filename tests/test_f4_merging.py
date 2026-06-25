"""Tests for F4: Merging Results."""
from __future__ import annotations

import pytest

from app.engine.merging import (
    answer_merge,
    concat,
    consensus,
    dedup,
    merge,
    narrative_order,
    ranking,
    rrf,
    voting,
    weighted,
)
from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef


# ── Strategy 1: concat ────────────────────────────────────────────────────────

def test_f4_concat_flattens_lists():
    result = concat([[1, 2], [3, 4], [5]], {})
    assert result == [1, 2, 3, 4, 5]


def test_f4_concat_mixed_types():
    result = concat([[1, 2], "x"], {})
    assert 1 in result and "x" in result


# ── Strategy 2: voting ────────────────────────────────────────────────────────

def test_f4_voting_majority_wins():
    result = voting([["yes", "yes", "no", "yes"]], {})
    assert result == "yes"


def test_f4_voting_single_item():
    result = voting([["only"]], {})
    assert result == "only"


# ── Strategy 3: ranking / scoring ────────────────────────────────────────────

def test_f4_ranking_sorts_by_score():
    items = [{"id": "a", "score": 1.0}, {"id": "b", "score": 3.0}, {"id": "c", "score": 2.0}]
    result = ranking([items], {"score_key": "score"})
    assert result[0]["id"] == "b"


def test_f4_ranking_top_n():
    items = [{"score": i} for i in range(10)]
    result = ranking([items], {"score_key": "score", "top_n": 3})
    assert len(result) == 3


# ── Strategy 4: dedup ─────────────────────────────────────────────────────────

def test_f4_dedup_removes_exact_duplicates():
    a = [{"id": "x", "score": 1.0}, {"id": "y", "score": 0.5}]
    b = [{"id": "x", "score": 0.8}, {"id": "z", "score": 0.3}]
    result = dedup([a, b], {"key": "id"})
    ids = [r["id"] for r in result]
    assert ids.count("x") == 1
    assert "y" in ids and "z" in ids


def test_f4_dedup_preserves_first_seen_order():
    result = dedup([[1, 2, 3, 2, 1]], {})
    assert result == [1, 2, 3]


# ── Strategy 5: consensus ─────────────────────────────────────────────────────

def test_f4_consensus_agrees_above_threshold():
    out = consensus([["yes", "yes", "yes", "no"]], {"threshold": 0.5})
    assert out["agreed"] is True
    assert out["value"] == "yes"


def test_f4_consensus_fails_below_threshold():
    out = consensus([["yes", "no", "maybe", "no"]], {"threshold": 0.9})
    assert out["agreed"] is False


# ── Strategy: RRF (hybrid search fusion) ─────────────────────────────────────

def test_f4_rrf_combines_ranked_lists():
    dense   = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.7}]
    keyword = [{"id": "b", "score": 0.8}, {"id": "c", "score": 0.6}]
    result  = rrf([dense, keyword], {"top_n": 3})
    ids = [r["id"] for r in result]
    # "b" appears in both lists → should rank highest
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_f4_rrf_top_n_respected():
    lists = [[{"id": str(i), "score": 1.0} for i in range(10)]]
    result = rrf([lists[0]], {"top_n": 3})
    assert len(result) == 3


# ── Strategy: answer_merge ────────────────────────────────────────────────────

def test_f4_answer_merge_finds_common_ground():
    answers = [
        {"provider": "llm1", "answer": "Machine learning is a subset of AI. It learns from data."},
        {"provider": "llm2", "answer": "Machine learning is a subset of AI. It uses algorithms."},
    ]
    result = answer_merge(answers, {})
    assert result["source_count"] == 2
    assert len(result["common_ground"]) >= 1


def test_f4_answer_merge_notes_differences():
    answers = [
        {"provider": "llm1", "answer": "The answer is definitely yes."},
        {"provider": "llm2", "answer": "The answer is probably no."},
    ]
    result = answer_merge(answers, {})
    assert result["source_count"] == 2
    assert isinstance(result["differences"], list)


def test_f4_answer_merge_single_source():
    result = answer_merge([{"answer": "Only one answer."}], {})
    assert result["source_count"] == 1
    assert result["common_ground"] == []


def test_f4_answer_merge_empty():
    result = answer_merge([], {})
    assert result["source_count"] == 0
    assert result["answer"] == ""


# ── Strategy: narrative_order ─────────────────────────────────────────────────

def test_f4_narrative_order_sorts_by_chunk_index():
    chunks = [
        {"id": "c3", "metadata": {"chunk_index": 2, "text": "End of Q3."}},
        {"id": "c1", "metadata": {"chunk_index": 0, "text": "Start of Q3."}},
        {"id": "c2", "metadata": {"chunk_index": 1, "text": "Mid Q3."}},
    ]
    result = narrative_order([chunks], {})
    assert [r["id"] for r in result] == ["c1", "c2", "c3"]


def test_f4_narrative_order_handles_no_metadata():
    chunks = [{"id": "x"}, {"id": "y"}]
    result = narrative_order([chunks], {})
    assert len(result) == 2


# ── Entity Search node ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f4_entity_search_finds_matching_records():
    """Index a doc then entity-search for a named entity it contains."""
    from app.api.pipelines import STORE, DIM, build_index_workflow
    from app.models.workflow import WorkflowDef

    # Index a document mentioning "Tesla" and "2023"
    idx = await WorkflowExecutor(build_index_workflow()).run({
        "tenant": "f4test",
        "text": "Tesla released its Q3 2023 earnings report showing strong growth.",
        "filename": "tesla.txt",
    })
    assert idx.status == "success"

    # Entity search for "Tesla 2023"
    wf = WorkflowDef(
        name="entity_search_test",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "es",  "type": "entity_search",
             "config": {"store": STORE, "namespace": "f4test",
                        "query": "$.inputs.query", "dimension": DIM, "top_k": 5}},
            {"id": "out", "type": "output", "config": {"value": "$.es"}},
        ],
        edges=[
            {"source": "in",  "target": "es"},
            {"source": "es",  "target": "out"},
        ],
    )
    result = await WorkflowExecutor(wf).run({"query": "Tesla 2023"})
    assert result.status == "success"
    hits = result.outputs.get("out") or []
    assert len(hits) >= 1
    assert any("Tesla" in str(h.get("matched_entities", [])) for h in hits)


# ── merge() dispatcher ────────────────────────────────────────────────────────

def test_f4_merge_dispatcher_unknown_strategy():
    import pytest
    with pytest.raises(ValueError, match="Unknown merge strategy"):
        merge("nonexistent", [[1, 2]], {})


def test_f4_all_strategies_callable():
    from app.engine.merging import STRATEGIES
    expected = {"concat", "voting", "ranking", "weighted", "dedup",
                "consensus", "rrf", "answer_merge", "narrative_order"}
    assert expected.issubset(set(STRATEGIES))


# ── Three-way merge pipeline (semantic + keyword + entity) ───────────────────

@pytest.mark.asyncio
async def test_f4_three_way_merge_via_workflow():
    """Parallel semantic+keyword+entity branches fused with rrf."""
    from app.api.pipelines import STORE, DIM, build_index_workflow

    # Index a document
    await WorkflowExecutor(build_index_workflow()).run({
        "tenant": "f4merge",
        "text": "Q3 2022 results: revenue increased by 20 percent driven by Cloud division.",
        "filename": "q3_2022.txt",
    })

    wf = WorkflowDef(
        name="three_way_merge",
        nodes=[
            {"id": "in",     "type": "input"},
            {"id": "dense",  "type": "vector_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.inputs.query", "dimension": DIM, "top_k": 3}},
            {"id": "bm25",   "type": "keyword_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.inputs.query", "dimension": DIM, "top_k": 3}},
            {"id": "entity", "type": "entity_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.inputs.query", "dimension": DIM, "top_k": 3}},
            {"id": "fuse",   "type": "merge",
             "config": {"strategy": "rrf", "top_n": 5}},
            {"id": "out",    "type": "output", "config": {"value": "$.fuse"}},
        ],
        edges=[
            {"source": "in",     "target": "dense"},
            {"source": "in",     "target": "bm25"},
            {"source": "in",     "target": "entity"},
            {"source": "dense",  "target": "fuse"},
            {"source": "bm25",   "target": "fuse"},
            {"source": "entity", "target": "fuse"},
            {"source": "fuse",   "target": "out"},
        ],
    )
    result = await WorkflowExecutor(wf).run({"tenant": "f4merge", "query": "Q3 2022 revenue"})
    assert result.status == "success"
    hits = result.outputs.get("out") or []
    assert isinstance(hits, list)
