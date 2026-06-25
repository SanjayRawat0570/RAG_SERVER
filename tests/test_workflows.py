"""Tests for the orchestration engine, organized by feature F1-F4."""
import pytest

from app.engine.conditions import evaluate
from app.engine.context import ExecutionContext
from app.engine.executor import WorkflowExecutor
from app.engine.graph import WorkflowGraphError, build_graph
from app.engine.merging import merge
from app.models.workflow import WorkflowDef


def _wf(nodes, edges, name="t"):
    return WorkflowDef(name=name, nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- F1
async def test_f1_linear_chain_passes_context():
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input", "config": {"key": "text"}},
            {"id": "trim", "type": "processing", "config": {"operation": "strip"}},
            {"id": "up", "type": "processing", "config": {"operation": "uppercase"}},
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in", "target": "trim"},
            {"source": "trim", "target": "up"},
            {"source": "up", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({"text": "  hi there  "})
    assert res.status == "success"
    assert res.outputs["out"] == "HI THERE"


async def test_f1_cycle_is_rejected():
    wf = _wf(
        nodes=[
            {"id": "a", "type": "processing"},
            {"id": "b", "type": "processing"},
        ],
        edges=[
            {"source": "a", "target": "b"},
            {"source": "b", "target": "a"},
        ],
    )
    with pytest.raises(WorkflowGraphError):
        build_graph(wf)


async def test_f1_retry_then_fallback():
    # 'add' of a string + int raises -> exhaust retries -> fallback kicks in.
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input", "config": {"key": "v"}},
            {
                "id": "boom",
                "type": "processing",
                "config": {"operation": "add", "amount": 2},
                "retry": {"max_attempts": 3},
                "fallback": "RECOVERED",
            },
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in", "target": "boom"},
            {"source": "boom", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({"v": "not-a-number"})
    boom = next(r for r in res.results if r.node_id == "boom")
    assert boom.status == "fallback"
    assert boom.attempts == 3
    assert res.outputs["out"] == "RECOVERED"
    assert res.status == "success"


# --------------------------------------------------------------------------- F2
def test_f2_condition_operators():
    ctx = ExecutionContext("r", inputs={"score": 0.9, "tags": ["a", "b"], "q": "hello"})
    assert evaluate({"left": "$.inputs.score", "op": ">", "right": 0.8}, ctx)
    assert evaluate({"left": "$.inputs.q", "op": "contains", "right": "ell"}, ctx)
    assert evaluate({"left": "$.inputs.tags", "op": "length_eq", "right": 2}, ctx)
    assert evaluate({"left": "$.inputs.tags", "op": "contains_item", "right": "a"}, ctx)
    assert evaluate(
        {"and": [
            {"left": "$.inputs.score", "op": ">=", "right": 0.5},
            {"not": {"left": "$.inputs.q", "op": "==", "right": "bye"}},
        ]},
        ctx,
    )


async def test_f2_branching_takes_one_path():
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "d", "type": "decision", "config": {"input": "$.inputs.confidence"}},
            {"id": "high", "type": "processing", "config": {"operation": "set", "value": "HIGH"}},
            {"id": "low", "type": "processing", "config": {"operation": "set", "value": "LOW"}},
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in", "target": "d"},
            {"source": "d", "target": "high",
             "condition": {"left": "$.inputs.confidence", "op": ">", "right": 0.8}},
            {"source": "d", "target": "low",
             "condition": {"left": "$.inputs.confidence", "op": "<=", "right": 0.8}},
            {"source": "high", "target": "out"},
            {"source": "low", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({"confidence": 0.5})
    statuses = {r.node_id: r.status for r in res.results}
    assert statuses["high"] == "skipped"
    assert statuses["low"] == "success"
    assert res.outputs["out"] == "LOW"


# --------------------------------------------------------------------------- F3
async def test_f3_parallel_branches_all_run():
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "a", "type": "processing", "config": {"operation": "set", "value": 1}},
            {"id": "b", "type": "processing", "config": {"operation": "set", "value": 2}},
            {"id": "c", "type": "processing", "config": {"operation": "set", "value": 3}},
            {"id": "m", "type": "merge", "config": {"strategy": "concat"}},
        ],
        edges=[
            {"source": "in", "target": "a"},
            {"source": "in", "target": "b"},
            {"source": "in", "target": "c"},
            {"source": "a", "target": "m"},
            {"source": "b", "target": "m"},
            {"source": "c", "target": "m"},
        ],
    )
    res = await WorkflowExecutor(wf).run({})
    assert sorted(res.outputs["m"]) == [1, 2, 3]


# --------------------------------------------------------------------------- F4
def test_f4_strategies():
    assert merge("concat", [[1, 2], [3]], {}) == [1, 2, 3]
    assert merge("voting", [["x", "y"], ["x"]], {}) == "x"
    assert merge("dedup", [[{"id": 1}, {"id": 1}, {"id": 2}]], {"key": "id"}) == [
        {"id": 1}, {"id": 2}
    ]
    ranked = merge(
        "ranking",
        [[{"d": "A", "score": 0.2}], [{"d": "B", "score": 0.9}]],
        {"score_key": "score", "top_n": 1},
    )
    assert ranked == [{"d": "B", "score": 0.9}]
    cons = merge("consensus", [["yes", "yes", "no"]], {"threshold": 0.6})
    assert cons["agreed"] is True and cons["value"] == "yes"


async def test_f4_merge_node_ranking_end_to_end():
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "s1", "type": "processing",
             "config": {"operation": "set", "value": [{"d": "A", "score": 0.4}]}},
            {"id": "s2", "type": "processing",
             "config": {"operation": "set", "value": [{"d": "B", "score": 0.95}]}},
            {"id": "m", "type": "merge",
             "config": {"strategy": "ranking", "score_key": "score", "top_n": 1}},
            {"id": "out", "type": "output", "config": {"value": "$.m"}},
        ],
        edges=[
            {"source": "in", "target": "s1"},
            {"source": "in", "target": "s2"},
            {"source": "s1", "target": "m"},
            {"source": "s2", "target": "m"},
            {"source": "m", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({})
    assert res.outputs["out"] == [{"d": "B", "score": 0.95}]
