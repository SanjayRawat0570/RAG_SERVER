"""Tests for F5 (decision trees), F6 (streaming), F7 (resilience),
and the F1 round-out (nested sub-workflows + loops)."""
import pytest

from app.engine.circuit_breaker import OPEN, registry
from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef


def _wf(nodes, edges, name="t"):
    return WorkflowDef(name=name, nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- F5
async def test_f5_switch_routes_to_matching_case():
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "sw", "type": "switch", "config": {
                "input": "$.inputs.score",
                "cases": [
                    {"label": "high", "when": {"left": "$.inputs.score", "op": ">", "right": 0.8}},
                    {"label": "mid", "when": {"left": "$.inputs.score", "op": ">=", "right": 0.5}},
                ],
                "default": "low",
            }},
            {"id": "h", "type": "processing", "config": {"operation": "set", "value": "H"}},
            {"id": "m", "type": "processing", "config": {"operation": "set", "value": "M"}},
            {"id": "l", "type": "processing", "config": {"operation": "set", "value": "L"}},
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in", "target": "sw"},
            {"source": "sw", "target": "h", "condition": {"left": "$.sw.case", "op": "==", "right": "high"}},
            {"source": "sw", "target": "m", "condition": {"left": "$.sw.case", "op": "==", "right": "mid"}},
            {"source": "sw", "target": "l", "condition": {"left": "$.sw.case", "op": "==", "right": "low"}},
            {"source": "h", "target": "out"},
            {"source": "m", "target": "out"},
            {"source": "l", "target": "out"},
        ],
    )
    res = await WorkflowExecutor(wf).run({"score": 0.65})
    statuses = {r.node_id: r.status for r in res.results}
    assert statuses["m"] == "success"
    assert statuses["h"] == "skipped" and statuses["l"] == "skipped"
    assert res.outputs["out"] == "M"


# --------------------------------------------------------------------------- F6
async def test_f6_streaming_emits_events_in_order():
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "p", "type": "processing", "config": {"operation": "set", "value": 1}},
            {"id": "out", "type": "output"},
        ],
        edges=[{"source": "in", "target": "p"}, {"source": "p", "target": "out"}],
    )
    events = [ev async for ev in WorkflowExecutor(wf).events({})]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "workflow_start"
    assert kinds[-1] == "workflow_end"
    assert kinds.count("node_complete") == 3
    assert events[-1]["status"] == "success"


# --------------------------------------------------------------------------- F7
async def test_f7_circuit_breaker_opens_after_threshold():
    registry.reset("svc")
    # 'add' string+int raises every time; threshold 2 => breaker opens.
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input", "config": {"key": "v"}},
            {"id": "x", "type": "processing", "config": {
                "operation": "add", "amount": 1,
                "breaker_key": "svc", "breaker_threshold": 2,
            }},
        ],
        edges=[{"source": "in", "target": "x"}],
    )
    executor = WorkflowExecutor(wf)
    await executor.run({"v": "bad"})
    await executor.run({"v": "bad"})
    assert registry.get("svc").state == OPEN
    registry.reset("svc")


# ----------------------------------------------------------------- F1 nested
async def test_f1_nested_subworkflow():
    child = {
        "name": "double",
        "nodes": [
            {"id": "i", "type": "input", "config": {"key": "n"}},
            {"id": "d", "type": "processing", "config": {"operation": "multiply", "factor": 2}},
            {"id": "o", "type": "output"},
        ],
        "edges": [{"source": "i", "target": "d"}, {"source": "d", "target": "o"}],
    }
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "sub", "type": "subworkflow", "config": {
                "workflow": child,
                "input_map": {"n": "$.inputs.value"},
                "output_key": "o",
            }},
            {"id": "out", "type": "output", "config": {"value": "$.sub"}},
        ],
        edges=[{"source": "in", "target": "sub"}, {"source": "sub", "target": "out"}],
    )
    res = await WorkflowExecutor(wf).run({"value": 21})
    assert res.outputs["out"] == 42


# ------------------------------------------------------------------- F1 loop
async def test_f1_loop_until_condition():
    body = {
        "name": "increment",
        "nodes": [
            {"id": "i", "type": "input", "config": {"key": "state"}},
            {"id": "inc", "type": "processing", "config": {"operation": "add", "amount": 10}},
            {"id": "o", "type": "output"},
        ],
        "edges": [{"source": "i", "target": "inc"}, {"source": "inc", "target": "o"}],
    }
    wf = _wf(
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "loop", "type": "loop", "config": {
                "workflow": body,
                "input_map": {"state": "$.inputs.start"},
                "state_key": "state",
                "output_key": "o",
                "until": {"left": "$.inputs.state", "op": ">=", "right": 50},
                "max_iterations": 20,
            }},
            {"id": "out", "type": "output", "config": {"value": "$.loop"}},
        ],
        edges=[{"source": "in", "target": "loop"}, {"source": "loop", "target": "out"}],
    )
    res = await WorkflowExecutor(wf).run({"start": 0})
    assert res.outputs["out"]["output"] == 50
    assert res.outputs["out"]["iterations"] == 5
