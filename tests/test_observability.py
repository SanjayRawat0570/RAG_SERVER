"""Tests for F8 observability: Prometheus metrics + trace context."""
from prometheus_client import REGISTRY

from app.engine.executor import WorkflowExecutor
from app.observability.metrics import render_latest
from app.observability.tracing import current_trace_id, get_tracer, init_tracing
from app.models.workflow import WorkflowDef


def _simple_wf():
    return WorkflowDef(
        name="obs_wf",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "p", "type": "processing", "config": {"operation": "set", "value": 1}},
            {"id": "out", "type": "output"},
        ],
        edges=[{"source": "in", "target": "p"}, {"source": "p", "target": "out"}],
    )


def _sample(name, labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def test_f8_metrics_increment_on_run():
    before_wf = _sample("workflow_runs_total", {"workflow": "obs_wf", "status": "success"})
    before_node = _sample("node_runs_total", {"type": "processing", "status": "success"})

    await WorkflowExecutor(_simple_wf()).run({})

    assert _sample("workflow_runs_total", {"workflow": "obs_wf", "status": "success"}) == before_wf + 1
    assert _sample("node_runs_total", {"type": "processing", "status": "success"}) >= before_node + 1

    payload, content_type = render_latest()
    assert b"workflow_duration_seconds" in payload
    assert "text/plain" in content_type


def test_f8_trace_id_present_inside_span():
    init_tracing()
    tracer = get_tracer("test")
    assert current_trace_id() is None  # no active span yet
    with tracer.start_as_current_span("unit"):
        tid = current_trace_id()
        assert tid is not None and len(tid) == 32  # 128-bit hex trace id
