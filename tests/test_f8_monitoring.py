"""Tests for F8: Monitoring, Observability & Distributed Tracing."""
from __future__ import annotations

import pytest


# ── Query log unit tests ───────────────────────────────────────────────────────

def test_f8_log_query_returns_id():
    from app.observability.query_log import log_query
    qid = log_query(
        user_id="u1", question="What is AI?", answer="AI is ...",
        confidence=0.9, sources_count=3, duration_ms=500.0, provider="stub",
    )
    assert isinstance(qid, str) and len(qid) > 0


def test_f8_log_query_appears_in_history():
    from app.observability.query_log import log_query, get_user_history
    qid = log_query(user_id="u_hist", question="Revenue?", answer="High",
                    confidence=0.8, sources_count=2, duration_ms=300.0)
    history = get_user_history("u_hist")
    ids = [e["id"] for e in history]
    assert qid in ids


def test_f8_history_scoped_to_user():
    from app.observability.query_log import log_query, get_user_history
    log_query(user_id="user_a", question="Q for A", answer="A", confidence=0.9, sources_count=1, duration_ms=100.0)
    log_query(user_id="user_b", question="Q for B", answer="B", confidence=0.7, sources_count=2, duration_ms=200.0)
    a_hist = get_user_history("user_a")
    b_hist = get_user_history("user_b")
    a_qs = [e["question"] for e in a_hist]
    b_qs = [e["question"] for e in b_hist]
    assert "Q for A" in a_qs
    assert "Q for B" not in a_qs
    assert "Q for B" in b_qs


def test_f8_history_most_recent_first():
    from app.observability.query_log import log_query, get_user_history
    q1 = log_query(user_id="u_order", question="First",  answer="1", confidence=0.5, sources_count=0, duration_ms=100.0)
    q2 = log_query(user_id="u_order", question="Second", answer="2", confidence=0.5, sources_count=0, duration_ms=100.0)
    history = get_user_history("u_order")
    ids = [e["id"] for e in history]
    assert ids.index(q2) < ids.index(q1)  # q2 (newer) appears first


def test_f8_submit_feedback_updates_rating():
    from app.observability.query_log import log_query, submit_feedback, get_user_history
    qid = log_query(user_id="u_rate", question="Q?", answer="A", confidence=0.8, sources_count=1, duration_ms=200.0)
    ok = submit_feedback(qid, "u_rate", 5)
    assert ok is True
    history = get_user_history("u_rate")
    entry = next(e for e in history if e["id"] == qid)
    assert entry["rating"] == 5


def test_f8_submit_feedback_clamps_rating():
    from app.observability.query_log import log_query, submit_feedback, get_user_history
    qid = log_query(user_id="u_clamp", question="Q?", answer="A", confidence=0.7, sources_count=1, duration_ms=150.0)
    submit_feedback(qid, "u_clamp", 99)  # should clamp to 5
    history = get_user_history("u_clamp")
    entry = next(e for e in history if e["id"] == qid)
    assert entry["rating"] == 5


def test_f8_submit_feedback_wrong_user_returns_false():
    from app.observability.query_log import log_query, submit_feedback
    qid = log_query(user_id="u_real", question="Q?", answer="A", confidence=0.7, sources_count=1, duration_ms=150.0)
    ok = submit_feedback(qid, "u_imposter", 5)
    assert ok is False


def test_f8_submit_feedback_unknown_id_returns_false():
    from app.observability.query_log import submit_feedback
    ok = submit_feedback("nonexistent-uuid", "u1", 3)
    assert ok is False


def test_f8_dashboard_stats_returns_correct_counts():
    from app.observability.query_log import log_query, dashboard_stats
    uid = "u_dash"
    log_query(user_id=uid, question="Q1", answer="A1", confidence=0.8, sources_count=2, duration_ms=400.0)
    log_query(user_id=uid, question="Q2", answer="A2", confidence=0.6, sources_count=1, duration_ms=600.0)
    stats = dashboard_stats(uid)
    assert stats["total_queries"] >= 2
    assert 0 < stats["avg_duration_ms"]
    assert 0 < stats["avg_confidence"] <= 1.0


def test_f8_dashboard_stats_empty_user():
    from app.observability.query_log import dashboard_stats
    stats = dashboard_stats("no_such_user_xyz")
    assert stats["total_queries"] == 0


# ── Span store unit tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f8_spans_captured_after_workflow():
    """Running a workflow should populate the in-memory span store."""
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef
    from app.observability.span_store import get_exporter

    wf = WorkflowDef(
        name="span_test",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": "done"}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    await WorkflowExecutor(wf).run({})
    spans = get_exporter().get_finished_spans()
    names = [s.name for s in spans]
    assert any("workflow" in n or "node" in n for n in names)


@pytest.mark.asyncio
async def test_f8_list_traces_returns_summaries():
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef
    from app.observability.span_store import list_traces

    wf = WorkflowDef(
        name="trace_list_test",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": 42}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    await WorkflowExecutor(wf).run({})
    traces = list_traces(limit=50)
    assert isinstance(traces, list)
    assert len(traces) > 0
    first = traces[0]
    assert "trace_id" in first
    assert "span_count" in first
    assert "status" in first


@pytest.mark.asyncio
async def test_f8_get_trace_returns_spans():
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef
    from app.observability.span_store import list_traces, get_trace

    wf = WorkflowDef(
        name="trace_drill_test",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": "span_test"}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    await WorkflowExecutor(wf).run({})
    traces = list_traces(limit=1)
    assert traces
    trace_id = traces[0]["trace_id"]
    spans = get_trace(trace_id)
    assert len(spans) > 0
    for span in spans:
        assert "name" in span
        assert "duration_ms" in span
        assert span["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_f8_get_trace_unknown_id_returns_empty():
    from app.observability.span_store import get_trace
    result = get_trace("0000000000000000000000000000dead")
    assert result == []


# ── API endpoint tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f8_api_dashboard():
    """GET /monitoring/dashboard returns expected keys."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/monitoring/dashboard",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries"   in data
    assert "avg_duration_ms" in data
    assert "avg_confidence"  in data
    assert "metrics_endpoint" in data


@pytest.mark.asyncio
async def test_f8_api_history_returns_list():
    """GET /monitoring/history returns queries list."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/monitoring/history",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "queries" in data
    assert isinstance(data["queries"], list)


@pytest.mark.asyncio
async def test_f8_api_rag_ask_logs_query():
    """POST /rag/ask logs the query in the monitoring store."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.observability.query_log import get_user_history

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/rag/ask",
            json={"question": "What is machine learning?", "tenant": "f8test"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    # The query should now appear in dev user's history
    history = get_user_history("dev")
    questions = [e["question"] for e in history]
    assert "What is machine learning?" in questions


@pytest.mark.asyncio
async def test_f8_api_feedback_not_found():
    """POST /monitoring/feedback with unknown id returns 404."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/monitoring/feedback/unknown-uuid-1234",
            json={"rating": 4},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_f8_api_feedback_roundtrip():
    """Log a query, then submit feedback via API, then verify in history."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.observability.query_log import log_query

    # Create an entry directly for the dev user
    qid = log_query(
        user_id="dev", question="Test feedback?", answer="Yes",
        confidence=0.9, sources_count=1, duration_ms=100.0,
    )

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        resp = client.post(
            f"/api/v1/monitoring/feedback/{qid}",
            json={"rating": 4, "note": "Good answer"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rating"] == 4
    assert data["query_id"] == qid


@pytest.mark.asyncio
async def test_f8_api_traces_list():
    """GET /monitoring/traces returns traces list."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/monitoring/traces",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "traces" in data
    assert isinstance(data["traces"], list)


@pytest.mark.asyncio
async def test_f8_api_spans_unknown_trace():
    """GET /monitoring/spans/<unknown> returns 404."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/monitoring/spans/0000000000000000000000000000cafe",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_f8_api_spans_known_trace():
    """Run a workflow, then retrieve its spans via API."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef
    from app.observability.span_store import list_traces

    wf = WorkflowDef(
        name="api_span_test",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": "span"}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    await WorkflowExecutor(wf).run({})
    traces = list_traces(limit=1)
    trace_id = traces[0]["trace_id"]

    with TestClient(app) as client:
        resp = client.get(
            f"/api/v1/monitoring/spans/{trace_id}",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["trace_id"] == trace_id
    assert len(data["spans"]) > 0
    span = data["spans"][0]
    assert "name" in span
    assert "duration_ms" in span


@pytest.mark.asyncio
async def test_f8_prometheus_metrics_endpoint():
    """GET /metrics returns Prometheus text format."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    assert "workflow_runs_total" in resp.text or "node_runs_total" in resp.text


@pytest.mark.asyncio
async def test_f8_metrics_increment_after_workflow():
    """Running a workflow increments Prometheus counters."""
    from prometheus_client import REGISTRY
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef

    def _count(metric_name: str) -> float:
        try:
            return sum(
                sample.value
                for metric in REGISTRY.collect()
                if metric.name == metric_name
                for sample in metric.samples
                if sample.name.endswith("_total")
            )
        except Exception:
            return 0.0

    before = _count("workflow_runs")
    wf = WorkflowDef(
        name="prom_test",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": "ok"}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    await WorkflowExecutor(wf).run({})
    after = _count("workflow_runs")
    assert after > before
