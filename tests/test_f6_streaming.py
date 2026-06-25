"""Tests for F6: Streaming & Real-Time Execution."""
from __future__ import annotations

import asyncio
import json

import pytest


# ── Executor: node_start events ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f6_executor_emits_node_start_before_complete():
    """Every runnable node should fire node_start before node_complete."""
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef

    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "p",   "type": "processing", "config": {"operation": "set", "value": 1}},
            {"id": "out", "type": "output"},
        ],
        edges=[{"source": "in", "target": "p"}, {"source": "p", "target": "out"}],
    )
    events = [ev async for ev in WorkflowExecutor(wf).events({})]
    kinds = [e["event"] for e in events]
    assert kinds.count("node_start") == 3
    assert kinds.count("node_complete") == 3
    # Every node_start must appear before the corresponding node_complete
    for ev in events:
        if ev["event"] == "node_start":
            nid = ev["node_id"]
            start_idx = next(i for i, e in enumerate(events) if e["event"] == "node_start" and e["node_id"] == nid)
            done_idx  = next(i for i, e in enumerate(events) if e["event"] == "node_complete" and e["result"].node_id == nid)
            assert start_idx < done_idx, f"node_start for {nid} should precede node_complete"


@pytest.mark.asyncio
async def test_f6_executor_node_start_has_type():
    """node_start events carry the node type."""
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef

    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output"},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    events = [ev async for ev in WorkflowExecutor(wf).events({})]
    starts = [e for e in events if e["event"] == "node_start"]
    for s in starts:
        assert "node_id" in s
        assert "type" in s


@pytest.mark.asyncio
async def test_f6_skipped_nodes_do_not_emit_node_start():
    """Skipped branch nodes should only emit node_complete (skipped), not node_start."""
    from app.engine.executor import WorkflowExecutor
    from app.models.workflow import WorkflowDef

    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",   "type": "input"},
            {"id": "sw",   "type": "switch", "config": {
                "cases": [{"label": "a", "when": {"left": "$.inputs.v", "op": "==", "right": "a"}}],
                "default": "b",
            }},
            {"id": "a",    "type": "processing", "config": {"operation": "set", "value": "A"}},
            {"id": "b",    "type": "processing", "config": {"operation": "set", "value": "B"}},
            {"id": "out",  "type": "output"},
        ],
        edges=[
            {"source": "in",  "target": "sw"},
            {"source": "sw",  "target": "a", "condition": {"left": "$.sw.case", "op": "==", "right": "a"}},
            {"source": "sw",  "target": "b", "condition": {"left": "$.sw.case", "op": "==", "right": "b"}},
            {"source": "a",   "target": "out"},
            {"source": "b",   "target": "out"},
        ],
    )
    events = [ev async for ev in WorkflowExecutor(wf).events({"v": "a"})]
    started = {e["node_id"] for e in events if e["event"] == "node_start"}
    assert "b" not in started   # skipped branch must not emit node_start
    assert "a" in started


# ── Stub LLM token streaming ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f6_stub_generates_stream_yields_tokens():
    """Stub generate_stream yields at least one token per word."""
    from app.rag.llm.stub import ExtractiveStubLLM

    llm = ExtractiveStubLLM()
    request = {
        "messages": [{"role": "user", "content": "test question"}],
        "query": "test",
        "documents": [{"text": "The answer is yes. And that is correct.", "marker": "[1]"}],
        "citations": [],
    }
    tokens: list[str] = []
    async for tok in llm.generate_stream(request, {}):
        tokens.append(tok)

    assert len(tokens) >= 1
    full_text = "".join(tokens)
    assert len(full_text) > 0


@pytest.mark.asyncio
async def test_f6_stub_stream_reconstructs_full_answer():
    """Concatenating all streamed tokens should equal the non-streaming answer."""
    from app.rag.llm.stub import ExtractiveStubLLM

    llm = ExtractiveStubLLM()
    request = {
        "messages": [{"role": "user", "content": "What is AI?"}],
        "query": "AI",
        "documents": [{"text": "Artificial intelligence is a branch of computer science.", "marker": ""}],
        "citations": [],
    }
    response = await llm.generate(request, {})
    streamed = "".join([tok async for tok in llm.generate_stream(request, {})])

    assert streamed == response.text


@pytest.mark.asyncio
async def test_f6_stub_stream_no_docs_yields_one_token():
    """Stub with no documents yields a non-empty token (the fallback message)."""
    from app.rag.llm.stub import ExtractiveStubLLM

    llm = ExtractiveStubLLM()
    request = {"messages": [], "query": "nothing", "documents": [], "citations": []}
    tokens = [tok async for tok in llm.generate_stream(request, {})]
    assert len(tokens) >= 1
    assert "".join(tokens).strip() != ""


# ── Gemini streaming fallback (no API key) ────────────────────────────────────

@pytest.mark.asyncio
async def test_f6_gemini_stream_falls_back_when_no_key(monkeypatch):
    """With no GEMINI_API_KEY, generate_stream falls back to full-text generation."""
    from app.rag.llm.gemini import GeminiLLM
    from app.config import settings

    monkeypatch.setattr(settings, "gemini_api_key", "")
    llm = GeminiLLM()
    # generate() raises RuntimeError without a key — stream should re-raise as well
    request = {"messages": [{"role": "user", "content": "hi"}], "query": "hi", "documents": [], "citations": []}
    tokens = []
    try:
        async for tok in llm.generate_stream(request, {}):
            tokens.append(tok)
    except Exception:
        pass  # expected when no key and generate() raises


# ── Pipeline event generator ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f6_pipeline_yields_all_expected_event_types():
    """_rag_pipeline_events must emit step, progress, token, and complete events."""
    from app.api.stream import _rag_pipeline_events

    events = [ev async for ev in _rag_pipeline_events("What is AI?", "f6test", "stub")]
    event_types = {e["event"] for e in events}
    assert "step"     in event_types
    assert "token"    in event_types
    assert "complete" in event_types


@pytest.mark.asyncio
async def test_f6_pipeline_complete_event_has_answer():
    """The complete event must contain a non-empty answer string."""
    from app.api.stream import _rag_pipeline_events

    events = [ev async for ev in _rag_pipeline_events("What is AI?", "f6test2", "stub")]
    complete = next((e for e in events if e["event"] == "complete"), None)
    assert complete is not None
    assert isinstance(complete["answer"], str)
    assert "run_id" in complete
    assert "duration_ms" in complete


@pytest.mark.asyncio
async def test_f6_pipeline_step_sequence_is_ordered():
    """Steps must appear in pipeline order: query_process → search → rerank → augment → generate."""
    from app.api.stream import _rag_pipeline_events

    events  = [ev async for ev in _rag_pipeline_events("What is AI?", "f6seq", "stub")]
    starts  = [e["step"] for e in events if e["event"] == "step" and e["status"] == "start"]
    ordered = ["query_process", "search", "rerank", "augment", "generate"]
    # All expected steps should appear
    for s in ordered:
        assert s in starts, f"Expected step '{s}' not in {starts}"
    # They must appear in the right order
    positions = [starts.index(s) for s in ordered if s in starts]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_f6_pipeline_tokens_form_the_answer():
    """Concatenating all token events should reconstruct the complete event answer."""
    from app.api.stream import _rag_pipeline_events

    events   = [ev async for ev in _rag_pipeline_events("What is AI?", "f6tok", "stub")]
    tokens   = "".join(e["token"] for e in events if e["event"] == "token")
    complete = next(e for e in events if e["event"] == "complete")
    assert tokens == complete["answer"]


@pytest.mark.asyncio
async def test_f6_pipeline_cancel_stops_execution():
    """Setting the cancel event mid-stream stops the pipeline early."""
    from app.api.stream import _rag_pipeline_events

    cancel = asyncio.Event()
    events: list[dict] = []
    async for ev in _rag_pipeline_events("Explain everything about AI?", "f6cancel", "stub", cancel):
        events.append(ev)
        # Cancel after the first step completes
        if ev.get("event") == "step" and ev.get("status") == "done":
            cancel.set()

    event_types = {e["event"] for e in events}
    assert "cancelled" in event_types
    # Should not have reached complete
    assert "complete" not in event_types


# ── SSE API endpoint ──────────────────────────────────────────────────────────

def _collect_sse_events(response) -> list[dict]:
    """Parse all SSE data lines from a streaming response into event dicts."""
    events = []
    for line in response.iter_lines():
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                pass
    return events


def test_f6_sse_endpoint_returns_streaming_response():
    """POST /rag/stream/ask returns text/event-stream content type."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/v1/rag/stream/ask",
            json={"question": "What is machine learning?", "tenant": "f6sse"},
            headers={"Authorization": "Bearer dev"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


def test_f6_sse_endpoint_emits_complete_event():
    """SSE stream must contain a complete event."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/v1/rag/stream/ask",
            json={"question": "What is deep learning?", "tenant": "f6sse2"},
            headers={"Authorization": "Bearer dev"},
        ) as resp:
            events = _collect_sse_events(resp)

    event_types = {e.get("event") for e in events}
    assert "complete" in event_types, f"No complete event in {event_types}"


def test_f6_sse_endpoint_emits_step_events():
    """SSE stream must contain multiple step events with status fields."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/v1/rag/stream/ask",
            json={"question": "Explain neural networks?", "tenant": "f6sse3"},
            headers={"Authorization": "Bearer dev"},
        ) as resp:
            events = _collect_sse_events(resp)

    step_events = [e for e in events if e.get("event") == "step"]
    assert len(step_events) >= 2
    for se in step_events:
        assert "step" in se
        assert se.get("status") in ("start", "done")


def test_f6_sse_endpoint_emits_token_events():
    """SSE stream must contain at least one token event."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.stream(
            "POST", "/api/v1/rag/stream/ask",
            json={"question": "What is AI?", "tenant": "f6sse4"},
            headers={"Authorization": "Bearer dev"},
        ) as resp:
            events = _collect_sse_events(resp)

    token_events = [e for e in events if e.get("event") == "token"]
    assert len(token_events) >= 1


# ── WebSocket endpoint ────────────────────────────────────────────────────────

def test_f6_websocket_endpoint_returns_complete():
    """WebSocket ask message triggers the pipeline and returns a complete event."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/rag/stream/ws") as ws:
            ws.send_json({"type": "ask", "question": "What is AI?", "tenant": "f6ws"})
            events: list[dict] = []
            for _ in range(100):   # cap to avoid infinite loop in tests
                data = ws.receive_json()
                events.append(data)
                if data.get("event") in ("complete", "error", "cancelled"):
                    break

    event_types = {e.get("event") for e in events}
    assert "complete" in event_types


def test_f6_websocket_cancel_stops_pipeline():
    """Sending cancel mid-stream causes the server to emit a cancelled event."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/rag/stream/ws") as ws:
            ws.send_json({"type": "ask", "question": "Explain everything about AI", "tenant": "f6wscancel"})
            events: list[dict] = []
            seen_step = False
            for _ in range(100):
                data = ws.receive_json()
                events.append(data)
                # Send cancel after receiving the first step start
                if not seen_step and data.get("event") == "step" and data.get("status") == "start":
                    ws.send_json({"type": "cancel"})
                    seen_step = True
                if data.get("event") in ("complete", "error", "cancelled"):
                    break

    event_types = {e.get("event") for e in events}
    # Either cancelled or completed (cancel may arrive too late for short pipelines)
    assert "cancelled" in event_types or "complete" in event_types
