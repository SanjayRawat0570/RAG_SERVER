"""F6: Streaming & Real-Time Execution

Server-Sent Events endpoint and WebSocket endpoint that run the full RAG
pipeline step-by-step and stream LLM tokens as they are generated.

Event protocol (same shape for SSE data and WebSocket messages)
---------------------------------------------------------------
{"event": "step",     "step": "query_process",  "status": "start", "message": "..."}
{"event": "step",     "step": "search",         "status": "done",  "hit_count": N}
{"event": "progress", "step": "search",         "pct": 50,         "message": "..."}
{"event": "token",    "token": "word "}
{"event": "complete", "answer": "...", "citations": [...], "duration_ms": N}
{"event": "error",    "detail": "..."}
{"event": "cancelled"}

Endpoints
---------
POST /rag/stream/ask   Server-Sent Events (SSE) with cancel via client disconnect
WS   /rag/stream/ws    WebSocket with cancel via {"type": "cancel"} message
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.api.pipelines import STORE, DIM, default_provider
from app.engine.context import ExecutionContext
from app.engine.merging import rrf
from app.engine.nodes import get_node
from app.models.workflow import NodeDef, NodeType
from app.rag.llm import get_llm

router = APIRouter(prefix="/rag/stream", tags=["streaming"])


# ── Request models ─────────────────────────────────────────────────────────────

class StreamAskRequest(BaseModel):
    question: str
    tenant: str = "default"
    provider: str | None = None


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _nd(node_type: NodeType, node_id: str, config: dict[str, Any]) -> Any:
    """Instantiate a node from its type + config."""
    return get_node(NodeDef(id=node_id, type=node_type, config=config))


async def _rag_pipeline_events(
    question: str,
    tenant: str,
    provider_name: str,
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the full RAG pipeline step-by-step, yielding event dicts.

    Steps (mirrors the build_ask_workflow pipeline):
        query_process → vector_search + keyword_search → rrf → rerank → augment → generate
    """
    run_id = str(uuid.uuid4())
    ctx = ExecutionContext(run_id=run_id, inputs={"question": question, "tenant": tenant})
    t0 = time.perf_counter()

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    # ── Step 1: Query understanding ────────────────────────────────────────────
    yield {"event": "step", "step": "query_process", "status": "start",
           "message": "Understanding your question..."}
    qp = _nd(NodeType.QUERY_PROCESS, "qp", {"query": "$.inputs.question"})
    qp_out = await qp.run(ctx, {})
    ctx.set_output("qp", qp_out)
    yield {"event": "step", "step": "query_process", "status": "done",
           "intent": qp_out.get("intent"), "normalized": qp_out.get("normalized")}

    if cancelled():
        yield {"event": "cancelled"}
        return

    # ── Step 2: Hybrid search ──────────────────────────────────────────────────
    yield {"event": "step", "step": "search", "status": "start",
           "message": "Searching knowledge base..."}
    yield {"event": "progress", "step": "search", "pct": 10,
           "message": "Running semantic search..."}

    dense_node = _nd(NodeType.VECTOR_SEARCH, "dense", {
        "store": STORE, "namespace": "$.inputs.tenant",
        "query": "$.qp.normalized", "dimension": DIM, "top_k": 5,
    })
    sparse_node = _nd(NodeType.KEYWORD_SEARCH, "sparse", {
        "store": STORE, "namespace": "$.inputs.tenant",
        "query": "$.qp.expanded_query", "dimension": DIM, "top_k": 5,
    })
    dense_out, sparse_out = await asyncio.gather(
        dense_node.run(ctx, {"qp": qp_out}),
        sparse_node.run(ctx, {"qp": qp_out}),
    )
    ctx.set_output("dense", dense_out)
    ctx.set_output("sparse", sparse_out)

    yield {"event": "progress", "step": "search", "pct": 70,
           "message": "Merging and deduplicating results..."}
    merged = rrf([dense_out or [], sparse_out or []], {"top_n": 5})
    ctx.set_output("fuse", merged)
    yield {"event": "step", "step": "search", "status": "done",
           "hit_count": len(merged)}

    if cancelled():
        yield {"event": "cancelled"}
        return

    # ── Step 3: Rerank ─────────────────────────────────────────────────────────
    yield {"event": "step", "step": "rerank", "status": "start",
           "message": "Reranking results by relevance..."}
    rerank_node = _nd(NodeType.RERANK, "rerank", {
        "method": "cross_encoder", "query": "$.qp.normalized", "top_n": 3,
    })
    reranked = await rerank_node.run(ctx, {"fuse": merged})
    ctx.set_output("rerank", reranked)
    yield {"event": "step", "step": "rerank", "status": "done",
           "result_count": len(reranked) if isinstance(reranked, list) else 0}

    # ── Step 4: Augment (build prompt context) ─────────────────────────────────
    yield {"event": "step", "step": "augment", "status": "start",
           "message": "Building context window..."}
    augment_node = _nd(NodeType.AUGMENT, "augment", {
        "query": "$.qp.normalized", "max_context_tokens": 600,
    })
    augment_out = await augment_node.run(ctx, {"rerank": reranked})
    ctx.set_output("augment", augment_out)
    token_est = augment_out.get("token_estimate", 0) if isinstance(augment_out, dict) else 0
    yield {"event": "step", "step": "augment", "status": "done",
           "context_tokens": token_est,
           "doc_count": augment_out.get("included", 0) if isinstance(augment_out, dict) else 0}

    if cancelled():
        yield {"event": "cancelled"}
        return

    # ── Step 5: Generate (token streaming) ────────────────────────────────────
    yield {"event": "step", "step": "generate", "status": "start",
           "message": "Generating response..."}

    llm = get_llm(provider_name)
    if isinstance(augment_out, dict) and "messages" in augment_out:
        llm_request = {
            "messages":  augment_out["messages"],
            "query":     qp_out.get("normalized", question),
            "documents": augment_out.get("documents", []),
            "citations": augment_out.get("citations", []),
        }
    else:
        llm_request = {
            "messages": [{"role": "user", "content": question}],
            "query": question, "documents": [], "citations": [],
        }

    llm_config = {"provider": provider_name, "max_tokens": 2048}
    full_answer = ""
    token_count = 0

    if hasattr(llm, "generate_stream"):
        async for token in llm.generate_stream(llm_request, llm_config):
            if cancelled():
                break
            full_answer += token
            token_count += 1
            yield {"event": "token", "token": token}
    else:
        # Fallback: non-streaming provider → yield full text as one token
        response = await llm.generate(llm_request, llm_config)
        full_answer = response.text
        yield {"event": "token", "token": full_answer}
        token_count = 1

    yield {"event": "step", "step": "generate", "status": "done",
           "token_count": token_count}

    # ── Complete ───────────────────────────────────────────────────────────────
    citations = augment_out.get("citations", []) if isinstance(augment_out, dict) else []
    yield {
        "event":       "complete",
        "answer":      full_answer,
        "citations":   citations,
        "intent":      qp_out.get("intent"),
        "hit_count":   len(merged),
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        "run_id":      run_id,
    }


# ── SSE endpoint ───────────────────────────────────────────────────────────────

@router.post("/ask")
async def stream_rag_ask(
    body: StreamAskRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """
    Full RAG pipeline streamed as Server-Sent Events (F6).

    Emits progress events for each pipeline step, then streams LLM tokens
    one-by-one as the answer is generated. Stops immediately if the client
    disconnects (cancel on HTTP close).

    Event types: ``step`` · ``progress`` · ``token`` · ``complete`` · ``error`` · ``cancelled``
    """
    provider = body.provider or default_provider()

    async def event_source() -> AsyncIterator[str]:
        try:
            async for ev in _rag_pipeline_events(body.question, body.tenant, provider):
                if await http_request.is_disconnected():
                    break
                yield f"event: {ev['event']}\ndata: {json.dumps(ev)}\n\n"
        except Exception as exc:  # noqa: BLE001
            err_ev = {"event": "error", "detail": str(exc)}
            yield f"event: error\ndata: {json.dumps(err_ev)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
        },
    )


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@router.websocket("/ws")
async def ws_rag_ask(ws: WebSocket) -> None:
    """
    Full RAG pipeline over WebSocket (F6 — bidirectional).

    Protocol:
      Client → Server:  ``{"type": "ask",    "question": "...", "tenant": "...", "provider": "..."}``
      Server → Client:  step / progress / token / complete / error / cancelled events
      Client → Server:  ``{"type": "cancel"}``  — abort the current pipeline

    The connection stays open after completion so the client can ask follow-up
    questions. Send ``{"type": "close"}`` or disconnect to end the session.
    """
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "ask")

            if msg_type == "close":
                await ws.close()
                return

            if msg_type != "ask":
                await ws.send_json({"event": "error", "detail": f"Unknown type: {msg_type}"})
                continue

            question    = str(data.get("question", ""))
            tenant      = str(data.get("tenant", "default"))
            provider    = str(data.get("provider") or default_provider())
            cancel      = asyncio.Event()

            # Listen for cancel messages concurrently while the pipeline runs
            async def _monitor_cancel() -> None:
                while not cancel.is_set():
                    try:
                        incoming = await asyncio.wait_for(ws.receive_json(), timeout=0.05)
                        if incoming.get("type") == "cancel":
                            cancel.set()
                            return
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        cancel.set()
                        return

            monitor_task = asyncio.create_task(_monitor_cancel())
            try:
                async for ev in _rag_pipeline_events(question, tenant, provider, cancel):
                    await ws.send_json(ev)
                    if ev.get("event") in ("complete", "cancelled", "error"):
                        break
            finally:
                monitor_task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.send_json({"event": "error", "detail": "Internal server error"})
        except Exception:
            pass
