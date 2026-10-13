"""F6: Streaming & Real-Time Execution

Server-Sent Events endpoint and WebSocket endpoint that run the full RAG
pipeline step-by-step and stream LLM tokens as they are generated.

Event protocol (same shape for SSE data and WebSocket messages)
---------------------------------------------------------------
{"event": "step",     "step": "query_process",  "status": "start", "message": "..."}
{"event": "step",     "step": "search",         "status": "done",  "hit_count": N}
{"event": "progress", "step": "search",         "pct": 50,         "message": "..."}
{"event": "token",    "token": "word "}             (text of the JSON "answer" field)
{"event": "sources",  "answer": "...", "replaced": bool, "citations": [...cited],
                      "confidence": "high|medium|low", "answerable": bool}
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
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.api.pipelines import DIM, STORE, build_agentic_retrieval_workflow, default_provider
from app.engine.context import ExecutionContext
from app.engine.executor import WorkflowExecutor
from app.engine.merging import rrf
from app.engine.nodes import get_node
from app.models.workflow import NodeDef, NodeType
from app.rag.cost import BudgetExceededError
from app.rag.memory import service as memory
from app.rag.tenants.ai_settings import effective_examples
from app.rag.llm.gateway import redact, stream_llm
from app.rag.llm.structured import (
    AnswerStreamExtractor,
    generate_structured,
    structured_config,
    with_instructions,
)

router = APIRouter(prefix="/rag/stream", tags=["streaming"])


# ── Request models ─────────────────────────────────────────────────────────────

class StreamAskRequest(BaseModel):
    question: str
    tenant: str = "default"
    provider: str | None = None
    few_shot: bool | None = None   # None → tenant setting
    mode: Literal["fast", "agentic"] = "fast"
    conversation_id: str | None = None


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _nd(node_type: NodeType, node_id: str, config: dict[str, Any]) -> Any:
    """Instantiate a node from its type + config."""
    return get_node(NodeDef(id=node_id, type=node_type, config=config))


async def _fast_retrieval(
    ctx: ExecutionContext,
    question: str,
    tenant: str,
    few_shot: bool | None,
    cancelled,
    out: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Fixed pipeline: query_process → dense + sparse → RRF → rerank → augment."""
    # ── Step 1: Query understanding ────────────────────────────────────────────
    yield {"event": "step", "step": "query_process", "status": "start",
           "message": "Understanding your question..."}
    qp = _nd(NodeType.QUERY_PROCESS, "qp", {"query": "$.inputs.question"})
    qp_out = await qp.run(ctx, {})
    ctx.set_output("qp", qp_out)
    yield {"event": "step", "step": "query_process", "status": "done",
           "intent": qp_out.get("intent"), "normalized": qp_out.get("normalized")}

    if cancelled():
        out["cancelled"] = True
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
        out["cancelled"] = True
        return

    # ── Step 3: Rerank ─────────────────────────────────────────────────────────
    yield {"event": "step", "step": "rerank", "status": "start",
           "message": "Reranking results by relevance..."}
    rerank_node = _nd(NodeType.RERANK, "rerank", {
        "method": "cross_encoder", "query": "$.qp.normalized", "top_n": 3,
    })
    reranked = await rerank_node.run(ctx, {"fuse": merged})
    # Profile + long-term memory interests (no-op for anonymous users).
    reranked = await _nd(NodeType.PERSONALIZE, "personalize",
                         {"user_id": "$.inputs.user_id"}).run(ctx, {"rerank": reranked})
    ctx.set_output("rerank", reranked)
    yield {"event": "step", "step": "rerank", "status": "done",
           "result_count": len(reranked) if isinstance(reranked, list) else 0}

    # ── Step 4: Augment (build prompt context) ─────────────────────────────────
    yield {"event": "step", "step": "augment", "status": "start",
           "message": "Building context window..."}
    augment_node = _nd(NodeType.AUGMENT, "augment", {
        "query": "$.qp.normalized", "max_context_tokens": 600,
        "examples": [e.model_dump() for e in effective_examples(tenant, few_shot)],
        "example_format": "json",
    })
    augment_out = await augment_node.run(ctx, {"rerank": reranked})
    ctx.set_output("augment", augment_out)
    token_est = augment_out.get("token_estimate", 0) if isinstance(augment_out, dict) else 0
    yield {"event": "step", "step": "augment", "status": "done",
           "context_tokens": token_est,
           "doc_count": augment_out.get("included", 0) if isinstance(augment_out, dict) else 0}

    if cancelled():
        out["cancelled"] = True
        return

    out.update(qp=qp_out, augment=augment_out, hit_count=len(merged),
               answer_query=qp_out.get("normalized", question))


async def _agentic_retrieval(
    question: str,
    tenant: str,
    provider_name: str,
    few_shot: bool | None,
    out: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Agentic retrieval through the DAG engine (plan → search → evaluate loop → augment).

    The agent's steps are emitted as one ``agent_steps`` event when the loop ends.
    """
    yield {"event": "step", "step": "agent", "status": "start",
           "message": "Agent is planning tool calls..."}
    examples = [e.model_dump() for e in effective_examples(tenant, few_shot)]
    result = await WorkflowExecutor(build_agentic_retrieval_workflow(provider_name)).run(
        {"question": question, "tenant": tenant, "few_shot": examples})
    if result.status != "success":
        errors = "; ".join(r.error for r in result.results if r.error)
        raise RuntimeError(f"agentic retrieval failed: {redact(errors)}")
    res = result.outputs["out"]
    agent = res.get("agent") or {}
    yield {"event": "step", "step": "agent", "status": "done",
           "rounds": agent.get("round"), "stop_reason": agent.get("stop_reason"),
           "planner": agent.get("planner")}
    yield {"event": "agent_steps", "steps": agent.get("steps", []),
           "rewritten_query": agent.get("rewritten_query")}
    out.update(qp=res.get("qp") or {}, augment=res.get("augment") or {},
               hit_count=len(agent.get("evidence", [])),
               answer_query=agent.get("answer_query") or question,
               steps=agent.get("steps", []), rewritten_query=agent.get("rewritten_query"))


async def _rag_pipeline_events(
    question: str,
    tenant: str,
    provider_name: str,
    cancel: asyncio.Event | None = None,
    few_shot: bool | None = None,
    mode: str = "fast",
    user_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the full RAG pipeline step-by-step, yielding event dicts.

    Retrieval depends on ``mode``:
        fast:    query_process → vector_search + keyword_search → rrf → rerank → augment
        agentic: plan → search → evaluate (→ rewrite …) agent loop → augment
    then structured generation streams the answer for both.
    """
    run_id = str(uuid.uuid4())
    ctx = ExecutionContext(run_id=run_id,
                           inputs={"question": question, "tenant": tenant, "user_id": user_id})
    t0 = time.perf_counter()

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    # ── Steps 1-4: retrieval (fixed pipeline or agent) ─────────────────────────
    retrieved: dict[str, Any] = {}
    if mode == "agentic":
        source = _agentic_retrieval(question, tenant, provider_name, few_shot, retrieved)
    else:
        source = _fast_retrieval(ctx, question, tenant, few_shot, cancelled, retrieved)
    async for ev in source:
        yield ev
    if cancelled() or retrieved.get("cancelled"):
        yield {"event": "cancelled"}
        return
    qp_out, augment_out = retrieved["qp"], retrieved["augment"]

    # ── Step 5: Generate (token streaming) ────────────────────────────────────
    yield {"event": "step", "step": "generate", "status": "start",
           "message": "Generating response..."}

    if isinstance(augment_out, dict) and "messages" in augment_out:
        llm_request = {
            "messages":  augment_out["messages"],
            "query":     retrieved["answer_query"],
            "documents": augment_out.get("documents", []),
            "citations": augment_out.get("citations", []),
        }
    else:
        llm_request = {
            "messages": [{"role": "user", "content": question}],
            "query": question, "documents": [], "citations": [],
        }

    # Structured output, streamed: the model emits the JSON answer object and
    # only its "answer" string is forwarded as tokens. Citations and confidence
    # arrive afterwards in a single "sources" event.
    llm_config = {"provider": provider_name, "max_tokens": 2048}
    stream_request = {**llm_request, "messages": with_instructions(llm_request["messages"])}
    extractor = AnswerStreamExtractor()
    raw_reply = ""
    streamed = ""
    token_count = 0
    stream_error: str | None = None

    try:
        async for chunk in stream_llm(provider_name, stream_request, structured_config(llm_config),
                                      tenant=tenant):
            if cancelled():
                break
            raw_reply += chunk
            delta = extractor.feed(chunk)
            if delta:
                streamed += delta
                token_count += 1
                yield {"event": "token", "token": delta}
    except BudgetExceededError:
        raise
    except Exception as exc:  # provider down / unconfigured / circuit open
        stream_error = redact(str(exc))

    if cancelled():
        yield {"event": "cancelled"}
        return

    # Validate the streamed JSON; repair once or fall back to extractive.
    result = await generate_structured(
        provider_name, llm_request, llm_config, tenant=tenant,
        first_reply=None if stream_error else raw_reply,
    )
    if stream_error:
        result["structured"]["errors"].insert(0, f"stream error: {stream_error}")
    result.pop("response", None)
    answer = result["answer"]

    yield {"event": "step", "step": "generate", "status": "done",
           "token_count": token_count}

    citations = augment_out.get("citations", []) if isinstance(augment_out, dict) else []
    cited = [c for c in citations if c.get("marker") in result["cited_markers"]]
    yield {
        "event":         "sources",
        "answer":        answer,
        "replaced":      answer != streamed,  # client should swap in the validated text
        "citations":     cited,
        "cited_markers": result["cited_markers"],
        "confidence":    result["confidence"],
        "answerable":    result["answerable"],
        "structured":    result["structured"],
    }

    # ── Complete ───────────────────────────────────────────────────────────────
    yield {
        "event":         "complete",
        "answer":        answer,
        "citations":     citations,
        "cited_markers": result["cited_markers"],
        "confidence":    result["confidence"],
        "answerable":    result["answerable"],
        "intent":        qp_out.get("intent"),
        "hit_count":     retrieved["hit_count"],
        "mode":          mode,
        "steps":         retrieved.get("steps"),
        "rewritten_query": retrieved.get("rewritten_query"),
        "duration_ms":   round((time.perf_counter() - t0) * 1000, 1),
        "run_id":        run_id,
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

    Event types: ``memory`` · ``step`` · ``progress`` · ``agent_steps`` · ``token`` ·
    ``sources`` · ``complete`` · ``error`` · ``cancelled``

    With ``conversation_id`` (or a new conversation), a follow-up is first
    rewritten into a standalone question, announced in a ``memory`` event.
    """
    provider = body.provider or default_provider()

    async def event_source() -> AsyncIterator[str]:
        try:
            mem = await memory.before(tenant=body.tenant, user_id=user["id"],
                                      conversation_id=body.conversation_id,
                                      question=body.question, provider=provider)
            first = {"event": "memory", "conversation_id": mem.conversation_id,
                     "standalone_question": mem.standalone, "steps": mem.steps}
            yield f"event: memory\ndata: {json.dumps(first)}\n\n"
            async for ev in _rag_pipeline_events(mem.standalone, body.tenant, provider,
                                                 few_shot=body.few_shot, mode=body.mode,
                                                 user_id=user["id"]):
                if await http_request.is_disconnected():
                    break
                if ev["event"] == "complete":
                    memory.after(mem, tenant=body.tenant, user_id=user["id"], out=ev,
                                 provider=provider)
                    ev = memory.present(mem, ev)
                yield f"event: {ev['event']}\ndata: {json.dumps(ev)}\n\n"
        except Exception as exc:  # noqa: BLE001
            err_ev = {"event": "error", "detail": redact(str(exc))}
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
                async for ev in _rag_pipeline_events(question, tenant, provider, cancel,
                                                     few_shot=data.get("few_shot"),
                                                     mode="agentic" if data.get("mode") == "agentic" else "fast"):
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
