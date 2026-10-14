"""HTTP API for the workflow orchestration engine (F1-F8)."""
from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_optional_user, require_tenant
from app.api.pipelines import (
    build_agentic_workflow,
    build_ask_workflow,
    build_index_workflow,
    default_provider,
)
from app.engine.executor import WorkflowExecutor
from app.engine.graph import WorkflowGraphError, build_graph, generations
from app.engine.merging import STRATEGIES
from app.models.workflow import RunRequest, RunResponse, WorkflowDef
from app.observability.metrics import render_latest
from app.observability.query_log import log_query
from app.observability.tracing import current_trace_id
from app.rag.cache import cache_stats
from app.rag.cost import budget_stats
from app.rag.memory import service as memory
from app.rag.tenants.ai_settings import effective_examples
from app.rag.vectorstore import store_stats

router = APIRouter()

# Structured-output confidence labels → numeric score for the F8 query log.
_CONFIDENCE_SCORE = {"high": 0.9, "medium": 0.6, "low": 0.3}


class IndexRequest(BaseModel):
    tenant: str = "default"
    text: str
    filename: str = "document.md"


class SearchRequest(BaseModel):
    tenant: str = "default"
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class AskRequest(BaseModel):
    tenant: str = "default"
    question: str
    provider: str | None = Field(default=None, description="LLM provider; defaults to server config")
    few_shot: bool | None = Field(
        default=None, description="Force few-shot examples on/off; default = tenant setting")
    mode: Literal["fast", "agentic"] = Field(
        default="fast",
        description="fast = fixed retrieval pipeline; agentic = tool-calling agent "
                    "(plan → search → evaluate → rewrite, max 3 rounds)")
    conversation_id: str | None = Field(
        default=None,
        description="Continue a conversation (last 5 turns are used to rewrite follow-ups); "
                    "omit to start a new one. Requires an authenticated user.")


@router.post("/workflows/run", response_model=RunResponse, tags=["workflows"])
async def run_workflow(request: RunRequest) -> RunResponse:
    """Execute a workflow definition against the provided inputs."""
    try:
        executor = WorkflowExecutor(request.workflow)
    except (WorkflowGraphError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await executor.run(request.inputs)


@router.post("/workflows/stream", tags=["workflows"])
async def stream_workflow(request: RunRequest) -> StreamingResponse:
    """Execute a workflow and stream results as Server-Sent Events (F6).

    Emits one SSE message per engine event (``workflow_start``,
    ``node_complete``, ``workflow_end``) as each node finishes, instead of
    waiting for the whole pipeline.
    """
    try:
        executor = WorkflowExecutor(request.workflow)
    except (WorkflowGraphError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async def event_source():
        async for ev in executor.events(request.inputs):
            payload = dict(ev)
            if "result" in payload:  # NodeResult -> serialisable dict
                payload["result"] = payload["result"].model_dump()
            yield f"event: {payload['event']}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@router.post("/workflows/validate", tags=["workflows"])
async def validate_workflow(workflow: WorkflowDef) -> dict:
    """Validate a workflow forms a DAG and return its execution plan."""
    try:
        graph = build_graph(workflow)
    except WorkflowGraphError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "valid": True,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "execution_plan": generations(graph),
    }


@router.post("/rag/index", tags=["rag"])
async def rag_index(
    request: IndexRequest,
    user: Annotated[dict, Depends(get_optional_user)],
) -> dict:
    """Ingest → chunk → embed → upsert a document into the tenant's namespace."""
    require_tenant(user, request.tenant, "ingest")
    executor = WorkflowExecutor(build_index_workflow())
    result = await executor.run(request.model_dump())
    if result.status != "success":
        raise HTTPException(status_code=500, detail="indexing failed")
    return {"status": "ok", **result.outputs["out"], "run_id": result.run_id}


@router.post("/rag/ask", tags=["rag"])
async def rag_ask(
    request: AskRequest,
    user: Annotated[dict, Depends(get_optional_user)],
) -> dict:
    """Full RAG answer.

    * ``mode="fast"`` (default): query → hybrid search → rerank → augment → generate.
    * ``mode="agentic"``: the agent plans tool calls (search_docs, get_document,
      lookup_entity), evaluates the evidence and may rewrite the query and search
      again; the response includes the ``steps`` trace.
    """
    require_tenant(user, request.tenant, "query")
    provider = request.provider or default_provider()
    # Conversation memory: rewrite a follow-up into a standalone question first.
    memory_ctx = await memory.before(tenant=request.tenant, user_id=user["id"],
                                     conversation_id=request.conversation_id,
                                     question=request.question, provider=provider)
    build = build_agentic_workflow if request.mode == "agentic" else build_ask_workflow
    executor = WorkflowExecutor(build(provider))
    examples = effective_examples(request.tenant, request.few_shot)
    result = await executor.run({
        "tenant": request.tenant,
        "question": memory_ctx.standalone,
        "user_id": user["id"],
        "few_shot": [e.model_dump() for e in examples],
    })
    if result.status != "success":
        raise HTTPException(status_code=500, detail="answering failed")
    out = result.outputs.get("out", {})
    memory.after(memory_ctx, tenant=request.tenant, user_id=user["id"], out=out, provider=provider)
    out = memory.present(memory_ctx, out)
    # Log query for F8 monitoring (best-effort, never fails the request).
    try:
        log_query(
            user_id=user["id"],
            question=request.question,
            answer=out.get("answer", ""),
            confidence=_CONFIDENCE_SCORE.get(out.get("confidence"), 0.0),
            sources_count=len(out.get("sources", [])),
            duration_ms=result.duration_ms,
            trace_id=current_trace_id(),
            provider=provider,
        )
    except Exception:
        pass
    return {**out, "run_id": result.run_id, "duration_ms": result.duration_ms}


@router.post("/rag/search", tags=["rag"])
async def rag_search(
    request: SearchRequest,
    user: Annotated[dict, Depends(get_optional_user)],
) -> dict:
    """Hybrid search (dense + BM25 → RRF → rerank) in one tenant's knowledge base.

    Returns passages without generating an answer — used by the MCP
    ``search_knowledge_base`` tool. Tenant access is checked like ``/rag/ask``.
    """
    from app.rag.agent.tools import ToolContext, hybrid_search

    require_tenant(user, request.tenant, "query")
    hits = await hybrid_search(request.query, ToolContext(tenant=request.tenant, top_k=request.top_k))
    return {
        "tenant": request.tenant,
        "query": request.query,
        "results": [
            {
                "rank": i + 1,
                "doc_id": h.get("metadata", {}).get("document_id"),
                "chunk_id": h.get("id"),
                "source": h.get("metadata", {}).get("source"),
                "heading": h.get("metadata", {}).get("heading"),
                "page": h.get("metadata", {}).get("page"),
                "score": round(float(h.get("score", 0.0)), 4),
                "text": h.get("metadata", {}).get("text", ""),
            }
            for i, h in enumerate(hits)
        ],
    }


@router.get("/merge-strategies", tags=["meta"])
async def list_merge_strategies() -> dict:
    """List available F4 merge strategies."""
    return {"strategies": sorted(STRATEGIES)}


@router.get("/metrics", tags=["meta"])
async def metrics() -> Response:
    """Prometheus metrics exposition endpoint (F8)."""
    payload, content_type = render_latest()
    return Response(content=payload, media_type=content_type)


@router.get("/vectorstores", tags=["rag"])
async def vectorstores() -> dict:
    """Inspect vector stores: namespaces, counts, index type (F12)."""
    return {"stores": store_stats()}


@router.get("/caches", tags=["rag"])
async def caches() -> dict:
    """Cache hit/miss stats per named cache (F17)."""
    return {"caches": cache_stats()}


@router.get("/budgets", tags=["rag"])
async def budgets() -> dict:
    """Per-tenant spend vs. limit (F24)."""
    return {"budgets": budget_stats()}
