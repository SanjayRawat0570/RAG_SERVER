"""HTTP API for the workflow orchestration engine (F1-F8)."""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_optional_user
from app.api.pipelines import build_ask_workflow, build_index_workflow, default_provider
from app.engine.executor import WorkflowExecutor
from app.engine.graph import WorkflowGraphError, build_graph, generations
from app.engine.merging import STRATEGIES
from app.models.workflow import RunRequest, RunResponse, WorkflowDef
from app.observability.metrics import render_latest
from app.observability.query_log import log_query
from app.observability.tracing import current_trace_id
from app.rag.cache import cache_stats
from app.rag.cost import budget_stats
from app.rag.vectorstore import store_stats

router = APIRouter()


class IndexRequest(BaseModel):
    tenant: str = "default"
    text: str
    filename: str = "document.md"


class AskRequest(BaseModel):
    tenant: str = "default"
    question: str
    provider: str | None = Field(default=None, description="LLM provider; defaults to server config")


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
async def rag_index(request: IndexRequest) -> dict:
    """Ingest → chunk → embed → upsert a document into the tenant's namespace."""
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
    """Full RAG answer: query → hybrid search → rerank → augment → generate."""
    provider = request.provider or default_provider()
    executor = WorkflowExecutor(build_ask_workflow(provider))
    result = await executor.run(
        {"tenant": request.tenant, "question": request.question}
    )
    if result.status != "success":
        raise HTTPException(status_code=500, detail="answering failed")
    out = result.outputs.get("out", {})
    # Log query for F8 monitoring (best-effort, never fails the request).
    try:
        log_query(
            user_id=user["id"],
            question=request.question,
            answer=out.get("answer", ""),
            confidence=float(out.get("confidence", 0.0)),
            sources_count=len(out.get("sources", [])),
            duration_ms=result.duration_ms,
            trace_id=current_trace_id(),
            provider=provider,
        )
    except Exception:
        pass
    return {**out, "run_id": result.run_id, "duration_ms": result.duration_ms}


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
