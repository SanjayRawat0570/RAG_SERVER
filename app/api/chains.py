"""F3: Chain API — sequential multi-step execution with Supabase history.

Endpoints
---------
POST /chains/run/document    Document upload chain: ingest→chunk→embed→store
POST /chains/run/ask         RAG query chain: query→search→rerank→generate
POST /chains/run/multihop    Multi-hop reasoning: decompose→answer each→synthesize
POST /chains/run             Run an arbitrary ChainDef from the request body
GET  /chains/history         Caller's last 50 executions (Supabase)
GET  /chains/executions/{id} Single execution with step-level outputs (Supabase)
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.api.pipelines import (
    build_ask_workflow,
    build_decompose_workflow,
    build_index_workflow,
    build_synthesize_workflow,
    default_provider,
)
from app.engine.chain import ChainDef, ChainExecutor, ChainResult, ChainStep, StepResult

router = APIRouter(prefix="/chains", tags=["chains"])


# ── request models ─────────────────────────────────────────────────────────────

class DocumentChainRequest(BaseModel):
    tenant: str = "default"
    text: str
    filename: str = "document.md"
    on_error: str = "fail_fast"


class AskChainRequest(BaseModel):
    tenant: str = "default"
    question: str
    provider: str | None = None
    on_error: str = "fail_fast"


class MultihopRequest(BaseModel):
    tenant: str = "default"
    question: str
    provider: str | None = None
    on_error: str = "continue"


class ChainRunRequest(BaseModel):
    chain: ChainDef
    inputs: dict[str, Any] = {}


# ── Supabase helper ────────────────────────────────────────────────────────────

def _sb():
    from app.config import settings
    if settings.supabase_url and settings.supabase_key:
        from supabase import create_client  # type: ignore[import]
        return create_client(settings.supabase_url, settings.supabase_key)
    return None


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("/run/document", response_model=ChainResult)
async def run_document_chain(
    request: DocumentChainRequest,
    user: dict = Depends(get_current_user),
) -> ChainResult:
    """
    Document Upload Chain — 4 sequential steps:
      1. ingest  → parse file, clean text
      2. chunk   → split into overlapping chunks
      3. embed   → convert chunks to vectors
      4. upsert  → store in vector DB
    """
    chain = ChainDef(
        name="document_upload",
        description="Ingest, chunk, embed, and index a document",
        on_error=request.on_error,
        steps=[ChainStep(name="index", workflow=build_index_workflow())],
    )
    result = await ChainExecutor(chain).run(
        {"tenant": request.tenant, "text": request.text, "filename": request.filename},
        user_id=user["id"],
    )
    if result.status == "error":
        raise HTTPException(500, detail="Document chain failed")
    return result


@router.post("/run/ask", response_model=ChainResult)
async def run_ask_chain(
    request: AskChainRequest,
    user: dict = Depends(get_current_user),
) -> ChainResult:
    """
    Query Processing Chain — 6 sequential steps:
      1. query_process  → normalize, extract intent & keywords
      2. vector_search  → dense semantic retrieval
      3. keyword_search → BM25 sparse retrieval
      4. rerank         → cross-encoder re-scoring
      5. augment        → build prompt context with citations
      6. generate       → LLM answer generation
    """
    provider = request.provider or default_provider()
    chain = ChainDef(
        name="rag_ask",
        description="Full RAG answer pipeline",
        on_error=request.on_error,
        steps=[ChainStep(name="ask", workflow=build_ask_workflow(provider))],
    )
    result = await ChainExecutor(chain).run(
        {"tenant": request.tenant, "question": request.question},
        user_id=user["id"],
    )
    if result.status == "error":
        raise HTTPException(500, detail="Ask chain failed")
    return result


@router.post("/run/multihop", response_model=ChainResult)
async def run_multihop_chain(
    request: MultihopRequest,
    user: dict = Depends(get_current_user),
) -> ChainResult:
    """
    Multi-Hop Reasoning Chain — 3 phases:
      Phase 1: decompose  → break complex question into sub-questions
      Phase 2: answer     → run full RAG ask for EACH sub-question
      Phase 3: synthesize → merge all answers into one coherent response
    """
    provider = request.provider or default_provider()
    chain_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    step_results: list[StepResult] = []
    had_error = False

    # ── Phase 1: Decompose ────────────────────────────────────────────────────
    t_s = time.perf_counter()
    try:
        decomp_run = await ChainExecutor(
            ChainDef(name="decompose", steps=[
                ChainStep(name="decompose", workflow=build_decompose_workflow())
            ])
        ).run({"question": request.question}, user_id=user["id"])

        sub_questions: list[str] = decomp_run.final_output.get(
            "sub_questions", [request.question]
        )
        step_results.append(StepResult(
            step="decompose", status="success",
            outputs=decomp_run.final_output,
            duration_ms=(time.perf_counter() - t_s) * 1000,
        ))
    except Exception as exc:
        step_results.append(StepResult(
            step="decompose", status="error", error=str(exc),
            duration_ms=(time.perf_counter() - t_s) * 1000,
        ))
        sub_questions = [request.question]
        had_error = True

    # ── Phase 2: Answer each sub-question ────────────────────────────────────
    sub_answers: list[dict[str, str]] = []
    for sq in sub_questions:
        t_s = time.perf_counter()
        try:
            ask_run = await ChainExecutor(
                ChainDef(
                    name="sub_ask",
                    on_error=request.on_error,
                    steps=[ChainStep(name="ask", workflow=build_ask_workflow(provider))],
                )
            ).run({"tenant": request.tenant, "question": sq}, user_id=user["id"])

            answer = ask_run.final_output.get("answer", "")
            sub_answers.append({"question": sq, "answer": answer})
            step_results.append(StepResult(
                step=f"ask:{sq[:40]}",
                status="success" if ask_run.status != "error" else "error",
                outputs=ask_run.final_output,
                duration_ms=(time.perf_counter() - t_s) * 1000,
            ))
        except Exception as exc:
            had_error = True
            step_results.append(StepResult(
                step=f"ask:{sq[:40]}", status="error", error=str(exc),
                duration_ms=(time.perf_counter() - t_s) * 1000,
            ))
            if request.on_error == "fail_fast":
                break

    # ── Phase 3: Synthesize ───────────────────────────────────────────────────
    t_s = time.perf_counter()
    try:
        synth_run = await ChainExecutor(
            ChainDef(name="synthesize", steps=[
                ChainStep(name="synthesize", workflow=build_synthesize_workflow())
            ])
        ).run(
            {"sub_answers": sub_answers, "question": request.question},
            user_id=user["id"],
        )
        final_answer = synth_run.final_output
        step_results.append(StepResult(
            step="synthesize", status="success",
            outputs=synth_run.final_output,
            duration_ms=(time.perf_counter() - t_s) * 1000,
        ))
    except Exception as exc:
        had_error = True
        final_answer = {"answer": "; ".join(a["answer"] for a in sub_answers)}
        step_results.append(StepResult(
            step="synthesize", status="error", error=str(exc),
            duration_ms=(time.perf_counter() - t_s) * 1000,
        ))

    n_ok = sum(1 for s in step_results if s.status == "success")
    status = (
        "success" if not had_error
        else ("partial" if n_ok > 0 and request.on_error == "continue" else "error")
    )

    return ChainResult(
        chain_id=chain_id,
        chain="multihop",
        status=status,
        steps=step_results,
        final_output=final_answer,
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


@router.post("/run", response_model=ChainResult)
async def run_custom_chain(
    request: ChainRunRequest,
    user: dict = Depends(get_current_user),
) -> ChainResult:
    """Run an arbitrary ChainDef submitted in the request body."""
    result = await ChainExecutor(request.chain).run(request.inputs, user_id=user["id"])
    if result.status == "error":
        raise HTTPException(500, detail=f"Chain '{request.chain.name}' failed")
    return result


@router.get("/history")
async def chain_history(user: dict = Depends(get_current_user)) -> dict:
    """List the caller's last 50 chain executions from Supabase."""
    sb = _sb()
    if not sb:
        return {"executions": [], "note": "Supabase not configured — history unavailable offline"}
    try:
        resp = (
            sb.table("workflow_executions")
            .select("id, filename, status, completed_at, tenant")
            .eq("user_id", user["id"])
            .order("completed_at", desc=True)
            .limit(50)
            .execute()
        )
        return {"executions": resp.data}
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/executions/{exec_id}")
async def chain_execution_detail(
    exec_id: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """Retrieve a single execution record including per-step outputs."""
    sb = _sb()
    if not sb:
        raise HTTPException(503, detail="Supabase not configured")
    try:
        resp = (
            sb.table("workflow_executions")
            .select("*")
            .eq("id", exec_id)
            .eq("user_id", user["id"])
            .single()
            .execute()
        )
        if not resp.data:
            raise HTTPException(404, detail="Execution not found")
        return resp.data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc
