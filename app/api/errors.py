"""F7: Error Handling & Fallbacks API

Implements the four spec scenarios using FallbackExecutor, plus error history
and pattern endpoints backed by Supabase ``audit_logs``.

Every fallback attempt is logged as ``operation: "fallback"`` (success with
degradation) or ``operation: "error"`` (all options exhausted).

Scenarios
---------
Scenario 1 — Vector DB fails     semantic → keyword → entity cascade
Scenario 2 — LLM API timeout     primary provider → fallback provider → graceful
Scenario 3 — No results          direct query → expanded query → entity search
Scenario 4 — Bad document        PDF extract → OCR route → re-upload prompt

Endpoints
---------
POST /errors/search-fallback  Scenario 1: cascading search with graceful degradation
POST /errors/ask-fallback     Scenario 2: cascading LLM providers
POST /errors/query-expansion  Scenario 3: expand query on empty results
POST /errors/ingest-fallback  Scenario 4: document parsing with OCR fallback
POST /errors/run              Generic: caller supplies FallbackChainDef + inputs
GET  /errors/history          Recent fallback/error events from Supabase
GET  /errors/patterns         Error type counts from Supabase
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.api.pipelines import (
    STORE, DIM,
    build_ask_workflow,
    build_semantic_only_workflow,
    build_keyword_only_workflow,
    build_entity_only_workflow,
    build_expanded_search_workflow,
    default_provider,
)
from app.engine.fallback import (
    FallbackChainDef,
    FallbackExecutor,
    FallbackOption,
    FallbackResult,
)
from app.models.workflow import WorkflowDef

router = APIRouter(prefix="/errors", tags=["errors"])


# ── Supabase helper ────────────────────────────────────────────────────────────

def _sb():
    from app.config import settings
    if settings.supabase_url and settings.supabase_key:
        try:
            from supabase import create_client  # type: ignore[import]
            return create_client(settings.supabase_url, settings.supabase_key)
        except Exception:
            return None
    return None


def _log_error(user_id: str, error_type: str, detail: str, chain: str) -> None:
    """Log a pure error event (all fallbacks exhausted) to Supabase."""
    try:
        sb = _sb()
        if not sb:
            return
        sb.table("audit_logs").insert({
            "id":            str(uuid.uuid4()),
            "user_id":       user_id,
            "operation":     "error",
            "decision_tree": chain,
            "decision_path": [error_type],
            "outcome":       "none",
            "confidence":    0.0,
            "created_at":    datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:  # noqa: BLE001
        pass


# ── Request models ─────────────────────────────────────────────────────────────

class SearchFallbackRequest(BaseModel):
    question: str
    tenant: str = "default"
    top_k: int = 5


class AskFallbackRequest(BaseModel):
    question: str
    tenant: str = "default"
    primary_provider: str | None = None
    fallback_providers: list[str] = Field(default_factory=list)


class QueryExpansionRequest(BaseModel):
    question: str
    tenant: str = "default"


class IngestFallbackRequest(BaseModel):
    filename: str
    text: str = ""
    tenant: str = "default"


class FallbackRunRequest(BaseModel):
    chain: FallbackChainDef
    inputs: dict[str, Any] = Field(default_factory=dict)


# ── Scenario helpers ──────────────────────────────────────────────────────────

def _to_response(result: FallbackResult, scenario: str) -> dict[str, Any]:
    """Build the API response dict from a FallbackResult."""
    resp: dict[str, Any] = {
        "scenario":       scenario,
        "succeeded":      result.succeeded,
        "degraded":       result.degraded,
        "used_option":    result.used_option,
        "fallback_depth": result.fallback_depth,
        "message":        result.message,
        "outputs":        result.outputs,
        "attempts": [
            {
                "option":      a.option,
                "status":      a.status,
                "error":       a.error,
                "duration_ms": round(a.duration_ms, 1),
            }
            for a in result.attempts
        ],
        "duration_ms": round(result.duration_ms, 1),
    }
    if result.partial_note:
        resp["partial_note"] = result.partial_note
    if result.degraded and result.succeeded:
        # Graceful degradation: explain what the user is seeing
        resp["degradation_info"] = {
            "primary_failed": True,
            "fallback_used":  result.used_option,
            "note": result.partial_note or (
                f"Results are from '{result.used_option}' (fallback), "
                "not the primary method. Relevance may be lower."
            ),
        }
    return resp


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/search-fallback")
async def search_with_fallback(
    request: SearchFallbackRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 1: Vector DB Fails

    Tries three search methods in cascade. Each is treated as a failure if it
    returns no results.

      1. Semantic search (vector/dense) — best relevance
      2. Keyword search (BM25/sparse)   — no vector DB needed
      3. Entity search (name/term match) — last resort

    Returns results from the first method that finds anything, with a
    ``degraded=true`` flag and explanation when a fallback was used.
    """
    chain = FallbackChainDef(
        name="search_fallback",
        description="Cascade: semantic → keyword → entity",
        options=[
            FallbackOption(
                name="semantic_search",
                description="Semantic (dense) vector search",
                workflow=build_semantic_only_workflow(),
            ),
            FallbackOption(
                name="keyword_search",
                description="BM25 keyword search",
                workflow=build_keyword_only_workflow(),
            ),
            FallbackOption(
                name="entity_search",
                description="Named-entity pattern search",
                workflow=build_entity_only_workflow(),
            ),
        ],
        on_all_fail="empty",
        skip_empty_results=True,
    )

    try:
        result = await FallbackExecutor(chain).run(
            {"question": request.question, "tenant": request.tenant},
            user_id=user["id"],
        )
    except RuntimeError as exc:
        _log_error(user["id"], "all_search_failed", str(exc), "search_fallback")
        raise HTTPException(503, detail=str(exc)) from exc

    return _to_response(result, "search_fallback")


@router.post("/ask-fallback")
async def ask_with_fallback(
    request: AskFallbackRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 2: LLM API Timeout

    Tries each LLM provider in order until one succeeds. Falls back to the
    offline stub (always available) as the last resort, ensuring an answer is
    always returned even when all cloud APIs are down.

      1. Primary provider  (e.g., gemini)
      2. Fallback provider (e.g., stub)
      ... any additional providers in fallback_providers
    """
    primary = request.primary_provider or default_provider()
    fallback_list = request.fallback_providers or (
        ["stub"] if primary != "stub" else []
    )

    options: list[FallbackOption] = [
        FallbackOption(
            name=f"provider_{primary}",
            description=f"LLM: {primary}",
            workflow=build_ask_workflow(primary),
        )
    ]
    for fb in fallback_list:
        options.append(FallbackOption(
            name=f"provider_{fb}",
            description=f"LLM: {fb} (fallback)",
            workflow=build_ask_workflow(fb),
        ))

    chain = FallbackChainDef(
        name="ask_fallback",
        description="Cascade LLM providers",
        options=options,
        on_all_fail="empty",
        # Answers are strings, not lists — don't skip on empty list
        skip_empty_results=False,
    )

    try:
        result = await FallbackExecutor(chain).run(
            {"question": request.question, "tenant": request.tenant},
            user_id=user["id"],
        )
    except RuntimeError as exc:
        _log_error(user["id"], "all_providers_failed", str(exc), "ask_fallback")
        raise HTTPException(503, detail=str(exc)) from exc

    return _to_response(result, "ask_fallback")


@router.post("/query-expansion")
async def search_with_query_expansion(
    request: QueryExpansionRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 3: Search Returns Nothing

    If the original query finds nothing, automatically expands it with
    synonyms and related terms, then falls back to entity search.

      1. Direct query      — exact normalized query
      2. Expanded query    — query_process adds synonyms/related terms
      3. Entity search     — searches by named entities in the query

    A ``partial_note`` explains which expansion was used.
    """
    chain = FallbackChainDef(
        name="query_expansion",
        description="Direct → expanded → entity cascade on empty results",
        options=[
            FallbackOption(
                name="direct_query",
                description="Direct normalized query",
                workflow=build_semantic_only_workflow(),
            ),
            FallbackOption(
                name="expanded_query",
                description="Expanded query with synonyms",
                workflow=build_expanded_search_workflow(),
            ),
            FallbackOption(
                name="entity_search",
                description="Entity/term search (related terms)",
                workflow=build_entity_only_workflow(),
            ),
        ],
        on_all_fail="empty",
        skip_empty_results=True,
    )

    try:
        result = await FallbackExecutor(chain).run(
            {"question": request.question, "tenant": request.tenant},
            user_id=user["id"],
        )
    except RuntimeError as exc:
        _log_error(user["id"], "no_results_after_expansion", str(exc), "query_expansion")
        raise HTTPException(503, detail=str(exc)) from exc

    resp = _to_response(result, "query_expansion")

    # Add query expansion metadata
    if result.fallback_depth > 0:
        resp["expansion_applied"] = True
        resp["expansion_note"] = (
            f"Original query returned no results. "
            f"Showing results after '{result.used_option}' expansion."
        )
    else:
        resp["expansion_applied"] = False

    return resp


@router.post("/ingest-fallback")
async def ingest_with_fallback(
    request: IngestFallbackRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 4: Document Processing Fails

    Routes document to the correct parser with graceful fallback:

      1. Normal ingest       — standard text extraction (PDF/DOCX/HTML)
      2. OCR fallback        — treated as image, OCR text extraction
      3. Re-upload prompt    — returns guidance when all parsing fails

    In this implementation the "OCR" option is simulated (it re-ingests the
    raw text as-is). In production replace with a real OCR node.
    """
    from app.api.pipelines import build_index_workflow
    from app.models.workflow import WorkflowDef

    def _ocr_wf() -> WorkflowDef:
        """Simulate OCR: ingest as plain text with 'ocr' filename suffix."""
        return WorkflowDef(
            name="ocr_ingest",
            nodes=[
                {"id": "in",     "type": "input"},
                {"id": "ingest", "type": "ingest",
                 "config": {"text": "$.inputs.text",
                            "filename": "$.inputs.filename",
                            "metadata": {"tenant": "$.inputs.tenant",
                                         "parser": "ocr"}}},
                {"id": "chunk",  "type": "chunk",
                 "config": {"strategy": "sentence", "chunk_size": 256,
                            "size_unit": "tokens"}},
                {"id": "embed",  "type": "embed",  "config": {"dimension": DIM}},
                {"id": "upsert", "type": "upsert",
                 "config": {"store": STORE, "namespace": "$.inputs.tenant",
                            "dimension": DIM}},
                {"id": "out",    "type": "output", "config": {"value": "$.upsert"}},
            ],
            edges=[
                {"source": "in",     "target": "ingest"},
                {"source": "ingest", "target": "chunk"},
                {"source": "chunk",  "target": "embed"},
                {"source": "embed",  "target": "upsert"},
                {"source": "upsert", "target": "out"},
            ],
        )

    def _guidance_wf() -> WorkflowDef:
        """Last resort: return re-upload guidance (no actual processing)."""
        return WorkflowDef(
            name="reupload_guidance",
            nodes=[
                {"id": "in",  "type": "input"},
                {"id": "out", "type": "output", "config": {"value": {
                    "status":     "manual_action_required",
                    "action":     "re_upload",
                    "message":    "Document could not be processed automatically.",
                    "suggestion": "Try converting to plain text (.txt) or re-upload.",
                    "chunks_indexed": 0,
                }}},
            ],
            edges=[{"source": "in", "target": "out"}],
        )

    chain = FallbackChainDef(
        name="ingest_fallback",
        description="PDF → OCR → re-upload guidance",
        options=[
            FallbackOption(
                name="standard_ingest",
                description="Standard text extraction",
                workflow=build_index_workflow(),
            ),
            FallbackOption(
                name="ocr_ingest",
                description="OCR (image-based text extraction)",
                workflow=_ocr_wf(),
            ),
            FallbackOption(
                name="reupload_prompt",
                description="Re-upload guidance (manual action)",
                workflow=_guidance_wf(),
            ),
        ],
        on_all_fail="empty",
        skip_empty_results=False,
    )

    result = await FallbackExecutor(chain).run(
        {"tenant": request.tenant, "text": request.text,
         "filename": request.filename},
        user_id=user["id"],
    )
    return _to_response(result, "ingest_fallback")


@router.post("/run")
async def run_fallback_chain(
    request: FallbackRunRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Generic fallback chain execution.

    The caller provides a FallbackChainDef (list of options with workflows)
    and the inputs dict. Useful for custom fallback scenarios not covered by
    the built-in endpoints.
    """
    try:
        result = await FallbackExecutor(request.chain).run(
            request.inputs, user_id=user["id"]
        )
    except RuntimeError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    return _to_response(result, request.chain.name)


@router.get("/history")
async def error_history(
    limit: int = 50,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Recent fallback and error events for the current user (Supabase).

    Returns records where ``operation IN ('fallback', 'error')``, most
    recent first. Used by the frontend to show the user what happened and
    which fallbacks were triggered.
    """
    sb = _sb()
    if not sb:
        return {"events": [], "note": "Supabase not configured — history unavailable offline"}
    try:
        resp = (
            sb.table("audit_logs")
            .select("id, user_id, operation, decision_tree, outcome, confidence, created_at")
            .eq("user_id", user["id"])
            .in_("operation", ["fallback", "error"])
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"events": resp.data, "count": len(resp.data)}
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc


@router.get("/patterns")
async def error_patterns(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Error pattern summary for the current user (Supabase).

    Groups events by ``decision_tree`` to show which scenarios fail most often.
    Helps identify system health problems at a glance.
    """
    sb = _sb()
    if not sb:
        return {"patterns": [], "note": "Supabase not configured"}
    try:
        resp = (
            sb.table("audit_logs")
            .select("decision_tree, operation, outcome")
            .eq("user_id", user["id"])
            .in_("operation", ["fallback", "error"])
            .execute()
        )
        # Count by (decision_tree, operation) in Python (no GROUP BY in Supabase JS client)
        counts: dict[tuple[str, str], int] = {}
        for row in (resp.data or []):
            key = (row.get("decision_tree", "unknown"), row.get("operation", "error"))
            counts[key] = counts.get(key, 0) + 1

        patterns = [
            {"scenario": tree, "type": op, "count": cnt}
            for (tree, op), cnt in sorted(counts.items(), key=lambda x: -x[1])
        ]
        return {"patterns": patterns, "total_events": sum(counts.values())}
    except Exception as exc:
        raise HTTPException(500, detail=str(exc)) from exc
