"""F8: Monitoring, Observability & Distributed Tracing — HTTP API.

Endpoints
---------
GET  /monitoring/dashboard          Aggregated stats for the current user
GET  /monitoring/history            Paginated query history
POST /monitoring/feedback/{qid}     Submit 1-5 star rating for a query
GET  /monitoring/traces             List recent distributed traces (in-memory)
GET  /monitoring/spans/{trace_id}   All spans for one trace

All endpoints require a Bearer token; offline dev auto-accepts any token.
Query logs are tied to user_id so each user sees only their own history.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.observability.query_log import (
    dashboard_stats,
    get_user_history,
    submit_feedback,
)
from app.observability.span_store import get_trace, list_traces

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# ── Request models ─────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Star rating 1–5")
    note: str = ""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/dashboard")
async def monitoring_dashboard(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Aggregated performance stats for the current user.

    Returns today-scope stats: total queries, average response time, average
    confidence, rating distribution, and the set of providers used.  All data
    comes from the in-memory query log (F8) — no Supabase required.

    Example response
    ----------------
    .. code-block:: json

        {
          "total_queries":   42,
          "avg_duration_ms": 1200.0,
          "avg_confidence":  0.87,
          "rated_count":     10,
          "avg_rating":      4.2,
          "providers":       ["gemini", "stub"],
          "most_recent":     "2026-06-24T14:00:00+00:00",
          "metrics_endpoint": "/api/v1/metrics"
        }
    """
    stats = dashboard_stats(user_id=user["id"])
    stats["metrics_endpoint"] = "/api/v1/metrics"
    stats["traces_endpoint"]  = "/api/v1/monitoring/traces"
    return stats


@router.get("/history")
async def query_history(
    limit: int = 50,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Paginated query history for the current user (most recent first).

    Each entry includes the question asked, answer snippet, confidence score,
    number of source documents retrieved, latency, the LLM provider used, and
    an optional user rating.

    Example entry
    -------------
    .. code-block:: json

        {
          "id":            "uuid",
          "question":      "What is revenue?",
          "answer":        "Revenue for Q3 was ...",
          "confidence":    0.95,
          "sources_count": 3,
          "duration_ms":   1200.0,
          "provider":      "gemini",
          "rating":        5,
          "created_at":    "2026-06-24T14:00:00+00:00"
        }
    """
    entries = get_user_history(user_id=user["id"], limit=max(1, min(limit, 200)))
    return {"queries": entries, "count": len(entries)}


@router.post("/feedback/{query_id}")
async def submit_query_feedback(
    query_id: str,
    body: FeedbackRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Attach a 1–5 star rating to a completed query.

    The rating is stored in the in-memory log and propagated to Supabase
    ``query_logs`` when available.  Returns 404 if the query id is not found
    in the current session's log (entries older than 1 000 queries are evicted).
    """
    updated = submit_feedback(
        query_id=query_id,
        user_id=user["id"],
        rating=body.rating,
    )
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Query '{query_id}' not found in current session log.",
        )
    return {"query_id": query_id, "rating": body.rating, "status": "recorded"}


@router.get("/traces")
async def list_recent_traces(
    limit: int = 20,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    List recent distributed traces captured by the in-memory span store.

    Each summary shows the trace id, root workflow name, number of spans, total
    wall-clock duration of all node spans, and overall status.  Use the trace
    id with ``GET /monitoring/spans/{trace_id}`` to drill into individual spans.

    Does not require Jaeger — spans are always stored in process memory (F8).

    Example trace entry
    -------------------
    .. code-block:: json

        {
          "trace_id":   "abc123...",
          "root_name":  "workflow.run",
          "workflow":   "ask_workflow",
          "span_count": 8,
          "duration_ms": 1650.0,
          "status":     "OK"
        }
    """
    traces = list_traces(limit=max(1, min(limit, 100)))
    return {"traces": traces, "count": len(traces)}


@router.get("/spans/{trace_id}")
async def trace_spans(
    trace_id: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    All spans for a single distributed trace, sorted by start time.

    Reproduces the package-tracking model from the spec: each span shows
    the operation name, start timestamp (nanoseconds), duration, status, and
    any attributes attached by the executor (node id, node type, etc.).

    Example span
    ------------
    .. code-block:: json

        {
          "trace_id":    "abc123...",
          "span_id":     "def456...",
          "name":        "node.embed",
          "duration_ms": 200.5,
          "status":      "OK",
          "attributes":  {"node.id": "embed", "node.type": "embed", "node.status": "success"}
        }
    """
    spans = get_trace(trace_id)
    if not spans:
        raise HTTPException(
            status_code=404,
            detail=f"Trace '{trace_id}' not found. It may have been evicted from the in-memory store.",
        )
    return {"trace_id": trace_id, "spans": spans, "span_count": len(spans)}
