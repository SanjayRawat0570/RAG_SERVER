"""Per-query log for F8 monitoring.

Every RAG answer is recorded here — in a fixed-size deque (last 1 000 queries)
and optionally in Supabase ``query_logs`` for persistence across restarts.

Schema (Supabase)
-----------------
  CREATE TABLE query_logs (
    id            UUID PRIMARY KEY,
    user_id       TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT,
    confidence    FLOAT,
    sources_count INT,
    duration_ms   FLOAT,
    trace_id      TEXT,
    provider      TEXT,
    rating        INT,          -- 1-5 stars, NULL until user rates
    created_at    TIMESTAMPTZ DEFAULT NOW()
  );

Entry points
------------
log_query(...)           record a completed RAG query
submit_feedback(...)     attach a 1-5 star rating to an existing entry
get_user_history(...)    retrieve recent entries for a user
dashboard_stats(...)     aggregate stats for the dashboard
"""
from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class QueryLog(BaseModel):
    id:            str
    user_id:       str
    question:      str
    answer:        str   = ""
    confidence:    float = 0.0
    sources_count: int   = 0
    duration_ms:   float = 0.0
    trace_id:      str | None = None
    provider:      str   = "stub"
    rating:        int | None = None
    created_at:    str


# Global ring-buffer — survives the process lifetime, no disk needed.
_store: deque[QueryLog] = deque(maxlen=1000)


def log_query(
    *,
    user_id:       str,
    question:      str,
    answer:        str   = "",
    confidence:    float = 0.0,
    sources_count: int   = 0,
    duration_ms:   float = 0.0,
    trace_id:      str | None = None,
    provider:      str   = "stub",
) -> str:
    """Record a RAG query and return the generated query id."""
    qid = str(uuid.uuid4())
    entry = QueryLog(
        id=qid,
        user_id=user_id,
        question=question,
        answer=answer,
        confidence=confidence,
        sources_count=sources_count,
        duration_ms=duration_ms,
        trace_id=trace_id,
        provider=provider,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _store.append(entry)
    _persist(entry)
    return qid


def submit_feedback(query_id: str, user_id: str, rating: int) -> bool:
    """Attach a 1-5 star rating to an existing log entry.

    Returns True if the entry was found and updated; False otherwise.
    """
    clamped = max(1, min(5, rating))
    for entry in _store:
        if entry.id == query_id and entry.user_id == user_id:
            entry.rating = clamped
            _update_rating_sb(query_id, user_id, clamped)
            return True
    return False


def get_user_history(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Most-recent-first list of query logs for a specific user."""
    entries = [e for e in reversed(list(_store)) if e.user_id == user_id]
    return [e.model_dump() for e in entries[:limit]]


def dashboard_stats(user_id: str | None = None) -> dict[str, Any]:
    """Aggregated stats for the dashboard endpoint.

    When ``user_id`` is given, scopes to that user; otherwise returns
    system-wide aggregates.
    """
    entries = list(_store)
    if user_id:
        entries = [e for e in entries if e.user_id == user_id]

    total = len(entries)
    if total == 0:
        return {
            "total_queries":  0,
            "avg_duration_ms": 0.0,
            "avg_confidence":  0.0,
            "rated_count":     0,
            "avg_rating":      None,
            "providers":       [],
            "most_recent":     None,
        }

    rated = [e for e in entries if e.rating is not None]
    return {
        "total_queries":   total,
        "avg_duration_ms": round(sum(e.duration_ms for e in entries) / total, 1),
        "avg_confidence":  round(sum(e.confidence for e in entries) / total, 3),
        "rated_count":     len(rated),
        "avg_rating":      round(sum(e.rating for e in rated) / len(rated), 2) if rated else None,
        "providers":       sorted({e.provider for e in entries}),
        "most_recent":     entries[-1].created_at,
    }


# ── Supabase helpers ──────────────────────────────────────────────────────────

def _sb():
    try:
        from app.config import settings
        if not (settings.supabase_url and settings.supabase_key):
            return None
        from supabase import create_client  # type: ignore[import]
        return create_client(settings.supabase_url, settings.supabase_key)
    except Exception:
        return None


def _persist(entry: QueryLog) -> None:
    try:
        sb = _sb()
        if not sb:
            return
        sb.table("query_logs").insert(entry.model_dump()).execute()
    except Exception:
        pass


def _update_rating_sb(query_id: str, user_id: str, rating: int) -> None:
    try:
        sb = _sb()
        if not sb:
            return
        sb.table("query_logs").update({"rating": rating}).eq("id", query_id).eq("user_id", user_id).execute()
    except Exception:
        pass
