"""Usage ledger — records every cost event (F24)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.rag.cost.pricing import estimate_cost

_MAX_EVENTS = 50_000
_events: deque[CostEvent] = deque(maxlen=_MAX_EVENTS)


@dataclass
class CostEvent:
    id:            str
    user_id:       str
    tenant:        str
    model:         str
    operation:     str   # llm | embedding | vectordb | storage
    input_tokens:  int
    output_tokens: int
    cost:          float
    cached:        bool
    timestamp:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def record_usage(
    user_id:       str = "",
    tenant:        str = "default",
    model:         str = "",
    operation:     str = "llm",
    input_tokens:  int = 0,
    output_tokens: int = 0,
    cached:        bool = False,
) -> CostEvent:
    cost  = 0.0 if cached else estimate_cost(model, input_tokens, output_tokens)
    event = CostEvent(
        id=str(uuid4()),
        user_id=user_id,
        tenant=tenant,
        model=model,
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        cached=cached,
    )
    _events.appendleft(event)
    return event


def get_usage(
    user_id:   str | None = None,
    tenant:    str | None = None,
    model:     str | None = None,
    operation: str | None = None,
    limit:     int = 100,
) -> list[CostEvent]:
    results: list[CostEvent] = list(_events)
    if user_id   is not None: results = [e for e in results if e.user_id   == user_id]
    if tenant    is not None: results = [e for e in results if e.tenant    == tenant]
    if model     is not None: results = [e for e in results if e.model     == model]
    if operation is not None: results = [e for e in results if e.operation == operation]
    return results[:limit]


def usage_summary(
    user_id: str | None = None,
    tenant:  str | None = None,
) -> dict[str, Any]:
    events = get_usage(user_id=user_id, tenant=tenant, limit=_MAX_EVENTS)
    total_requests = len(events)
    cached_count   = sum(1 for e in events if e.cached)
    total_cost     = round(sum(e.cost for e in events), 6)

    by_model:     dict[str, float] = {}
    by_operation: dict[str, float] = {}
    for e in events:
        by_model[e.model]         = round(by_model.get(e.model, 0.0)         + e.cost, 6)
        by_operation[e.operation] = round(by_operation.get(e.operation, 0.0) + e.cost, 6)

    return {
        "total_cost":      total_cost,
        "total_requests":  total_requests,
        "cached_requests": cached_count,
        "cache_hit_rate":  round(cached_count / total_requests, 3) if total_requests else 0.0,
        "by_model":        by_model,
        "by_operation":    by_operation,
    }


def reset_tracker() -> None:
    _events.clear()
