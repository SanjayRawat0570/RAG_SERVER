"""Indexing task model (F19)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Priority(IntEnum):
    LOW    = 10
    NORMAL = 5
    HIGH   = 2
    URGENT = 1   # lowest value = highest priority in heapq


class TaskStatus(str):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"
    RETRYING   = "retrying"


class IndexingTask(BaseModel):
    id:          str
    status:      str           = TaskStatus.PENDING
    priority:    int           = Priority.NORMAL
    user_id:     str           = "dev"
    tenant:      str           = "default"
    namespace:   str           = "default"
    filename:    str           = "document.txt"
    text:        str           = ""
    store:       str           = "kb"
    progress:    int           = 0    # 0-100
    step:        str           = "queued"
    message:     str           = "Queued for indexing"
    chunks_done: int           = 0
    attempts:    int           = 0
    max_attempts: int          = 3
    error:       str | None    = None
    metadata:    dict[str, Any] = Field(default_factory=dict)
    submitted_at: datetime     = Field(default_factory=_now)
    started_at:  datetime | None = None
    finished_at: datetime | None = None

    # Events accumulated for SSE streaming.
    events: list[dict[str, Any]] = Field(default_factory=list)

    def push_event(self, **kw: Any) -> None:
        self.events.append({"ts": _now().isoformat(), **kw})

    @property
    def wait_seconds(self) -> float | None:
        if self.started_at and self.submitted_at:
            return (self.started_at - self.submitted_at).total_seconds()
        return None

    @property
    def run_seconds(self) -> float | None:
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None
