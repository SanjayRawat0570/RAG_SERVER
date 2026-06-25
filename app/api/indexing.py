"""Real-Time Indexing API (F19).

Endpoints
---------
POST /indexing/submit          Submit text for immediate background indexing
POST /indexing/batch           Submit multiple texts at once
POST /indexing/priority        Submit with URGENT priority (jumps the queue)
GET  /indexing/{task_id}       Task status + progress
GET  /indexing/{task_id}/stream  SSE real-time progress stream
DELETE /indexing/{task_id}     Cancel a pending task
GET  /indexing/queue/stats     Queue depth, throughput, worker count
GET  /indexing/queue/tasks     List my tasks (filterable by status)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.indexing import (
    IndexingTask, Priority, TaskStatus,
    cancel, get_task, list_tasks, queue_stats, start_workers, submit,
)
from app.rag.tenants import resolve_namespace

router = APIRouter(prefix="/indexing", tags=["indexing"])


# ── Request models ─────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    text:      str   = Field(..., min_length=1)
    filename:  str   = "document.txt"
    store:     str   = "kb"
    tenant:    str   = "default"
    namespace: str   = "default"
    org_slug:  str | None = None
    team_slug: str | None = None
    metadata:  dict[str, Any] = Field(default_factory=dict)
    priority:  int   = Priority.NORMAL


class BatchSubmitRequest(BaseModel):
    documents: list[SubmitRequest] = Field(..., min_length=1, max_length=50)
    store:     str = "kb"
    tenant:    str = "default"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_task(req: SubmitRequest, user_id: str, priority: int | None = None) -> IndexingTask:
    ns = req.namespace
    if ns == "default" and (req.org_slug or req.team_slug):
        ns = resolve_namespace(user_id, org_slug=req.org_slug, team_slug=req.team_slug)
    return IndexingTask(
        id=str(uuid4()),
        user_id=user_id,
        tenant=req.tenant,
        namespace=ns,
        filename=req.filename,
        text=req.text,
        store=req.store,
        priority=priority if priority is not None else req.priority,
        metadata=req.metadata,
    )


def _task_response(task: IndexingTask, include_events: bool = False) -> dict[str, Any]:
    d = task.model_dump(exclude={"text", "events"} if not include_events else {"text"})
    d["wait_seconds"] = task.wait_seconds
    d["run_seconds"]  = task.run_seconds
    return d


async def _ensure_workers() -> None:
    await start_workers(n=3)


# ── Submit endpoints ────────────────────────────────────────────────────────────

@router.post("/submit", status_code=202)
async def submit_document(req: SubmitRequest, user: CurrentUser) -> dict[str, Any]:
    """Submit a document for background indexing.  Returns immediately with a task ID."""
    await _ensure_workers()
    task = _make_task(req, user["id"])
    submit(task)
    return {
        "task_id":  task.id,
        "status":   task.status,
        "priority": task.priority,
        "message":  task.message,
        "stream_url": f"/api/v1/indexing/{task.id}/stream",
    }


@router.post("/priority", status_code=202)
async def submit_priority(req: SubmitRequest, user: CurrentUser) -> dict[str, Any]:
    """Submit with URGENT priority — document jumps to the front of the queue."""
    await _ensure_workers()
    task = _make_task(req, user["id"], priority=Priority.URGENT)
    submit(task)
    return {
        "task_id":  task.id,
        "status":   task.status,
        "priority": task.priority,
        "message":  "Queued with URGENT priority",
        "stream_url": f"/api/v1/indexing/{task.id}/stream",
    }


@router.post("/batch", status_code=202)
async def submit_batch(req: BatchSubmitRequest, user: CurrentUser) -> dict[str, Any]:
    """Submit multiple documents in one call.  All are queued at NORMAL priority."""
    await _ensure_workers()
    tasks = []
    for doc in req.documents:
        doc = doc.model_copy(update={"store": req.store, "tenant": req.tenant})
        task = _make_task(doc, user["id"])
        submit(task)
        tasks.append({"task_id": task.id, "filename": task.filename, "status": task.status})
    return {
        "submitted":  len(tasks),
        "tasks":      tasks,
        "message":    f"{len(tasks)} documents queued for indexing",
    }


# ── Status + cancel ────────────────────────────────────────────────────────────

@router.get("/queue/stats")
async def get_queue_stats(_: CurrentUser) -> dict[str, Any]:
    """Return queue depth, worker count, and throughput stats."""
    return queue_stats()


@router.get("/queue/tasks")
async def get_my_tasks(
    user:   CurrentUser,
    status: str | None = Query(None, description="Filter by status"),
) -> dict[str, Any]:
    """List this user's indexing tasks, most recent first."""
    tasks = list_tasks(user_id=user["id"], status=status)
    return {
        "total": len(tasks),
        "tasks": [_task_response(t) for t in tasks],
    }


@router.get("/{task_id}")
async def get_task_status(task_id: str, user: CurrentUser) -> dict[str, Any]:
    task = get_task(task_id)
    if task is None or task.user_id != user["id"]:
        raise HTTPException(404, "Task not found")
    return _task_response(task, include_events=True)


@router.delete("/{task_id}")
async def cancel_task(task_id: str, user: CurrentUser) -> dict[str, Any]:
    task = get_task(task_id)
    if task is None or task.user_id != user["id"]:
        raise HTTPException(404, "Task not found")
    if task.status != TaskStatus.PENDING:
        raise HTTPException(409, f"Cannot cancel a task in '{task.status}' state")
    cancel(task_id)
    return {"task_id": task_id, "status": TaskStatus.CANCELLED, "message": "Task cancelled"}


# ── SSE real-time stream ───────────────────────────────────────────────────────

@router.get("/{task_id}/stream")
async def stream_task(task_id: str, user: CurrentUser) -> StreamingResponse:
    """Stream indexing progress as Server-Sent Events.

    The client receives one event per pipeline step and a final ``done``
    event when the task completes or fails.  The stream closes automatically.
    """
    task = get_task(task_id)
    if task is None or task.user_id != user["id"]:
        raise HTTPException(404, "Task not found")

    async def _generate():
        sent = 0  # index into task.events we've already sent
        while True:
            new_events = task.events[sent:]
            for ev in new_events:
                data = json.dumps(ev)
                yield f"data: {data}\n\n"
                sent += 1

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                               TaskStatus.CANCELLED):
                # Send a terminal event so the client knows the stream ended.
                yield f"data: {json.dumps({'step': 'stream_end', 'status': task.status})}\n\n"
                return

            await asyncio.sleep(0.1)

    return StreamingResponse(_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
