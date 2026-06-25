"""Priority indexing queue with worker pool and auto-retry (F19).

Architecture
------------
- A min-heap (heapq) acts as the priority queue: (priority, seq, task_id).
  Lower priority integer = higher urgency (URGENT=1 < HIGH=2 < NORMAL=5 < LOW=10).
- A configurable number of async workers drain the queue concurrently.
- Failed tasks are automatically re-queued up to max_attempts with exponential
  back-off (0.5 s → 1 s → 2 s) so transient errors don't block the queue.
- All state lives in-process (offline-first). A production system would
  persist to Redis / Supabase, but nothing here requires that.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
from datetime import datetime, timezone
from typing import Any

from app.rag.indexing.task import IndexingTask, Priority, TaskStatus

# ── Global state ────────────────────────────────────────────────────────────────

_heap:    list[tuple[int, int, str]]  = []        # (priority, seq, task_id)
_counter: itertools.count             = itertools.count()
_tasks:   dict[str, IndexingTask]    = {}          # task_id → task
_workers: list[asyncio.Task[None]]   = []
_running: bool                       = False
_lock:    asyncio.Lock | None        = None


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# ── Public API ───────────────────────────────────────────────────────────────────

def submit(task: IndexingTask) -> IndexingTask:
    """Add a task to the priority queue. Returns the task (queued status)."""
    task.status  = TaskStatus.PENDING
    task.step    = "queued"
    task.message = "Queued for indexing"
    task.push_event(step="queued", priority=task.priority)
    _tasks[task.id] = task
    heapq.heappush(_heap, (task.priority, next(_counter), task.id))
    return task


def cancel(task_id: str) -> bool:
    """Cancel a pending task. Running tasks are not interrupted."""
    task = _tasks.get(task_id)
    if not task or task.status != TaskStatus.PENDING:
        return False
    task.status      = TaskStatus.CANCELLED
    task.step        = "cancelled"
    task.message     = "Task cancelled by user"
    task.finished_at = datetime.now(timezone.utc)
    task.push_event(step="cancelled", message=task.message)
    return True


def get_task(task_id: str) -> IndexingTask | None:
    return _tasks.get(task_id)


def list_tasks(user_id: str | None = None,
               status: str | None = None) -> list[IndexingTask]:
    tasks = list(_tasks.values())
    if user_id:
        tasks = [t for t in tasks if t.user_id == user_id]
    if status:
        tasks = [t for t in tasks if t.status == status]
    tasks.sort(key=lambda t: t.submitted_at, reverse=True)
    return tasks


def queue_stats() -> dict[str, Any]:
    all_tasks = list(_tasks.values())
    counts: dict[str, int] = {}
    for t in all_tasks:
        counts[t.status] = counts.get(t.status, 0) + 1

    completed = [t for t in all_tasks if t.status == TaskStatus.COMPLETED and t.run_seconds is not None]
    avg_run   = (sum(t.run_seconds for t in completed) / len(completed)) if completed else 0.0  # type: ignore[arg-type]

    return {
        "pending":          counts.get(TaskStatus.PENDING,   0),
        "running":          counts.get(TaskStatus.RUNNING,   0),
        "completed":        counts.get(TaskStatus.COMPLETED, 0),
        "failed":           counts.get(TaskStatus.FAILED,    0),
        "cancelled":        counts.get(TaskStatus.CANCELLED, 0),
        "retrying":         counts.get(TaskStatus.RETRYING,  0),
        "total":            len(all_tasks),
        "queue_depth":      len([t for t in all_tasks if t.status == TaskStatus.PENDING]),
        "avg_run_seconds":  round(avg_run, 2),
        "workers_active":   len([w for w in _workers if not w.done()]),
    }


# ── Worker loop ───────────────────────────────────────────────────────────────────

async def _drain_queue(worker_id: int) -> None:
    from app.rag.indexing.worker import run_task

    while True:
        async with _get_lock():
            # Find the next pending task in the heap.
            task: IndexingTask | None = None
            skipped: list[tuple[int, int, str]] = []
            while _heap:
                entry = heapq.heappop(_heap)
                tid   = entry[2]
                t     = _tasks.get(tid)
                if t and t.status == TaskStatus.PENDING:
                    task = t
                    break
                skipped.append(entry)
            for e in skipped:
                heapq.heappush(_heap, e)

        if task is None:
            await asyncio.sleep(0.05)
            continue

        task.attempts += 1
        try:
            await run_task(task)
        except Exception:
            if task.attempts < task.max_attempts:
                # Exponential back-off then re-queue.
                delay = 0.5 * (2 ** (task.attempts - 1))
                task.status  = TaskStatus.RETRYING
                task.step    = "retrying"
                task.message = f"Retrying (attempt {task.attempts}/{task.max_attempts}) in {delay:.1f}s…"
                task.push_event(step="retry", attempt=task.attempts,
                                delay=delay, message=task.message)
                await asyncio.sleep(delay)
                task.status   = TaskStatus.PENDING
                task.error    = None
                heapq.heappush(_heap, (task.priority, next(_counter), task.id))
            # else: already marked FAILED by run_task


async def start_workers(n: int = 3) -> None:
    """Start *n* background worker coroutines that drain the queue."""
    global _running, _workers
    if _running:
        return
    _running = True
    loop = asyncio.get_event_loop()
    _workers = [loop.create_task(_drain_queue(i)) for i in range(n)]


def stop_workers() -> None:
    global _running
    _running = False
    for w in _workers:
        w.cancel()
    _workers.clear()


def reset_queue() -> None:
    """Clear all state (for tests)."""
    global _heap, _counter, _tasks, _lock
    stop_workers()
    _heap    = []
    _counter = itertools.count()
    _tasks   = {}
    _lock    = None
    _running = False
