"""Tests for F19: Real-Time Indexing."""
from __future__ import annotations

import asyncio
import time

import pytest

from app.rag.indexing import (
    IndexingTask, Priority, TaskStatus,
    cancel, get_task, list_tasks, queue_stats, reset_queue, submit,
)
from app.rag.indexing.worker import run_task
from app.rag.vectorstore import reset_stores


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean():
    reset_queue()
    reset_stores()
    yield
    reset_queue()
    reset_stores()


def _task(text: str = "Hello world. This is a test document.",
          filename: str = "doc.txt",
          priority: int = Priority.NORMAL,
          user_id: str = "user-1") -> IndexingTask:
    import uuid
    return IndexingTask(
        id=str(uuid.uuid4()),
        text=text,
        filename=filename,
        priority=priority,
        user_id=user_id,
        store="test-store",
        namespace="test-ns",
    )


# ── Task model ─────────────────────────────────────────────────────────────────

def test_f19_task_default_status_pending():
    t = _task()
    assert t.status == TaskStatus.PENDING


def test_f19_task_push_event():
    t = _task()
    t.push_event(step="test", progress=50)
    assert len(t.events) == 1
    assert t.events[0]["step"] == "test"


def test_f19_task_wait_seconds_none_before_start():
    t = _task()
    assert t.wait_seconds is None


def test_f19_task_run_seconds_none_before_finish():
    t = _task()
    assert t.run_seconds is None


def test_f19_priority_ordering():
    assert Priority.URGENT < Priority.HIGH < Priority.NORMAL < Priority.LOW


# ── Queue: submit / cancel ─────────────────────────────────────────────────────

def test_f19_submit_adds_to_queue():
    t = submit(_task())
    assert t.status == TaskStatus.PENDING
    assert get_task(t.id) is t


def test_f19_submit_returns_task():
    t = _task()
    result = submit(t)
    assert result is t


def test_f19_cancel_pending_task():
    t = submit(_task())
    assert cancel(t.id) is True
    assert get_task(t.id).status == TaskStatus.CANCELLED


def test_f19_cancel_nonexistent_returns_false():
    assert cancel("no-such-id") is False


def test_f19_cancel_completed_returns_false():
    import uuid
    from app.rag.indexing.queue import _tasks
    # Insert a task directly in COMPLETED state, bypassing submit() which forces PENDING.
    t = IndexingTask(id=str(uuid.uuid4()), text="x", user_id="user-1",
                     status=TaskStatus.COMPLETED)
    _tasks[t.id] = t
    assert cancel(t.id) is False


def test_f19_list_tasks_by_user():
    t1 = submit(_task(user_id="alice"))
    t2 = submit(_task(user_id="bob"))
    alice_tasks = list_tasks(user_id="alice")
    assert t1 in alice_tasks
    assert t2 not in alice_tasks


def test_f19_list_tasks_by_status():
    t1 = submit(_task())
    t2 = submit(_task())
    cancel(t2.id)
    pending = list_tasks(status=TaskStatus.PENDING)
    cancelled = list_tasks(status=TaskStatus.CANCELLED)
    assert t1 in pending
    assert t2 in cancelled


def test_f19_queue_stats_counts_correctly():
    submit(_task())
    submit(_task())
    t3 = submit(_task())
    cancel(t3.id)
    stats = queue_stats()
    assert stats["pending"]   == 2
    assert stats["cancelled"] == 1
    assert stats["total"]     == 3


def test_f19_queue_stats_queue_depth():
    submit(_task())
    submit(_task())
    stats = queue_stats()
    assert stats["queue_depth"] == 2


# ── Priority ordering ──────────────────────────────────────────────────────────

def test_f19_urgent_submitted_before_normal():
    urgent = _task(priority=Priority.URGENT)
    normal = _task(priority=Priority.NORMAL)
    # Submit normal first, then urgent — urgent should be popped first.
    submit(normal)
    submit(urgent)
    # Verify heap ordering: smallest priority value = highest urgency.
    import heapq
    from app.rag.indexing.queue import _heap
    first_priority = _heap[0][0]
    assert first_priority == Priority.URGENT


# ── Worker pipeline ────────────────────────────────────────────────────────────

def test_f19_worker_runs_to_completion():
    t = _task(text="Quarterly revenue was $5 million. Expenses were $3 million. "
                    "Net profit was $2 million for the period.")
    asyncio.run(run_task(t))
    assert t.status  == TaskStatus.COMPLETED
    assert t.progress == 100
    assert t.chunks_done >= 1


def test_f19_worker_sets_started_and_finished():
    t = _task(text="Revenue was $1M. Expenses were $0.5M.")
    asyncio.run(run_task(t))
    assert t.started_at  is not None
    assert t.finished_at is not None


def test_f19_worker_pushes_events():
    t = _task(text="Some document content here.")
    asyncio.run(run_task(t))
    steps = [e["step"] for e in t.events]
    assert "start"   in steps
    assert "chunk"   in steps
    assert "embed"   in steps
    assert "upsert"  in steps
    assert "done"    in steps


def test_f19_worker_indexes_to_vector_store():
    from app.rag.vectorstore import get_store
    from app.rag.embeddings import embed_texts
    from app.rag.embeddings.registry import DEFAULT_DIMENSION, DEFAULT_MODEL

    text = "Revenue was $10M in Q4. Expenses grew by 5%. Net income improved."
    t = _task(text=text, filename="report.txt")
    asyncio.run(run_task(t))

    store = get_store(t.store, DEFAULT_DIMENSION)
    q_vec = embed_texts(["revenue"], DEFAULT_MODEL, DEFAULT_DIMENSION)[0]
    hits  = store.search(q_vec, top_k=5, namespace=t.namespace)
    assert len(hits) >= 1


def test_f19_worker_fails_on_empty_text():
    t = _task(text="   ")  # empty after strip
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(run_task(t))
    assert t.status == TaskStatus.FAILED


def test_f19_worker_records_quality_metadata():
    t = _task(text="Detailed financial analysis with sufficient content for quality check.")
    asyncio.run(run_task(t))
    assert "quality" in t.metadata


def test_f19_worker_run_seconds_positive():
    t = _task(text="Some text content for timing measurement.")
    asyncio.run(run_task(t))
    assert t.run_seconds is not None
    assert t.run_seconds >= 0.0


# ── End-to-end queue drain ─────────────────────────────────────────────────────

def test_f19_queue_drains_to_completed():
    async def _run():
        from app.rag.indexing.queue import start_workers
        await start_workers(n=2)
        t = submit(_task(text="Financial report: revenue $5M, expenses $3M."))
        deadline = time.time() + 5
        while time.time() < deadline:
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                break
            await asyncio.sleep(0.05)
        return t

    t = asyncio.run(_run())
    assert t.status == TaskStatus.COMPLETED


def test_f19_queue_retries_on_failure():
    async def _run():
        from app.rag.indexing.queue import start_workers
        await start_workers(n=1)
        t = submit(_task(text="   ", priority=Priority.URGENT))
        t.max_attempts = 2
        deadline = time.time() + 5
        while time.time() < deadline:
            if t.status == TaskStatus.FAILED:
                break
            await asyncio.sleep(0.05)
        return t

    t = asyncio.run(_run())
    assert t.status == TaskStatus.FAILED
    assert t.attempts >= 1


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}
DOC_TEXT = "Annual report: Revenue $10M, Expenses $6M, Net profit $4M."


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f19_api_submit():
    with _client() as c:
        resp = c.post("/api/v1/indexing/submit",
                      json={"text": DOC_TEXT, "filename": "report.txt"},
                      headers=AUTH)
    assert resp.status_code == 202
    data = resp.json()
    assert "task_id"    in data
    assert "status"     in data
    assert "stream_url" in data
    assert data["status"] == TaskStatus.PENDING


def test_f19_api_submit_priority():
    with _client() as c:
        resp = c.post("/api/v1/indexing/priority",
                      json={"text": DOC_TEXT, "filename": "urgent.txt"},
                      headers=AUTH)
    assert resp.status_code == 202
    assert resp.json()["priority"] == Priority.URGENT


def test_f19_api_batch_submit():
    docs = [
        {"text": "Document one about revenue."},
        {"text": "Document two about expenses."},
        {"text": "Document three about strategy."},
    ]
    with _client() as c:
        resp = c.post("/api/v1/indexing/batch",
                      json={"documents": docs, "store": "kb", "tenant": "default"},
                      headers=AUTH)
    assert resp.status_code == 202
    data = resp.json()
    assert data["submitted"] == 3
    assert len(data["tasks"]) == 3


def test_f19_api_get_task():
    with _client() as c:
        sub  = c.post("/api/v1/indexing/submit",
                      json={"text": DOC_TEXT}, headers=AUTH)
        tid  = sub.json()["task_id"]
        resp = c.get(f"/api/v1/indexing/{tid}", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"]     == tid
    assert "status"       in data
    assert "progress"     in data
    assert "events"       in data


def test_f19_api_get_task_not_found():
    with _client() as c:
        resp = c.get("/api/v1/indexing/no-such-task", headers=AUTH)
    assert resp.status_code == 404


def test_f19_api_cancel_task():
    # Submit directly via the queue (no workers) so task stays PENDING.
    task = submit(_task(text=DOC_TEXT, user_id="dev"))
    with _client() as c:
        resp = c.delete(f"/api/v1/indexing/{task.id}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == TaskStatus.CANCELLED


def test_f19_api_cancel_not_pending():
    # Cancel the same task twice — second call should be 409.
    task = submit(_task(text=DOC_TEXT, user_id="dev"))
    with _client() as c:
        c.delete(f"/api/v1/indexing/{task.id}", headers=AUTH)
        resp = c.delete(f"/api/v1/indexing/{task.id}", headers=AUTH)
    assert resp.status_code == 409


def test_f19_api_queue_stats():
    with _client() as c:
        c.post("/api/v1/indexing/submit", json={"text": DOC_TEXT}, headers=AUTH)
        resp = c.get("/api/v1/indexing/queue/stats", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "pending"      in data
    assert "completed"    in data
    assert "queue_depth"  in data
    assert "total"        in data


def test_f19_api_list_my_tasks():
    with _client() as c:
        c.post("/api/v1/indexing/submit", json={"text": DOC_TEXT, "filename": "a.txt"},
               headers=AUTH)
        c.post("/api/v1/indexing/submit", json={"text": DOC_TEXT, "filename": "b.txt"},
               headers=AUTH)
        resp = c.get("/api/v1/indexing/queue/tasks", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2


def test_f19_api_list_tasks_status_filter():
    with _client() as c:
        sub = c.post("/api/v1/indexing/submit", json={"text": DOC_TEXT}, headers=AUTH)
        tid = sub.json()["task_id"]
        c.delete(f"/api/v1/indexing/{tid}", headers=AUTH)
        resp = c.get("/api/v1/indexing/queue/tasks?status=cancelled", headers=AUTH)
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert all(t["status"] == "cancelled" for t in tasks)


def test_f19_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/indexing/queue/stats")
    assert resp.status_code == 401
