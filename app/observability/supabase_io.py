"""Shared Supabase client and a non-blocking background writer.

The Supabase Python client is synchronous: every ``.execute()`` is an HTTPS
round trip. Called inline from an async request handler it blocks the event
loop, and every other request waits — the audit measured 0.6 req/s under load
with query logging on. So:

* :func:`client` creates the client **once** per process (it used to be created
  on every call, which costs a TLS handshake each time);
* :func:`insert` / :func:`update` only enqueue. One daemon thread drains the
  queue and sends inserts in batches (per table and column set), so writes
  never run on the event loop and many rows cost one round trip;
* the queue is bounded: when Supabase is down or slow, rows beyond
  ``SUPABASE_WRITE_QUEUE`` are dropped and counted rather than growing memory;
* :func:`run` runs a read in a worker thread for endpoints that must await one.

Writes are best-effort observability records (query logs, audit events,
execution history); losing one never fails a request.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any, Callable

from app.config import settings

log = logging.getLogger(__name__)

_client: Any = None
_client_lock = threading.Lock()
_queue: queue.Queue | None = None
_thread: threading.Thread | None = None
_stats = {"enqueued": 0, "written": 0, "dropped": 0, "failed": 0}
BATCH = 100


def enabled() -> bool:
    return bool(settings.supabase_url and settings.supabase_key)


def client() -> Any:
    """The process-wide Supabase client, or None when Supabase is not configured."""
    global _client
    if not enabled():
        return None
    if _client is None:
        with _client_lock:
            if _client is None:
                from supabase import create_client  # type: ignore[import]

                _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def _ensure_writer() -> queue.Queue:
    global _queue, _thread
    if _queue is None or _thread is None or not _thread.is_alive():
        with _client_lock:
            if _queue is None:
                _queue = queue.Queue(maxsize=settings.supabase_write_queue)
            if _thread is None or not _thread.is_alive():
                _thread = threading.Thread(target=_drain, args=(_queue,), name="supabase-writer",
                                           daemon=True)
                _thread.start()
    return _queue


def _enqueue(op: tuple) -> None:
    if not enabled():
        return
    try:
        _ensure_writer().put_nowait(op)
        _stats["enqueued"] += 1
    except queue.Full:
        _stats["dropped"] += 1


def insert(table: str, row: dict[str, Any]) -> None:
    """Queue *row* for insertion into *table*; returns immediately."""
    _enqueue(("insert", table, row))


def update(table: str, values: dict[str, Any], **match: Any) -> None:
    """Queue ``UPDATE table SET values WHERE match``; returns immediately."""
    _enqueue(("update", table, values, match))


def _execute(ops: list[tuple]) -> None:
    sb = client()
    inserts: dict[tuple[str, frozenset], list[dict]] = {}
    for op in ops:
        if op[0] == "insert":
            inserts.setdefault((op[1], frozenset(op[2])), []).append(op[2])
    for (table, _), rows in inserts.items():
        try:
            sb.table(table).insert(rows).execute()
            _stats["written"] += len(rows)
        except Exception as exc:  # noqa: BLE001 - observability must never break anything
            _stats["failed"] += len(rows)
            log.warning("supabase insert into %s failed: %s", table, exc)
    for op in ops:
        if op[0] == "update":
            _, table, values, match = op
            try:
                q = sb.table(table).update(values)
                for k, v in match.items():
                    q = q.eq(k, v)
                q.execute()
                _stats["written"] += 1
            except Exception as exc:  # noqa: BLE001
                _stats["failed"] += 1
                log.warning("supabase update of %s failed: %s", table, exc)


def _drain(q: queue.Queue) -> None:
    while True:
        ops = [q.get()]
        while len(ops) < BATCH:
            try:
                ops.append(q.get_nowait())
            except queue.Empty:
                break
        try:
            _execute(ops)
        finally:
            for _ in ops:
                q.task_done()


def flush(timeout: float = 10.0) -> bool:
    """Wait until queued writes are sent (tests, shutdown). True if drained in time."""
    if _queue is None:
        return True
    done = threading.Event()
    threading.Thread(target=lambda: (_queue.join(), done.set()), daemon=True).start()
    return done.wait(timeout)


def stats() -> dict[str, int]:
    return {**_stats, "queued": _queue.qsize() if _queue else 0}


async def run(fn: Callable[[Any], Any]) -> Any:
    """Run ``fn(client)`` (a blocking Supabase read) in a worker thread."""
    sb = client()
    return await asyncio.to_thread(fn, sb)


def reset_for_tests(fake_client: Any = None) -> None:
    """Swap the client and start over with a fresh queue (and writer) on next use."""
    global _client, _queue, _thread
    _client = fake_client
    _queue = _thread = None
    for k in _stats:
        _stats[k] = 0
