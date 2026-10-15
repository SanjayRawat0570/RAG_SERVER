"""Supabase writes never block a request (P1 throughput fix)."""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.observability import query_log, supabase_io

DELAY = 1.0


class SlowSupabase:
    """Stands in for the Supabase client: every round trip takes DELAY seconds."""

    def __init__(self) -> None:
        self.rows: dict[str, list[dict]] = {}
        self.calls = 0
        self.threads: set[str] = set()

    def table(self, name: str):
        outer = self

        class Query:
            def insert(self, rows):
                self.rows = rows if isinstance(rows, list) else [rows]
                return self

            def update(self, values):
                self.rows = []
                return self

            def eq(self, *_):
                return self

            def execute(self):
                outer.calls += 1
                outer.threads.add(threading.current_thread().name)
                time.sleep(DELAY)
                outer.rows.setdefault(name, []).extend(self.rows)

        return Query()


@pytest.fixture
def slow_supabase(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_key", "test-key")
    fake = SlowSupabase()
    supabase_io.reset_for_tests(fake)
    yield fake
    supabase_io.flush(30)
    supabase_io.reset_for_tests(None)


def test_log_query_returns_immediately_and_rows_arrive_in_batches(slow_supabase):
    t0 = time.perf_counter()
    for i in range(50):
        query_log.log_query(user_id="u", question=f"q{i}")
    assert time.perf_counter() - t0 < 0.2  # 50 logs, each a 1 s round trip if inline
    assert supabase_io.flush(30)
    assert len(slow_supabase.rows["query_logs"]) == 50
    assert slow_supabase.calls <= 2  # batched (first row alone, the other 49 together)
    assert slow_supabase.threads == {"supabase-writer"}


def test_rag_ask_is_not_slowed_by_a_slow_supabase(slow_supabase):
    from app.main import app

    with TestClient(app) as client:
        auth = {"Authorization": "Bearer dev"}
        client.post("/api/v1/rag/index", headers=auth,
                    json={"tenant": "sb-t", "filename": "a.md", "text": "# A\n\nThe garage opens at 7am."})
        t0 = time.perf_counter()
        for _ in range(3):
            assert client.post("/api/v1/rag/ask", headers=auth,
                               json={"tenant": "sb-t", "question": "When does the garage open?"}).status_code == 200
        elapsed = time.perf_counter() - t0
    assert elapsed < DELAY, f"3 asks took {elapsed:.2f}s with a {DELAY}s Supabase"
    assert supabase_io.flush(30)
    assert len(slow_supabase.rows["query_logs"]) == 3


def test_client_is_created_once(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_key", "test-key")
    created = []
    import supabase

    monkeypatch.setattr(supabase, "create_client", lambda *a: created.append(a) or object())
    supabase_io.reset_for_tests(None)
    assert supabase_io.client() is supabase_io.client() is supabase_io.client()
    assert len(created) == 1
    supabase_io.reset_for_tests(None)


def test_full_queue_drops_instead_of_blocking(slow_supabase, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "supabase_write_queue", 5)
    supabase_io.reset_for_tests(slow_supabase)  # fresh queue with the small bound
    t0 = time.perf_counter()
    for i in range(200):
        supabase_io.insert("audit_logs", {"i": i})
    assert time.perf_counter() - t0 < 0.2
    assert supabase_io.stats()["dropped"] > 0
