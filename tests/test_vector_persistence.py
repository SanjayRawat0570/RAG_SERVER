"""Indexed documents survive a restart (P1: vector store journal)."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from app.config import settings
from app.engine.executor import WorkflowExecutor
from app.api.pipelines import build_index_workflow
from app.rag.vectorstore import get_store, reset_stores
from app.rag.vectorstore import journal as journal_mod
from app.rag.vectorstore.base import VectorRecord

BACKEND = Path(__file__).resolve().parent.parent
CORPUS = BACKEND / "eval" / "corpus"
DOCS = ("hr_leave_policy.md", "teams.md", "projects.md")


@pytest.fixture
def persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_persist", True)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    reset_stores()
    yield tmp_path
    reset_stores()


def _store():
    from app.api.pipelines import DIM, STORE

    return get_store(STORE, DIM)


async def _index(tenant: str, names=DOCS) -> None:
    for name in names:
        await WorkflowExecutor(build_index_workflow()).run(
            {"tenant": tenant, "filename": name, "text": (CORPUS / name).read_text(encoding="utf-8")})


async def test_documents_survive_a_store_restart(persisted):
    await _index("acme")
    before = _store().count("acme")
    hits_before = [h.id for h in _store().search([0.1] * _store().dimension, top_k=5, namespace="acme")]
    assert before > 0

    reset_stores()  # the process forgets everything in memory
    assert _store().count("acme") == before
    assert [h.id for h in _store().search([0.1] * _store().dimension, top_k=5, namespace="acme")] == hits_before
    assert _store().list_records("acme")[0][1]["text"]  # chunk text (BM25 corpus) is back too


async def test_deletes_and_updates_are_journalled(persisted):
    store = _store()
    dim = store.dimension
    store.upsert([VectorRecord(id="a", vector=[1.0] + [0.0] * (dim - 1), metadata={"text": "old"}),
                  VectorRecord(id="b", vector=[0.0, 1.0] + [0.0] * (dim - 2), metadata={"text": "b"})], "t")
    store.upsert([VectorRecord(id="a", vector=[1.0] + [0.0] * (dim - 1), metadata={"text": "new"})], "t")
    store.delete(["b"], "t")
    reset_stores()
    assert dict(_store().list_records("t")) == {"a": {"text": "new"}}


async def test_namespaces_stay_separate_on_disk(persisted):
    await _index("acme", DOCS[:1])
    await _index("globex/sales", DOCS[1:2])
    reset_stores()
    assert _store().count("acme") > 0 and _store().count("globex/sales") > 0
    sources = {m["source"] for _, m in _store().list_records("globex/sales")}
    assert sources == {DOCS[1]}


async def test_many_small_writes_are_compacted(persisted, monkeypatch):
    monkeypatch.setattr(journal_mod, "COMPACT_AFTER", 5)
    store = _store()
    dim = store.dimension
    for i in range(12):
        store.upsert([VectorRecord(id=str(i), vector=[float(i + 1)] + [0.0] * (dim - 1), metadata={})], "c")
    segs = list(store._journal._segments("c"))
    assert len(segs) <= 6
    reset_stores()
    assert _store().count("c") == 12


# ── A real server restart ──────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(data_dir: Path, port: int) -> subprocess.Popen:
    env = {**os.environ, "DATA_DIR": str(data_dir), "VECTOR_PERSIST": "true", "APP_ENV": "development",
           "LLM_PROVIDER": "stub", "SUPABASE_URL": "", "SUPABASE_KEY": "", "EMBEDDING_MODEL": "local-hash"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port),
                             "--log-level", "warning"], cwd=BACKEND, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not start")


def test_server_restart_keeps_the_document_count(tmp_path):
    port = _free_port()
    api = f"http://127.0.0.1:{port}/api/v1"
    auth = {"Authorization": "Bearer dev"}
    question = {"tenant": "acme", "question": "Who owns Project Beacon?"}

    proc = _serve(tmp_path, port)
    try:
        for name in DOCS:
            r = httpx.post(f"{api}/rag/index", headers=auth, timeout=60, json={
                "tenant": "acme", "filename": name, "text": (CORPUS / name).read_text(encoding="utf-8")})
            assert r.status_code == 200
        count = httpx.get(f"{api}/vectorstores", headers=auth).json()["stores"]["kb"]["namespaces"]["acme"]
        answer = httpx.post(f"{api}/rag/ask", headers=auth, json=question, timeout=60).json()["answer"]
    finally:
        proc.kill()
        proc.wait()

    proc = _serve(tmp_path, port)
    try:
        after = httpx.get(f"{api}/vectorstores", headers=auth).json()["stores"]["kb"]["namespaces"]["acme"]
        answer_after = httpx.post(f"{api}/rag/ask", headers=auth, json=question, timeout=60).json()["answer"]
    finally:
        proc.kill()
        proc.wait()
    assert count > 0 and after == count
    assert answer_after == answer and "Mobile" in answer
