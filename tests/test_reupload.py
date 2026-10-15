"""Re-uploading a changed file replaces its old chunks (P1 data correctness)."""
from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

AUTH = {"Authorization": "Bearer dev"}
V1 = "# Parking\n\nThe parking garage opens at 7am. Each employee gets one parking badge."
V2 = "# Parking\n\nThe parking garage opens at 6am. Each employee gets two parking badges."
QUESTION = "When does the parking garage open?"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _count(client, tenant: str) -> int:
    stores = client.get("/api/v1/vectorstores", headers=AUTH).json()["stores"]
    return stores["kb"]["namespaces"].get(tenant, 0)


def _upload(client, tenant: str, text: str, name: str = "parking.md") -> None:
    r = client.post("/api/v1/documents/upload", headers=AUTH, data={"tenant": tenant},
                    files={"file": (name, text.encode(), "text/markdown")})
    exec_id = r.json()["execution_id"]
    for _ in range(100):
        status = client.get(f"/api/v1/documents/executions/{exec_id}", headers=AUTH).json()["status"]
        if status in ("completed", "failed"):
            assert status == "completed"
            return
        time.sleep(0.05)
    raise AssertionError("upload did not finish")


def _ask(client, tenant: str) -> str:
    return client.post("/api/v1/rag/ask", headers=AUTH,
                       json={"tenant": tenant, "question": QUESTION}).json()["answer"]


def test_upload_of_changed_file_replaces_old_chunks(client):
    tenant = f"re-{uuid.uuid4().hex[:6]}"
    _upload(client, tenant, V1)
    _upload(client, tenant, V1)  # identical: idempotent
    assert _count(client, tenant) == 1
    assert "7am" in _ask(client, tenant)
    _upload(client, tenant, V2)  # changed content, same file name
    assert _count(client, tenant) == 1
    answer = _ask(client, tenant)
    assert "6am" in answer and "7am" not in answer


def test_rag_index_replaces_too_and_other_files_are_untouched(client):
    tenant = f"re-{uuid.uuid4().hex[:6]}"
    for name, text in (("parking.md", V1), ("canteen.md", "# Canteen\n\nThe canteen opens at noon.")):
        assert client.post("/api/v1/rag/index", headers=AUTH,
                           json={"tenant": tenant, "filename": name, "text": text}).status_code == 200
    r = client.post("/api/v1/rag/index", headers=AUTH, json={"tenant": tenant, "filename": "parking.md", "text": V2})
    assert r.json()["replaced"] == 1
    assert _count(client, tenant) == 2
    sources = {h["source"] for h in client.post("/api/v1/rag/search", headers=AUTH, json={
        "tenant": tenant, "query": "opens", "top_k": 10}).json()["results"]}
    assert sources == {"parking.md", "canteen.md"}


def test_queued_indexing_replaces_changed_resubmission(client):
    tenant = f"re-{uuid.uuid4().hex[:6]}"
    for text in (V1, V1, V2):
        task = client.post("/api/v1/indexing/submit", headers=AUTH, json={
            "tenant": tenant, "namespace": tenant, "filename": "parking.md", "text": text}).json()
        for _ in range(200):
            if client.get(f"/api/v1/indexing/{task['task_id']}", headers=AUTH).json()["status"] in (
                    "completed", "failed"):
                break
            time.sleep(0.05)
    texts = [h["text"] for h in client.post("/api/v1/rag/search", headers=AUTH, json={
        "tenant": tenant, "query": "parking garage opens", "top_k": 10}).json()["results"]]
    assert texts and all("6am" in t for t in texts)
