"""The extractive fallback is streamed and labelled like any answer (P2)."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import settings

AUTH = {"Authorization": "Bearer dev"}


def _events(client, body: dict) -> list[dict]:
    with client.stream("POST", "/api/v1/rag/stream/ask", headers=AUTH, json=body) as r:
        text = "".join(r.iter_text())
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_fallback_answer_arrives_as_tokens_and_is_labelled(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "gemini_api_key", "")  # provider unavailable
    with TestClient(app) as client:
        client.post("/api/v1/rag/index", headers=AUTH, json={
            "tenant": "fb-t", "filename": "parking.md",
            "text": "# Parking\n\nThe parking garage opens at 7am. Each employee gets one badge."})
        events = _events(client, {"tenant": "fb-t", "question": "When does the parking garage open?",
                                  "provider": "gemini"})
    tokens = [e["token"] for e in events if e["event"] == "token"]
    sources = next(e for e in events if e["event"] == "sources")
    complete = next(e for e in events if e["event"] == "complete")
    assert complete["provider"] == sources["provider"] == "extractive-fallback"
    assert tokens and "".join(tokens) == complete["answer"]
    assert "7am" in complete["answer"] and sources["replaced"] is False
