"""Conversation memory is opt-in and bounded (P2)."""
from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.rag.memory import store

AUTH = {"Authorization": "Bearer dev"}


@pytest.fixture
def client():
    from app.main import app

    store.reset()
    with TestClient(app) as c:
        yield c
    store.reset()


def _ask(client, question: str, **extra) -> dict:
    return client.post("/api/v1/rag/ask", headers=AUTH,
                       json={"tenant": "conv-t", "question": question, **extra}).json()


def test_stateless_questions_store_nothing(client):
    for i in range(100):
        assert _ask(client, f"Question number {i}?")["conversation_id"] is None
    assert store.count() == 0


def test_conversations_are_created_on_request_and_continued(client):
    first = _ask(client, "Who leads the Mobile team?", new_conversation=True)
    conv_id = first["conversation_id"]
    assert conv_id and store.count() == 1
    assert _ask(client, "And where are they based?", conversation_id=conv_id)["conversation_id"] == conv_id
    assert store.count() == 1


def test_per_user_cap_drops_the_least_recently_used(client, monkeypatch):
    monkeypatch.setattr(settings, "conversation_max_per_user", 3)
    ids = [_ask(client, f"Topic {i}?", new_conversation=True)["conversation_id"] for i in range(5)]
    listed = {c["id"] for c in client.get("/api/v1/conversations?tenant=conv-t", headers=AUTH)
              .json()["conversations"]}
    assert listed == set(ids[2:])


def test_untouched_conversations_expire(client):
    conv_id = _ask(client, "Who leads the Mobile team?", new_conversation=True)["conversation_id"]
    conv = store.get("conv-t", "dev", conv_id)
    conv.updated_at -= timedelta(hours=settings.conversation_ttl_hours + 1)
    assert store.get("conv-t", "dev", conv_id) is None
    # Continuing an expired id starts a fresh chat instead of failing.
    again = _ask(client, "Who leads the Mobile team?", conversation_id=conv_id)
    assert again["conversation_id"] and again["conversation_id"] != conv_id
