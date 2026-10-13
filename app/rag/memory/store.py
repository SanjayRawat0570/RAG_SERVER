"""Short-term conversation memory: the last few turns of each chat.

Conversations are keyed by ``(tenant, user_id, conversation_id)``: a user only
ever sees their own chats, and a chat in one tenant is invisible from another.
Only the last :data:`MAX_TURNS` turns are kept — enough to resolve follow-ups
("what about part-time staff?") without growing prompts or storing more
personal data than needed. In-memory like the other stores; GDPR export and
erasure cover it (:func:`export_user`, :func:`erase_user`).
"""
from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MAX_TURNS = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Turn:
    question: str               # what the user typed
    standalone_question: str    # after follow-up rewriting
    answer: str
    cited_sources: list[str] = field(default_factory=list)
    at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "standalone_question": self.standalone_question,
                "answer": self.answer, "cited_sources": self.cited_sources, "at": self.at.isoformat()}


@dataclass
class Conversation:
    id: str
    tenant: str
    user_id: str
    title: str = "New chat"
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    turns: deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS))

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "tenant": self.tenant, "title": self.title,
                "created_at": self.created_at.isoformat(), "updated_at": self.updated_at.isoformat(),
                "turns": len(self.turns)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "messages": [t.to_dict() for t in self.turns]}


_conversations: dict[tuple[str, str, str], Conversation] = {}


def create(tenant: str, user_id: str) -> Conversation:
    conv = Conversation(id=uuid.uuid4().hex[:16], tenant=tenant, user_id=user_id)
    _conversations[(tenant, user_id, conv.id)] = conv
    return conv


def get(tenant: str, user_id: str, conversation_id: str) -> Conversation | None:
    return _conversations.get((tenant, user_id, conversation_id))


def get_or_create(tenant: str, user_id: str, conversation_id: str | None) -> Conversation:
    """The caller's conversation, or a fresh one if the id is unknown to *this* user."""
    if conversation_id:
        conv = get(tenant, user_id, conversation_id)
        if conv is not None:
            return conv
    return create(tenant, user_id)


def history(conv: Conversation) -> list[Turn]:
    return list(conv.turns)


def add_turn(conv: Conversation, turn: Turn) -> None:
    if not conv.turns:
        conv.title = turn.question[:80]
    conv.turns.append(turn)
    conv.updated_at = _now()


def list_conversations(tenant: str, user_id: str) -> list[Conversation]:
    convs = [c for (t, u, _), c in _conversations.items() if t == tenant and u == user_id]
    return sorted(convs, key=lambda c: c.updated_at, reverse=True)


def delete(tenant: str, user_id: str, conversation_id: str) -> bool:
    return _conversations.pop((tenant, user_id, conversation_id), None) is not None


def export_user(user_id: str) -> list[dict[str, Any]]:
    return [c.to_dict() for (_, u, _), c in _conversations.items() if u == user_id]


def erase_user(user_id: str) -> int:
    keys = [k for k in _conversations if k[1] == user_id]
    for k in keys:
        del _conversations[k]
    return len(keys)


def reset() -> None:
    _conversations.clear()
