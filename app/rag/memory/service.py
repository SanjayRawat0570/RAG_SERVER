"""Conversation memory around one question — shared by REST, SSE and MCP.

``before`` (prior to retrieval):
    load the caller's conversation and rewrite a follow-up into a standalone
    question, producing a ``memory_rewrite`` trace step when it changed.

``after`` (once answered):
    store the turn (short-term, last 5 per conversation) and record the topics
    of the *cited sources* in the user's history (long-term). Those topics feed
    the existing personalization boosts via :func:`effective_profile`.

Anonymous callers get no memory at all — a shared "anonymous" history would
leak one visitor's questions into another's answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.rag.memory import store
from app.rag.memory.rewriter import rewrite_follow_up
from app.rag.personalization.profile import QueryRecord, UserProfile
from app.rag.personalization.store import detect_topics, get_profile, infer_interests, record_query
from app.rag.vectorstore import lookup_store

ANONYMOUS = "anonymous"


@dataclass
class MemoryContext:
    question: str                       # as typed
    standalone: str                     # used for retrieval and generation
    conversation: store.Conversation | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def conversation_id(self) -> str | None:
        return self.conversation.id if self.conversation else None


async def before(*, tenant: str, user_id: str, conversation_id: str | None, question: str,
                 provider: str) -> MemoryContext:
    if not user_id or user_id == ANONYMOUS:
        return MemoryContext(question=question, standalone=question)
    conv = store.get_or_create(tenant, user_id, conversation_id)
    result = await rewrite_follow_up(question, store.history(conv), provider=provider,
                                     tenant=tenant, user_id=user_id)
    ctx = MemoryContext(question=question, standalone=result["question"], conversation=conv)
    if result["rewritten"]:
        ctx.steps.append({"type": "memory_rewrite", "round": 0, "query": result["question"],
                          "original": question, "method": result["method"]})
    return ctx


def _cited(out: dict[str, Any]) -> list[dict[str, Any]]:
    markers = set(out.get("cited_markers") or [])
    return [c for c in out.get("citations") or [] if c.get("marker") in markers]


def _texts(tenant: str, ids: set[str], store_name: str = "kb") -> list[str]:
    vs = lookup_store(store_name)
    if vs is None or not ids:
        return []
    return [str(meta.get("text", "")) for rid, meta in vs.list_records(namespace=tenant) if rid in ids]


def after(ctx: MemoryContext, *, tenant: str, user_id: str, out: dict[str, Any],
          provider: str = "") -> list[str]:
    """Store the turn and the cited topics; returns the topics recorded."""
    if ctx.conversation is None:
        return []
    cited = _cited(out)
    sources = list(dict.fromkeys(c.get("source") for c in cited if c.get("source")))
    store.add_turn(ctx.conversation, store.Turn(
        question=ctx.question, standalone_question=ctx.standalone,
        answer=str(out.get("answer", "")), cited_sources=sources))

    ids = {c["id"] for c in cited if c.get("id")}
    topics = detect_topics(" ".join(_texts(tenant, ids))) if ids else []
    if ids:
        record_query(user_id, QueryRecord(query=ctx.standalone, doc_ids=sorted(ids),
                                          provider=provider, topics=topics))
    return topics


def effective_profile(user_id: str) -> UserProfile:
    """The user's profile with long-term interests (cited topics) merged in."""
    profile = get_profile(user_id).model_copy(deep=True)
    inferred = [i["topic"] for i in infer_interests(user_id, top_n=3)]
    profile.content.interests = list(dict.fromkeys([*profile.content.interests, *inferred]))
    return profile


def present(ctx: MemoryContext, out: dict[str, Any]) -> dict[str, Any]:
    """Add memory fields (and the rewrite step) to an answer payload."""
    return {
        **out,
        "conversation_id": ctx.conversation_id,
        "original_question": ctx.question,
        "standalone_question": ctx.standalone,
        "steps": ctx.steps + list(out.get("steps") or []) or out.get("steps"),
    }
