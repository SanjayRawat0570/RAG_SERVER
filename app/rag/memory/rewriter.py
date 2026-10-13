"""Follow-up rewriting: turn "what about part-time staff?" into a standalone question.

Retrieval only sees the current question, so a follow-up that leans on earlier
turns ("what about…", "how often are they rotated?") would search for the
wrong thing. Before retrieval the question is rewritten using the recent turns:

* **LLM rewrite** when a real model is configured (through the LLM gateway, so
  breaker/budget/cost/tracing apply).
* **Heuristic rewrite** offline or when the model fails:
  - "what about X?" / "how about X?" / "and X?" — substitute X into the previous
    standalone question, replacing the phrase that shares X's head noun
    ("What is the leave policy for full-time staff?" + "what about part-time
    staff?" → "What is the leave policy for part-time staff?") or its leading
    word ("Project Atlas" → "Project Cedar");
  - pronoun follow-ups ("is it…", "are they…") — attach the previous question
    as context so retrieval still finds the right passages.

Self-contained questions are returned unchanged.
"""
from __future__ import annotations

import re
from typing import Any

from app.rag.llm.gateway import call_llm, is_configured
from app.rag.memory.store import Turn

_SWAP = re.compile(r"^\s*(?:and\s+)?(?:what|how)\s+about\s+(?P<x>.+?)\s*\??\s*$|^\s*and\s+(?P<y>.+?)\s*\?\s*$",
                   re.IGNORECASE)
_PRONOUN = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|him|her)\b", re.IGNORECASE)
_ENTITY = re.compile(r"\b[A-Z][\w-]+")
_LEADING_DET = re.compile(r"^(?:the|a|an|our|their)\s+", re.IGNORECASE)

REWRITE_SYSTEM = (
    "Rewrite the user's latest question as a single standalone question that can be "
    "understood without the conversation. Resolve pronouns and references like "
    "'what about X' using the earlier turns. Keep the user's wording where possible. "
    "If the question is already standalone, return it unchanged. Reply with the "
    "question only."
)


def is_follow_up(question: str, turns: list[Turn]) -> bool:
    if not turns:
        return False
    if _SWAP.match(question):
        return True
    words = question.split()
    # Short questions with a pronoun and no named entity of their own lean on context.
    named = [w for w in _ENTITY.findall(question) if w.lower() not in {"what", "how", "who", "is", "are",
                                                                       "does", "do", "and", "which", "when",
                                                                       "where", "why", "can", "i"}]
    return len(words) <= 10 and bool(_PRONOUN.search(question)) and not named


def heuristic_rewrite(question: str, turns: list[Turn]) -> str:
    previous = turns[-1].standalone_question.strip().rstrip("?").strip()
    m = _SWAP.match(question)
    if m:
        new = _LEADING_DET.sub("", (m.group("x") or m.group("y") or "").strip().rstrip("?"))
        new_words = new.split()
        head = new_words[-1].lower() if new_words else ""
        prev_words = previous.split()
        lowered = [w.lower().strip(",.") for w in prev_words]
        if head and head in lowered:
            end = len(lowered) - 1 - lowered[::-1].index(head)   # last occurrence of the head noun
            start = max(0, end - len(new_words) + 1)
            # Do not swallow the preposition/determiner before the phrase.
            while start < end and lowered[start] in {"for", "of", "to", "in", "on", "the", "a", "an", "do", "does"}:
                start += 1
            return " ".join(prev_words[:start] + new_words + prev_words[end + 1:]) + "?"
        first = new_words[0].lower() if new_words else ""
        if first and first in lowered:
            # Same leading word ("Project Cedar" vs "Project Atlas"): swap that span.
            start = lowered.index(first)
            end = min(len(prev_words), start + len(new_words))
            return " ".join(prev_words[:start] + new_words + prev_words[end:]) + "?"
        for prep in (" for ", " of ", " in ", " about "):
            if prep in previous:
                return previous.rsplit(prep, 1)[0] + prep + new + "?"
        return f"{previous} for {new}?"
    # Pronoun follow-up: keep the question, add the previous one as context.
    return f"{question.strip().rstrip('?')}? (context: {previous}?)"


async def rewrite_follow_up(
    question: str,
    turns: list[Turn],
    *,
    provider: str,
    tenant: str | None = None,
    user_id: str = "",
) -> dict[str, Any]:
    """Return ``{"question": standalone, "rewritten": bool, "method": ...}``."""
    if not is_follow_up(question, turns):
        return {"question": question, "rewritten": False, "method": None}

    if provider != "stub" and is_configured(provider):
        transcript = "\n".join(
            f"User: {t.standalone_question}\nAssistant: {t.answer[:300]}" for t in turns[-3:])
        try:
            response = await call_llm(
                provider,
                {"messages": [
                    {"role": "system", "content": REWRITE_SYSTEM},
                    {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest question: {question}"},
                ]},
                {"temperature": 0.0, "max_tokens": 128},
                tenant=tenant, user_id=user_id, operation="rewrite_query",
            )
            text = response.text.strip().strip('"').splitlines()[0].strip() if response.text.strip() else ""
            if 3 <= len(text) <= 400:
                return {"question": text, "rewritten": text != question, "method": "llm"}
        except Exception:  # noqa: BLE001 - fall through to the offline rewrite
            pass

    return {"question": heuristic_rewrite(question, turns), "rewritten": True, "method": "heuristic"}
