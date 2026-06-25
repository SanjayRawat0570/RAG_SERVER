"""Synthesize node — merges multiple Q&A pairs into one coherent answer (F3).

Offline-first: deduplicates sentences across all sub-answers and assembles
a single paragraph that directly addresses the original question.

Config::

    {
      "answers":           "$.inputs.sub_answers",   # list of {question, answer}
      "original_question": "$.inputs.question"
    }

Output::

    {
      "answer":            "...",
      "source_count":      N,
      "synthesized":       true,
      "original_question": "..."
    }
"""
from __future__ import annotations

import re
from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _dedupe_sentences(texts: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for sent in _SENT_SPLIT.split(text.strip()):
            s = sent.strip()
            norm = s.lower().rstrip(".!?")
            if s and norm not in seen:
                seen.add(norm)
                out.append(s)
    return out


@register
class SynthesizeNode(Node):
    type = "synthesize"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        # Resolve sub-answers: list of {question, answer} dicts
        sub_answers: list[Any] = []
        if "answers" in self.config:
            raw = ctx.resolve(self.config["answers"])
            if isinstance(raw, list):
                sub_answers = raw
        elif upstream:
            raw = _single_upstream(upstream)
            if isinstance(raw, list):
                sub_answers = raw

        original = str(ctx.resolve(self.config.get("original_question", "")) or "")

        # Extract answer texts, skip empty/don't-know responses
        _DONT_KNOW = re.compile(r"don.t know|no information|not found|unavailable", re.I)
        texts: list[str] = []
        for item in sub_answers:
            if isinstance(item, dict):
                ans = str(item.get("answer") or item.get("text") or "").strip()
            else:
                ans = str(item).strip()
            if ans and not _DONT_KNOW.search(ans):
                texts.append(ans)

        if not texts:
            return {
                "answer": "I could not find enough information to answer this question.",
                "source_count": 0,
                "synthesized": True,
                "original_question": original,
            }

        sentences = _dedupe_sentences(texts)
        answer = " ".join(sentences)

        return {
            "answer": answer,
            "source_count": len(texts),
            "synthesized": True,
            "original_question": original,
        }
