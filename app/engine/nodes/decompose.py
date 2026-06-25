"""Decompose node — breaks a complex question into focused sub-questions (F3).

Offline-first heuristics (no LLM required):
  • Temporal comparison  → per-year question + "difference?" question
  • Multi-part "?"       → split on "?" boundaries
  • "and" conjunction    → split on " and "
  • Default             → return question as-is (single-hop)

Config::

    {
      "question":           "$.inputs.question",
      "max_sub_questions":  5
    }

Output::

    {"original": "...", "sub_questions": ["q1", "q2", ...]}
"""
from __future__ import annotations

import re
from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_AND_SPLIT = re.compile(r"\s+and\s+", re.I)
_COMPARISON_RE = re.compile(
    r"\b(compar|differ|change|changed|vs\.?|versus|between|from .+ to )\b", re.I
)


def _decompose(question: str, max_sub: int) -> list[str]:
    q = question.strip()

    # Temporal comparison: "How did budget change from 2022 to 2023?"
    years = _YEAR_RE.findall(q)
    if len(years) >= 2 and _COMPARISON_RE.search(q):
        base = _YEAR_RE.sub("", q).strip(" ,?").lower()
        # Remove leading question words for cleaner sub-questions
        base = re.sub(r"^(how did|what was|how much did)\s+", "", base, flags=re.I).strip()
        sub: list[str] = [f"What was {base} in {yr}?" for yr in years[: max_sub - 1]]
        sub.append(f"What is the difference between {years[0]} and {years[1]}?")
        return sub[:max_sub]

    # Multi-part question split on "?" boundaries
    parts = [p.strip() for p in q.split("?") if p.strip()]
    if len(parts) > 1:
        return [(p + "?") for p in parts[:max_sub]]

    # "and" conjunction: "What is X and Y?"
    and_parts = [p.strip() for p in _AND_SPLIT.split(q) if p.strip()]
    if len(and_parts) > 1:
        return [(p.rstrip("?") + "?") for p in and_parts[:max_sub]]

    # Single question — return as-is
    return [q if q.endswith("?") else q + "?"]


@register
class DecomposeNode(Node):
    type = "decompose"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        if "question" in self.config:
            question = str(ctx.resolve(self.config["question"]) or "")
        else:
            question = str(_single_upstream(upstream) or "")

        max_sub = int(self.config.get("max_sub_questions", 5))
        sub_questions = _decompose(question, max_sub)
        return {"original": question, "sub_questions": sub_questions}
