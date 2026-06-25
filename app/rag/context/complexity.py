"""Query complexity estimator (F15).

Analyses a user query to recommend how much context the LLM needs.

Simple   → 1-2 chunks,   ~500  tokens  (single-fact lookup)
Moderate → 3-5 chunks,   ~1500 tokens  (multi-part question)
Complex  → 6-10 chunks,  ~3000 tokens  (temporal, comparative, causal)
"""
from __future__ import annotations

import re

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "who",
    "in", "on", "of", "to", "for", "and", "or", "it", "this", "that",
    "do", "does", "did", "can", "with", "from", "at", "by", "be", "has",
    "have", "had", "will", "would", "could", "should", "may", "might",
}

_TEMPORAL   = {"trend", "over", "years", "history", "since", "before", "after",
               "when", "year", "month", "quarter", "annual", "monthly", "weekly",
               "period", "timeline", "progression", "growth", "change", "shift"}

_COMPARATIVE = {"compare", "vs", "versus", "difference", "between", "than",
                "better", "worse", "higher", "lower", "more", "less",
                "which", "contrast", "relative", "against"}

_CAUSAL      = {"why", "because", "reason", "cause", "explain", "impact",
                "effect", "result", "due", "lead", "affect", "influence",
                "drove", "driven", "factor", "contribute"}

_MULTI_WORD  = {"list", "all", "every", "summarize", "overview", "breakdown",
                "detail", "describe", "outline", "breakdown", "enumerate"}


def _keywords(query: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", query) if w.lower() not in _STOP}


def estimate_complexity(query: str) -> dict[str, object]:
    """Return complexity level and recommended context parameters.

    Returns
    -------
    dict with keys:
      level               : "simple" | "moderate" | "complex"
      score               : raw signal count (0–4)
      signals             : which complexity signals fired
      recommended_chunks  : suggested number of chunks to retrieve
      recommended_tokens  : suggested max context tokens
    """
    words = _keywords(query)
    word_count = len(re.findall(r"\w+", query))

    signals: list[str] = []

    if words & _TEMPORAL:
        signals.append("temporal")
    if words & _COMPARATIVE:
        signals.append("comparative")
    if words & _CAUSAL:
        signals.append("causal")
    if words & _MULTI_WORD or word_count > 12:
        signals.append("multi_concept")

    score = len(signals)

    if score == 0:
        level, chunks, tokens = "simple",   2,  500
    elif score == 1:
        level, chunks, tokens = "moderate", 5, 1500
    else:
        level, chunks, tokens = "complex", 10, 3000

    return {
        "level":               level,
        "score":               score,
        "signals":             signals,
        "recommended_chunks":  chunks,
        "recommended_tokens":  tokens,
    }
