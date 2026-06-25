"""Query understanding: intent, entities, expansion, normalization (F13).

Heuristic and dependency-free (a production system would use spaCy/an LLM here).
The result drives downstream branching — e.g. route questions to dense search,
keyword queries to BM25 — and provides an expanded query for recall.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"\w+")
_WS = re.compile(r"\s+")

_QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would",
}
_COMMAND_VERBS = {
    "find", "show", "list", "get", "give", "fetch", "search", "compare",
    "summarize", "explain", "describe", "tell",
}
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "my", "your", "it", "this", "that", "with", "at", "by", "be", "as",
}


def detect_intent(query: str) -> str:
    text = query.strip().lower()
    if not text:
        return "empty"
    first = text.split()[0]
    if text.endswith("?") or first in _QUESTION_WORDS:
        return "question"
    if first in _COMMAND_VERBS:
        return "command"
    return "keyword"


def extract_entities(query: str) -> dict[str, list[str]]:
    return {
        "capitalized": re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", query),
        "numbers": re.findall(r"\b\d+(?:\.\d+)?\b", query),
        "dates": re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query),
    }


def normalize(query: str) -> str:
    return _WS.sub(" ", query).strip()


def keywords(query: str) -> list[str]:
    """Content tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(query.lower()) if t not in _STOPWORDS]


def expand(tokens: list[str], synonyms: dict[str, list[str]] | None) -> list[str]:
    if not synonyms:
        return []
    extra: list[str] = []
    for tok in tokens:
        extra.extend(synonyms.get(tok, []))
    return extra


def process_query(
    query: str, synonyms: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    normalized = normalize(query)
    kw = keywords(normalized)
    expansion = expand(kw, synonyms)
    expanded_terms = kw + expansion
    return {
        "raw": query,
        "normalized": normalized,
        "intent": detect_intent(normalized),
        "entities": extract_entities(query),
        "keywords": kw,
        "expansion": expansion,
        # A recall-oriented query string usable by either search backend.
        "expanded_query": " ".join(expanded_terms) if expanded_terms else normalized,
    }
