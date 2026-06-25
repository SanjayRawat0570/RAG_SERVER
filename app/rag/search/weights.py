"""Hybrid search weight profiles and query-type classifier (F20).

Weight convention: (semantic_alpha, keyword_alpha) where both sum to 1.0.
These are fed into hybrid_search as (dense_weight, sparse_weight) after
normalisation so that dense_weight + sparse_weight == 1.0.

Query classification uses lightweight heuristics (no ML model needed):
- Technical:   code syntax, version numbers, exact identifiers
- Conceptual:  abstract/domain terminology, reasoning words
- Factual:     who/what/when/where interrogatives
- Analytical:  compare/analyse/trend verbs
- General:     everything else → balanced
"""
from __future__ import annotations

import re
from typing import NamedTuple


class WeightProfile(NamedTuple):
    name:           str
    semantic_alpha: float    # dense weight (0–1)
    keyword_alpha:  float    # sparse weight (0–1)
    description:    str


# ── Named profiles ─────────────────────────────────────────────────────────────

PROFILES: dict[str, WeightProfile] = {
    "balanced": WeightProfile(
        "balanced", 0.6, 0.4,
        "60% semantic + 40% keyword — good default for mixed queries",
    ),
    "semantic": WeightProfile(
        "semantic", 0.8, 0.2,
        "80% semantic + 20% keyword — meaning matters more than exact words",
    ),
    "keyword": WeightProfile(
        "keyword", 0.3, 0.7,
        "30% semantic + 70% keyword — exact terminology is critical",
    ),
    "technical": WeightProfile(
        "technical", 0.3, 0.7,
        "30% semantic + 70% keyword — code / identifiers / version numbers",
    ),
    "conceptual": WeightProfile(
        "conceptual", 0.8, 0.2,
        "80% semantic + 20% keyword — abstract or high-level reasoning",
    ),
    "equal": WeightProfile(
        "equal", 0.5, 0.5,
        "50/50 split — neutral baseline",
    ),
}

DEFAULT_PROFILE = "balanced"


# ── Query-type classifier ───────────────────────────────────────────────────────

_TECHNICAL_SIGNALS = re.compile(
    r"\b(def |class |import |return |function|error|exception|bug|fix|null|"
    r"api|sdk|cli|regex|json|xml|sql|http|url|uri|v\d+\.\d+|"
    r"[a-z_]+\(\)|[A-Z][A-Z_]{2,}|[a-z]+_[a-z_]+)\b"
)

_CONCEPTUAL_SIGNALS = re.compile(
    r"\b(concept|theory|meaning|philosophy|principle|framework|paradigm|"
    r"architecture|strategy|approach|overview|introduction|fundamentals|"
    r"explain|describe|understand|insight|perspective|impact|effect)\b",
    re.IGNORECASE,
)

_FACTUAL_SIGNALS = re.compile(
    r"\b(who|what|when|where|which|how many|how much|did|does|is|are|was|"
    r"were|name|list|show|tell|give)\b",
    re.IGNORECASE,
)

_ANALYTICAL_SIGNALS = re.compile(
    r"\b(compare|contrast|analyse|analyze|trend|difference|versus|vs\.?|"
    r"better|worse|tradeoff|pros|cons|advantages|disadvantages|evaluate|"
    r"assess|rank|benchmark)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """Return one of: technical | conceptual | factual | analytical | general."""
    tech       = len(_TECHNICAL_SIGNALS.findall(query))
    conceptual = len(_CONCEPTUAL_SIGNALS.findall(query))
    factual    = len(_FACTUAL_SIGNALS.findall(query))
    analytical = len(_ANALYTICAL_SIGNALS.findall(query))

    scores = {
        "technical":  tech * 3,        # strong signal if present
        "conceptual": conceptual * 2,
        "analytical": analytical * 2,
        "factual":    factual,
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "general"
    return best


_CLASS_TO_PROFILE: dict[str, str] = {
    "technical":  "technical",
    "conceptual": "conceptual",
    "analytical": "semantic",
    "factual":    "balanced",
    "general":    "balanced",
}


def auto_weights(query: str) -> tuple[WeightProfile, str]:
    """Return (profile, query_type) automatically chosen for *query*."""
    qtype   = classify_query(query)
    profile = PROFILES[_CLASS_TO_PROFILE[qtype]]
    return profile, qtype


def get_profile(name: str) -> WeightProfile:
    """Return a named profile, falling back to DEFAULT_PROFILE if unknown."""
    return PROFILES.get(name, PROFILES[DEFAULT_PROFILE])


def normalize_to_unit(semantic: float, keyword: float) -> tuple[float, float]:
    """Scale arbitrary (semantic, keyword) values so they sum to 1.0."""
    total = semantic + keyword
    if total == 0:
        return 0.5, 0.5
    return semantic / total, keyword / total
