"""In-memory personalization store (F22).

Holds UserProfiles and per-user QueryHistory.
Production: swap out for Supabase/Redis; API contract unchanged.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from app.rag.personalization.profile import ContentPreferences, QueryRecord, SearchPreferences, UserProfile

_MAX_HISTORY = 200   # per-user query records kept in memory

_profiles: dict[str, UserProfile]          = {}
_history:  dict[str, deque[QueryRecord]]   = {}


# ── Profile CRUD ────────────────────────────────────────────────────────────────

def get_profile(user_id: str) -> UserProfile:
    if user_id not in _profiles:
        _profiles[user_id] = UserProfile(user_id=user_id)
    return _profiles[user_id]


def upsert_profile(user_id: str, **updates: Any) -> UserProfile:
    profile = get_profile(user_id)
    if "search" in updates:
        profile.search = SearchPreferences(**updates["search"])
    if "content" in updates:
        profile.content = ContentPreferences(**updates["content"])
    if "metadata" in updates:
        profile.metadata.update(updates["metadata"])
    profile.updated_at = datetime.now(timezone.utc)
    return profile


def reset_profile(user_id: str) -> None:
    _profiles[user_id] = UserProfile(user_id=user_id)
    _history.pop(user_id, None)


# ── Query history ────────────────────────────────────────────────────────────────

def record_query(user_id: str, record: QueryRecord) -> None:
    if user_id not in _history:
        _history[user_id] = deque(maxlen=_MAX_HISTORY)
    _history[user_id].appendleft(record)


def get_history(user_id: str, limit: int = 50) -> list[QueryRecord]:
    dq = _history.get(user_id, deque())
    return list(dq)[:limit]


# ── Interest inference ───────────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Finance":    ["revenue", "profit", "loss", "income", "earnings", "budget",
                   "financial", "fiscal", "accounting", "balance sheet", "cash flow"],
    "Sales":      ["sales", "deal", "pipeline", "quota", "crm", "customer",
                   "leads", "conversion", "closed won"],
    "HR":         ["employee", "headcount", "hiring", "onboarding", "payroll",
                   "benefits", "performance review", "retention"],
    "Technology": ["api", "software", "code", "deploy", "infrastructure",
                   "cloud", "database", "architecture", "engineering"],
    "Marketing":  ["campaign", "brand", "advertising", "seo", "content",
                   "social media", "conversion rate", "funnel"],
    "Legal":      ["contract", "compliance", "regulation", "gdpr", "policy",
                   "terms", "liability", "agreement", "intellectual property"],
    "Operations": ["supply chain", "logistics", "manufacturing", "inventory",
                   "process", "efficiency", "workflow", "capacity"],
}


def detect_topics(text: str) -> list[str]:
    tl = text.lower()
    return [topic for topic, kws in _TOPIC_KEYWORDS.items()
            if any(kw in tl for kw in kws)]


def infer_interests(user_id: str, top_n: int = 3) -> list[dict[str, Any]]:
    """Count topic hits in recent history and return ranked interests."""
    history = get_history(user_id, limit=_MAX_HISTORY)
    counts: dict[str, int] = {}
    for record in history:
        for topic in record.topics:
            counts[topic] = counts.get(topic, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"topic": t, "count": c} for t, c in ranked[:top_n]]


# ── Registry reset ───────────────────────────────────────────────────────────────

def reset_store() -> None:
    _profiles.clear()
    _history.clear()
