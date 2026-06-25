"""Personalization module (F22)."""
from app.rag.personalization.profile import (
    ContentPreferences, QueryRecord, SearchPreferences, UserProfile,
)
from app.rag.personalization.store import (
    detect_topics, get_history, get_profile, infer_interests,
    record_query, reset_profile, reset_store, upsert_profile,
)
from app.rag.personalization.ranker import personalize_hits, recommend_documents

__all__ = [
    "UserProfile", "SearchPreferences", "ContentPreferences", "QueryRecord",
    "get_profile", "upsert_profile", "reset_profile", "reset_store",
    "record_query", "get_history", "detect_topics", "infer_interests",
    "personalize_hits", "recommend_documents",
]
