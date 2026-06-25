"""User preference profile (F22)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SearchPreferences(BaseModel):
    cite_sources:       bool  = True
    min_confidence:     float = 0.0      # 0.0 = no filter; 0.7 = only high-confidence
    preferred_format:   str   = "prose"  # prose | bullets | summary | structured
    max_answer_tokens:  int   = 512
    preferred_provider: str | None = None


class ContentPreferences(BaseModel):
    interests:           list[str] = Field(default_factory=list)   # e.g. ["Finance", "Sales"]
    disinterests:        list[str] = Field(default_factory=list)   # e.g. ["HR"]
    preferred_doc_types: list[str] = Field(default_factory=list)   # e.g. ["report", "pdf"]
    prefer_recent:       bool  = True
    recency_weight:      float = 0.3   # 0.0–1.0 boost for recent docs
    prefer_authoritative: bool = False
    authority_sources:   list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    user_id:          str
    search:           SearchPreferences   = Field(default_factory=SearchPreferences)
    content:          ContentPreferences  = Field(default_factory=ContentPreferences)
    created_at:       datetime = Field(default_factory=_now)
    updated_at:       datetime = Field(default_factory=_now)
    metadata:         dict[str, Any] = Field(default_factory=dict)


class QueryRecord(BaseModel):
    """One entry in the per-user interaction history."""
    query:       str
    timestamp:   datetime = Field(default_factory=_now)
    doc_ids:     list[str] = Field(default_factory=list)   # docs clicked/used
    rating:      int | None = None                          # 1-5 stars if given
    provider:    str = ""
    topics:      list[str] = Field(default_factory=list)   # detected topics
