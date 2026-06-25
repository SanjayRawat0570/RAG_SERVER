"""Tenant / isolation data models (F18)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _slug_re(v: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{0,61}[a-z0-9]?", v):
        raise ValueError(f"Invalid slug '{v}': use lowercase letters, digits and hyphens")
    return v


class Organization(BaseModel):
    id:          str
    slug:        str
    name:        str
    owner_id:    str
    created_at:  datetime = Field(default_factory=_now)
    settings:    dict[str, Any] = Field(default_factory=dict)

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, v: str) -> str:
        return _slug_re(v)


class Team(BaseModel):
    id:         str
    org_slug:   str
    slug:       str
    name:       str
    created_at: datetime = Field(default_factory=_now)

    @field_validator("slug")
    @classmethod
    def _validate_slug(cls, v: str) -> str:
        return _slug_re(v)


class Membership(BaseModel):
    user_id:    str
    org_slug:   str
    team_slug:  str | None = None   # None = org-level membership (no specific team)
    role:       str = "member"      # owner | admin | member | viewer
    joined_at:  datetime = Field(default_factory=_now)


class UsageStat(BaseModel):
    """Aggregated usage counters for an org / team / user."""
    org_slug:       str
    team_slug:      str | None = None
    user_id:        str | None = None
    total_queries:  int = 0
    total_tokens:   int = 0
    total_cost_usd: float = 0.0
    cache_hits:     int = 0
    last_active:    datetime | None = None
