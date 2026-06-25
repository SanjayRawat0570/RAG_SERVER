"""Feedback data models (F23)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class FeedbackRecord(BaseModel):
    id:         str  = Field(default_factory=lambda: str(uuid4()))
    query_id:   str
    user_id:    str  = ""
    rating:     int  = Field(..., ge=1, le=5)
    comment:    str  = ""
    query:      str  = ""
    answer:     str  = ""
    sources:    list[str]        = Field(default_factory=list)
    signals:    dict[str, Any]   = Field(default_factory=dict)
    tags:       list[str]        = Field(default_factory=list)
    timestamp:  datetime         = Field(default_factory=lambda: datetime.now(timezone.utc))


class ABVariant(BaseModel):
    name:    str
    ratings: list[float] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.ratings)

    @property
    def avg_rating(self) -> float | None:
        return round(sum(self.ratings) / len(self.ratings), 4) if self.ratings else None


class ABTest(BaseModel):
    id:          str
    name:        str
    variants:    dict[str, ABVariant] = Field(default_factory=dict)
    description: str = ""
    created_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def winner(self) -> str | None:
        avgs = {n: v.avg_rating for n, v in self.variants.items() if v.avg_rating is not None}
        return max(avgs, key=lambda k: avgs[k]) if avgs else None  # type: ignore[arg-type]


class PatternReport(BaseModel):
    analyzed_count:         int
    high_quality_patterns:  list[str] = Field(default_factory=list)
    low_quality_patterns:   list[str] = Field(default_factory=list)
    recommendations:        list[str] = Field(default_factory=list)
    high_avg_confidence:    float | None = None
    low_avg_confidence:     float | None = None
    high_avg_sources:       float | None = None
    low_avg_sources:        float | None = None
    timestamp:              datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
