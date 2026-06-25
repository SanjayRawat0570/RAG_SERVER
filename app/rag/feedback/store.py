"""In-memory feedback store (F23)."""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Any

from app.rag.feedback.models import ABTest, ABVariant, FeedbackRecord

_MAX_RECORDS = 10_000

_records:  deque[FeedbackRecord] = deque(maxlen=_MAX_RECORDS)
_ab_tests: dict[str, ABTest]     = {}


# ── Feedback records ──────────────────────────────────────────────────────────

def submit_feedback(record: FeedbackRecord) -> FeedbackRecord:
    _records.appendleft(record)
    return record


def get_feedback(
    query_id:   str | None = None,
    user_id:    str | None = None,
    rating_min: int | None = None,
    rating_max: int | None = None,
    limit:      int = 100,
) -> list[FeedbackRecord]:
    results: list[FeedbackRecord] = list(_records)
    if query_id  is not None:
        results = [r for r in results if r.query_id == query_id]
    if user_id   is not None:
        results = [r for r in results if r.user_id  == user_id]
    if rating_min is not None:
        results = [r for r in results if r.rating   >= rating_min]
    if rating_max is not None:
        results = [r for r in results if r.rating   <= rating_max]
    return results[:limit]


def feedback_stats() -> dict[str, Any]:
    all_records = list(_records)
    if not all_records:
        return {
            "count": 0,
            "average_rating": 0.0,
            "distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "high_rated_count": 0,
            "low_rated_count":  0,
        }
    ratings = [r.rating for r in all_records]
    dist    = {i: 0 for i in range(1, 6)}
    for r in ratings:
        dist[r] = dist.get(r, 0) + 1
    return {
        "count":           len(all_records),
        "average_rating":  round(sum(ratings) / len(ratings), 4),
        "distribution":    dist,
        "high_rated_count": sum(1 for r in ratings if r >= 4),
        "low_rated_count":  sum(1 for r in ratings if r <= 2),
    }


# ── A/B tests ─────────────────────────────────────────────────────────────────

def create_ab_test(name: str, variants: list[str], description: str = "") -> ABTest:
    if name in _ab_tests:
        raise ValueError(f"A/B test '{name}' already exists")
    test = ABTest(
        id=name,
        name=name,
        variants={v: ABVariant(name=v) for v in variants},
        description=description,
    )
    _ab_tests[name] = test
    return test


def list_ab_tests() -> list[ABTest]:
    return list(_ab_tests.values())


def get_ab_test(test_id: str) -> ABTest | None:
    return _ab_tests.get(test_id)


def assign_variant(test_id: str, user_id: str) -> str:
    test = _ab_tests.get(test_id)
    if test is None:
        raise KeyError(f"A/B test '{test_id}' not found")
    variants = sorted(test.variants.keys())
    key = f"{test_id}:{user_id}".encode()
    idx = int(hashlib.md5(key).hexdigest(), 16) % len(variants)
    return variants[idx]


def record_ab_result(test_id: str, variant: str, rating: float) -> None:
    test = _ab_tests.get(test_id)
    if test is None:
        raise KeyError(f"A/B test '{test_id}' not found")
    if variant not in test.variants:
        raise ValueError(f"Variant '{variant}' not in test '{test_id}'")
    test.variants[variant].ratings.append(rating)


def get_ab_results(test_id: str) -> dict[str, Any]:
    test = _ab_tests.get(test_id)
    if test is None:
        raise KeyError(f"A/B test '{test_id}' not found")
    results: dict[str, Any] = {
        name: {"count": v.count, "avg_rating": v.avg_rating}
        for name, v in test.variants.items()
    }
    results["winner"] = test.winner
    return results


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_feedback() -> None:
    _records.clear()
    _ab_tests.clear()
