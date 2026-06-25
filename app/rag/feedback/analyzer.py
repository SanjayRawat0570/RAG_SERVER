"""Pattern analysis and improvement insights (F23)."""
from __future__ import annotations

from typing import Any

from app.rag.feedback.models import PatternReport


def _avg(records: list, key: str) -> float | None:
    vals = [r.signals[key] for r in records if key in r.signals]
    return round(sum(vals) / len(vals), 3) if vals else None


def analyze_patterns() -> PatternReport:
    from app.rag.feedback.store import get_feedback

    all_fb = get_feedback(limit=10_000)
    high   = [f for f in all_fb if f.rating >= 4]
    low    = [f for f in all_fb if f.rating <= 2]

    high_patterns: list[str] = []
    low_patterns:  list[str] = []
    recommendations: list[str] = []

    high_conf = _avg(high, "confidence")
    low_conf  = _avg(low,  "confidence")
    high_src  = _avg(high, "source_count")
    low_src   = _avg(low,  "source_count")

    if high:
        if high_conf is not None and high_conf > 0.80:
            high_patterns.append(
                f"High-rated answers average {high_conf:.0%} confidence"
            )
        if high_src is not None and high_src > 1.5:
            high_patterns.append(
                f"High-rated answers cite {high_src:.1f} sources on average"
            )
        nums_count = sum(1 for f in high if f.signals.get("has_numbers"))
        if nums_count > len(high) * 0.5:
            high_patterns.append(
                "High-rated answers frequently contain specific numbers"
            )
        if not high_patterns:
            high_patterns.append(f"{len(high)} high-rated answers recorded")

    if low:
        if low_conf is not None and low_conf < 0.65:
            low_patterns.append(
                f"Low-rated answers average only {low_conf:.0%} confidence"
            )
        if low_src is not None and low_src < 1.5:
            low_patterns.append(
                f"Low-rated answers cite fewer sources ({low_src:.1f} avg)"
            )
        if not low_patterns:
            low_patterns.append(f"{len(low)} low-rated answers recorded")

    if low_conf is not None and low_conf < 0.65:
        recommendations.append(
            "Increase minimum confidence threshold to filter low-confidence answers"
        )
    if low_src is not None and low_src < 2:
        recommendations.append(
            "Require at least 2 sources to improve answer credibility"
        )
    if not recommendations and (high or low):
        recommendations.append(
            "Continue collecting feedback to identify improvement patterns"
        )

    return PatternReport(
        analyzed_count=len(all_fb),
        high_quality_patterns=high_patterns,
        low_quality_patterns=low_patterns,
        recommendations=recommendations,
        high_avg_confidence=high_conf,
        low_avg_confidence=low_conf,
        high_avg_sources=high_src,
        low_avg_sources=low_src,
    )


def get_insights() -> dict[str, Any]:
    from app.rag.feedback.store import feedback_stats

    stats  = feedback_stats()
    report = analyze_patterns()
    total  = max(stats["count"], 1)

    return {
        "summary": {
            "total_feedback":   stats["count"],
            "average_rating":   stats["average_rating"],
            "high_rated_pct":   round(stats["high_rated_count"] / total * 100, 1),
            "low_rated_pct":    round(stats["low_rated_count"]  / total * 100, 1),
        },
        "patterns": {
            "high_quality": report.high_quality_patterns,
            "low_quality":  report.low_quality_patterns,
        },
        "recommendations":  report.recommendations,
        "last_analyzed_at": report.timestamp.isoformat(),
    }
