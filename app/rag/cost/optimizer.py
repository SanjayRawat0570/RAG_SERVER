"""Model selection and cost-saving analysis (F24)."""
from __future__ import annotations

from typing import Any

from app.rag.cost.pricing import PRICING, estimate_cost

# (model, quality_tier) — higher tier = better quality, usually higher cost.
_MODEL_QUALITY: list[tuple[str, int]] = [
    ("extractive-stub",   0),
    ("gemini-2.5-flash",  1),
    ("gemini-2.0-flash",  1),
    ("gemini-1.5-flash",  1),
    ("gpt-4o-mini",       2),
    ("gemini-1.5-pro",    3),
    ("gemini-2.5-pro",    3),
    ("claude-3-5-sonnet", 4),
    ("gpt-4o",            5),
]


def recommend_model(
    budget_remaining:  float,
    avg_input_tokens:  int  = 1000,
    avg_output_tokens: int  = 200,
    prefer_quality:    bool = True,
) -> dict[str, Any]:
    """Return the best (or cheapest) model whose per-call cost fits the budget."""
    affordable: list[tuple[str, int, float]] = []
    for model, quality in _MODEL_QUALITY:
        cost = estimate_cost(model, avg_input_tokens, avg_output_tokens)
        if cost <= budget_remaining:
            affordable.append((model, quality, cost))

    if not affordable:
        return {"model": "extractive-stub", "estimated_cost": 0.0, "quality_tier": 0}

    if prefer_quality:
        best = max(affordable, key=lambda x: (x[1], -x[2]))
    else:
        # cheapest; among ties prefer lowest quality (most conservative)
        best = min(affordable, key=lambda x: (x[2], x[1]))

    return {"model": best[0], "estimated_cost": best[2], "quality_tier": best[1]}


def model_comparison(
    models:        list[str],
    input_tokens:  int = 1000,
    output_tokens: int = 200,
) -> list[dict[str, Any]]:
    """Compare per-call cost across models, sorted cheapest first."""
    results = [
        {
            "model":         m,
            "cost":          estimate_cost(m, input_tokens, output_tokens),
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        }
        for m in models
    ]
    return sorted(results, key=lambda x: x["cost"])


def cache_savings(
    total_queries:     int,
    cache_hit_rate:    float,
    model:             str,
    avg_input_tokens:  int = 500,
    avg_output_tokens: int = 200,
) -> dict[str, Any]:
    """Estimate cost savings from a cache with the given hit rate."""
    cost_per_query  = estimate_cost(model, avg_input_tokens, avg_output_tokens)
    original_cost   = cost_per_query * total_queries
    new_queries     = round(total_queries * (1.0 - cache_hit_rate))
    cached_queries  = total_queries - new_queries
    with_cache_cost = cost_per_query * new_queries
    savings         = original_cost - with_cache_cost

    return {
        "model":           model,
        "total_queries":   total_queries,
        "cache_hit_rate":  cache_hit_rate,
        "original_cost":   round(original_cost,   6),
        "with_cache_cost": round(with_cache_cost, 6),
        "savings":         round(savings,          6),
        "savings_pct":     round(cache_hit_rate * 100, 1),
        "new_queries":     new_queries,
        "cached_queries":  cached_queries,
    }
