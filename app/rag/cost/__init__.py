"""Cost optimization & token counting (F24)."""
from app.rag.cost.budget import (
    BudgetExceededError,
    budget_stats,
    record_spend,
    reserve,
    reset_budgets,
    upsert_budget,
)
from app.rag.cost.pricing import count_tokens, estimate_cost
from app.rag.cost.tracker import get_usage, record_usage, reset_tracker, usage_summary
from app.rag.cost.optimizer import cache_savings, model_comparison, recommend_model

__all__ = [
    # pricing
    "count_tokens",
    "estimate_cost",
    # budget
    "BudgetExceededError",
    "budget_stats",
    "record_spend",
    "reserve",
    "reset_budgets",
    "upsert_budget",
    # tracker
    "get_usage",
    "record_usage",
    "reset_tracker",
    "usage_summary",
    # optimizer
    "cache_savings",
    "model_comparison",
    "recommend_model",
]
