"""Per-tenant spend budgets (F24).

A process-wide registry tracks cumulative spend per budget key (typically a
tenant or org). ``reserve`` rejects a call whose estimated cost would breach the
limit *before* it runs; ``record_spend`` books the actual cost after. Raises
:class:`BudgetExceededError`, which the engine turns into a node fallback/error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Budget:
    key: str
    limit: float
    spent: float = 0.0
    requests: int = 0
    rejected: int = 0


_budgets: dict[str, Budget] = {}


def _get(key: str, limit: float) -> Budget:
    budget = _budgets.get(key)
    if budget is None:
        budget = Budget(key=key, limit=limit)
        _budgets[key] = budget
    else:
        budget.limit = limit  # allow limit updates
    return budget


def reserve(key: str, limit: float, estimated_cost: float) -> None:
    """Raise if booking ``estimated_cost`` would exceed the budget limit."""
    budget = _get(key, limit)
    if budget.spent + estimated_cost > budget.limit:
        budget.rejected += 1
        raise BudgetExceededError(
            f"budget '{key}' exceeded: spent={budget.spent:.6f} "
            f"+ est={estimated_cost:.6f} > limit={limit:.6f}"
        )


def record_spend(key: str, limit: float, cost: float) -> Budget:
    budget = _get(key, limit)
    budget.spent = round(budget.spent + cost, 6)
    budget.requests += 1
    return budget


def budget_stats() -> dict[str, Any]:
    return {
        k: {
            "limit": b.limit,
            "spent": b.spent,
            "remaining": round(b.limit - b.spent, 6),
            "requests": b.requests,
            "rejected": b.rejected,
        }
        for k, b in _budgets.items()
    }


def upsert_budget(key: str, limit: float) -> dict[str, Any]:
    """Create or update a budget limit without recording a spend event."""
    b = _get(key, limit)
    return {
        "key":       b.key,
        "limit":     b.limit,
        "spent":     b.spent,
        "remaining": round(b.limit - b.spent, 6),
        "requests":  b.requests,
        "rejected":  b.rejected,
    }


def reset_budgets(key: str | None = None) -> None:
    if key is None:
        _budgets.clear()
    else:
        _budgets.pop(key, None)
