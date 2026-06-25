"""Condition evaluation for graph branching (F2).

A condition is a JSON object. Two shapes are supported:

Boolean combinators::

    {"and": [<cond>, <cond>, ...]}
    {"or":  [<cond>, <cond>, ...]}
    {"not": <cond>}

Comparison (leaf)::

    {"left": "$.classify.confidence", "op": ">", "right": 0.8}

``left`` and ``right`` are resolved against the run context, so either side may
reference node outputs/inputs or be a literal.

Supported ops:
    numeric/equality : >  <  >=  <=  ==  !=
    string           : contains  startswith  endswith  regex
    array            : length_eq  length_gt  length_lt  contains_item  in
"""
from __future__ import annotations

import re
from typing import Any

from app.engine.context import ExecutionContext


def evaluate(condition: dict[str, Any] | None, ctx: ExecutionContext) -> bool:
    """Evaluate a condition tree to a boolean. ``None`` means unconditional True."""
    if condition is None:
        return True
    if not isinstance(condition, dict):
        raise ValueError(f"Condition must be an object, got {type(condition).__name__}")

    if "and" in condition:
        return all(evaluate(c, ctx) for c in condition["and"])
    if "or" in condition:
        return any(evaluate(c, ctx) for c in condition["or"])
    if "not" in condition:
        return not evaluate(condition["not"], ctx)

    if "op" not in condition:
        raise ValueError(f"Leaf condition missing 'op': {condition}")

    left = ctx.resolve(condition.get("left"))
    right = ctx.resolve(condition.get("right"))
    return _apply(str(condition["op"]), left, right)


def _apply(op: str, left: Any, right: Any) -> bool:  # noqa: C901 - explicit dispatch
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op in (">", "<", ">=", "<="):
        if left is None or right is None:
            return False
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        return left <= right

    if op == "contains":
        return right in left if left is not None else False
    if op == "startswith":
        return str(left).startswith(str(right))
    if op == "endswith":
        return str(left).endswith(str(right))
    if op == "regex":
        return re.search(str(right), str(left)) is not None

    if op == "length_eq":
        return _length(left) == right
    if op == "length_gt":
        return _length(left) > right
    if op == "length_lt":
        return _length(left) < right
    if op == "contains_item":
        return right in left if isinstance(left, (list, tuple, set)) else False
    if op == "in":
        return left in right if isinstance(right, (list, tuple, set, str)) else False

    raise ValueError(f"Unknown condition op: {op!r}")


def _length(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0
