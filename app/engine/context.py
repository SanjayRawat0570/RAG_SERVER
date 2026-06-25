"""Execution context — state passed between nodes during a run (F1/F3).

Every node's output is stored under its id. Nodes read upstream outputs and
the initial inputs through a single resolver, which also powers condition
evaluation (F2) via dotted reference strings like ``$.node_id.field``.
"""
from __future__ import annotations

from typing import Any

_MISSING = object()


class ExecutionContext:
    def __init__(self, run_id: str, inputs: dict[str, Any] | None = None) -> None:
        self.run_id = run_id
        self.inputs: dict[str, Any] = inputs or {}
        # node_id -> output value
        self.outputs: dict[str, Any] = {}
        # node_id -> status (success | error | skipped | fallback)
        self.status: dict[str, str] = {}

    def set_output(self, node_id: str, value: Any, status: str = "success") -> None:
        self.outputs[node_id] = value
        self.status[node_id] = status

    def resolve(self, ref: Any) -> Any:
        """Resolve a reference.

        - ``"$.inputs.foo"``    -> initial input ``foo``
        - ``"$.node_id.bar"``   -> field ``bar`` of node ``node_id`` output
        - ``"$.node_id"``       -> full output of ``node_id``
        - anything else is treated as a literal value.
        """
        if not isinstance(ref, str) or not ref.startswith("$."):
            return ref

        parts = ref[2:].split(".")
        root = parts[0]
        current: Any
        if root == "inputs":
            current = self.inputs
            parts = parts[1:]
        else:
            current = self.outputs.get(root, _MISSING)
            parts = parts[1:]

        for p in parts:
            if current is _MISSING or current is None:
                return None
            if isinstance(current, dict):
                current = current.get(p, _MISSING)
            elif isinstance(current, list) and p.lstrip("-").isdigit():
                idx = int(p)
                current = current[idx] if -len(current) <= idx < len(current) else _MISSING
            else:
                current = getattr(current, p, _MISSING)

        return None if current is _MISSING else current
