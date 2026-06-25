"""Loop node — repeats a sub-workflow until a condition holds (F1 looping).

The DAG itself stays acyclic; iteration is contained inside this node. Each
iteration runs the embedded ``workflow``. The iteration's output is fed into the
next iteration under ``state_key`` (default ``"state"``) alongside the initial
inputs, so the body can read ``$.inputs.state`` to see the running value.

Termination (whichever comes first):
* ``until`` condition evaluates True against the iteration's context, or
* ``max_iterations`` reached (also capped by the global safety limit).

Config::

    {
      "workflow": { ...WorkflowDef... },
      "input_map": {"state": "$.inputs.start"},
      "state_key": "state",
      "until": {"left": "$.inputs.state", "op": ">=", "right": 100},
      "max_iterations": 10
    }
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.engine.conditions import evaluate
from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.external import _resolve_map
from app.engine.nodes.registry import register
from app.models.workflow import WorkflowDef


@register
class LoopNode(Node):
    type = "loop"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        from app.engine.executor import WorkflowExecutor

        child = WorkflowDef.model_validate(self.config["workflow"])
        executor = WorkflowExecutor(child)
        state_key = self.config.get("state_key", "state")
        until = self.config.get("until")
        max_iter = min(
            self.config.get("max_iterations", settings.max_loop_iterations),
            settings.max_loop_iterations,
        )

        # Seed inputs for the first iteration.
        loop_inputs = _resolve_map(ctx, self.config.get("input_map", {}))
        output_key = self.config.get("output_key")
        last_output: Any = None
        iterations = 0

        while iterations < max_iter:
            iterations += 1
            result = await executor.run(loop_inputs)
            if result.status == "error":
                raise RuntimeError(f"loop body '{child.name}' failed on iteration {iterations}")
            last_output = (
                result.outputs.get(output_key) if output_key else result.outputs
            )

            # Evaluate the termination condition against the iteration output.
            iter_ctx = ExecutionContext(run_id=ctx.run_id, inputs={state_key: last_output})
            if until is not None and evaluate(until, iter_ctx):
                break
            # Carry the output forward as the next iteration's state.
            loop_inputs = {**loop_inputs, state_key: last_output}

        return {"iterations": iterations, "output": last_output}
