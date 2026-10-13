"""Agentic retrieval nodes — the agent state machine inside the DAG engine.

One iteration of the agent is a small sub-workflow run by the existing ``loop``
node (the outer DAG stays acyclic)::

    in ──▶ agent_plan ──▶ agent_search ──▶ agent_evaluate ──▶ out
     ▲                                                          │
     └──────────── loop: state carried in $.inputs.state ◀──────┘   until phase == "answer"

Each node reads the running state from its upstream (or ``$.inputs.state`` on
the first node), advances it one transition, and passes it on. The tenant comes
from the loop's server-side ``input_map`` — never from the model.

Config (``agent_plan`` only, used when the state does not exist yet)::

    {"question": "$.inputs.question", "tenant": "$.inputs.tenant",
     "provider": "gemini", "max_rounds": 3, "timeout_s": 20, "token_budget": 6000}

Also here: ``graph_index`` — feeds indexed chunks into the tenant's knowledge
graph so the ``lookup_entity`` tool has something to find.
"""
from __future__ import annotations

import copy
from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.agent import planner


def _state(ctx: ExecutionContext, upstream: dict[str, Any]) -> dict[str, Any] | None:
    value = _single_upstream(upstream) if upstream else None
    if not (isinstance(value, dict) and "phase" in value):
        value = ctx.resolve("$.inputs.state")
    return copy.deepcopy(value) if isinstance(value, dict) and "phase" in value else None


@register
class AgentPlanNode(Node):
    type = "agent_plan"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        state = _state(ctx, upstream)
        if state is None:
            state = planner.new_state(
                question=str(ctx.resolve(self.config.get("question", "$.inputs.question")) or ""),
                tenant=str(ctx.resolve(self.config.get("tenant", "$.inputs.tenant")) or "default"),
                provider=str(ctx.resolve(self.config.get("provider", "$.inputs.provider")) or "stub"),
                model=self.config.get("model"),
                max_rounds=self.config.get("max_rounds"),
                timeout_s=self.config.get("timeout_s"),
                token_budget=self.config.get("token_budget"),
            )
        return await planner.plan(state)


@register
class AgentSearchNode(Node):
    type = "agent_search"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        return await planner.search(_state(ctx, upstream))


@register
class AgentEvaluateNode(Node):
    type = "agent_evaluate"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        return await planner.evaluate(_state(ctx, upstream))


@register
class GraphIndexNode(Node):
    """Extract entities/relations from chunks into the tenant's knowledge graph.

    Config: ``{"tenant": "$.inputs.tenant"}``. Passes its input through unchanged
    so it can sit inline in an indexing pipeline.
    """

    type = "graph_index"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        from app.rag.graph import ingest_document

        items = _single_upstream(upstream) if upstream else []
        tenant = str(ctx.resolve(self.config.get("tenant", "$.inputs.tenant")) or "default")
        for item in items if isinstance(items, list) else [items]:
            if isinstance(item, dict) and item.get("text"):
                ingest_document(str(item["text"]),
                                doc_id=str(item.get("document_id") or item.get("chunk_id") or ""),
                                tenant=tenant)
        return items
