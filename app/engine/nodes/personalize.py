"""Personalize node — re-rank hits with the user's profile (F22 + long-term memory).

Interests are the user's explicit ones plus those inferred from the topics of
sources cited in their past answers (conversation long-term memory). Passes
hits through unchanged for anonymous users or when there is nothing to apply.

Config::

    {"user_id": "$.inputs.user_id"}
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register


@register
class PersonalizeNode(Node):
    type = "personalize"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        from app.rag.memory.service import ANONYMOUS, effective_profile
        from app.rag.personalization.ranker import personalize_hits

        hits = _single_upstream(upstream) if upstream else []
        user_id = ctx.resolve(self.config.get("user_id", "$.inputs.user_id"))
        if not isinstance(hits, list) or not hits or not user_id or user_id == ANONYMOUS:
            return hits
        profile = effective_profile(str(user_id))
        if not (profile.content.interests or profile.content.disinterests):
            return hits
        return personalize_hits(hits, profile)
