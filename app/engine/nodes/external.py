"""External service node — calls an external HTTP API (F1 node type / F7).

Resilience (retry, fallback, circuit breaker) is applied by the executor around
every node, so this node only has to perform the call and raise on failure.

Config::

    {
      "method": "GET",
      "url": "https://api.example.com/search",
      "params": {"q": "$.inputs.query"},
      "json": {...},
      "headers": {...},
      "timeout": 10,
      "breaker_key": "search-api"          # opt-in circuit breaker (F7)
    }
"""
from __future__ import annotations

from typing import Any

import httpx

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.registry import register


def _resolve_map(ctx: ExecutionContext, value: Any) -> Any:
    """Recursively resolve ``$.`` references inside dicts/lists/strings."""
    if isinstance(value, dict):
        return {k: _resolve_map(ctx, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_map(ctx, v) for v in value]
    return ctx.resolve(value)


@register
class ExternalServiceNode(Node):
    type = "external"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        method = self.config.get("method", "GET").upper()
        url = ctx.resolve(self.config["url"])
        timeout = self.config.get("timeout", 10)
        request_kwargs: dict[str, Any] = {}
        for key in ("params", "json", "headers"):
            if key in self.config:
                request_kwargs[key] = _resolve_map(ctx, self.config[key])

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, **request_kwargs)
            response.raise_for_status()
            ctype = response.headers.get("content-type", "")
            return response.json() if "application/json" in ctype else response.text
