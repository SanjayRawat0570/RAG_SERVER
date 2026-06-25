"""Anthropic Claude provider (F16) — Claude 4.x via Messages API.

Raises RuntimeError at call time when the API key is absent.
Supports haiku (fast/cheap), sonnet (balanced), opus (best quality).
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.rag.llm.base import LLMResponse

_ENDPOINT       = "https://api.anthropic.com/v1/messages"
_API_VERSION    = "2023-06-01"
_DEFAULT_MODEL  = "claude-haiku-4-5-20251001"

# Map short tier names to current model IDs.
_TIER_MAP: dict[str, str] = {
    "fast":     "claude-haiku-4-5-20251001",
    "balanced": "claude-sonnet-4-6",
    "best":     "claude-opus-4-8",
    "haiku":    "claude-haiku-4-5-20251001",
    "sonnet":   "claude-sonnet-4-6",
    "opus":     "claude-opus-4-8",
}


def _resolve_model(name: str) -> str:
    return _TIER_MAP.get(name, name)


class ClaudeLLM:
    name = "claude"

    def _key(self, config: dict[str, Any]) -> str:
        key = config.get("api_key") or getattr(settings, "anthropic_api_key", "")
        if not key:
            raise RuntimeError(
                "Claude provider requires ANTHROPIC_API_KEY "
                "(set it in .env or use provider 'stub' for offline mode)"
            )
        return key

    def _payload(self, request: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        model    = _resolve_model(config.get("model", _DEFAULT_MODEL))
        messages = request.get("messages", [])
        system   = None
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append({"role": m["role"], "content": m["content"]})
        if not user_msgs:
            user_msgs = [{"role": "user", "content": request.get("query", "")}]

        payload: dict[str, Any] = {
            "model":       model,
            "messages":    user_msgs,
            "max_tokens":  config.get("max_tokens", 512),
        }
        if system:
            payload["system"] = system
        return payload

    async def generate(self, request: dict[str, Any], config: dict[str, Any]) -> LLMResponse:
        key     = self._key(config)
        model   = _resolve_model(config.get("model", _DEFAULT_MODEL))
        payload = self._payload(request, config)

        async with httpx.AsyncClient(timeout=config.get("timeout", 30)) as client:
            resp = await client.post(
                _ENDPOINT,
                headers={
                    "x-api-key":          key,
                    "anthropic-version":  _API_VERSION,
                    "Content-Type":       "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("content", [])
        text    = "".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
        usage   = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            finish_reason=data.get("stop_reason", "stop"),
            usage={
                "input_tokens":  usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens":  (usage.get("input_tokens") or 0)
                                 + (usage.get("output_tokens") or 0),
            },
        )

    async def generate_stream(
        self, request: dict[str, Any], config: dict[str, Any]
    ) -> AsyncIterator[str]:
        key     = self._key(config)
        payload = {**self._payload(request, config), "stream": True}

        try:
            async with httpx.AsyncClient(timeout=config.get("timeout", 30)) as client:
                async with client.stream(
                    "POST", _ENDPOINT,
                    headers={
                        "x-api-key":         key,
                        "anthropic-version": _API_VERSION,
                        "Content-Type":      "application/json",
                    },
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            event = json.loads(line[6:])
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if text := delta.get("text"):
                                    yield text
                        except (json.JSONDecodeError, KeyError):
                            continue
        except Exception:
            response = await self.generate(request, config)
            yield response.text
