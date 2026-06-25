"""OpenAI provider (F16) — GPT-4o, GPT-4o-mini, GPT-3.5-turbo via REST.

Raises RuntimeError at call time (not import time) when the API key is absent,
so the rest of the pipeline works offline without any key.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.rag.llm.base import LLMResponse

_ENDPOINT        = "https://api.openai.com/v1/chat/completions"
_DEFAULT_MODEL   = "gpt-4o-mini"


class OpenAILLM:
    name = "openai"

    def _key(self, config: dict[str, Any]) -> str:
        key = config.get("api_key") or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OpenAI provider requires OPENAI_API_KEY "
                "(set it in .env or use provider 'stub' for offline mode)"
            )
        return key

    def _payload(self, request: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        model    = config.get("model", _DEFAULT_MODEL)
        messages = request.get("messages", [])
        if not messages:
            query = request.get("query", "")
            messages = [{"role": "user", "content": query}]
        return {
            "model":       model,
            "messages":    messages,
            "temperature": config.get("temperature", 0.2),
            "max_tokens":  config.get("max_tokens", 512),
        }

    async def generate(self, request: dict[str, Any], config: dict[str, Any]) -> LLMResponse:
        key     = self._key(config)
        model   = config.get("model", _DEFAULT_MODEL)
        payload = self._payload(request, config)

        async with httpx.AsyncClient(timeout=config.get("timeout", 30)) as client:
            resp = await client.post(
                _ENDPOINT,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice  = (data.get("choices") or [{}])[0]
        text    = choice.get("message", {}).get("content", "").strip()
        usage   = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "input_tokens":  usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens":  usage.get("total_tokens"),
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
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        try:
                            chunk = json.loads(line[6:])
                            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                            if text := delta.get("content"):
                                yield text
                        except (json.JSONDecodeError, IndexError):
                            continue
        except Exception:
            response = await self.generate(request, config)
            yield response.text
