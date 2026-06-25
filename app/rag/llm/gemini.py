"""Google Gemini provider (F16) — free-tier generative API via REST.

Used only when a key is configured (``GEMINI_API_KEY``). No SDK dependency: a
single ``httpx`` POST to the Generative Language API. Chat ``messages`` are
mapped to Gemini's ``system_instruction`` + ``contents`` shape.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.config import settings
from app.rag.llm.base import LLMResponse

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_STREAM_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"


class GeminiLLM:
    name = "gemini"

    async def generate(self, request: dict[str, Any], config: dict[str, Any]) -> LLMResponse:
        api_key = config.get("api_key") or settings.gemini_api_key
        if not api_key:
            raise RuntimeError(
                "Gemini provider requires GEMINI_API_KEY (set it or use provider 'stub')"
            )
        model = config.get("model", settings.gemini_model)
        messages = request.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": config.get("temperature", 0.2),
                "maxOutputTokens": config.get("max_tokens", 512),
            },
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        url = _ENDPOINT.format(model=model)
        async with httpx.AsyncClient(timeout=config.get("timeout", settings.llm_timeout)) as client:
            resp = await client.post(url, params={"key": api_key}, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidate = (data.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            finish_reason=candidate.get("finishReason", "stop"),
            usage={
                "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
                "total_tokens": usage.get("totalTokenCount"),
            },
        )

    async def generate_stream(
        self, request: dict[str, Any], config: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Stream tokens from Gemini's streamGenerateContent endpoint (SSE)."""
        api_key = config.get("api_key") or settings.gemini_api_key
        if not api_key:
            # No key — fall back to non-streaming generate
            response = await self.generate(request, config)
            yield response.text
            return

        model = config.get("model", settings.gemini_model)
        messages = request.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": config.get("temperature", 0.2),
                "maxOutputTokens": config.get("max_tokens", 512),
            },
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        url = _STREAM_ENDPOINT.format(model=model)
        try:
            async with httpx.AsyncClient(
                timeout=config.get("timeout", settings.llm_timeout)
            ) as client:
                async with client.stream(
                    "POST", url,
                    params={"key": api_key, "alt": "sse"},
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            chunk = json.loads(line[6:])
                            parts = (
                                (chunk.get("candidates") or [{}])[0]
                                .get("content", {})
                                .get("parts", [])
                            )
                            for part in parts:
                                if text := part.get("text"):
                                    yield text
                        except (json.JSONDecodeError, IndexError):
                            continue
        except Exception:
            # Network/API error — yield the full response as one chunk
            try:
                response = await self.generate(request, config)
                yield response.text
            except Exception:
                yield ""
