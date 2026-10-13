"""LLM provider interface and response model (F16).

A provider turns an augmented request (chat ``messages`` plus the retrieved
``documents``/``query`` for offline providers) into an answer. Resilience
(retry, fallback, circuit breaker) is supplied by the executor around the node,
so providers just perform the call and raise on failure.
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    text: str
    provider: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, Any] = Field(default_factory=dict)
    # Citation markers (e.g. ["[1]"]) the answer relies on, when known.
    citations: list[str] = Field(default_factory=list)
    # Tool calling: functions the model asked to call ({"name", "args"}), plus the
    # provider's raw content parts so the turn can be replayed in the transcript.
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    parts: list[dict[str, Any]] = Field(default_factory=list)


class LLMProvider(Protocol):
    name: str

    async def generate(self, request: dict[str, Any], config: dict[str, Any]) -> LLMResponse:
        ...
