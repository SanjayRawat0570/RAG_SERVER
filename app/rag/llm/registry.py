"""LLM provider registry (F16).

Providers are looked up by name; the default comes from settings. Register a new
provider (Claude/OpenAI/local Ollama) by adding a factory here — the generate
node and the rest of the pipeline are provider-agnostic.
"""
from __future__ import annotations

from typing import Callable

from app.config import settings
from app.rag.llm.base import LLMProvider
from app.rag.llm.claude_llm import ClaudeLLM
from app.rag.llm.gemini import GeminiLLM
from app.rag.llm.openai_llm import OpenAILLM
from app.rag.llm.stub import ExtractiveStubLLM

_FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "stub":   ExtractiveStubLLM,
    "gemini": GeminiLLM,
    "openai": OpenAILLM,
    "claude": ClaudeLLM,
}
_instances: dict[str, LLMProvider] = {}


def register_provider(name: str, factory: Callable[[], LLMProvider]) -> None:
    _FACTORIES[name] = factory


def get_llm(provider: str | None = None) -> LLMProvider:
    name = provider or settings.llm_provider
    if name not in _FACTORIES:
        raise ValueError(f"Unknown LLM provider {name!r}. Available: {sorted(_FACTORIES)}")
    if name not in _instances:
        _instances[name] = _FACTORIES[name]()
    return _instances[name]
