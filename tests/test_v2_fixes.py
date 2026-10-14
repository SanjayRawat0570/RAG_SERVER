"""v2 Step 0 fixes: follow-up rewrite truncation, and fallback answers never cached."""
from __future__ import annotations

import json
import uuid

import pytest

from app.api.pipelines import build_ask_workflow, build_index_workflow
from app.config import settings
from app.engine.executor import WorkflowExecutor
from app.rag.cache import reset_caches
from app.rag.llm import registry as llm_registry
from app.rag.llm.base import LLMResponse
from app.rag.llm.gateway import fast_model
from app.rag.memory.rewriter import rewrite_follow_up
from app.rag.memory.store import Turn


@pytest.fixture
def provider():
    """Register a fake provider under a unique name; yields (name, install)."""
    name = f"fake-{uuid.uuid4().hex[:6]}"

    def install(cls):
        llm_registry.register_provider(name, cls)
        llm_registry._instances.pop(name, None)
        return name
    yield install
    llm_registry._FACTORIES.pop(name, None)
    llm_registry._instances.pop(name, None)


def _turns(q: str) -> list[Turn]:
    return [Turn(question=q, standalone_question=q, answer="The Mobile team is led by Tom Becker.")]


async def test_truncated_llm_rewrite_falls_back_to_heuristic(provider):
    class Truncating:
        name = "truncating"

        async def generate(self, request, config):
            return LLMResponse(text="How much does SKU", provider="t", model="m", finish_reason="MAX_TOKENS")

    out = await rewrite_follow_up("Where is that team based?", _turns("Who leads the Mobile team?"),
                                  provider=provider(Truncating))
    assert out["method"] == "heuristic"
    assert "SKU" not in out["question"] and "Mobile team" in out["question"]


async def test_rewrite_uses_the_requested_model_and_room_to_answer(provider):
    seen: list[dict] = []

    class Recorder:
        name = "recorder"

        async def generate(self, request, config):
            seen.append(config)
            return LLMResponse(text="Where is the Mobile team based?", provider="r", model=config.get("model", "?"))

    out = await rewrite_follow_up("Where is that team based?", _turns("Who leads the Mobile team?"),
                                  provider=provider(Recorder), model="lite-model")
    assert out == {"question": "Where is the Mobile team based?", "rewritten": True, "method": "llm"}
    assert seen[0]["model"] == "lite-model" and seen[0]["max_tokens"] >= 256


def test_gemini_rewrites_default_to_the_fast_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_fast_model", "gemini-flash-lite-latest")
    assert fast_model("gemini") == "gemini-flash-lite-latest"
    assert fast_model("stub") is None
    monkeypatch.setattr(settings, "gemini_fast_model", "")
    assert fast_model("gemini") is None


async def test_fallback_answers_are_not_cached(provider):
    """A provider outage must not keep serving the extractive fallback after recovery."""
    class Flaky:
        name = "flaky"
        down = True

        async def generate(self, request, config):
            if Flaky.down:
                raise RuntimeError("503 Service Unavailable")
            return LLMResponse(text=json.dumps({"answer": "Full-time employees get 25 days [1].",
                                                "citations": [1], "confidence": "high", "answerable": True}),
                               provider="flaky", model="m")

    name = provider(Flaky)
    tenant = f"cache-{uuid.uuid4().hex[:6]}"
    reset_caches("llm-response")
    await WorkflowExecutor(build_index_workflow()).run(
        {"tenant": tenant, "text": "# Leave\n\nFull-time employees get 25 days of leave.", "filename": "l.md"})

    async def ask():
        result = await WorkflowExecutor(build_ask_workflow(name)).run(
            {"tenant": tenant, "question": "How many leave days do full-time employees get?"})
        return result.outputs["out"]

    first = await ask()                       # provider down → extractive fallback
    assert first["provider"] == "extractive-fallback"
    Flaky.down = False                        # provider recovers
    second = await ask()
    assert second["provider"] == "flaky" and second["cache_hit"] is False
    third = await ask()
    assert third["cache_hit"] is True and third["provider"] == "flaky"
