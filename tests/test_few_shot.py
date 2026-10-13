"""Few-shot prompting: default examples, per-tenant configuration, prompt rendering."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.rag.context.prompt import build_prompt, render_examples
from app.rag.tenants import ai_settings
from app.rag.tenants.ai_settings import (
    DEFAULT_EXAMPLES,
    FewShotExample,
    effective_examples,
    get_settings,
    update_settings,
)

AUTH = {"Authorization": "Bearer dev"}


@pytest.fixture(autouse=True)
def _clean_settings():
    ai_settings.reset_settings()
    yield
    ai_settings.reset_settings()


def test_defaults_are_one_answerable_and_one_idk():
    assert len(DEFAULT_EXAMPLES) == 2
    assert [e.answerable for e in DEFAULT_EXAMPLES] == [True, False]
    assert DEFAULT_EXAMPLES[0].citations == [1]
    assert "don't know" in DEFAULT_EXAMPLES[1].answer
    assert effective_examples("any-tenant") == DEFAULT_EXAMPLES


def test_examples_render_into_system_prompt_as_structured_json():
    prompt = build_prompt("Q?", "[1] ctx", {"examples": DEFAULT_EXAMPLES, "example_format": "json"})
    system = prompt["system"]
    assert "Example 1" in system and "Example 2" in system
    assert "[1] (handbook.md) Employees can expense up to 40 USD" in system
    # The answer line is the exact JSON the model must produce.
    answer_line = next(line for line in system.splitlines()
                       if line.startswith("Answer: {") and '"answerable": false' in line)
    assert json.loads(answer_line[len("Answer: "):])["citations"] == []
    # The user turn (context + question) is unchanged by few-shot examples.
    assert "Example" not in prompt["prompt"]


def test_text_format_renders_plain_answers():
    block = render_examples(DEFAULT_EXAMPLES, "text")
    assert "Answer: Employees can expense up to 40 USD per day" in block
    assert '"citations"' not in block


def test_no_examples_leaves_prompt_unchanged():
    assert build_prompt("Q?", "ctx")["system"] == build_prompt("Q?", "ctx", {"examples": []})["system"]


def test_per_tenant_examples_and_toggle():
    custom = [FewShotExample(question="Who approves leave?", context=["(hr.md) Managers approve leave."],
                             answer="Managers approve leave [1].", citations=[1])]
    update_settings("tenant-a", few_shot_examples=custom)
    update_settings("tenant-b", few_shot_enabled=False)

    assert effective_examples("tenant-a") == custom
    assert effective_examples("tenant-b") == []
    assert effective_examples("tenant-c") == DEFAULT_EXAMPLES
    # A per-request override wins over the tenant setting.
    assert effective_examples("tenant-b", override=True) == DEFAULT_EXAMPLES
    assert effective_examples("tenant-a", override=False) == []


def test_settings_persist_and_reload_from_disk():
    update_settings("persisted", few_shot_enabled=False, vision_enabled=True)
    ai_settings._cache = None          # simulate a fresh process (e.g. the MCP server)
    cfg = get_settings("persisted")
    assert cfg.few_shot_enabled is False and cfg.vision_enabled is True


@pytest.mark.asyncio
async def test_ask_workflow_puts_tenant_examples_in_the_prompt():
    from app.api.pipelines import build_ask_workflow
    from app.engine.executor import WorkflowExecutor

    examples = [e.model_dump() for e in effective_examples("fs-wf")]
    result = await WorkflowExecutor(build_ask_workflow("stub")).run(
        {"tenant": "fs-wf", "question": "What is AI?", "few_shot": examples})
    augment = next(r for r in result.results if r.node_id == "augment").output
    assert "Follow these examples" in augment["system"]

    bare = await WorkflowExecutor(build_ask_workflow("stub")).run(
        {"tenant": "fs-wf", "question": "What is AI?", "few_shot": []})
    augment = next(r for r in bare.results if r.node_id == "augment").output
    assert "Follow these examples" not in augment["system"]


def test_settings_api_roundtrip():
    from app.main import app

    with TestClient(app) as client:
        got = client.get("/api/v1/tenant-settings/fs-api", headers=AUTH).json()
        assert got["few_shot_enabled"] is True and got["few_shot_custom"] is False
        assert len(got["few_shot_examples"]) == 2

        custom = {"question": "Q?", "context": ["(a.md) A."], "answer": "A [1].", "citations": [1]}
        put = client.put("/api/v1/tenant-settings/fs-api", headers=AUTH,
                         json={"few_shot_examples": [custom], "vision_enabled": True}).json()
        assert put["few_shot_custom"] is True and put["few_shot_examples"][0]["question"] == "Q?"
        assert put["vision_enabled"] is True

        reset = client.put("/api/v1/tenant-settings/fs-api", headers=AUTH,
                           json={"reset_examples": True}).json()
        assert reset["few_shot_custom"] is False and len(reset["few_shot_examples"]) == 2
        assert reset["vision_enabled"] is True   # untouched fields are kept

        bad = client.put("/api/v1/tenant-settings/fs-api", headers=AUTH,
                         json={"few_shot_examples": [{"question": "", "answer": "x"}]})
        assert bad.status_code == 422
