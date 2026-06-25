"""Prompt assembly with templates and chain-of-thought (F15).

Produces both a single user-prompt string and a chat ``messages`` list, ready
for the F16 LLM node. Supports a custom ``template`` ({context}/{question}),
optional chain-of-thought, and an answer-format instruction. Prompt size is
adjusted to query complexity (F15 dynamic adjustment) via the caller's budget.
"""
from __future__ import annotations

from typing import Any

_SYSTEM_PROMPTS: dict[str, str] = {
    "default": (
        "You are a precise assistant for an enterprise knowledge base. Answer the "
        "user's question using ONLY the provided context. Cite supporting sources "
        "with their bracket markers, e.g. [1]. If the answer is not contained in the "
        "context, say you don't know."
    ),
    "qa": (
        "You are a concise question-answering assistant. Give a direct answer using "
        "only the provided context. Cite your sources with bracket markers."
    ),
    "summarize": (
        "You are a summarization assistant. Summarize the key points from the "
        "provided context clearly and concisely. Attribute claims to their sources."
    ),
    "extract": (
        "You are a structured data extraction assistant. Extract the requested "
        "information from the provided context and return it in a structured format."
    ),
    "chain_of_thought": (
        "You are a reasoning assistant. Think step-by-step through the provided "
        "context before giving your final answer. Show your reasoning clearly."
    ),
}

DEFAULT_SYSTEM = _SYSTEM_PROMPTS["default"]

_COT = "Think through the relevant context step by step before giving the final answer."


def list_templates() -> dict[str, str]:
    """Return all template names mapped to their system prompt text."""
    return dict(_SYSTEM_PROMPTS)


def build_prompt(
    question: str, context: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = config or {}
    tmpl_name = config.get("template_name", "default")
    system = config.get("system") or _SYSTEM_PROMPTS.get(tmpl_name, DEFAULT_SYSTEM)

    template = config.get("template")
    if template:
        user = template.format(context=context, question=question)
    else:
        sections = [
            "Context:",
            context if context else "(no context retrieved)",
            "",
            f"Question: {question}",
        ]
        if config.get("answer_format"):
            sections.append(f"\nAnswer format: {config['answer_format']}")
        if config.get("chain_of_thought"):
            sections.append(f"\n{_COT}")
        sections.append("\nAnswer:")
        user = "\n".join(sections)

    return {
        "system": system,
        "prompt": user,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
