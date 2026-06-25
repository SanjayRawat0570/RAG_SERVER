"""Offline extractive LLM (F16 default).

Deterministic and dependency-free: scores the retrieved documents by lexical
overlap with the query, then returns the most relevant sentence(s) from the best
document with its citation marker. Not a generative model — it lets the whole
RAG pipeline (and its tests) run end-to-end with no API key, and serves as the
fallback when a real provider is unavailable.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator

from app.rag.llm.base import LLMResponse
from app.rag.search import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveStubLLM:
    name = "stub"

    async def generate(self, request: dict[str, Any], config: dict[str, Any]) -> LLMResponse:
        query = str(request.get("query", ""))
        documents = request.get("documents", []) or []
        qtok = set(tokenize(query))

        best, best_score = None, -1
        for doc in documents:
            overlap = len(qtok & set(tokenize(str(doc.get("text", "")))))
            if overlap > best_score:
                best, best_score = doc, overlap

        if not best or best_score <= 0:
            return LLMResponse(
                text="I don't know based on the provided context.",
                provider=self.name, model="extractive-stub",
                finish_reason="stop", citations=[],
            )

        n = int(config.get("answer_sentences", 2))
        sentences = [s for s in _SENTENCE.split(best["text"]) if s.strip()][:n]
        marker = best.get("marker", "")
        text = " ".join(sentences).strip()
        if marker:
            text = f"{text} {marker}"
        return LLMResponse(
            text=text,
            provider=self.name,
            model="extractive-stub",
            finish_reason="stop",
            usage={"input_tokens": len(qtok), "output_tokens": len(tokenize(text))},
            citations=[marker] if marker else [],
        )

    async def generate_stream(
        self, request: dict[str, Any], config: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Simulate token streaming by yielding one word at a time."""
        response = await self.generate(request, config)
        words = response.text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0)  # yield control to event loop between tokens
