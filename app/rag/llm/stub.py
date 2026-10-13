"""Offline extractive LLM (F16 default).

Deterministic and dependency-free: scores the retrieved documents by lexical
overlap with the query, then returns the most relevant sentence(s) from the best
document with its citation marker. Not a generative model — it lets the whole
RAG pipeline (and its tests) run end-to-end with no API key, and serves as the
fallback when a real provider is unavailable.

When ``config["response_schema"]`` is set (structured output), the reply is the
JSON answer object instead. Confidence and answerability come from how much of
the question's content words the best passage covers.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator

from app.rag.llm.base import LLMResponse
from app.rag.search import tokenize

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MARKER_NUM = re.compile(r"\[(\d+)\]")
_IDK = "I don't know based on the provided context."
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
    "my", "our", "your", "it", "this", "that", "with", "at", "by", "be", "as", "do", "does",
    "did", "what", "which", "who", "whom", "whose", "how", "when", "where", "why", "can",
    "could", "should", "would", "many", "much", "there", "any", "get", "i", "we", "you",
}


def _content(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP}


class ExtractiveStubLLM:
    name = "stub"

    def _best(self, request: dict[str, Any]) -> tuple[dict | None, int, float]:
        """Best document by raw overlap, its overlap, and content-word coverage."""
        query = str(request.get("query", ""))
        qtok = set(tokenize(query))
        qcontent = _content(query)
        best, best_score = None, -1
        for doc in request.get("documents", []) or []:
            overlap = len(qtok & set(tokenize(str(doc.get("text", "")))))
            if overlap > best_score:
                best, best_score = doc, overlap
        coverage = 0.0
        if best is not None and qcontent:
            coverage = len(qcontent & _content(str(best.get("text", "")))) / len(qcontent)
        return best, best_score, coverage

    async def generate(self, request: dict[str, Any], config: dict[str, Any]) -> LLMResponse:
        best, best_score, coverage = self._best(request)
        n_in = len(tokenize(str(request.get("query", ""))))

        if config.get("response_schema"):
            return self._structured(best, best_score, coverage, n_in, config)

        if not best or best_score <= 0:
            return LLMResponse(
                text=_IDK, provider=self.name, model="extractive-stub",
                finish_reason="stop", citations=[],
            )

        text, marker = self._extract(best, config)
        return LLMResponse(
            text=text,
            provider=self.name,
            model="extractive-stub",
            finish_reason="stop",
            usage={"input_tokens": n_in, "output_tokens": len(tokenize(text))},
            citations=[marker] if marker else [],
        )

    def _extract(self, doc: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
        n = int(config.get("answer_sentences", 2))
        sentences = [s for s in _SENTENCE.split(doc["text"]) if s.strip()][:n]
        marker = doc.get("marker", "")
        text = " ".join(sentences).strip()
        return (f"{text} {marker}" if marker else text), marker

    def _structured(self, best, best_score, coverage, n_in, config) -> LLMResponse:
        if not best or best_score <= 0 or coverage <= 0.5:
            payload = {"answer": _IDK, "citations": [], "confidence": "low", "answerable": False}
        else:
            text, marker = self._extract(best, config)
            payload = {
                "answer": text,
                "citations": [int(n) for n in _MARKER_NUM.findall(marker)],
                "confidence": "high" if coverage >= 0.75 else "medium",
                "answerable": True,
            }
        body = json.dumps(payload)
        return LLMResponse(
            text=body, provider=self.name, model="extractive-stub", finish_reason="stop",
            usage={"input_tokens": n_in, "output_tokens": len(tokenize(body))},
            citations=[f"[{c}]" for c in payload["citations"]],
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
