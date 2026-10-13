"""Structured output — the model answers in validated JSON.

Contract::

    {"answer": str, "citations": [int], "confidence": "high"|"medium"|"low", "answerable": bool}

Flow (:func:`generate_structured`):

1. Ask the provider with its native JSON/schema mode (Gemini ``responseSchema``)
   plus explicit format instructions in the system prompt.
2. Validate with the :class:`StructuredAnswer` Pydantic model.
3. On invalid JSON / schema errors, retry **once**, feeding the validation error
   back to the model.
4. If that also fails (or the provider is unavailable), fall back to the offline
   extractive answer, which is always valid.

Citation numbers are clamped to the sources actually supplied, so a hallucinated
``[7]`` with three sources is dropped rather than shown.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.rag.cost import BudgetExceededError
from app.rag.llm.gateway import call_llm

Confidence = Literal["high", "medium", "low"]
IDK_ANSWER = "I don't know based on the provided context."


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    citations: list[int] = Field(default_factory=list)
    confidence: Confidence
    answerable: bool


# Gemini responseSchema (OpenAPI subset). ``answer`` is ordered first so it can be
# streamed to the client before the rest of the object has been generated.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "citations": {"type": "ARRAY", "items": {"type": "INTEGER"}},
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
        "answerable": {"type": "BOOLEAN"},
    },
    "required": ["answer", "citations", "confidence", "answerable"],
    "propertyOrdering": ["answer", "citations", "confidence", "answerable"],
}

STRUCTURED_INSTRUCTIONS = (
    "Respond with a single JSON object and nothing else, in this exact shape:\n"
    '{"answer": "...", "citations": [1], "confidence": "high", "answerable": true}\n'
    "- answer: the answer, with inline bracket markers such as [1] for each supporting source.\n"
    "- citations: the numbers of the context sources the answer relies on.\n"
    '- confidence: "high", "medium" or "low" — how directly the context supports the answer.\n'
    f'- answerable: false when the context does not contain the answer; then answer "{IDK_ANSWER}" '
    "with citations []."
)

RETRY_PROMPT = (
    "Your previous reply was rejected: {error}\n"
    "Reply again with ONLY a valid JSON object matching the required shape."
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class StructuredOutputError(ValueError):
    """The model's reply was not valid JSON for :class:`StructuredAnswer`."""


def parse_structured(text: str) -> StructuredAnswer:
    raw = _FENCE.sub("", text or "").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"invalid JSON ({exc.msg} at position {exc.pos})") from exc
    try:
        return StructuredAnswer.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'root'}: {e['msg']}" for e in exc.errors()
        )
        raise StructuredOutputError(f"schema validation failed: {problems}") from exc


def clamp_citations(answer: StructuredAnswer, n_sources: int) -> StructuredAnswer:
    valid = list(dict.fromkeys(c for c in answer.citations if 1 <= c <= n_sources))
    return answer.model_copy(update={"citations": valid})


def with_instructions(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append the JSON contract to the system message (adding one if absent)."""
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = f"{m['content']}\n\n{STRUCTURED_INSTRUCTIONS}"
            return out
    return [{"role": "system", "content": STRUCTURED_INSTRUCTIONS}, *out]


def structured_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, "response_schema": ANSWER_SCHEMA}


async def extractive_answer(request: dict[str, Any]) -> StructuredAnswer:
    """The offline fallback: the extractive stub's answer, always schema-valid."""
    from app.rag.llm.stub import ExtractiveStubLLM

    response = await ExtractiveStubLLM().generate(request, {"response_schema": ANSWER_SCHEMA})
    return parse_structured(response.text)


def _result(answer: StructuredAnswer, *, n_sources: int, attempts: int, fallback: bool,
            errors: list[str], response=None) -> dict[str, Any]:
    answer = clamp_citations(answer, n_sources)
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "cited_markers": [f"[{c}]" for c in answer.citations],
        "confidence": answer.confidence,
        "answerable": answer.answerable,
        "structured": {"attempts": attempts, "fallback": fallback, "errors": errors},
        "response": response,
    }


async def generate_structured(
    provider: str,
    request: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    tenant: str | None = None,
    user_id: str = "",
    budget_limit: float | None = None,
    first_reply: str | None = None,
) -> dict[str, Any]:
    """Generate → validate → retry once with the error → extractive fallback.

    ``first_reply`` lets a caller that already streamed attempt 1 hand over the
    raw text, so only the repair attempt is spent here.
    """
    cfg = structured_config(config or {})
    messages = with_instructions(request.get("messages", []))
    req = {**request, "messages": messages}
    n_sources = len(request.get("documents") or request.get("citations") or [])
    errors: list[str] = []
    attempts = 0
    last_response = None
    reply = first_reply

    for attempt in (1, 2):
        if reply is None:
            attempts += 1
            try:
                last_response = await call_llm(
                    provider, req, cfg, tenant=tenant, user_id=user_id, budget_limit=budget_limit,
                    operation="generate" if attempt == 1 else "generate_retry",
                )
            except BudgetExceededError:
                raise
            except Exception as exc:  # provider down / unconfigured / circuit open
                errors.append(f"provider error: {exc}")
                break
            reply = last_response.text
        else:
            attempts += 1
        try:
            answer = parse_structured(reply)
            return _result(answer, n_sources=n_sources, attempts=attempts, fallback=False,
                           errors=errors, response=last_response)
        except StructuredOutputError as exc:
            errors.append(str(exc))
            req = {**req, "messages": [
                *messages,
                {"role": "assistant", "content": reply[:4000]},
                {"role": "user", "content": RETRY_PROMPT.format(error=exc)},
            ]}
            reply = None

    answer = await extractive_answer(request)
    return _result(answer, n_sources=n_sources, attempts=attempts, fallback=True, errors=errors)


class AnswerStreamExtractor:
    """Incrementally decode the ``"answer"`` string out of a streaming JSON reply.

    ``feed`` receives raw JSON fragments and returns only the newly decoded answer
    text, so the UI can show the answer as it is generated even in JSON mode.
    """

    _ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self, field: str = "answer") -> None:
        self._key = f'"{field}"'
        self._buf = ""
        self._pos = 0
        self._state = "seek"  # seek → colon → open → string → done

    @property
    def done(self) -> bool:
        return self._state == "done"

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        buf = self._buf
        while self._state != "done":
            if self._state == "seek":
                i = buf.find(self._key, self._pos)
                if i < 0:
                    self._pos = max(0, len(buf) - len(self._key))
                    break
                self._pos, self._state = i + len(self._key), "colon"
            elif self._state in ("colon", "open"):
                while self._pos < len(buf) and buf[self._pos].isspace():
                    self._pos += 1
                if self._pos >= len(buf):
                    break
                want = ":" if self._state == "colon" else '"'
                if buf[self._pos] != want:
                    self._state = "done"  # not the shape we expected; give up quietly
                    break
                self._pos += 1
                self._state = "open" if self._state == "colon" else "string"
            else:  # string
                while self._pos < len(buf):
                    c = buf[self._pos]
                    if c == "\\":
                        if self._pos + 1 >= len(buf):
                            break
                        nxt = buf[self._pos + 1]
                        if nxt == "u":
                            if self._pos + 6 > len(buf):
                                break
                            out.append(chr(int(buf[self._pos + 2:self._pos + 6], 16)))
                            self._pos += 6
                        else:
                            out.append(self._ESCAPES.get(nxt, nxt))
                            self._pos += 2
                    elif c == '"':
                        self._pos += 1
                        self._state = "done"
                        break
                    else:
                        out.append(c)
                        self._pos += 1
                break
        return "".join(out)
