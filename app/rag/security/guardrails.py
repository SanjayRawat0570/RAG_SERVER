"""Output PII guardrail (F25).

Every generated answer — model, extractive stub or fallback, JSON or streamed —
passes through :func:`guard_answer` before it leaves the server. It removes:

* **fields marked confidential in the sources.** A clause of a source line that
  carries a marker such as ``(confidential)`` / ``[internal]``, every clause of a
  line that starts ``Confidential:``, and every line under a heading containing
  "confidential". An answer sentence that repeats all the content words of such a
  field is replaced with :data:`CONFIDENTIAL`.
* **phone numbers** (international ``+…`` numbers and ``(555) 010-0100`` style).
* **personal email addresses.** Functional mailboxes whose local part is in
  ``settings.pii_allowed_email_prefixes`` (``helpdesk@``, ``security@``, ``dpo@``…)
  are published contact points and are kept.

:class:`StreamGuard` applies the same rules to a token stream by releasing text
one sentence at a time, so redacted text is never streamed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.config import settings

EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)[\s.-]?)?\d{1,4}(?:[\s.-]?\d{2,4}){1,3}"
    r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}"
    r"|\b\d{3}[\s.]\d{3}[\s.]\d{4})\b"
)
_MARKER = re.compile(r"[(\[]\s*(confidential|internal only|internal|restricted|private)\s*[)\]]", re.I)
_LINE_MARKER = re.compile(r"^\s*(?:[-*]\s*)?(confidential|restricted)\s*[:\-–]", re.I)
_HEADING = re.compile(r"^\s*#{1,6}\s*(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-@_/][a-z0-9]+)*")
_STOP = set("""a an the and or of to in on at by for with from as is are was were be it its this that
these those their they he she his her we our you your i me my which who what when where how do does
has have had can may not no than then so also per each any all into over under up about""".split())

EMAIL_REDACTED = "[email withheld]"
PHONE_REDACTED = "[phone withheld]"
CONFIDENTIAL = "[withheld: marked confidential in the source]"


def _content(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP}


def confidential_fields(documents: Iterable[dict[str, Any]]) -> list[set[str]]:
    """Content-word sets of every field the sources mark as confidential."""
    fields: list[set[str]] = []
    for doc in documents or []:
        text = str(doc.get("text") or doc.get("metadata", {}).get("text") or "")
        heading_confidential = bool(_MARKER.search(str(doc.get("heading") or "")) or
                                    "confidential" in str(doc.get("heading") or "").lower())
        for line in text.splitlines():
            m = _HEADING.match(line)
            if m:
                heading_confidential = "confidential" in m.group(1).lower()
                continue
            whole = heading_confidential or bool(_LINE_MARKER.match(line))
            for clause in re.split(r"[,;]|\s+-\s+", line):
                if whole or _MARKER.search(clause):
                    words = _content(_LINE_MARKER.sub("", _MARKER.sub("", clause)))
                    words -= {"confidential", "internal", "restricted", "private", "only"}
                    if words:
                        fields.append(words)
    return fields


@dataclass
class GuardReport:
    redactions: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, n: int = 1) -> None:
        if n:
            self.redactions[kind] = self.redactions.get(kind, 0) + n

    def as_dict(self) -> dict[str, Any]:
        return {"applied": bool(self.redactions), "redactions": dict(self.redactions)}


def _redact_patterns(text: str, report: GuardReport) -> str:
    allowed = {p.lower() for p in settings.pii_allowed_email_prefixes}

    def email(m: re.Match) -> str:
        if m.group(1).lower() in allowed:
            return m.group(0)
        report.add("email")
        return EMAIL_REDACTED

    def phone(m: re.Match) -> str:
        if sum(c.isdigit() for c in m.group(0)) < 7:
            return m.group(0)
        report.add("phone")
        return PHONE_REDACTED

    return PHONE.sub(phone, EMAIL.sub(email, text))


def _redact_sentence(sentence: str, fields: list[set[str]], report: GuardReport) -> str:
    words = _content(sentence)
    if any(f <= words for f in fields):
        report.add("confidential")
        # Keep the citation marker so the sentence still points at its source.
        markers = " ".join(re.findall(r"\[\d+\]", sentence))
        return CONFIDENTIAL + (f" {markers}" if markers else "")
    return sentence


def guard_answer(answer: str, documents: Iterable[dict[str, Any]] | None = None
                 ) -> tuple[str, dict[str, Any]]:
    """Return (*answer* with PII removed, report)."""
    report = GuardReport()
    if not settings.pii_guardrail or not answer:
        return answer, report.as_dict()
    fields = confidential_fields(documents or [])
    if fields:
        parts = re.split(r"((?<=[.!?])\s+|\n+)", answer)
        answer = "".join(p if i % 2 else _redact_sentence(p, fields, report) for i, p in enumerate(parts))
    return _redact_patterns(answer, report), report.as_dict()


class StreamGuard:
    """Incremental :func:`guard_answer` for token streams: emits whole sentences only."""

    def __init__(self, documents: Iterable[dict[str, Any]] | None = None) -> None:
        self.documents = list(documents or [])
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        last = max((m.end() for m in _SENTENCE.finditer(self._buf)), default=0)
        if not last:
            return ""
        ready, self._buf = self._buf[:last], self._buf[last:]
        return guard_answer(ready, self.documents)[0]

    def flush(self) -> str:
        ready, self._buf = self._buf, ""
        return guard_answer(ready, self.documents)[0]
