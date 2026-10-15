"""Output PII guardrail (P0): phone numbers, personal emails, confidential fields."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.rag.security.guardrails import (
    CONFIDENTIAL, EMAIL_REDACTED, PHONE_REDACTED, StreamGuard, confidential_fields, guard_answer,
)

HR_DOC = {"text": "# HR\n\nThe HR contact is Dana Lee, phone +1-555-0100, salary band C (confidential).",
          "source": "hr.md"}


@pytest.mark.parametrize("phone", ["+1-555-0100", "+44 20 7946 0958", "(555) 010-0100", "555.010.0100",
                                   "+49 (30) 1234 5678"])
def test_phone_numbers_are_withheld(phone):
    out, report = guard_answer(f"Call {phone} for help.")
    assert phone not in out and PHONE_REDACTED in out
    assert report == {"applied": True, "redactions": {"phone": 1}}


@pytest.mark.parametrize("text", [
    "Error E-4012 means the token expired.", "PO 2025-00418 was approved.", "Order LAP-14-STD.",
    "Released on 2024-01-15 as version 3.2.1.", "Passwords need at least 14 characters.",
    "The budget is $1,500,000.", "Ticket INC-2024-041 closed.", "Build 2025-1182 shipped.",
])
def test_codes_dates_and_amounts_are_not_phone_numbers(text):
    assert guard_answer(text) == (text, {"applied": False, "redactions": {}})


def test_personal_email_withheld_functional_mailbox_kept():
    out, report = guard_answer("Write to dana.lee@example.com or helpdesk@example.com.")
    assert "dana.lee@example.com" not in out and EMAIL_REDACTED in out
    assert "helpdesk@example.com" in out
    assert report["redactions"] == {"email": 1}


def test_confidential_field_is_withheld_but_the_rest_of_the_answer_stays():
    answer = "Dana Lee is the HR contact [1]. Dana Lee is in salary band C [1]."
    out, report = guard_answer(answer, [HR_DOC])
    assert out.startswith("Dana Lee is the HR contact [1].")
    assert "band C" not in out and CONFIDENTIAL in out and out.endswith("[1]")
    assert report["redactions"] == {"confidential": 1}


def test_confidential_line_and_heading_markers():
    docs = [{"text": "# Pay\n\nConfidential: the bonus pool is 2 million.\nThe office opens at 9."},
            {"text": "# Confidential salaries\n\nAlex earns 90000 per year."}]
    assert {"bonus", "pool", "2", "million"} in confidential_fields(docs)
    out, _ = guard_answer("The bonus pool is 2 million. Alex earns 90000 per year. It opens at 9.", docs)
    assert "2 million" not in out and "90000" not in out and "opens at 9" in out


def test_guardrail_can_be_switched_off(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "pii_guardrail", False)
    assert guard_answer("Call +1-555-0100.")[0] == "Call +1-555-0100."


def test_stream_guard_never_emits_a_number_split_across_tokens():
    guard = StreamGuard([HR_DOC])
    tokens = ["Her phone is +1-", "555-", "01", "00. She is in salary ", "band C.", " Thanks"]
    emitted = "".join(guard.feed(t) for t in tokens) + guard.flush()
    assert "555" not in emitted and "band C" not in emitted
    assert PHONE_REDACTED in emitted and emitted.endswith("Thanks")


# ── Through the API ────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/v1/rag/index", headers=AUTH,
                   json={"tenant": "pii-t", "filename": "hr.md", "text": HR_DOC["text"]})
        assert r.status_code == 200
        yield c


def test_rag_ask_never_returns_the_phone_or_confidential_band(client):
    out = client.post("/api/v1/rag/ask", headers=AUTH, json={
        "tenant": "pii-t", "question": "What is Dana Lee's phone number and salary band?"}).json()
    assert "555-0100" not in out["answer"] and "band C" not in out["answer"]
    # The stub quotes the whole source sentence, which carries the confidential field.
    assert CONFIDENTIAL in out["answer"] and out["guardrail"]["applied"] is True


def test_stream_never_sends_the_phone_or_confidential_band(client):
    with client.stream("POST", "/api/v1/rag/stream/ask", headers=AUTH, json={
            "tenant": "pii-t", "question": "What is Dana Lee's phone number and salary band?"}) as r:
        body = "".join(r.iter_text())
    events = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]
    tokens = "".join(e.get("token", "") for e in events if e.get("event") == "token")
    complete = next(e for e in events if e.get("event") == "complete")
    for text in (tokens, complete["answer"]):
        assert "555-0100" not in text and "band C" not in text
