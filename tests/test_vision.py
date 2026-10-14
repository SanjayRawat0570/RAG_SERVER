"""Multimodal ingestion: scanned PDF pages, figures and images via (mocked) Gemini vision.

PDFs are built in-test: Pillow writes an image-only PDF (a real "scanned" page,
no text layer) and pypdf writes a page with a text layer. The vision provider is
a fake registered in the LLM registry, so calls still run through the gateway.
"""
from __future__ import annotations

import io
import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.api.pipelines import STORE
from app.config import settings
from app.main import app
from app.observability.span_store import get_exporter
from app.rag.cost import get_usage
from app.rag.ingestion import vision
from app.rag.ingestion.parsers import parse_pdf
from app.rag.llm import registry as llm_registry
from app.rag.llm.base import LLMResponse
from app.rag.llm.gateway import IMAGE_TOKENS_ESTIMATE, _prompt_tokens
from app.rag.tenants.ai_settings import update_settings
from app.rag.vectorstore import lookup_store

AUTH = {"Authorization": "Bearer dev"}
TEXT_LINES = ["Annual leave policy for all staff members.",
              "Full-time staff receive 25 days of annual leave per year."]
SCAN_TEXT = "Part-time staff receive leave pro rata to their contracted hours."


# ── PDF builders ───────────────────────────────────────────────────────────────

def _text_pdf(lines: list[str]) -> bytes:
    w = PdfWriter()
    page = w.add_blank_page(612, 792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): w._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td 14 TL " + b"".join(b"(%s) Tj T* " % l.encode() for l in lines) + b"ET")
    page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def _image(size=(600, 800)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).rectangle([50, 50, 300, 200], fill="black")
    return img


def _scan_pdf(size=(600, 800)) -> bytes:
    buf = io.BytesIO()
    _image(size).save(buf, "PDF")
    return buf.getvalue()


def _pdf(*pages: bytes) -> bytes:
    w = PdfWriter()
    for p in pages:
        w.append(PdfReader(io.BytesIO(p)))
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def _text_page_with_figure() -> bytes:
    page = PdfReader(io.BytesIO(_scan_pdf((800, 600)))).pages[0]
    page.merge_page(PdfReader(io.BytesIO(_text_pdf(TEXT_LINES))).pages[0])
    w = PdfWriter()
    w.add_page(page)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


# ── Fixtures ───────────────────────────────────────────────────────────────────

class FakeVision:
    """Stands in for Gemini: answers the vision prompt with fixed JSON."""

    name = "fakevision"

    def __init__(self, fail: bool = False):
        self.calls: list[tuple[dict, dict]] = []
        self.fail = fail

    async def generate(self, request, config):
        self.calls.append((request, config))
        if self.fail:
            raise RuntimeError("vision backend exploded")
        prompt = request["messages"][0]["parts"][0]["text"]
        if "scanned document page" in prompt:
            payload = {"text": SCAN_TEXT, "description": "A scanned HR policy page."}
        else:
            payload = {"text": "Leave days: 2023 = 22, 2024 = 25",
                       "description": "Bar chart of annual leave days rising from 22 to 25."}
        return LLMResponse(text=json.dumps(payload), provider=self.name, model="fake-vision",
                           usage={"input_tokens": 1100, "output_tokens": 40})


@pytest.fixture
def fake_vision(monkeypatch):
    def install(fail: bool = False) -> FakeVision:
        bot = FakeVision(fail)
        llm_registry.register_provider("fakevision", lambda: bot)
        llm_registry._instances.pop("fakevision", None)
        monkeypatch.setattr(settings, "vision_provider", "fakevision")
        return bot
    yield install
    llm_registry._FACTORIES.pop("fakevision", None)
    llm_registry._instances.pop("fakevision", None)


@pytest.fixture
def tenant():
    return f"vis-{uuid.uuid4().hex[:8]}"


def _upload(client: TestClient, tenant: str, name: str, content: bytes, mime="application/pdf") -> dict:
    r = client.post("/api/v1/documents/upload", headers=AUTH, data={"tenant": tenant},
                    files={"file": (name, content, mime)})
    assert r.status_code == 200, r.text
    return client.get(f"/api/v1/documents/executions/{r.json()['execution_id']}", headers=AUTH).json()


def _chunks(tenant: str) -> list[dict]:
    store = lookup_store(STORE)
    return [meta for _, meta in store.list_records(namespace=tenant)] if store else []


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_scanned_page_gets_text_via_vision(fake_vision, tenant):
    bot = fake_vision()
    update_settings(tenant, vision_enabled=True)
    pdf = _pdf(_text_pdf(TEXT_LINES), _scan_pdf())
    with TestClient(app) as client:
        ex = _upload(client, tenant, "handbook.pdf", pdf)
        assert ex["status"] == "completed", ex
        assert ex["vision"]["status"] == "applied" and ex["vision"]["pages"] == [2]

        # One vision call, with the page image inline.
        assert len(bot.calls) == 1
        request, config = bot.calls[0]
        image = request["messages"][0]["parts"][1]["inline_data"]
        assert image["mime_type"] == "image/png" and len(image["data"]) > 100
        assert config["response_schema"]["required"] == ["text", "description"]

        chunks = _chunks(tenant)
        by_page = {c["page"]: c for c in chunks}
        assert by_page[1]["source_type"] == "text" and "25 days" in by_page[1]["text"]
        assert by_page[2]["source_type"] == "vision" and SCAN_TEXT in by_page[2]["text"]
        assert by_page[2]["source"] == "handbook.pdf"

        # The vision text is retrievable and cited with its page.
        out = client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": tenant, "question": "How much leave do part-time staff receive?"}).json()
    assert "pro rata" in out["answer"]
    cited = [c for c in out["citations"] if c["marker"] in out["cited_markers"]]
    assert any(c.get("page") == 2 and c.get("source_type") == "vision" for c in cited)


def test_vision_calls_go_through_the_gateway(fake_vision, tenant):
    fake_vision()
    update_settings(tenant, vision_enabled=True)
    with TestClient(app) as client:
        _upload(client, tenant, "scan.pdf", _scan_pdf())
    # Cost ledger …
    events = get_usage(tenant=tenant, operation="llm:vision")
    assert len(events) == 1 and events[0].input_tokens == 1100
    # … and an OpenTelemetry span.
    spans = [s for s in get_exporter().get_finished_spans()
             if s.name == "llm.call" and (s.attributes or {}).get("tenant.id") == tenant]
    assert [s.attributes["llm.operation"] for s in spans] == ["vision"]


def test_figure_on_a_text_page_is_described(fake_vision, tenant):
    bot = fake_vision()
    update_settings(tenant, vision_enabled=True)
    with TestClient(app) as client:
        ex = _upload(client, tenant, "report.pdf", _text_page_with_figure())
    assert ex["status"] == "completed"
    assert "figure" in bot.calls[0][0]["messages"][0]["parts"][0]["text"]
    chunks = _chunks(tenant)
    figure = [c for c in chunks if c.get("source_type") == "vision"]
    assert len(figure) == 1 and figure[0]["page"] == 1 and figure[0]["vision_kind"] == "figure"
    assert "[Figure on page 1]" in figure[0]["text"] and "22 to 25" in figure[0]["text"]
    # The page's own text layer is kept alongside the figure.
    assert any(c.get("source_type") == "text" and "25 days" in c["text"] for c in chunks)


def test_disabled_tenant_makes_no_vision_calls(fake_vision, tenant):
    bot = fake_vision()   # vision_enabled defaults to False
    with TestClient(app) as client:
        mixed = _upload(client, tenant, "mixed.pdf", _pdf(_text_pdf(TEXT_LINES), _scan_pdf()))
        scan_only = _upload(client, tenant, "scan.pdf", _scan_pdf())
    assert bot.calls == []
    assert mixed["status"] == "completed"
    assert {c["page"] for c in _chunks(tenant)} == {1}
    assert scan_only["status"] == "failed"
    assert "turn on vision ingestion" in scan_only["message"]


def test_no_api_key_skips_with_a_warning(monkeypatch, tenant, caplog):
    monkeypatch.setattr(settings, "vision_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    update_settings(tenant, vision_enabled=True)
    with caplog.at_level(logging.WARNING, logger="app.rag.ingestion.vision"):
        with TestClient(app) as client:
            ex = _upload(client, tenant, "mixed.pdf", _pdf(_text_pdf(TEXT_LINES), _scan_pdf()))
    assert ex["status"] == "completed" and ex["vision"] == {"status": "no_api_key"}
    assert any("no API key" in r.getMessage() for r in caplog.records)


def test_provider_failure_keeps_the_text_layer(fake_vision, tenant):
    fake_vision(fail=True)
    update_settings(tenant, vision_enabled=True)
    with TestClient(app) as client:
        ex = _upload(client, tenant, "mixed.pdf", _pdf(_text_pdf(TEXT_LINES), _scan_pdf()))
    assert ex["status"] == "completed"
    assert ex["vision"]["status"] == "failed" and ex["vision"]["errors"] == 1
    assert {c["source_type"] for c in _chunks(tenant)} == {"text"}


def test_per_document_image_cap(fake_vision, tenant, monkeypatch):
    bot = fake_vision()
    monkeypatch.setattr(settings, "vision_max_images", 2)
    update_settings(tenant, vision_enabled=True)
    with TestClient(app) as client:
        ex = _upload(client, tenant, "scans.pdf", _pdf(_scan_pdf(), _scan_pdf(), _scan_pdf()))
    assert len(bot.calls) == 2
    assert ex["vision"]["skipped_over_cap"] == 1


def test_standalone_image_uses_vision(fake_vision, tenant):
    fake_vision()
    update_settings(tenant, vision_enabled=True)
    buf = io.BytesIO()
    _image().save(buf, "PNG")
    with TestClient(app) as client:
        ex = _upload(client, tenant, "notice.png", buf.getvalue(), "image/png")
    assert ex["status"] == "completed"
    [chunk] = _chunks(tenant)
    assert chunk["source_type"] == "vision" and SCAN_TEXT in chunk["text"]
    assert "OCR text extraction requires" not in chunk["text"]


def test_text_pdfs_get_page_numbers_without_vision():
    text, meta = parse_pdf(_pdf(_text_pdf(["Page one says hello there, with a full line of text."]),
                                _text_pdf(["Page two says goodbye now, also with enough text."])), "two.pdf")
    assert [(s["page"], s["source_type"]) for s in meta["segments"]] == [(1, "text"), (2, "text")]
    assert meta["scanned_pages"] == [] and "hello" in text and "goodbye" in text


def test_gateway_budgets_images():
    req = {"messages": [{"role": "user", "content": "hi",
                         "parts": [{"text": "hi"}, {"inline_data": {"mime_type": "image/png", "data": "x"}}]}]}
    assert _prompt_tokens(req) >= IMAGE_TOKENS_ESTIMATE


def test_vision_toggle_requires_manage_rights():
    from app.rag.security.apikeys import create_api_key, reset_api_keys
    reset_api_keys()
    _, member = create_api_key("alice", "m", tenant="acme", role="member")
    with TestClient(app) as client:
        r = client.put("/api/v1/tenant-settings/acme", json={"vision_enabled": True}, headers={"X-API-Key": member})
        assert r.status_code == 403
        r = client.put("/api/v1/tenant-settings/acme", json={"vision_enabled": True}, headers=AUTH)
        assert r.status_code == 200 and r.json()["vision_enabled"] is True
        client.put("/api/v1/tenant-settings/acme", json={"vision_enabled": False}, headers=AUTH)
    reset_api_keys()


def test_availability_states(monkeypatch, tenant):
    assert vision.availability(tenant) == "disabled"
    update_settings(tenant, vision_enabled=True)
    monkeypatch.setattr(settings, "vision_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert vision.availability(tenant) == "no_api_key"
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    assert vision.availability(tenant) == "ok"
