"""Multimodal ingestion — Gemini vision for scanned pages, figures and images.

PDF pages with (almost) no text layer — scans — and the charts/figures embedded
in text pages are sent to Gemini's vision input, which returns the text in the
image and a short description. The result becomes ordinary segments
(``source_type="vision"``, with the page number), so it is chunked, embedded,
retrieved and cited like any other text. Standalone images (``.png``, ``.jpg``)
are handled the same way.

Opt-in per tenant (``vision_enabled`` in the tenant AI settings) because every
image is a paid-tier-sized model call. Without an API key, or with vision off,
ingestion carries on with the text layer and logs why images were skipped.

Every call goes through the LLM gateway (circuit breaker, budget guard, cost
ledger, OpenTelemetry span) as ``operation="vision"``.

Page images come from the images embedded in the PDF (pypdf) — a scanned page
is one full-page image. Vector-drawn charts have no image to send and are not
seen; rasterising whole pages would need a renderer such as PyMuPDF.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.rag.ingestion.cleaning import clean_text

log = logging.getLogger(__name__)

MAX_SIDE_PX = 1536   # longer side; ≤ 2×2 Gemini 768-px tiles

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "text": {"type": "STRING"},
        "description": {"type": "STRING"},
    },
    "required": ["text", "description"],
}

_PROMPTS = {
    "page": ("This is a scanned document page. Transcribe all of its text exactly, in reading "
             "order. End every line with a newline character (\\n) so that headings, lines, list "
             "items and table rows stay separate; never join two lines. Then describe the "
             "page in one sentence. Reply as JSON {\"text\": ..., \"description\": ...}."),
    "figure": ("This image is a figure from a document (chart, diagram, table or photo). In "
               "\"text\", copy any text, labels and numbers it contains. In \"description\", say in "
               "1-3 sentences what it shows, including the key values or trend. Reply as JSON "
               "{\"text\": ..., \"description\": ...}."),
}


@dataclass
class PageImage:
    page: int
    png: bytes
    width: int
    height: int


def availability(tenant: str) -> str:
    """``"ok"``, ``"disabled"`` (tenant setting off) or ``"no_api_key"``."""
    from app.rag.llm.gateway import is_configured
    from app.rag.tenants.ai_settings import get_settings

    if not get_settings(tenant).vision_enabled:
        return "disabled"
    if not is_configured(settings.vision_provider):
        return "no_api_key"
    return "ok"


def _to_png(img) -> tuple[bytes, int, int]:
    """Normalise any PIL image to an RGB PNG no larger than MAX_SIDE_PX."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE_PX:
        img = img.copy()
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), img.width, img.height


def pdf_page_images(content: bytes) -> dict[int, list[PageImage]]:
    """Embedded images per page (1-based), largest first, icons dropped."""
    from pypdf import PdfReader

    out: dict[int, list[PageImage]] = {}
    for number, page in enumerate(PdfReader(io.BytesIO(content)).pages, start=1):
        try:
            files = list(page.images)
        except Exception as exc:   # malformed image streams must not fail ingestion
            log.warning("vision: cannot read images on page %d: %s", number, exc)
            continue
        images = []
        for f in files:
            try:
                img = f.image
                if img is None or min(img.size) < settings.vision_min_image_px:
                    continue
                png, w, h = _to_png(img)
                images.append(PageImage(number, png, w, h))
            except Exception as exc:
                log.warning("vision: skipping an image on page %d: %s", number, exc)
        if images:
            out[number] = sorted(images, key=lambda i: i.width * i.height, reverse=True)
    return out


async def describe_image(png: bytes, kind: str, *, tenant: str, user_id: str = "") -> dict[str, str]:
    """One vision call: ``{"text", "description"}`` for a page (``kind="page"``) or figure."""
    from app.rag.llm.gateway import call_llm

    request = {"messages": [{
        "role": "user",
        "content": _PROMPTS[kind],
        "parts": [
            {"text": _PROMPTS[kind]},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode("ascii")}},
        ],
    }]}
    config: dict[str, Any] = {"response_schema": _SCHEMA, "temperature": 0.0, "max_tokens": 2048}
    if settings.vision_model:
        config["model"] = settings.vision_model
    response = await call_llm(settings.vision_provider, request, config,
                              tenant=tenant, user_id=user_id, operation="vision")
    try:
        data = json.loads(response.text)
        return {"text": clean_text(str(data.get("text", ""))),
                "description": clean_text(str(data.get("description", "")))}
    except (ValueError, AttributeError):
        # Not JSON after all: keep whatever text came back.
        return {"text": clean_text(response.text), "description": ""}


def _segment(result: dict[str, str], page: int, kind: str) -> dict[str, Any] | None:
    text, desc = result["text"], result["description"]
    if kind == "figure":
        body = "\n".join(filter(None, [f"[Figure on page {page}] {desc}".strip(), text]))
    else:
        body = "\n\n".join(filter(None, [text, f"[Page {page} image: {desc}]" if desc else ""]))
    if not body.strip():
        return None
    return {"text": body, "page": page, "source_type": "vision", "vision_kind": kind,
            "description": desc or None}


async def enrich(content: bytes, fmt: str, text: str, meta: dict[str, Any], *,
                 tenant: str, user_id: str = "") -> tuple[str, dict[str, Any]]:
    """Add vision segments to a parsed PDF or image; returns the new ``(text, meta)``.

    ``meta["vision"]`` records what happened: ``status`` is ``applied``,
    ``not_needed``, ``disabled``, ``no_api_key`` or ``failed``.
    """
    if fmt not in ("pdf", "image"):
        return text, meta
    meta = dict(meta)
    status = availability(tenant)
    if status != "ok":
        if fmt == "image" or meta.get("scanned_pages"):
            reason = ("vision ingestion is off for this workspace" if status == "disabled"
                      else f"no API key for vision provider '{settings.vision_provider}'")
            level = logging.WARNING if status == "no_api_key" else logging.INFO
            log.log(level, "vision: skipped for tenant %s (%s)", tenant, reason)
        meta["vision"] = {"status": status}
        return text, meta

    jobs: list[tuple[int, str, bytes]] = []   # (page, kind, png)
    if fmt == "image":
        from PIL import Image

        png, _, _ = _to_png(Image.open(io.BytesIO(content)))
        jobs.append((1, "page", png))
    else:
        scanned = set(meta.get("scanned_pages") or [])
        for page, images in sorted(pdf_page_images(content).items()):
            if page in scanned:
                jobs.append((page, "page", images[0].png))     # the scan itself
                images = images[1:]
            jobs.extend((page, "figure", img.png) for img in images)
    if not jobs:
        meta["vision"] = {"status": "not_needed"}
        return text, meta

    skipped = max(0, len(jobs) - settings.vision_max_images)
    if skipped:
        log.warning("vision: %d image(s) over the per-document cap (%d) were skipped",
                    skipped, settings.vision_max_images)
    from app.rag.cost import BudgetExceededError
    from app.rag.llm.gateway import LLMUnavailableError

    segments: list[dict[str, Any]] = []
    errors = 0
    for page, kind, png in jobs[:settings.vision_max_images]:
        try:
            seg = _segment(await describe_image(png, kind, tenant=tenant, user_id=user_id), page, kind)
        except (LLMUnavailableError, BudgetExceededError) as exc:
            log.warning("vision: stopped (%s); the rest of the document is indexed without it", exc)
            errors += 1
            break
        except Exception as exc:   # one bad image must not fail the upload
            log.warning("vision: page %d %s failed: %s", page, kind, exc)
            errors += 1
            continue
        if seg:
            segments.append(seg)

    meta["vision"] = {"status": "applied" if segments else "failed", "images": len(segments),
                      "errors": errors, "skipped_over_cap": skipped,
                      "pages": sorted({s["page"] for s in segments})}
    if not segments:
        return text, meta

    if fmt == "image":
        meta["segments"] = segments
        meta.pop("quality_flag", None)          # the OCR stub text is replaced
        return segments[0]["text"], meta

    # Merge: a scanned page's vision text replaces its (near-empty) text layer;
    # figures follow their page's text.
    scanned_done = {s["page"] for s in segments if s["vision_kind"] == "page"}
    merged = [s for s in meta.get("segments") or [] if s["page"] not in scanned_done]
    merged += segments
    merged.sort(key=lambda s: (s["page"], s.get("source_type") == "vision"))
    meta["segments"] = merged
    return "\n\n".join(s["text"] for s in merged), meta
