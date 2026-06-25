"""F9: Document Ingestion & Format Support — HTTP API.

Endpoints
---------
GET  /ingest/formats           List all supported formats and extensions
POST /ingest/analyze           Parse + quality-assess without indexing
POST /ingest/url               Fetch a web page and index it
POST /ingest/text              Ingest inline text with quality assessment
"""
from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.api.pipelines import build_index_workflow
from app.engine.executor import WorkflowExecutor
from app.rag.ingestion.cleaning import clean_text, derive_title
from app.rag.ingestion.parsers import MissingDependencyError, fetch_url
from app.rag.ingestion.quality import assess_quality
from app.rag.ingestion.registry import (
    PARSERS,
    detect_format,
    ingest,
    supported_formats,
)
from app.rag.models import estimate_tokens

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ── Request / response models ─────────────────────────────────────────────────

class TextIngestRequest(BaseModel):
    text: str
    filename: str = "document.txt"
    tenant: str = "default"
    index: bool = True


class UrlIngestRequest(BaseModel):
    url: str
    tenant: str = "default"
    index: bool = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_analyze_response(
    text: str,
    parser_meta: dict[str, Any],
    filename: str,
    fmt: str,
    content_size: int,
) -> dict[str, Any]:
    """Build the analysis response dict without indexing."""
    quality = assess_quality(text, parser_meta)
    words = text.split()
    return {
        "filename":       filename,
        "format":         fmt,
        "file_size":      content_size,
        "char_count":     len(text),
        "word_count":     len(words),
        "token_estimate": estimate_tokens(text),
        "title":          parser_meta.get("title") or derive_title(text),
        "parser_meta":    {k: v for k, v in parser_meta.items() if k != "title"},
        "quality":        quality.model_dump(),
        "preview":        text[:500].strip() if text else "",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/formats")
async def list_formats() -> dict:
    """
    List all supported document formats with their file extensions.

    Use this to check whether a particular file type is supported before
    uploading it. Formats that require optional libraries (OCR, transcription)
    are included with a note about required dependencies.
    """
    fmts = supported_formats()
    notes = {
        "image": "OCR text extraction: requires pytesseract + Tesseract binary.",
        "audio": "Transcription: requires openai-whisper (pip install openai-whisper).",
        "video": "Transcription: requires openai-whisper (pip install openai-whisper).",
    }
    return {
        "formats": {
            fmt: {"extensions": exts, "note": notes.get(fmt)}
            for fmt, exts in fmts.items()
        },
        "total_extensions": sum(len(v) for v in fmts.values()),
    }


@router.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Parse a document and run quality assessment without indexing.

    Returns extracted text preview, word/token counts, detected language,
    quality score, and any quality flags (OCR noise, too short, etc.).
    Use this to preview what will be indexed before committing.

    Supports all formats listed by ``GET /ingest/formats``.
    """
    content = await file.read()
    if not content:
        raise HTTPException(422, "Uploaded file is empty.")

    filename = file.filename or "document"
    fmt = detect_format(filename)

    try:
        _, parser = PARSERS[fmt]
        raw_text, parser_meta = parser(content, filename)
    except MissingDependencyError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Failed to parse '{filename}': {exc}") from exc

    text = clean_text(raw_text)
    canonical_fmt = PARSERS[fmt][0]

    return _build_analyze_response(text, parser_meta, filename, canonical_fmt, len(content))


@router.post("/url")
async def ingest_url(
    request: UrlIngestRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Fetch a web page and ingest it.

    Downloads the page at ``url``, strips HTML, runs quality assessment, and
    (when ``index=true``) indexes it into the vector store under ``tenant``.
    Returns the same quality + metadata response as ``POST /ingest/analyze``.
    """
    try:
        raw_text, parser_meta = await fetch_url(request.url)
    except Exception as exc:
        raise HTTPException(422, f"Failed to fetch URL: {exc}") from exc

    text = clean_text(raw_text)
    quality = assess_quality(text, parser_meta)

    filename = request.url.split("//", 1)[-1].split("/")[0] + ".html"
    response = _build_analyze_response(text, parser_meta, filename, "html", len(text.encode()))

    if request.index and quality.action != "reject":
        try:
            executor = WorkflowExecutor(build_index_workflow())
            result = await executor.run(
                {"tenant": request.tenant, "text": text, "filename": filename}
            )
            chunks = result.outputs.get("out", {}).get("upserted", 0)
            response["indexed"] = True
            response["chunks_indexed"] = chunks
            response["tenant"] = request.tenant
        except Exception as exc:
            response["indexed"] = False
            response["index_error"] = str(exc)
    else:
        response["indexed"] = False
        if quality.action == "reject":
            response["index_skipped_reason"] = "quality_too_low"

    return response


@router.post("/text")
async def ingest_text(
    request: TextIngestRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Ingest raw text with quality assessment.

    Useful for programmatic ingestion when you already have the text extracted
    (e.g., from a pipeline, a database export, or an API response).
    Set ``index=false`` to run analysis only.
    """
    text = clean_text(request.text)
    fmt = detect_format(request.filename)
    canonical_fmt = PARSERS.get(fmt, ("text", None))[0]
    quality = assess_quality(text)

    response = _build_analyze_response(text, {}, request.filename, canonical_fmt, len(text.encode()))

    if request.index and quality.action != "reject":
        try:
            executor = WorkflowExecutor(build_index_workflow())
            result = await executor.run(
                {"tenant": request.tenant, "text": text, "filename": request.filename}
            )
            chunks = result.outputs.get("out", {}).get("upserted", 0)
            response["indexed"] = True
            response["chunks_indexed"] = chunks
            response["tenant"] = request.tenant
        except Exception as exc:
            response["indexed"] = False
            response["index_error"] = str(exc)
    else:
        response["indexed"] = False
        if quality.action == "reject":
            response["index_skipped_reason"] = "quality_too_low"

    return response
