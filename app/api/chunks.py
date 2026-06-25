"""F10: Advanced Chunking Strategies — HTTP API.

Endpoints
---------
GET  /chunks/strategies          List all strategies with descriptions + parameters
POST /chunks/preview             Chunk a document and return chunks (no indexing)
POST /chunks/compare             Run multiple strategies and compare results side-by-side
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.rag.chunking import STRATEGIES, chunk_document
from app.rag.ingestion.cleaning import clean_text
from app.rag.ingestion.parsers import MissingDependencyError
from app.rag.ingestion.registry import PARSERS, detect_format, ingest
from app.rag.models import Document, estimate_tokens

router = APIRouter(prefix="/chunks", tags=["chunks"])


# ── Strategy registry ─────────────────────────────────────────────────────────

_STRATEGY_INFO: dict[str, dict[str, Any]] = {
    "fixed": {
        "description": (
            "Fixed-size windows with configurable character/token overlap. "
            "Simple and predictable; good baseline for uniform-length text."
        ),
        "best_for": ["plain text", "reports", "articles"],
        "parameters": {
            "chunk_size": {"type": "int", "default": 512, "unit": "tokens",
                           "description": "Maximum chunk size."},
            "overlap":    {"type": "int", "default": 64,  "unit": "tokens",
                           "description": "Characters to repeat from previous chunk."},
            "size_unit":  {"type": "str", "default": "tokens",
                           "description": "'tokens' or 'chars'"},
        },
    },
    "recursive": {
        "description": (
            "Tries progressively finer separators: paragraph → line → sentence → "
            "word → character.  Preserves natural boundaries while guaranteeing "
            "the size limit is never exceeded."
        ),
        "best_for": ["general documents", "mixed content", "emails"],
        "parameters": {
            "chunk_size": {"type": "int", "default": 512, "unit": "tokens",
                           "description": "Maximum chunk size."},
            "size_unit":  {"type": "str", "default": "tokens"},
        },
    },
    "semantic": {
        "description": (
            "Packs complete sentences into each chunk without ever splitting a "
            "sentence mid-way.  Sentences that exceed the budget alone are hard-split."
        ),
        "best_for": ["prose", "Q&A documents", "legal text"],
        "parameters": {
            "chunk_size": {"type": "int", "default": 512, "unit": "tokens",
                           "description": "Maximum chunk size."},
        },
    },
    "sentence": {
        "description": "Alias for 'semantic' — split on sentence boundaries.",
        "best_for": ["same as semantic"],
        "parameters": {
            "chunk_size": {"type": "int", "default": 512},
        },
    },
    "structure": {
        "description": (
            "Respects markdown heading hierarchy.  Each heading and its body "
            "become one chunk; oversized sections are recursively split while "
            "keeping the section heading in every sub-chunk's metadata."
        ),
        "best_for": ["markdown", "wiki pages", "documentation", "reports with headings"],
        "parameters": {
            "chunk_size": {"type": "int", "default": 512, "unit": "tokens",
                           "description": "Maximum size per section chunk."},
        },
    },
    "code": {
        "description": (
            "Splits source code at function/class declaration boundaries.  "
            "Keeps entire functions or classes together wherever they fit within "
            "the budget; oversized blocks are recursively split.  Works across "
            "Python, JavaScript/TypeScript, Go, Rust, Java, C#, Ruby, and more."
        ),
        "best_for": ["source code", "Jupyter notebooks", "SQL scripts"],
        "parameters": {
            "chunk_size": {"type": "int", "default": 512, "unit": "tokens",
                           "description": "Maximum chunk size."},
        },
    },
}


# ── Request models ─────────────────────────────────────────────────────────────

class PreviewRequest(BaseModel):
    text: str
    filename: str = "document.txt"
    strategy: str = "recursive"
    chunk_size: int = Field(default=512, ge=1)
    overlap: int = Field(default=64, ge=0)
    size_unit: str = "tokens"


class CompareRequest(BaseModel):
    text: str
    filename: str = "document.txt"
    strategies: list[str] = Field(default=["fixed", "recursive", "semantic", "structure"])
    chunk_size: int = Field(default=512, ge=1)
    overlap: int = Field(default=64, ge=0)
    size_unit: str = "tokens"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_summary(chunk: dict) -> dict[str, Any]:
    """Summarise one chunk for API responses (omit the full text)."""
    return {
        "chunk_id":      chunk["chunk_id"],
        "index":         chunk["index"],
        "char_count":    chunk["char_count"],
        "token_estimate": chunk["token_estimate"],
        "start_char":    chunk["start_char"],
        "end_char":      chunk["end_char"],
        "preview":       chunk["text"][:200].strip(),
        "metadata":      {
            k: v for k, v in (chunk.get("metadata") or {}).items()
            if k in ("heading", "level", "block_index", "chunk_position",
                     "language", "quality_score", "strategy", "source", "title")
        },
    }


def _do_chunk(text: str, filename: str, strategy: str, chunk_size: int,
              overlap: int, size_unit: str) -> list[dict]:
    if strategy not in STRATEGIES:
        raise HTTPException(
            422,
            f"Unknown strategy '{strategy}'. "
            f"Available: {sorted(STRATEGIES)}"
        )
    doc = Document(
        document_id="preview",
        text=clean_text(text),
        format=detect_format(filename),
        source=filename,
        metadata={"title": filename},
    )
    config = {
        "chunk_size": chunk_size,
        "overlap":    overlap,
        "size_unit":  size_unit,
        "strategy":   strategy,
    }
    return [c.model_dump() for c in chunk_document(doc, strategy=strategy, config=config)]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/strategies")
async def list_strategies() -> dict:
    """
    List all available chunking strategies with descriptions, best-use cases,
    and supported parameters.

    Use this to help the user pick the right strategy for their document type.
    """
    return {
        "strategies": _STRATEGY_INFO,
        "default":    "recursive",
        "count":      len(_STRATEGY_INFO),
    }


@router.post("/preview")
async def preview_chunks(
    request: PreviewRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Chunk text using the chosen strategy and return the results.

    No indexing is performed — this is a preview / dry-run endpoint.
    Each chunk's metadata includes position, section heading (for structure),
    block index (for code), language, and quality score.

    Use the ``preview`` field (first 200 chars) to visually inspect chunk
    boundaries before committing to an ingestion strategy.
    """
    chunks = _do_chunk(
        request.text, request.filename, request.strategy,
        request.chunk_size, request.overlap, request.size_unit,
    )
    return {
        "strategy":    request.strategy,
        "chunk_count": len(chunks),
        "total_chars": sum(c["char_count"] for c in chunks),
        "total_tokens": sum(c["token_estimate"] for c in chunks),
        "avg_chars":   round(
            sum(c["char_count"] for c in chunks) / len(chunks), 1
        ) if chunks else 0,
        "chunks":      [_chunk_summary(c) for c in chunks],
    }


@router.post("/preview/upload")
async def preview_chunks_upload(
    file: UploadFile = File(...),
    strategy: str = Form(default="recursive"),
    chunk_size: int = Form(default=512),
    overlap: int = Form(default=64),
    size_unit: str = Form(default="tokens"),
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Upload a file, parse it, then preview how it would be chunked.

    Accepts any format supported by ``GET /ingest/formats``.  No indexing.
    """
    content = await file.read()
    if not content:
        raise HTTPException(422, "Uploaded file is empty.")
    filename = file.filename or "document"
    fmt = detect_format(filename)
    try:
        _, parser = PARSERS[fmt]
        raw_text, _ = parser(content, filename)
    except MissingDependencyError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, f"Failed to parse '{filename}': {exc}") from exc

    return await preview_chunks(
        PreviewRequest(
            text=raw_text,
            filename=filename,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
            size_unit=size_unit,
        ),
        user=user,
    )


@router.post("/compare")
async def compare_strategies(
    request: CompareRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Run multiple chunking strategies on the same text and compare results.

    Returns per-strategy stats (chunk count, avg chars, min/max chars) plus a
    ``recommendation`` based on the document type detected from the filename.
    Useful for choosing the optimal strategy before a bulk ingestion job.
    """
    if len(request.strategies) > 6:
        raise HTTPException(422, "Compare supports at most 6 strategies at once.")

    results: dict[str, Any] = {}
    for strat in request.strategies:
        if strat not in STRATEGIES:
            results[strat] = {"error": f"Unknown strategy '{strat}'"}
            continue
        chunks = _do_chunk(
            request.text, request.filename, strat,
            request.chunk_size, request.overlap, request.size_unit,
        )
        chars = [c["char_count"] for c in chunks]
        results[strat] = {
            "chunk_count": len(chunks),
            "total_chars": sum(chars),
            "avg_chars":   round(sum(chars) / len(chars), 1) if chars else 0,
            "min_chars":   min(chars, default=0),
            "max_chars":   max(chars, default=0),
            "total_tokens": sum(c["token_estimate"] for c in chunks),
            "sample_chunks": [_chunk_summary(c) for c in chunks[:3]],
        }

    # Heuristic recommendation based on filename / content signals.
    fmt = detect_format(request.filename)
    text = request.text
    if fmt in ("pptx",) or "### Slide" in text:
        recommended = "structure"
        reason = "Slide-structured content — preserve section boundaries"
    elif fmt in ("xlsx",) or "\t" in text[:500]:
        recommended = "fixed"
        reason = "Tabular / spreadsheet content — fixed windows work best"
    elif "#" in text[:200] and "\n#" in text:
        recommended = "structure"
        reason = "Markdown headings detected — section-aware chunking preserves structure"
    elif any(kw in text[:500] for kw in ("def ", "class ", "function ", "func ")):
        recommended = "code"
        reason = "Source code detected — split at function/class boundaries"
    elif len(text) < 2000:
        recommended = "semantic"
        reason = "Short document — sentence-level chunking avoids over-splitting"
    else:
        recommended = "recursive"
        reason = "General document — recursive strategy handles mixed content well"

    return {
        "strategies":   results,
        "recommended":  recommended if recommended in request.strategies else None,
        "reason":       reason,
        "input_chars":  len(request.text),
        "input_tokens": estimate_tokens(request.text),
    }
