"""Ingestion registry: format detection and Document assembly (F9)."""
from __future__ import annotations

import hashlib
from typing import Any

from app.rag.ingestion.cleaning import clean_text, derive_title
from app.rag.ingestion.parsers import (
    Parser,
    parse_audio,
    parse_docx,
    parse_html,
    parse_image,
    parse_json,
    parse_markdown,
    parse_pdf,
    parse_pptx,
    parse_text,
    parse_xlsx,
    parse_xml,
)
from app.rag.models import Document, estimate_tokens

# Extension → (canonical-format-name, parser).
PARSERS: dict[str, tuple[str, Parser]] = {
    # Plain text
    "txt":      ("text",     parse_text),
    "text":     ("text",     parse_text),
    # Markdown
    "md":       ("markdown", parse_markdown),
    "markdown": ("markdown", parse_markdown),
    # Web
    "html":     ("html",     parse_html),
    "htm":      ("html",     parse_html),
    # Structured data
    "json":     ("json",     parse_json),
    "xml":      ("xml",      parse_xml),
    # Documents
    "pdf":      ("pdf",      parse_pdf),
    "docx":     ("docx",     parse_docx),
    # Spreadsheets / presentations
    "xlsx":     ("xlsx",     parse_xlsx),
    "pptx":     ("pptx",     parse_pptx),
    # Images (OCR)
    "jpg":      ("image",    parse_image),
    "jpeg":     ("image",    parse_image),
    "png":      ("image",    parse_image),
    "bmp":      ("image",    parse_image),
    "tiff":     ("image",    parse_image),
    "tif":      ("image",    parse_image),
    "webp":     ("image",    parse_image),
    # Audio / Video (transcription)
    "mp3":      ("audio",    parse_audio),
    "wav":      ("audio",    parse_audio),
    "m4a":      ("audio",    parse_audio),
    "mp4":      ("video",    parse_audio),
    "mov":      ("video",    parse_audio),
    "webm":     ("video",    parse_audio),
}

DEFAULT_FORMAT = "text"


def detect_format(filename: str | None, explicit: str | None = None) -> str:
    """Resolve a format key from an explicit override or a filename extension."""
    if explicit:
        return explicit.lower()
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in PARSERS:
            return ext
    return DEFAULT_FORMAT


def supported_formats() -> dict[str, list[str]]:
    """Return canonical format names mapped to their supported extensions."""
    result: dict[str, list[str]] = {}
    for ext, (fmt, _) in PARSERS.items():
        result.setdefault(fmt, []).append(f".{ext}")
    return {k: sorted(v) for k, v in sorted(result.items())}


def parse(content: bytes, fmt: str, filename: str | None = None) -> tuple[str, dict[str, Any]]:
    if fmt not in PARSERS:
        raise ValueError(f"Unsupported format {fmt!r}. Supported: {sorted(set(PARSERS))}")
    _, parser = PARSERS[fmt]
    return parser(content, filename)


def ingest(
    content: bytes,
    *,
    filename: str | None = None,
    fmt: str | None = None,
    document_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    run_quality: bool = True,
) -> Document:
    """Parse + clean + quality-assess raw bytes into a :class:`Document` (F9)."""
    from app.rag.ingestion.quality import assess_quality

    resolved_fmt = detect_format(filename, fmt)
    format_name = PARSERS[resolved_fmt][0]
    raw_text, parser_meta = parse(content, resolved_fmt, filename)
    text = clean_text(raw_text)

    doc_id = document_id or hashlib.sha1(content).hexdigest()[:16]
    metadata: dict[str, Any] = {
        "char_count":     len(text),
        "word_count":     len(text.split()),
        "token_estimate": estimate_tokens(text),
        "file_size":      len(content),
        "title":          parser_meta.get("title") or derive_title(text),
    }
    metadata.update({k: v for k, v in parser_meta.items() if k != "title"})
    if extra_metadata:
        metadata.update(extra_metadata)

    if run_quality:
        quality = assess_quality(text, metadata)
        metadata["quality"] = quality.model_dump()

    return Document(
        document_id=doc_id,
        text=text,
        format=format_name,
        source=filename,
        metadata=metadata,
    )
