"""Text cleaning and normalization shared by all parsers (F9)."""
from __future__ import annotations

import re
import unicodedata

_TRAILING_WS = re.compile(r"[ \t]+(\n|$)")
_INLINE_WS = re.compile(r"[ \t]{2,}")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace while preserving paragraph breaks.

    * NFC unicode normalization
    * normalize CRLF/CR to LF
    * collapse runs of spaces/tabs to a single space
    * strip trailing spaces on each line
    * collapse 3+ blank lines to a single blank line (paragraph boundary)
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    text = _TRAILING_WS.sub(r"\1", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def derive_title(text: str) -> str | None:
    """Best-effort title: first markdown heading, else first non-empty line."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.lstrip("#").strip()[:200]
    return None
