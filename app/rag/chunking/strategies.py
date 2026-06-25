"""Chunking strategies (F10).

Each strategy is a function ``(text, size_chars, overlap_chars) -> list[(text, meta)]``
where ``meta`` carries per-chunk hints (e.g. the section heading for
structure-aware chunking). :func:`chunk_document` converts a token/char budget,
runs the chosen strategy, and wraps the pieces into :class:`Chunk` objects with
full provenance — including language and quality score propagated from the
parent document.

Strategies
----------
* ``fixed``      — fixed-size windows with configurable overlap
* ``recursive``  — split on a separator hierarchy (¶ → line → sentence → word)
* ``semantic``   — pack whole sentences up to the budget, never splitting one
* ``structure``  — split on markdown headings, keeping each section together
* ``sentence``   — alias for semantic (split on sentence boundaries)
* ``code``       — split at function/class boundaries; preserve entire blocks
"""
from __future__ import annotations

import re
from typing import Any, Callable

from app.rag.models import CHARS_PER_TOKEN, Chunk, Document, estimate_tokens

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Zero-width lookahead: match positions just before function/class definitions
# across the most common programming languages (Python, JS/TS, Go, Rust, Java,
# C/C++, Ruby, PHP).  The pattern fires on lines that start with one of these
# keywords, giving us clean split points between top-level declarations.
_CODE_BOUNDARY = re.compile(
    r"(?m)^(?=[ \t]*(?:"
    r"def |async def |class |"            # Python
    r"function |async function |"         # JavaScript / TypeScript
    r"const \w|let \w|var \w|"           # JS arrow / var declarations
    r"export (?:default )?(?:function|class|const)|"  # ES modules
    r"func |"                             # Go / Swift
    r"pub fn |fn |"                       # Rust
    r"(?:public|private|protected|internal|static)\s+\w|"  # Java / C#
    r"sub |"                              # Perl / Ruby
    r"def self\.|"                        # Ruby class methods
    r"#\[|@"                              # Rust attr / Java annotation
    r"))"
)

Piece = tuple[str, dict[str, Any]]


# -- raw splitters (return plain strings) -----------------------------------

def _fixed(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    step = max(1, size - overlap)
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start += step
    return out


def _recursive(text: str, size: int, separators: list[str] | None = None) -> list[str]:
    separators = separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]
    sep, rest = separators[0], separators[1:]

    if sep == "":  # last resort: hard character split
        return [text[i:i + size] for i in range(0, len(text), size)] or [text]

    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = f"{buf}{sep}{part}" if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(part) > size and rest:
            out.extend(_recursive(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        out.append(buf)
    return [c for c in out if c.strip()]


def _semantic(text: str, size: int) -> list[str]:
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    out: list[str] = []
    buf = ""
    for sentence in sentences:
        if not buf:
            buf = sentence
        elif len(buf) + 1 + len(sentence) <= size:
            buf = f"{buf} {sentence}"
        else:
            out.append(buf)
            buf = sentence
        while len(buf) > size:  # a single oversized sentence: fall back to fixed
            out.append(buf[:size])
            buf = buf[size:]
    if buf:
        out.append(buf)
    return out


# -- strategy wrappers (attach per-chunk metadata) --------------------------

def fixed(text: str, size: int, overlap: int) -> list[Piece]:
    return [(c, {}) for c in _fixed(text, size, overlap)]


def recursive(text: str, size: int, overlap: int) -> list[Piece]:
    return [(c, {}) for c in _recursive(text, size)]


def semantic(text: str, size: int, overlap: int) -> list[Piece]:
    return [(c, {}) for c in _semantic(text, size)]


def structure(text: str, size: int, overlap: int) -> list[Piece]:
    """Split on markdown headings; sections larger than the budget are further
    split recursively while keeping the heading in each chunk's metadata."""
    sections: list[tuple[str | None, int, list[str]]] = []
    heading: str | None = None
    level = 0
    buf: list[str] = []

    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            if buf:
                sections.append((heading, level, buf))
            heading, level, buf = match.group(2).strip(), len(match.group(1)), [line]
        else:
            buf.append(line)
    if buf:
        sections.append((heading, level, buf))

    out: list[Piece] = []
    for sec_heading, sec_level, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        meta = {"heading": sec_heading, "level": sec_level} if sec_heading else {}
        if len(body) <= size:
            out.append((body, meta))
        else:
            out.extend((piece, meta) for piece in _recursive(body, size))
    return out


def sentence(text: str, size: int, overlap: int) -> list[Piece]:
    """Alias for semantic — split on sentence boundaries."""
    return semantic(text, size, overlap)


def code(text: str, size: int, overlap: int) -> list[Piece]:
    """Code-aware chunking: split at function/class declaration boundaries.

    Keeps each top-level declaration (function, class, method) together as one
    chunk whenever it fits within the size budget.  Oversized blocks are further
    split with the recursive strategy so they never exceed the limit.

    The chunk metadata includes ``block_index`` (the nth code block found) so
    callers can reconstruct ordering without relying on char offsets.
    """
    if not text.strip():
        return []

    # Find all split positions.
    positions = [m.start() for m in _CODE_BOUNDARY.finditer(text)]
    if not positions:
        # No recognisable boundaries — fall back to recursive splitting.
        return [(piece, {"block_index": i}) for i, piece in enumerate(_recursive(text, size))]

    # Ensure we capture any text before the first boundary.
    if positions[0] > 0:
        positions = [0] + positions

    blocks: list[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)

    out: list[Piece] = []
    block_idx = 0
    for block in blocks:
        if len(block) <= size:
            out.append((block, {"block_index": block_idx}))
            block_idx += 1
        else:
            # Block is too large — split recursively, keeping block_index.
            sub_pieces = _recursive(block, size)
            for piece in sub_pieces:
                out.append((piece, {"block_index": block_idx}))
            block_idx += 1

    return out


STRATEGIES: dict[str, Callable[[str, int, int], list[Piece]]] = {
    "fixed":     fixed,
    "recursive": recursive,
    "semantic":  semantic,
    "sentence":  sentence,
    "structure": structure,
    "code":      code,
}


def chunk_document(
    document: Document,
    strategy: str = "recursive",
    config: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split *document* into retrieval-ready :class:`Chunk` objects.

    Each chunk's ``metadata`` carries full provenance:
    - ``title`` / ``format`` / ``source`` — from the parent document
    - ``strategy`` — which chunking strategy was used
    - ``heading`` / ``level`` — populated by the *structure* strategy
    - ``block_index`` — populated by the *code* strategy
    - ``language`` — propagated from the document's quality assessment (F9)
    - ``quality_score`` — document-level quality score (F9)
    - ``chunk_position`` — ``"first"`` | ``"middle"`` | ``"last"`` | ``"only"``
    """
    config = config or {}
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}. Available: {sorted(STRATEGIES)}"
        )

    # Budget may be given in tokens (default) or characters.
    factor = 1 if config.get("size_unit") == "chars" else CHARS_PER_TOKEN
    size_chars = max(1, int(config.get("chunk_size", 512)) * factor)
    overlap_chars = max(0, int(config.get("overlap", 0)) * factor)

    pieces = STRATEGIES[strategy](document.text, size_chars, overlap_chars)

    # Propagate language and quality from the F9 quality assessment block.
    doc_quality = document.metadata.get("quality") or {}
    doc_language = doc_quality.get("language") or document.metadata.get("language")
    quality_score = doc_quality.get("score")

    total = len(pieces)
    chunks: list[Chunk] = []
    cursor = 0
    base_meta: dict[str, Any] = {
        "title":         document.metadata.get("title"),
        "format":        document.format,
        "source":        document.source,
        "strategy":      strategy,
        "language":      doc_language,
        "quality_score": quality_score,
    }
    for index, (text, extra) in enumerate(pieces):
        start = document.text.find(text, cursor)
        if start < 0:
            start = cursor
        end = start + len(text)
        cursor = max(cursor, start + 1)

        if total == 1:
            position = "only"
        elif index == 0:
            position = "first"
        elif index == total - 1:
            position = "last"
        else:
            position = "middle"

        meta = {**base_meta, **extra, "chunk_position": position}
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::{index}",
                document_id=document.document_id,
                index=index,
                text=text,
                start_char=start,
                end_char=end,
                char_count=len(text),
                token_estimate=estimate_tokens(text),
                metadata=meta,
            )
        )
    return chunks
