"""Core RAG data models: Document (F9) and Chunk (F10)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Rough average across English text / common tokenizers. Used to convert
# token-denominated chunk sizes to character budgets without a heavy tokenizer.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class Document(BaseModel):
    """A parsed, normalized document ready for chunking."""

    document_id: str
    text: str
    format: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunk(BaseModel):
    """A chunk of a document, carrying provenance for retrieval (F10)."""

    chunk_id: str
    document_id: str
    index: int
    text: str
    start_char: int = 0
    end_char: int = 0
    char_count: int = 0
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
