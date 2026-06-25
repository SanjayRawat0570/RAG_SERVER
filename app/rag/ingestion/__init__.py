"""Document ingestion (F9): format detection + extraction + normalization."""
from app.rag.ingestion.registry import detect_format, ingest, parse

__all__ = ["detect_format", "ingest", "parse"]
