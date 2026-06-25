"""Ingest node — turns raw input into a parsed Document (F9).

Content sources (in priority order):
* ``content_base64`` — base64-encoded bytes (any format)
* ``path``           — read bytes from a file on disk
* ``text`` / upstream — inline text (or the upstream node's output)

Config::

    {
      "text": "$.inputs.body",       # or content_base64 / path
      "filename": "report.pdf",       # drives format detection
      "format": "markdown",           # optional explicit override
      "metadata": {"tenant": "acme"}  # merged into Document.metadata
    }
"""
from __future__ import annotations

import base64
from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.ingestion import ingest


@register
class IngestNode(Node):
    type = "ingest"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        content = self._load_content(ctx, upstream)
        filename = ctx.resolve(self.config.get("filename"))
        fmt = self.config.get("format")
        extra = ctx.resolve(self.config.get("metadata")) if "metadata" in self.config else None
        document = ingest(
            content,
            filename=filename,
            fmt=fmt,
            document_id=ctx.resolve(self.config.get("document_id")),
            extra_metadata=extra if isinstance(extra, dict) else None,
        )
        return document.model_dump()

    def _load_content(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> bytes:
        if "content_base64" in self.config:
            return base64.b64decode(ctx.resolve(self.config["content_base64"]))
        if "path" in self.config:
            with open(ctx.resolve(self.config["path"]), "rb") as fh:
                return fh.read()
        if "text" in self.config:
            value = ctx.resolve(self.config["text"])
        else:
            value = _single_upstream(upstream) if upstream else ""
        if isinstance(value, bytes):
            return value
        return str(value).encode("utf-8")
