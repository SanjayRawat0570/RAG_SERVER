"""F1 — Document upload workflow with step-by-step progress tracking.

Flow: validate → extract text → chunk → embed → index → save → done
Progress is tracked in-memory and persisted to Supabase on completion.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_current_user
from app.api.pipelines import build_index_workflow
from app.config import settings
from app.engine.executor import WorkflowExecutor
from app.rag.ingestion.registry import PARSERS, detect_format

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_MB       = 50
ALLOWED_EXTS = {
    # text
    ".txt", ".md", ".markdown",
    # web
    ".html", ".htm",
    # structured
    ".json", ".xml",
    # documents
    ".pdf", ".docx",
    # spreadsheets / presentations
    ".xlsx", ".pptx",
    # images (OCR)
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
    # audio / video (transcription stub)
    ".mp3", ".wav", ".mp4", ".mov",
}

# In-memory store: exec_id → state dict
_execs: dict[str, dict[str, Any]] = {}


def _set(exec_id: str, **kw: Any) -> None:
    if exec_id in _execs:
        _execs[exec_id].update(kw)


def _sb():
    if settings.supabase_url and settings.supabase_key:
        from supabase import create_client  # type: ignore[import]
        return create_client(settings.supabase_url, settings.supabase_key)
    return None


async def _process(
    exec_id: str,
    content:  bytes,
    filename: str,
    tenant:   str,
    user_id:  str,
) -> None:
    try:
        # ── 1. Extract text ──────────────────────────────────────────────────
        _set(exec_id, progress=15, step="extract",
             message=f"Extracting text from {filename}…")
        await asyncio.sleep(0)

        fmt = detect_format(filename)
        if fmt not in PARSERS:
            raise ValueError(f"Unsupported file type '{os.path.splitext(filename)[1]}'.")
        _, parser = PARSERS[fmt]
        text, parser_meta = parser(content, filename)
        if not text.strip():
            raise ValueError("No readable text found in the file.")

        words = len(text.split())

        # ── 2. Quality assessment ─────────────────────────────────────────────
        _set(exec_id, progress=25, step="quality",
             message="Assessing document quality…")
        await asyncio.sleep(0)

        from app.rag.ingestion.quality import assess_quality
        quality = assess_quality(text, parser_meta)
        if quality.action == "reject":
            raise ValueError(
                f"Document quality too low (score {quality.score:.2f}): "
                + (quality.note or "text extraction likely failed")
            )
        quality_meta = quality.model_dump()

        # ── 3. Chunk ─────────────────────────────────────────────────────────
        _set(exec_id, progress=40, step="chunk",
             message="Splitting into chunks…")
        await asyncio.sleep(0)

        # ── 4. Embed ─────────────────────────────────────────────────────────
        _set(exec_id, progress=60, step="embed",
             message="Creating embeddings…")
        await asyncio.sleep(0)

        # ── 5. Index (chunk → embed → upsert run via the engine) ─────────────
        _set(exec_id, progress=75, step="index",
             message="Indexing into knowledge base…")
        executor = WorkflowExecutor(build_index_workflow())
        result   = await executor.run(
            {"tenant": tenant, "text": text, "filename": filename}
        )
        if result.status != "success":
            raise RuntimeError("Indexing pipeline returned failure status.")

        chunks = result.outputs["out"].get("upserted", 0)

        # ── 6. Persist to Supabase ───────────────────────────────────────────
        _set(exec_id, progress=92, step="save",
             message="Saving execution record…")
        await asyncio.sleep(0)

        sb = _sb()
        if sb:
            sb.table("workflow_executions").insert({
                "id":             exec_id,
                "user_id":        user_id,
                "tenant":         tenant,
                "filename":       filename,
                "status":         "completed",
                "chunks_indexed": chunks,
                "completed_at":   datetime.now(timezone.utc).isoformat(),
            }).execute()

        # ── Done ─────────────────────────────────────────────────────────────
        quality_note = f" ⚠ {quality.note}" if quality.action == "warn" else ""
        _set(exec_id,
             progress=100,
             step="done",
             status="completed",
             message=f"Done! {chunks} chunks indexed from {words:,} words.{quality_note}",
             chunks_indexed=chunks,
             quality=quality_meta)

    except Exception as exc:  # noqa: BLE001
        _set(exec_id, status="failed", progress=0,
             message=str(exc), step="error")
        try:
            sb = _sb()
            if sb:
                sb.table("workflow_executions").insert({
                    "id":            exec_id,
                    "user_id":       user_id,
                    "tenant":        tenant,
                    "filename":      filename,
                    "status":        "failed",
                    "error_message": str(exc),
                }).execute()
        except Exception:
            pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file:   UploadFile = File(...),
    tenant: str        = Form(default="acme"),
    user:   dict       = Depends(get_current_user),
) -> dict:
    filename = file.filename or "document"
    ext      = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            422,
            f"File type '{ext}' not supported. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    content = await file.read()
    if len(content) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_MB} MB limit.")
    if not content:
        raise HTTPException(422, "Uploaded file is empty.")

    exec_id = str(uuid4())
    _execs[exec_id] = {
        "id":             exec_id,
        "status":         "running",
        "progress":       5,
        "step":           "validate",
        "message":        "Validating file…",
        "filename":       filename,
        "tenant":         tenant,
        "user_id":        user["id"],
        "chunks_indexed": None,
    }
    background_tasks.add_task(_process, exec_id, content, filename, tenant, user["id"])
    return {"execution_id": exec_id}


@router.get("/executions/{exec_id}")
async def get_execution(
    exec_id: str,
    user: dict = Depends(get_current_user),
) -> dict:
    ex = _execs.get(exec_id)
    if not ex or ex["user_id"] != user["id"]:
        raise HTTPException(404, "Execution not found.")
    # Strip the internal user_id before sending to the client.
    return {k: v for k, v in ex.items() if k != "user_id"}
