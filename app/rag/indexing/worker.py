"""Indexing worker — runs the 6-step pipeline for one task (F19).

Steps: validate → quality → chunk → embed → upsert → done
Each step updates the task's progress and pushes an SSE event.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.indexing.task import IndexingTask


async def run_task(task: "IndexingTask") -> None:
    """Execute the full indexing pipeline for *task* in place (mutates it)."""
    from app.rag.indexing.task import TaskStatus

    task.status     = TaskStatus.RUNNING
    task.started_at = datetime.now(timezone.utc)
    task.push_event(step="start", progress=0, message="Pipeline started")

    try:
        await _step_validate(task)
        await _step_quality(task)
        await _step_chunk_embed_upsert(task)
        task.status      = TaskStatus.COMPLETED
        task.progress    = 100
        task.step        = "done"
        task.finished_at = datetime.now(timezone.utc)
        task.message     = f"Indexed {task.chunks_done} chunks from '{task.filename}'"
        task.push_event(step="done", progress=100, chunks=task.chunks_done,
                        message=task.message)
    except Exception as exc:
        task.status      = TaskStatus.FAILED
        task.error       = str(exc)
        task.step        = "error"
        task.finished_at = datetime.now(timezone.utc)
        task.message     = str(exc)
        task.push_event(step="error", progress=task.progress, message=str(exc))
        raise


async def _step_validate(task: "IndexingTask") -> None:
    task.step     = "validate"
    task.progress = 10
    task.message  = "Validating text…"
    task.push_event(step="validate", progress=10, message=task.message)
    await asyncio.sleep(0)
    if not task.text.strip():
        raise ValueError("Text is empty — nothing to index.")


async def _step_quality(task: "IndexingTask") -> None:
    from app.rag.ingestion.quality import assess_quality

    task.step     = "quality"
    task.progress = 25
    task.message  = "Assessing document quality…"
    task.push_event(step="quality", progress=25, message=task.message)
    await asyncio.sleep(0)

    quality = assess_quality(task.text, {})
    if quality.action == "reject":
        raise ValueError(
            f"Quality too low (score {quality.score:.2f}): "
            + (quality.note or "text extraction likely failed")
        )
    task.metadata["quality"] = quality.model_dump()
    task.push_event(step="quality_done", quality_score=quality.score,
                    quality_action=quality.action)


async def _step_chunk_embed_upsert(task: "IndexingTask") -> None:
    from app.rag.chunking      import chunk_document
    from app.rag.embeddings    import embed_texts
    from app.rag.models        import Document
    from app.rag.vectorstore   import VectorRecord, get_store

    # Chunk
    task.step     = "chunk"
    task.progress = 40
    task.message  = "Splitting into chunks…"
    task.push_event(step="chunk", progress=40, message=task.message)
    await asyncio.sleep(0)

    doc    = Document(document_id=task.id, text=task.text,
                      format="text", source=task.filename)
    chunks = chunk_document(doc, strategy="recursive",
                            config={"chunk_size": 512, "size_unit": "tokens"})
    texts  = [c.text for c in chunks]
    task.push_event(step="chunk_done", chunk_count=len(chunks))

    # Embed
    task.step     = "embed"
    task.progress = 65
    task.message  = f"Embedding {len(chunks)} chunks…"
    task.push_event(step="embed", progress=65, message=task.message)
    await asyncio.sleep(0)

    from app.rag.embeddings.registry import DEFAULT_DIMENSION, DEFAULT_MODEL
    dim  = DEFAULT_DIMENSION
    vecs = embed_texts(texts, DEFAULT_MODEL, dim)
    task.push_event(step="embed_done", vectors=len(vecs), dimension=dim)

    # Upsert
    task.step     = "upsert"
    task.progress = 85
    task.message  = "Storing in vector DB…"
    task.push_event(step="upsert", progress=85, message=task.message)
    await asyncio.sleep(0)

    store = get_store(task.store, dim)
    records = [
        VectorRecord(
            id=f"{task.id}:{i}",
            vector=vecs[i],
            metadata={
                "text":      c.text,
                "doc_id":    task.id,
                "filename":  task.filename,
                "chunk_idx": i,
                "user_id":   task.user_id,
                "tenant":    task.tenant,
                **task.metadata,
            },
        )
        for i, c in enumerate(chunks)
    ]
    store.upsert(records, namespace=task.namespace)
    task.chunks_done = len(records)
    task.push_event(step="upsert_done", upserted=task.chunks_done)
