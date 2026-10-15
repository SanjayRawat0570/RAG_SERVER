"""On-disk journal for the in-process vector stores (F12): data survives restarts.

Each namespace (one tenant, or ``org/team``) has its own directory under
``<DATA_DIR>/vectors/<store>/<namespace>/`` holding numbered segment files:

* an **upsert** segment — ids, a float32 vector matrix and the metadata (which
  includes the chunk text, so the BM25 index is rebuilt from it on load);
* a **delete** segment — ids only.

Every upsert/delete batch appends one small segment (atomic write: temp file +
rename), so a write costs the size of the batch, not of the namespace. On load
the segments are replayed in order; when a namespace has more than
:data:`COMPACT_AFTER` segments it is rewritten as a single snapshot.

Stores whose name starts with ``__`` (internal caches) are not journalled.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

import numpy as np

from app.rag.vectorstore.base import VectorRecord

if TYPE_CHECKING:
    from app.rag.vectorstore.memory import InMemoryVectorStore

log = logging.getLogger(__name__)
COMPACT_AFTER = 64
_SEG = re.compile(r"^seg-(\d{8})\.npz$")


class Journal:
    def __init__(self, root: Path) -> None:
        self.root = root

    # ── paths ──────────────────────────────────────────────────────────────
    def _dir(self, namespace: str) -> Path:
        return self.root / quote(namespace, safe="")

    def _segments(self, namespace: str) -> list[Path]:
        d = self._dir(namespace)
        if not d.is_dir():
            return []
        return sorted(p for p in d.iterdir() if _SEG.match(p.name))

    def _write(self, namespace: str, **arrays: Any) -> None:
        d = self._dir(namespace)
        d.mkdir(parents=True, exist_ok=True)
        segs = self._segments(namespace)
        n = int(_SEG.match(segs[-1].name).group(1)) + 1 if segs else 1
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".seg-", suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, d / f"seg-{n:08d}.npz")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── writes ─────────────────────────────────────────────────────────────
    def upsert(self, namespace: str, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._write(namespace, kind=np.array("upsert"),
                    ids=np.array(json.dumps([r.id for r in records])),
                    vectors=np.asarray([r.vector for r in records], dtype=np.float32),
                    metadata=np.array(json.dumps([r.metadata for r in records], default=str)))

    def delete(self, namespace: str, ids: list[str]) -> None:
        if ids:
            self._write(namespace, kind=np.array("delete"), ids=np.array(json.dumps(list(ids))))

    def snapshot(self, store: InMemoryVectorStore, namespace: str) -> None:
        """Rewrite *namespace* as one segment and drop the older ones."""
        old = self._segments(namespace)
        ns = store._namespaces.get(namespace)
        if ns is None or not ns.ids:
            for p in old:
                p.unlink(missing_ok=True)
            return
        self._write(namespace, kind=np.array("upsert"), ids=np.array(json.dumps(ns.ids)),
                    vectors=np.vstack(ns.vectors).astype(np.float32),
                    metadata=np.array(json.dumps(ns.metadata, default=str)))
        for p in old:
            p.unlink(missing_ok=True)

    def maybe_compact(self, store: InMemoryVectorStore, namespace: str) -> None:
        if len(self._segments(namespace)) > COMPACT_AFTER:
            self.snapshot(store, namespace)

    def drop(self, namespace: str) -> None:
        for p in self._segments(namespace):
            p.unlink(missing_ok=True)

    # ── load ───────────────────────────────────────────────────────────────
    def namespaces(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return [unquote(d.name) for d in self.root.iterdir() if d.is_dir()]

    def replay(self, store: InMemoryVectorStore) -> int:
        """Load every journalled namespace into *store*; returns the record count."""
        total = 0
        for namespace in self.namespaces():
            ns = store._ns(namespace)
            segments = self._segments(namespace)
            for seg in segments:
                try:
                    with np.load(seg, allow_pickle=False) as z:
                        ids = json.loads(str(z["ids"]))
                        if str(z["kind"]) == "delete":
                            ns.delete(ids)
                            continue
                        meta = json.loads(str(z["metadata"]))
                        for rid, vec, md in zip(ids, z["vectors"], meta):
                            ns.upsert_one(VectorRecord(id=rid, vector=vec.tolist(), metadata=md))
                except Exception as exc:  # noqa: BLE001 - one bad segment must not lose the rest
                    log.error("skipping unreadable vector segment %s: %s", seg, exc)
            total += len(ns.ids)
            if len(segments) > 1:
                self.snapshot(store, namespace)
        return total
