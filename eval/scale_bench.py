"""Retrieval scaling: per-query dense + BM25 time and process memory vs. chunk count.

Fills the in-process vector store (the one the ask pipeline uses) with synthetic
chunks: real eval-corpus chunk texts repeated with a numeric suffix, and random
unit vectors of the active embedding dimension (vector values don't change the
cost of FLAT search). Then times the same operations the ask pipeline's
``vector_search`` and ``keyword_search`` nodes run per question (keyword search
uses the namespace's incremental BM25 index), plus the insert cost per 1000
chunks (which now includes keeping that index up to date). Embedding cost is
excluded (bge indexing is ~100 ms/doc; see eval/benchmark.py).

``--compare-rebuild`` also times the old per-query BM25 rebuild.

Usage (from backend/)::

    python eval/scale_bench.py --sizes 1000 10000 50000 100000 --compare-rebuild
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import sys
import time
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.rag.search import BM25, tokenize  # noqa: E402
from app.rag.vectorstore.base import VectorRecord  # noqa: E402
from app.rag.vectorstore.memory import InMemoryVectorStore  # noqa: E402


def rss_mb() -> float | None:
    """Working set of this process (Windows); None elsewhere."""
    if sys.platform != "win32":
        return None

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PMC), ctypes.wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.WorkingSetSize / 2**20, 1)


def texts() -> list[str]:
    out = []
    for doc in sorted((EVAL_DIR / "corpus").rglob("*.md")):
        out += [p.strip() for p in doc.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000, 100000])
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--compare-rebuild", action="store_true")
    p.add_argument("--label", default="scale_bench")
    a = p.parse_args()
    base = texts()
    rng = np.random.default_rng(0)
    queries = ["What is the minimum password length?", "Who owns Project Beacon?",
               "What does error E-4012 mean?", "How many vacation days do employees get?"]
    store, n, results = InMemoryVectorStore("scale", a.dim), 0, []
    print(f"baseline rss {rss_mb()} MB")
    for size in sorted(a.sizes):
        batch = []
        while n < size:
            vec = rng.standard_normal(a.dim).astype(np.float32)
            batch.append(VectorRecord(id=f"c{n}", vector=vec.tolist(),
                                      metadata={"text": f"{base[n % len(base)]} ({n})", "source": f"d{n // 3}.md"}))
            n += 1
        t0 = time.perf_counter()
        store.upsert(batch, namespace="t")
        insert_ms_per_1k = (time.perf_counter() - t0) * 1000 / max(len(batch), 1) * 1000
        dense, sparse, rebuild = [], [], []
        for i in range(a.queries):
            q = queries[i % len(queries)]
            t0 = time.perf_counter()
            store.search(rng.standard_normal(a.dim).astype(np.float32).tolist(), top_k=5, namespace="t")
            dense.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            store.keyword_search(q, 5, "t")
            sparse.append((time.perf_counter() - t0) * 1000)
            if a.compare_rebuild:
                t0 = time.perf_counter()
                records = store.list_records("t")
                BM25([tokenize(str(m.get("text", ""))) for _, m in records]).scores(tokenize(q))
                rebuild.append((time.perf_counter() - t0) * 1000)
        r = {"chunks": size, "dense_ms_median": round(float(np.median(dense)), 1),
             "bm25_ms_median": round(float(np.median(sparse)), 2),
             **({"bm25_rebuild_ms_median": round(float(np.median(rebuild)), 1)} if rebuild else {}),
             "insert_ms_per_1k_chunks": round(insert_ms_per_1k, 1), "rss_mb": rss_mb()}
        print(json.dumps(r), flush=True)
        results.append(r)
    out = EVAL_DIR / "results" / f"{a.label}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
