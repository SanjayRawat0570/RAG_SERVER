"""Latency benchmark: retrieval time, time-to-first-token and total, per question.

Drives the same generator the SSE endpoint streams (``_rag_pipeline_events``),
so the timings are what a browser sees minus the network:

* index_ms_per_doc  — ingest → chunk → graph → embed → upsert, per corpus document
* retrieval_ms      — request start → retrieval done (generation starts)
* ttft_ms           — request start → first answer token
* total_ms          — request start → ``complete`` event

Usage (from backend/)::

    python eval/benchmark.py                                 # stub, all questions
    python eval/benchmark.py --provider gemini --model gemini-3.1-flash-lite --limit 30 --delay 4
    python eval/benchmark.py --label v2_baseline_stub        # save eval/results/bench_<label>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.api.stream import _rag_pipeline_events  # noqa: E402
from eval.run_eval import index_corpus, load_questions  # noqa: E402

TENANT = "bench"


def _pct(values: list[float], p: float) -> float | None:
    vals = sorted(values)
    return round(vals[int(p * (len(vals) - 1))], 1) if vals else None


def summary(values: list[float]) -> dict[str, float | None]:
    return {"p50": _pct(values, 0.5), "p95": _pct(values, 0.95),
            "mean": round(statistics.mean(values), 1) if values else None}


async def time_question(question: str, provider: str, mode: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    marks: dict[str, float] = {}
    async for ev in _rag_pipeline_events(question, TENANT, provider, mode=mode):
        now = (time.perf_counter() - t0) * 1000
        if ev["event"] == "step" and ev.get("step") == "generate" and ev.get("status") == "start":
            marks.setdefault("retrieval_ms", now)
        elif ev["event"] == "token":
            marks.setdefault("ttft_ms", now)
        elif ev["event"] in ("complete", "error"):
            marks["total_ms"] = now
            marks["error"] = ev["event"] == "error"
            break
    return marks


async def run(provider: str, mode: str, limit: int | None, delay: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    chunks, _ = await index_corpus(TENANT)
    docs = len(list((EVAL_DIR / "corpus").rglob("*.md")))
    index_ms = (time.perf_counter() - t0) * 1000

    questions = load_questions()[:limit] if limit else load_questions()
    rows = []
    for q in questions:
        rows.append({"id": q["id"], "category": q["category"],
                     **await time_question(q["question"], provider, mode)})
        if delay:
            await asyncio.sleep(delay)
    ok = [r for r in rows if not r.get("error")]
    return {
        "provider": provider, "mode": mode, "questions": len(rows), "errors": len(rows) - len(ok),
        "index": {"docs": docs, "chunks": chunks, "ms_total": round(index_ms, 1),
                  "ms_per_doc": round(index_ms / docs, 1)},
        "retrieval_ms": summary([r["retrieval_ms"] for r in ok if "retrieval_ms" in r]),
        "ttft_ms": summary([r["ttft_ms"] for r in ok if "ttft_ms" in r]),
        "total_ms": summary([r["total_ms"] for r in ok]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="stub")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=["fast", "agentic"], default="fast")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if args.model:
        from app.config import settings
        settings.gemini_model = args.model

    report = asyncio.run(run(args.provider, args.mode, args.limit, args.delay))
    print(f"\nprovider={report['provider']}  mode={report['mode']}  questions={report['questions']}  "
          f"errors={report['errors']}")
    print(f"  indexing           {report['index']['docs']} docs / {report['index']['chunks']} chunks in "
          f"{report['index']['ms_total']} ms ({report['index']['ms_per_doc']} ms/doc)")
    for k in ("retrieval_ms", "ttft_ms", "total_ms"):
        s = report[k]
        print(f"  {k:<18} p50 {s['p50']}   p95 {s['p95']}   mean {s['mean']}")
    if args.label:
        out = EVAL_DIR / "results" / f"bench_{args.label}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
