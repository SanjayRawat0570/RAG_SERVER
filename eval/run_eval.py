"""Offline RAG evaluation harness.

Indexes ``eval/corpus/*.md`` into an isolated tenant, asks every question in
``eval/questions.jsonl`` through the same workflow the API uses, and reports:

* citation_rate      — answerable/multihop answers that cite >= 1 source
* citation_precision — of those, answers citing at least one expected source
* answer_accuracy    — answerable/multihop answers containing every expected keyword
* idk_accuracy       — unanswerable questions answered with "I don't know"
* false_idk_rate     — answerable/multihop questions wrongly answered "I don't know"
* latency_ms (mean / p95)

Usage (from backend/)::

    python eval/run_eval.py                       # stub provider, deterministic
    python eval/run_eval.py --provider gemini     # real model (needs GEMINI_API_KEY)
    python eval/run_eval.py --label baseline      # results -> eval/results/baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.api.pipelines import build_ask_workflow, build_index_workflow  # noqa: E402
from app.engine.executor import WorkflowExecutor  # noqa: E402

TENANT = "eval"
_MARKER = re.compile(r"\[(\d+)\]")
_IDK = re.compile(
    r"(i\s+don'?t\s+know|i\s+do\s+not\s+know|not\s+(?:contained|mentioned|available|"
    r"specified|stated|provided)\s+in|cannot\s+(?:answer|find)|no\s+information)",
    re.IGNORECASE,
)


def load_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def index_corpus(tenant: str = TENANT) -> int:
    total = 0
    for doc in sorted((EVAL_DIR / "corpus").glob("*.md")):
        result = await WorkflowExecutor(build_index_workflow()).run(
            {"tenant": tenant, "text": doc.read_text(encoding="utf-8"), "filename": doc.name}
        )
        total += result.outputs["out"]["upserted"]
    return total


async def ask(question: str, provider: str, tenant: str = TENANT) -> dict[str, Any]:
    result = await WorkflowExecutor(build_ask_workflow(provider)).run(
        {"tenant": tenant, "question": question}
    )
    return result.outputs.get("out", {}) or {}


def cited_sources(out: dict[str, Any]) -> list[str]:
    """Sources the answer actually cites (markers in the text or structured ids)."""
    by_marker = {c.get("marker"): c.get("source") for c in out.get("citations") or []}
    markers = out.get("cited_markers")
    if markers is None:
        markers = [f"[{n}]" for n in _MARKER.findall(str(out.get("answer", "")))]
    return [by_marker[m] for m in markers if by_marker.get(m)]


def is_idk(out: dict[str, Any]) -> bool:
    if out.get("answerable") is False:
        return True
    return bool(_IDK.search(str(out.get("answer", ""))))


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    cited = [r for r in answerable if r["cited_sources"]]
    latencies = sorted(r["latency_ms"] for r in rows)

    def ratio(n: int, d: int) -> float:
        return round(n / d, 3) if d else 0.0

    return {
        "questions": len(rows),
        "citation_rate": ratio(len(cited), len(answerable)),
        "citation_precision": ratio(
            sum(1 for r in cited if set(r["cited_sources"]) & set(r["expected_sources"])), len(cited)
        ),
        "answer_accuracy": ratio(sum(1 for r in answerable if r["keywords_ok"]), len(answerable)),
        "multihop_accuracy": ratio(
            sum(1 for r in answerable if r["type"] == "multihop" and r["keywords_ok"]),
            sum(1 for r in answerable if r["type"] == "multihop"),
        ),
        "idk_accuracy": ratio(sum(1 for r in unanswerable if r["idk"]), len(unanswerable)),
        "false_idk_rate": ratio(sum(1 for r in answerable if r["idk"]), len(answerable)),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else 0.0,
    }


async def evaluate(provider: str, *, delay: float = 0.0, ask_fn=None) -> dict[str, Any]:
    ask_fn = ask_fn or (lambda q: ask(q, provider))
    questions = load_questions(EVAL_DIR / "questions.jsonl")
    chunks = await index_corpus()
    rows: list[dict[str, Any]] = []
    for q in questions:
        t0 = time.perf_counter()
        out = await ask_fn(q["question"])
        latency = (time.perf_counter() - t0) * 1000
        answer = str(out.get("answer", ""))
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": answer,
            "provider": out.get("provider"),
            "confidence": out.get("confidence"),
            "cited_sources": cited_sources(out),
            "expected_sources": q["expected_sources"],
            "keywords_ok": all(k.lower() in answer.lower() for k in q["expected_keywords"]),
            "idk": is_idk(out),
            "latency_ms": round(latency, 1),
        })
        if delay:
            await asyncio.sleep(delay)  # stay under free-tier rate limits
    return {"provider": provider, "chunks_indexed": chunks, "metrics": score(rows), "rows": rows}


def print_report(report: dict[str, Any]) -> None:
    print(f"\nprovider={report['provider']}  chunks={report['chunks_indexed']}")
    for k, v in report["metrics"].items():
        print(f"  {k:<20} {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="stub")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--label", default=None, help="save to eval/results/<label>.json")
    args = parser.parse_args()

    report = asyncio.run(evaluate(args.provider, delay=args.delay))
    print_report(report)
    if args.label:
        out = EVAL_DIR / "results" / f"{args.label}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
