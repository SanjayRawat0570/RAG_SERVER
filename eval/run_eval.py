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
    python eval/run_eval.py --few-shot off        # disable few-shot examples
    python eval/run_eval.py --mode agentic        # tool-calling agent instead of fast mode
    python eval/run_eval.py --compare-fewshot --provider gemini --model gemini-3.1-flash-lite --delay 7
        # runs with and without examples -> eval/results/fewshot_comparison_<provider>.json
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

from app.api.pipelines import (  # noqa: E402
    build_agentic_workflow,
    build_ask_workflow,
    build_index_workflow,
)
from app.engine.executor import WorkflowExecutor  # noqa: E402
from app.rag.tenants.ai_settings import effective_examples  # noqa: E402

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


async def ask(question: str, provider: str, tenant: str = TENANT,
              few_shot: bool | None = None, model: str | None = None,
              mode: str = "fast") -> dict[str, Any]:
    examples = [e.model_dump() for e in effective_examples(tenant, few_shot)]
    result = await WorkflowExecutor(
        (build_agentic_workflow if mode == "agentic" else build_ask_workflow)(provider, model)
    ).run(
        {"tenant": tenant, "question": question, "few_shot": examples}
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
        # Answers that came from the extractive fallback because the provider failed.
        "fallback_rate": ratio(sum(1 for r in rows if r["provider"] == "extractive-fallback"), len(rows)),
        "latency_ms_mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "latency_ms_p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1) if latencies else 0.0,
    }


async def evaluate(provider: str, *, delay: float = 0.0, few_shot: bool | None = None,
                   model: str | None = None, mode: str = "fast", ask_fn=None) -> dict[str, Any]:
    ask_fn = ask_fn or (lambda q: ask(q, provider, few_shot=few_shot, model=model, mode=mode))
    consecutive_fallbacks = 0
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
        # A real provider that keeps failing (quota, network) would silently turn the
        # run into an extractive-stub run — stop instead of saving misleading numbers.
        consecutive_fallbacks = consecutive_fallbacks + 1 if out.get("provider") == "extractive-fallback" else 0
        if provider != "stub" and consecutive_fallbacks >= 3:
            raise SystemExit(f"aborted: {provider} failed 3 times in a row (quota or network); "
                             "answers would come from the extractive fallback")
        if delay:
            await asyncio.sleep(delay)  # stay under free-tier rate limits
    return {"provider": provider, "model": model, "mode": mode, "few_shot": few_shot,
            "chunks_indexed": chunks,
            "metrics": score(rows), "rows": rows}


async def compare_fewshot(provider: str, delay: float, model: str | None = None) -> dict[str, Any]:
    """Same questions with and without few-shot examples; report the deltas."""
    without = await evaluate(provider, delay=delay, few_shot=False, model=model)
    with_ = await evaluate(provider, delay=delay, few_shot=True, model=model)
    metrics = ("citation_rate", "citation_precision", "answer_accuracy",
               "idk_accuracy", "false_idk_rate", "fallback_rate")
    return {
        "provider": provider,
        "summary": {
            m: {"without": without["metrics"][m], "with": with_["metrics"][m],
                "delta": round(with_["metrics"][m] - without["metrics"][m], 3)}
            for m in metrics
        },
        "without_examples": without,
        "with_examples": with_,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"\nprovider={report['provider']}  mode={report.get('mode', 'fast')}  "
          f"few_shot={report.get('few_shot')}  "
          f"chunks={report['chunks_indexed']}")
    for k, v in report["metrics"].items():
        print(f"  {k:<20} {v}")
    if report["provider"] != "stub" and report["metrics"]["fallback_rate"] > 0:
        print(f"  WARNING: {report['metrics']['fallback_rate']:.0%} of answers used the "
              "extractive fallback — the provider failed; these are not model results.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="stub")
    parser.add_argument("--mode", choices=["fast", "agentic"], default="fast")
    parser.add_argument("--model", default=None, help="provider model override, e.g. gemini-3.1-flash-lite")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--label", default=None, help="save to eval/results/<label>.json")
    parser.add_argument("--few-shot", choices=["on", "off"], default=None,
                        help="force few-shot examples (default: tenant setting = on)")
    parser.add_argument("--compare-fewshot", action="store_true",
                        help="run with and without few-shot examples and save the comparison")
    args = parser.parse_args()

    if args.compare_fewshot:
        report = asyncio.run(compare_fewshot(args.provider, args.delay, args.model))
        print(f"\nfew-shot comparison  provider={args.provider}")
        print(f"  {'metric':<20} {'without':>8} {'with':>8} {'delta':>8}")
        for m, v in report["summary"].items():
            print(f"  {m:<20} {v['without']:>8} {v['with']:>8} {v['delta']:>+8}")
        _save(report, args.label or f"fewshot_comparison_{args.provider}")
        return

    few_shot = None if args.few_shot is None else args.few_shot == "on"
    report = asyncio.run(evaluate(args.provider, delay=args.delay, few_shot=few_shot, model=args.model,
                                  mode=args.mode))
    print_report(report)
    if args.label:
        _save(report, args.label)


def _save(report: dict[str, Any], label: str) -> None:
    out = EVAL_DIR / "results" / f"{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
