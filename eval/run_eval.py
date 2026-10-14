"""RAG evaluation harness (v2): per-category quality and retrieval metrics.

Indexes ``eval/corpus/*.md`` into an isolated tenant, runs every question in
``eval/questions.jsonl`` through the same workflow the API uses, and scores each
answer. Follow-up questions replay their ``history`` into conversation memory
and go through the real follow-up rewriter first.

Question categories: factual, multihop, comparison, unanswerable, exact_id,
followup. Gold labels are section-level: ``evidence`` lists ``file#Heading``
units; ``expected_keywords`` are answer keywords (``a|b`` = either).

Metrics (answerable categories unless noted):

* recall_at_5     — share of evidence sections in the top 5 of the final
                    retrieval ranking (after fusion and reranking)
* mrr             — reciprocal rank of the first evidence section (0 if none in the ranking)
* context_recall  — share of evidence sections in the context sent to the LLM
* answer_accuracy — answer contains every expected keyword group
* citation_rate   — answer cites >= 1 source
* citation_precision — of cited answers, share citing an evidence document
* faithfulness    — share of answer sentences whose content words are (>= 75%)
                    present in the chunks the answer cites. A lexical proxy:
                    deterministic, offline, blind to paraphrase.
* idk_accuracy    — unanswerable questions answered "I don't know" (unanswerable only)
* false_idk_rate  — answerable questions wrongly answered "I don't know"
* latency_ms p50 / p95 — end to end per question, including the follow-up rewrite

Usage (from backend/)::

    python eval/run_eval.py                              # stub provider, deterministic
    python eval/run_eval.py --label v2_baseline_stub     # save eval/results/<label>.json
    python eval/run_eval.py --provider gemini --model gemini-3.1-flash-lite --delay 4
    python eval/run_eval.py --mode agentic               # tool-calling agent instead of fast mode
    python eval/run_eval.py --category exact_id          # one category only
    python eval/run_eval.py --compare-fewshot --provider gemini --delay 7
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
from app.rag.cache import reset_caches  # noqa: E402
from app.rag.memory import service as memory  # noqa: E402
from app.rag.memory import store as memory_store  # noqa: E402
from app.rag.tenants.ai_settings import effective_examples  # noqa: E402

TENANT = "eval"
EVAL_USER = "eval-user"
CATEGORIES = ("factual", "multihop", "comparison", "unanswerable", "exact_id", "followup")
SUPPORT_THRESHOLD = 0.75

_MARKER = re.compile(r"\[(\d+)\]")
_IDK = re.compile(
    r"(i\s+don'?t\s+know|i\s+do\s+not\s+know|not\s+(?:contained|mentioned|available|"
    r"specified|stated|provided)\s+in|cannot\s+(?:answer|find)|no\s+information|i'?m\s+not\s+sure)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,:\-@_/][a-z0-9]+)*")
_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those there their they them he she his her we our you your i me my which who whom
whose what when where why how do does did done has have had can could should would may might must will
shall not no than then so such also only per each every any all both more most other into over under up
about after before during while within without new""".split())


# ── Corpus and labels ──────────────────────────────────────────────────────────

def load_questions(path: Path = EVAL_DIR / "questions.jsonl") -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unit_of(meta: dict[str, Any]) -> str:
    return f"{meta.get('source')}#{meta.get('heading')}"


def expected_sources(q: dict[str, Any]) -> list[str]:
    return sorted({u.split("#", 1)[0] for u in q.get("evidence", [])})


async def index_corpus(tenant: str = TENANT) -> tuple[int, dict[str, str]]:
    """Index the corpus; returns (chunks, {unit: section text}) for label validation."""
    total, units = 0, {}
    # Gold documents plus the generated hard negatives in corpus/distractors/.
    for doc in sorted((EVAL_DIR / "corpus").rglob("*.md")):
        result = await WorkflowExecutor(build_index_workflow()).run(
            {"tenant": tenant, "text": doc.read_text(encoding="utf-8"), "filename": doc.name}
        )
        total += result.outputs["out"]["upserted"]
        for r in result.results:
            if r.node_id == "chunk":
                for c in r.output:
                    u = unit_of({"source": doc.name, "heading": c["metadata"].get("heading")})
                    units[u] = units.get(u, "") + "\n" + c["text"]
    return total, units


def validate_labels(questions: list[dict[str, Any]], units: dict[str, str]) -> None:
    """Every evidence unit must exist and contain each expected keyword group."""
    problems = []
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        problems.append("duplicate question ids")
    for q in questions:
        if q["category"] not in CATEGORIES:
            problems.append(f"{q['id']}: unknown category {q['category']!r}")
        missing = [u for u in q.get("evidence", []) if u not in units]
        if missing:
            problems.append(f"{q['id']}: evidence not in index: {missing}")
            continue
        text = " ".join(units[u] for u in q.get("evidence", [])).lower()
        for group in q.get("expected_keywords", []):
            if not any(alt.lower() in text for alt in group.split("|")):
                problems.append(f"{q['id']}: keyword {group!r} not in its evidence")
    if problems:
        raise SystemExit("invalid eval labels:\n  " + "\n  ".join(problems))


# ── Running one question ───────────────────────────────────────────────────────

async def run_question(q: dict[str, Any], provider: str, *, mode: str = "fast",
                       few_shot: bool | None = None, model: str | None = None,
                       tenant: str = TENANT) -> dict[str, Any]:
    """Ask one question the way the API does; returns the answer and node outputs."""
    question = q["question"]
    standalone = question
    if q.get("history"):
        conv = memory_store.create(tenant, EVAL_USER)
        for turn in q["history"]:
            memory_store.add_turn(conv, memory_store.Turn(
                question=turn["question"], standalone_question=turn["question"], answer=turn["answer"]))
        ctx = await memory.before(tenant=tenant, user_id=EVAL_USER, conversation_id=conv.id,
                                  question=question, provider=provider, model=model)
        standalone = ctx.standalone
        memory_store.delete(tenant, EVAL_USER, conv.id)

    examples = [e.model_dump() for e in effective_examples(tenant, few_shot)]
    build = build_agentic_workflow if mode == "agentic" else build_ask_workflow
    result = await WorkflowExecutor(build(provider, model)).run(
        {"tenant": tenant, "question": standalone, "few_shot": examples})
    nodes = {r.node_id: r.output for r in result.results}
    return {"out": result.outputs.get("out", {}) or {}, "nodes": nodes, "standalone": standalone}


def ranking(nodes: dict[str, Any]) -> list[dict[str, Any]]:
    """Final retrieval ranking: reranked hits, then the rest of the fused list (fast),
    or the agent's evidence in order (agentic)."""
    if isinstance(nodes.get("fuse"), list):
        reranked = nodes.get("rerank") if isinstance(nodes.get("rerank"), list) else []
        seen = {h["id"] for h in reranked}
        return reranked + [h for h in nodes["fuse"] if h["id"] not in seen]
    agent = nodes.get("agent") or {}
    return list((agent.get("output") or {}).get("evidence") or [])


# ── Scoring ────────────────────────────────────────────────────────────────────

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


def keywords_ok(answer: str, groups: list[str]) -> bool:
    a = answer.lower()
    return all(any(alt.lower() in a for alt in g.split("|")) for g in groups)


def _content(text: str) -> set[str]:
    return {t.rstrip(".,:").removesuffix("s") for t in _TOKEN.findall(text.lower()) if t not in _STOP}


def faithfulness(answer: str, cited_texts: list[str]) -> tuple[float | None, list[str]]:
    """Share of answer sentences supported by the cited chunks (lexical proxy)."""
    claims = [s.strip() for s in _SENTENCE.split(_MARKER.sub("", answer))
              if s.strip() and not _IDK.search(s) and _content(s)]
    if not claims:
        return None, []
    support = _content(" ".join(cited_texts))
    unsupported = [c for c in claims
                   if len(_content(c) & support) / len(_content(c)) < SUPPORT_THRESHOLD]
    return round(1 - len(unsupported) / len(claims), 3), unsupported


def score_row(q: dict[str, Any], run: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    out, nodes = run["out"], run["nodes"]
    answer = str(out.get("answer", ""))
    evidence = set(q.get("evidence", []))
    ranked_units = [unit_of(h.get("metadata", {})) for h in ranking(nodes)]
    docs = (nodes.get("augment") or {}).get("documents") or []
    context_units = {unit_of(d) for d in docs}
    markers = set(out.get("cited_markers") or [])
    cited_texts = [d.get("text", "") for d in docs if d.get("marker") in markers]
    idk = is_idk(out)
    faith, unsupported = (None, []) if idk else faithfulness(answer, cited_texts)
    first = next((i for i, u in enumerate(ranked_units, 1) if u in evidence), None)
    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "standalone_question": run["standalone"],
        "answer": answer,
        "provider": out.get("provider"),
        "confidence": out.get("confidence"),
        "cited_sources": cited_sources(out),
        "expected_sources": expected_sources(q),
        "recall_at_5": round(len(evidence & set(ranked_units[:5])) / len(evidence), 3) if evidence else None,
        "mrr": round(1 / first, 3) if (evidence and first) else (0.0 if evidence else None),
        "context_recall": round(len(evidence & context_units) / len(evidence), 3) if evidence else None,
        "keywords_ok": keywords_ok(answer, q.get("expected_keywords", [])),
        "faithfulness": faith,
        "unsupported_claims": unsupported,
        "idk": idk,
        "latency_ms": round(latency_ms, 1),
    }


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def _pct(sorted_vals: list[float], p: float) -> float:
    return round(sorted_vals[int(p * (len(sorted_vals) - 1))], 1) if sorted_vals else 0.0


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    cited = [r for r in answerable if r["cited_sources"]]
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "questions": len(rows),
        "recall_at_5": _mean([r["recall_at_5"] for r in answerable]),
        "mrr": _mean([r["mrr"] for r in answerable]),
        "context_recall": _mean([r["context_recall"] for r in answerable]),
        "answer_accuracy": _ratio(sum(r["keywords_ok"] for r in answerable), len(answerable)),
        "citation_rate": _ratio(len(cited), len(answerable)),
        "citation_precision": _ratio(
            sum(1 for r in cited if set(r["cited_sources"]) & set(r["expected_sources"])), len(cited)),
        "faithfulness": _mean([r["faithfulness"] for r in rows]),
        "idk_accuracy": _ratio(sum(r["idk"] for r in unanswerable), len(unanswerable)),
        "false_idk_rate": _ratio(sum(r["idk"] for r in answerable), len(answerable)),
        # Answers from the extractive fallback because the provider failed.
        "fallback_rate": _ratio(sum(1 for r in rows if r["provider"] == "extractive-fallback"), len(rows)),
        "latency_ms_p50": _pct(latencies, 0.50),
        "latency_ms_p95": _pct(latencies, 0.95),
    }


def score_by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c: score([r for r in rows if r["category"] == c])
            for c in CATEGORIES if any(r["category"] == c for r in rows)}


# ── Runs ───────────────────────────────────────────────────────────────────────

OUTAGE_BACKOFF_S = (15, 30, 60)


def _provider_outage(out: dict[str, Any]) -> bool:
    """The answer fell back to extractive because the provider call itself failed."""
    if out.get("provider") != "extractive-fallback":
        return False
    errors = (out.get("structured") or {}).get("errors") or []
    return any("provider error" in str(e) or "unavailable" in str(e).lower() for e in errors)


async def evaluate(provider: str, *, delay: float = 0.0, few_shot: bool | None = None,
                   model: str | None = None, mode: str = "fast", category: str | None = None,
                   limit: int | None = None, run_fn=None) -> dict[str, Any]:
    run_fn = run_fn or (lambda q: run_question(q, provider, mode=mode, few_shot=few_shot, model=model))
    questions = load_questions()
    chunks, units = await index_corpus()
    validate_labels(questions, units)
    if category:
        questions = [q for q in questions if q["category"] == category]
    questions = questions[:limit] if limit else questions

    rows: list[dict[str, Any]] = []
    consecutive_fallbacks = 0
    outage_retries = 0
    for q in questions:
        for attempt in range(len(OUTAGE_BACKOFF_S) + 1):
            t0 = time.perf_counter()
            run = await run_fn(q)
            latency = (time.perf_counter() - t0) * 1000
            if provider == "stub" or not _provider_outage(run["out"]) or attempt == len(OUTAGE_BACKOFF_S):
                break
            # A provider outage (503, timeout) says nothing about answer quality: wait
            # and ask again. The failed answer was cached, so drop the response cache.
            outage_retries += 1
            reset_caches("llm-response")
            await asyncio.sleep(OUTAGE_BACKOFF_S[attempt])
        rows.append(score_row(q, run, latency))
        # A real provider that keeps failing (quota, network) would silently turn the
        # run into an extractive-stub run — stop instead of saving misleading numbers.
        consecutive_fallbacks = consecutive_fallbacks + 1 if rows[-1]["provider"] == "extractive-fallback" else 0
        if provider != "stub" and consecutive_fallbacks >= 3:
            raise SystemExit(f"aborted after {len(rows)} questions: {provider} failed 3 times in a row "
                             "(quota or network); answers would come from the extractive fallback")
        if delay:
            await asyncio.sleep(delay)  # stay under free-tier rate limits
    return {"provider": provider, "model": model, "mode": mode, "few_shot": few_shot,
            "chunks_indexed": chunks, "outage_retries": outage_retries, "metrics": score(rows),
            "by_category": score_by_category(rows), "rows": rows}


async def compare_fewshot(provider: str, delay: float, model: str | None = None) -> dict[str, Any]:
    """Same questions with and without few-shot examples; report the deltas."""
    without = await evaluate(provider, delay=delay, few_shot=False, model=model)
    with_ = await evaluate(provider, delay=delay, few_shot=True, model=model)
    metrics = ("citation_rate", "citation_precision", "answer_accuracy", "faithfulness",
               "idk_accuracy", "false_idk_rate", "fallback_rate")
    return {
        "provider": provider,
        "summary": {
            m: {"without": without["metrics"][m], "with": with_["metrics"][m],
                "delta": None if None in (without["metrics"][m], with_["metrics"][m])
                else round(with_["metrics"][m] - without["metrics"][m], 3)}
            for m in metrics
        },
        "without_examples": without,
        "with_examples": with_,
    }


# ── Output ─────────────────────────────────────────────────────────────────────

TABLE_METRICS = ("recall_at_5", "mrr", "context_recall", "answer_accuracy", "citation_rate",
                 "citation_precision", "faithfulness", "idk_accuracy", "false_idk_rate",
                 "latency_ms_p50", "latency_ms_p95")


def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) and v <= 1 else str(v))


def print_report(report: dict[str, Any]) -> None:
    print(f"\nprovider={report['provider']}  model={report.get('model')}  mode={report.get('mode', 'fast')}  "
          f"few_shot={report.get('few_shot')}  chunks={report['chunks_indexed']}  "
          f"questions={report['metrics']['questions']}")
    cols = ["overall", *report["by_category"]]
    print(f"  {'metric':<19}" + "".join(f"{c[:12]:>13}" for c in cols))
    for m in TABLE_METRICS:
        vals = [report["metrics"][m], *(report["by_category"][c][m] for c in cols[1:])]
        print(f"  {m:<19}" + "".join(f"{_fmt(v):>13}" for v in vals))
    print(f"  {'n':<19}" + "".join(
        f"{v:>13}" for v in [report["metrics"]["questions"],
                              *(report["by_category"][c]["questions"] for c in cols[1:])]))
    if report["provider"] != "stub" and (report["metrics"]["fallback_rate"] or 0) > 0:
        print(f"  WARNING: {report['metrics']['fallback_rate']:.0%} of answers used the "
              "extractive fallback — the provider failed; these are not model results.")


def _save(report: dict[str, Any], label: str) -> None:
    out = EVAL_DIR / "results" / f"{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {out.relative_to(EVAL_DIR.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="stub")
    parser.add_argument("--mode", choices=["fast", "agentic"], default="fast")
    parser.add_argument("--model", default=None, help="provider model override, e.g. gemini-3.1-flash-lite")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--label", default=None, help="save to eval/results/<label>.json")
    parser.add_argument("--category", choices=CATEGORIES, default=None)
    parser.add_argument("--limit", type=int, default=None)
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
            print(f"  {m:<20} {_fmt(v['without']):>8} {_fmt(v['with']):>8} {_fmt(v['delta']):>8}")
        _save(report, args.label or f"fewshot_comparison_{args.provider}")
        return

    few_shot = None if args.few_shot is None else args.few_shot == "on"
    report = asyncio.run(evaluate(args.provider, delay=args.delay, few_shot=few_shot, model=args.model,
                                  mode=args.mode, category=args.category, limit=args.limit))
    print_report(report)
    if args.label:
        _save(report, args.label)


if __name__ == "__main__":
    main()
