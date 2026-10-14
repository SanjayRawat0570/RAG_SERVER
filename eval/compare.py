"""Compare two saved eval runs, overall and per category.

    python eval/compare.py v2_base_stub_fast v2_a0_bge_stub_fast
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
METRICS = ("recall_at_5", "mrr", "context_recall", "answer_accuracy", "citation_rate",
           "citation_precision", "faithfulness", "idk_accuracy", "false_idk_rate",
           "latency_ms_p50", "latency_ms_p95")
CATEGORIES = ("factual", "multihop", "comparison", "unanswerable", "exact_id", "followup")


def load(label: str) -> dict:
    return json.loads((RESULTS / f"{label}.json").read_text(encoding="utf-8"))


def delta(a, b) -> str:
    if a is None or b is None:
        return "-"
    d = b - a
    return f"{d:+.3f}" if abs(a) <= 1 and abs(b) <= 1 else f"{d:+.1f}"


def main(before: str, after: str) -> None:
    a, b = load(before), load(after)
    print(f"{before}  ->  {after}")
    print(f"  model {a.get('model')} -> {b.get('model')}   questions {a['metrics']['questions']} -> "
          f"{b['metrics']['questions']}   outage retries {a.get('outage_retries')} -> {b.get('outage_retries')}")
    cats = [c for c in CATEGORIES if c in a["by_category"] and c in b["by_category"]]
    print(f"  {'metric':<19}{'overall':>22}" + "".join(f"{c[:10]:>12}" for c in cats))
    for m in METRICS:
        x, y = a["metrics"][m], b["metrics"][m]
        row = f"  {m:<19}{f'{x} -> {y}':>15} {delta(x, y):>6}"
        row += "".join(f"{delta(a['by_category'][c][m], b['by_category'][c][m]):>12}" for c in cats)
        print(row)


if __name__ == "__main__":
    main(*sys.argv[1:3])
