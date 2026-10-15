"""Semantic answer cache: hit rate on paraphrases, false hits on near-misses.

``eval/cache_pairs.jsonl`` holds two kinds of pairs built on eval questions:

* ``paraphrase`` — a rewording of eval question ``seed``; should reuse its answer;
* ``near_miss``  — two *different* eval questions on the same topic
  (E-4012 vs E-4021, current vs 2023 meal allowance, …); must NOT reuse.

Part 1 (similarity sweep): the cosine similarity of every pair under the
knowledge base's embedding model, and the paraphrase hit rate / near-miss
false-hit rate at each threshold.

Part 2 (end to end, at ``SEMANTIC_CACHE_THRESHOLD``): ask the seed through the
same cache + pipeline as /rag/ask, then the pair's question; report whether it
was a cache hit and whether the answer it got contains the expected keywords
of the question actually asked.

Usage (from backend/)::

    python eval/semantic_cache_bench.py                     # stub provider
    python eval/semantic_cache_bench.py --provider gemini --model gemini-3.5-flash-lite --delay 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.config import settings  # noqa: E402
from app.rag.cache import answers as answer_cache  # noqa: E402
from app.rag.embeddings import active_embedding, embed_texts  # noqa: E402
from eval.run_eval import TENANT, index_corpus, keywords_ok, load_questions, run_question  # noqa: E402

THRESHOLDS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97)


def pairs() -> list[dict]:
    qs = {q["id"]: q for q in load_questions()}
    out = []
    for line in (EVAL_DIR / "cache_pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        seed = qs[p["seed"]]
        asked = {"question": p["question"], "expected_keywords": seed["expected_keywords"]} \
            if p["type"] == "paraphrase" else qs[p["other"]]
        out.append({"type": p["type"], "seed": seed, "asked": asked})
    return out


def sweep(ps: list[dict]) -> dict:
    model, dim = active_embedding()
    sims = []
    for p in ps:
        a, b = embed_texts([p["seed"]["question"], p["asked"]["question"]], model, dim, kind="query")
        a, b = np.asarray(a), np.asarray(b)
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    same = [answer_cache.specifics(p["seed"]["question"]) == answer_cache.specifics(p["asked"]["question"])
            for p in ps]
    para = [(s, g) for s, g, p in zip(sims, same, ps) if p["type"] == "paraphrase"]
    near = [(s, g) for s, g, p in zip(sims, same, ps) if p["type"] == "near_miss"]

    def rate(rows, t, guard):
        return round(sum(s >= t and (g or not guard) for s, g in rows) / len(rows), 3)

    table = {t: {"paraphrase_hit_rate": rate(para, t, False), "near_miss_false_hit_rate": rate(near, t, False),
                 "with_specifics_guard": {"paraphrase_hit_rate": rate(para, t, True),
                                          "near_miss_false_hit_rate": rate(near, t, True)}}
             for t in THRESHOLDS}
    return {"model": model, "paraphrase_sims": sorted(round(s, 3) for s, _ in para),
            "near_miss_sims": sorted(round(s, 3) for s, _ in near), "by_threshold": table,
            "pairs": [{"type": p["type"], "seed": p["seed"]["question"], "asked": p["asked"]["question"],
                       "similarity": round(s, 4)} for p, s in zip(ps, sims)]}


def eval_collisions(threshold: float) -> list[dict]:
    """Pairs of *different* eval questions that would answer each other from the cache."""
    model, dim = active_embedding()
    qs = load_questions()
    vecs = np.asarray(embed_texts([q["question"] for q in qs], model, dim, kind="query"))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs @ vecs.T
    out = []
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            same = answer_cache.specifics(qs[i]["question"]) == answer_cache.specifics(qs[j]["question"])
            if sims[i, j] >= threshold and same:
                out.append({"a": qs[i]["id"], "b": qs[j]["id"], "similarity": round(float(sims[i, j]), 4),
                            "questions": [qs[i]["question"], qs[j]["question"]]})
    return out


async def end_to_end(ps: list[dict], provider: str, model: str | None, delay: float) -> dict:
    await index_corpus()
    rows = []
    for p in ps:
        tenant = f"{TENANT}"
        for q in (p["seed"], p["asked"]):
            standalone = q["question"]
            cached = answer_cache.lookup(tenant, standalone, provider)
            if cached is None:
                run = await run_question({"question": standalone}, provider, model=model)
                out = run["out"]
                answer_cache.store(tenant, standalone, provider, out)
            else:
                out = cached
            if delay and cached is None:
                await asyncio.sleep(delay)
        correct = keywords_ok(str(out.get("answer", "")), p["asked"]["expected_keywords"])
        uncached = correct
        if cached is not None:  # what the pipeline would have answered without the cache
            fresh = (await run_question({"question": p["asked"]["question"]}, provider, model=model))["out"]
            uncached = keywords_ok(str(fresh.get("answer", "")), p["asked"]["expected_keywords"])
            if delay:
                await asyncio.sleep(delay)
        rows.append({"type": p["type"], "asked": p["asked"]["question"], "cache_hit": cached is not None,
                     "similarity": (cached or {}).get("cache_similarity"), "correct": correct,
                     "correct_without_cache": uncached, "answer": str(out.get("answer", ""))[:160]})
        # Fresh cache per pair, so one pair's entries never answer another pair.
        from app.rag.vectorstore import reset_stores
        reset_stores(f"__semcache__{answer_cache.NAME}:{tenant}:{provider}")

    def summary(kind: str) -> dict:
        rs = [r for r in rows if r["type"] == kind]
        hits = [r for r in rs if r["cache_hit"]]
        return {"n": len(rs), "hit_rate": round(len(hits) / len(rs), 3),
                "accuracy_with_cache": round(sum(r["correct"] for r in rs) / len(rs), 3),
                "accuracy_without_cache": round(sum(r["correct_without_cache"] for r in rs) / len(rs), 3),
                "hits_made_wrong": sum(1 for r in hits if r["correct_without_cache"] and not r["correct"]),
                "hits_made_right": sum(1 for r in hits if r["correct"] and not r["correct_without_cache"])}

    return {"threshold": settings.semantic_cache_threshold, "provider": provider,
            "paraphrase": summary("paraphrase"), "near_miss": summary("near_miss"), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="stub")
    ap.add_argument("--model", default=None)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--label", default="semantic_cache_bench")
    a = ap.parse_args()
    ps = pairs()
    report = {"sweep": sweep(ps)}
    print(f"model {report['sweep']['model']}")
    print(f"  paraphrase sims: {report['sweep']['paraphrase_sims']}")
    print(f"  near-miss sims:  {report['sweep']['near_miss_sims']}")
    for t, v in report["sweep"]["by_threshold"].items():
        g = v["with_specifics_guard"]
        print(f"  threshold {t:.2f}: cosine only - paraphrase hits {v['paraphrase_hit_rate']:.3f}, "
              f"near-miss false hits {v['near_miss_false_hit_rate']:.3f} | + specifics guard - "
              f"{g['paraphrase_hit_rate']:.3f}, {g['near_miss_false_hit_rate']:.3f}")
    report["eval_collisions"] = eval_collisions(settings.semantic_cache_threshold)
    print(f"eval questions (165) that would hit another eval question's entry at "
          f"{settings.semantic_cache_threshold}: {len(report['eval_collisions'])}")
    for c in report["eval_collisions"]:
        print(f"  {c['similarity']}  {c['questions'][0]}  <->  {c['questions'][1]}")
    report["end_to_end"] = asyncio.run(end_to_end(ps, a.provider, a.model, a.delay))
    e = report["end_to_end"]
    print(f"end to end at threshold {e['threshold']} ({e['provider']}):")
    for kind in ("paraphrase", "near_miss"):
        print(f"  {kind:<10} {e[kind]}")
    out = EVAL_DIR / "results" / f"{a.label}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
