"""HTTP load test: N concurrent users asking eval questions against a running API.

Each simulated user loops POST /api/v1/rag/ask (fast mode) with a random question
from eval/questions.jsonl until the duration ends; requests still in flight are
awaited (up to --timeout). Reports throughput, latency p50/p95/p99, error rate
(non-200 or timeout) and response-cache hit rate (``cache_hit`` in the reply).

Usage (from backend/, server already running and the tenant seeded)::

    python eval/loadtest.py --url http://localhost:8011 --tenant acme --users 50 200 500 --duration 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).resolve().parent


def _pct(vals: list[float], p: float) -> float | None:
    vals = sorted(vals)
    return round(vals[int(p * (len(vals) - 1))], 1) if vals else None


async def level(url: str, tenant: str, users: int, duration: float, timeout: float,
                questions: list[str], seed: int) -> dict:
    rng = random.Random(seed)
    lat, errors, hits, oks = [], {}, 0, 0
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits,
                                 headers={"Authorization": "Bearer dev"}) as client:
        deadline = time.perf_counter() + duration

        async def user() -> None:
            nonlocal hits, oks
            while time.perf_counter() < deadline:
                q = rng.choice(questions)
                t0 = time.perf_counter()
                try:
                    r = await client.post("/api/v1/rag/ask", json={"tenant": tenant, "question": q})
                    key = None if r.status_code == 200 else f"http_{r.status_code}"
                    if key is None:
                        oks += 1
                        hits += bool(r.json().get("cache_hit"))
                except httpx.TimeoutException:
                    key = "timeout"
                except httpx.HTTPError as exc:
                    key = type(exc).__name__
                lat.append((time.perf_counter() - t0) * 1000)
                if key:
                    errors[key] = errors.get(key, 0) + 1

        t0 = time.perf_counter()
        await asyncio.gather(*(user() for _ in range(users)))
        wall = time.perf_counter() - t0
    total = len(lat)
    return {
        "users": users, "duration_s": duration, "wall_s": round(wall, 1), "requests": total,
        "ok": oks, "errors": errors, "error_rate": round((total - oks) / total, 3) if total else None,
        "throughput_rps": round(oks / wall, 2),
        "latency_ms": {"p50": _pct(lat, .5), "p95": _pct(lat, .95), "p99": _pct(lat, .99)},
        "cache_hit_rate": round(hits / oks, 3) if oks else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--tenant", default="acme")
    p.add_argument("--users", type=int, nargs="+", default=[50, 200, 500])
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--label", default=None)
    a = p.parse_args()
    questions = [json.loads(l)["question"] for l in (EVAL_DIR / "questions.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    results = []
    for n in a.users:
        r = asyncio.run(level(a.url, a.tenant, n, a.duration, a.timeout, questions, a.seed))
        print(json.dumps(r), flush=True)
        results.append(r)
    if a.label:
        out = EVAL_DIR / "results" / f"load_{a.label}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
