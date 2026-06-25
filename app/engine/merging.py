"""Merge strategies for combining outputs of multiple branches (F4).

Each strategy takes a list of upstream values and returns a merged value.
Strategies are referenced by name from a merge node's config.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable


def _flatten(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for v in values:
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return out


def concat(values: list[Any], config: dict[str, Any]) -> list[Any]:
    """Append all results into a single list (lists are flattened one level)."""
    return _flatten(values)


def voting(values: list[Any], config: dict[str, Any]) -> Any:
    """Return the most common value (majority vote)."""
    items = _flatten(values)
    if not items:
        return None
    hashable = [_hashable(i) for i in items]
    winner, _ = Counter(hashable).most_common(1)[0]
    # Map back to the first original item matching the winning key.
    for orig, key in zip(items, hashable):
        if key == winner:
            return orig
    return None


def ranking(values: list[Any], config: dict[str, Any]) -> list[Any]:
    """Sort items by a numeric ``score_key`` (desc) and keep ``top_n``."""
    score_key = config.get("score_key", "score")
    top_n = config.get("top_n")
    reverse = not config.get("ascending", False)
    items = _flatten(values)

    def _score(item: Any) -> float:
        if isinstance(item, dict):
            return float(item.get(score_key, 0) or 0)
        return 0.0

    ranked = sorted(items, key=_score, reverse=reverse)
    return ranked[:top_n] if top_n else ranked


def weighted(values: list[Any], config: dict[str, Any]) -> list[Any]:
    """Combine a per-branch ``score`` with a per-branch ``weight``.

    ``weights`` is a list aligned with the incoming branch order. Each item is
    expected to be a dict; a ``weighted_score`` field is added.
    """
    weights = config.get("weights") or []
    score_key = config.get("score_key", "score")
    out: list[Any] = []
    for idx, branch in enumerate(values):
        w = weights[idx] if idx < len(weights) else 1.0
        for item in (branch if isinstance(branch, list) else [branch]):
            if isinstance(item, dict):
                merged = dict(item)
                merged["weighted_score"] = float(item.get(score_key, 0) or 0) * w
                out.append(merged)
            else:
                out.append({"value": item, "weighted_score": w})
    out.sort(key=lambda i: i["weighted_score"], reverse=True)
    top_n = config.get("top_n")
    return out[:top_n] if top_n else out


def dedup(values: list[Any], config: dict[str, Any]) -> list[Any]:
    """Remove duplicates while preserving order. Optional ``key`` for dicts."""
    key = config.get("key")
    items = _flatten(values)
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        marker = _hashable(item.get(key)) if (key and isinstance(item, dict)) else _hashable(item)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def rrf(values: list[Any], config: dict[str, Any]) -> list[Any]:
    """Reciprocal Rank Fusion — combine ranked hit lists for hybrid search (F13/F20).

    Each branch is a ranked list of hits (dicts with an ``id``). A document's
    fused score is ``sum(1 / (k + rank))`` across the lists it appears in, which
    blends dense and sparse rankings without needing comparable raw scores.
    """
    k = float(config.get("k", 60))
    id_field = config.get("id_field", "id")
    top_n = config.get("top_n")

    fused: dict[Any, float] = {}
    meta: dict[Any, Any] = {}
    for branch in values:
        if not isinstance(branch, list):
            continue
        for rank, hit in enumerate(branch):
            hid = hit.get(id_field) if isinstance(hit, dict) else hit
            fused[hid] = fused.get(hid, 0.0) + 1.0 / (k + rank + 1)
            if isinstance(hit, dict) and hid not in meta:
                meta[hid] = hit.get("metadata", {})

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    out = [{"id": hid, "score": score, "metadata": meta.get(hid, {})} for hid, score in ranked]
    return out[:top_n] if top_n else out


def consensus(values: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    """Require agreement of at least ``threshold`` fraction on a single value."""
    threshold = float(config.get("threshold", 0.5))
    items = _flatten(values)
    if not items:
        return {"agreed": False, "value": None, "support": 0.0}
    hashable = [_hashable(i) for i in items]
    winner, count = Counter(hashable).most_common(1)[0]
    support = count / len(items)
    value = next((o for o, k in zip(items, hashable) if k == winner), None)
    return {"agreed": support >= threshold, "value": value, "support": support}


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def answer_merge(values: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple LLM text answers: find common ground, note differences, synthesize.

    Each value may be a string or a dict with an ``answer`` / ``text`` key.
    Returns a comprehensive answer plus a structured comparison report.
    """
    import re as _re

    answers: list[str] = []
    sources: list[str] = []
    for i, v in enumerate(values):
        if isinstance(v, dict):
            text = str(v.get("answer") or v.get("text") or "").strip()
            sources.append(str(v.get("provider") or v.get("source") or f"source_{i}"))
        else:
            text = str(v).strip()
            sources.append(f"source_{i}")
        if text:
            answers.append(text)

    if not answers:
        return {"answer": "", "common_ground": [], "differences": [], "source_count": 0}
    if len(answers) == 1:
        return {
            "answer": answers[0], "common_ground": [],
            "differences": [], "source_count": 1, "sources": sources,
        }

    def _sents(text: str) -> set[str]:
        return {s.strip().lower().rstrip(".!?")
                for s in _re.split(r"[.!?]+", text) if len(s.strip()) > 5}

    sent_sets = [_sents(a) for a in answers]

    # Common ground — sentences present in ALL answers
    common: set[str] = sent_sets[0].copy()
    for ss in sent_sets[1:]:
        common &= ss
    common_ground = sorted(common)

    # Differences — sentences unique to exactly one answer
    differences: list[dict[str, Any]] = []
    for i, (ss, src) in enumerate(zip(sent_sets, sources)):
        others: set[str] = set().union(*(s for j, s in enumerate(sent_sets) if j != i))
        unique = sorted(ss - others)
        if unique:
            differences.append({"source": src, "unique_points": unique[:3]})

    # Comprehensive answer: deduplicated sentences preserving source order
    seen: set[str] = set()
    all_sents: list[str] = []
    for text in answers:
        for s in _re.split(r"(?<=[.!?])\s+", text):
            norm = s.strip().lower().rstrip(".!?")
            if s.strip() and norm not in seen:
                seen.add(norm)
                all_sents.append(s.strip())

    max_s = int(config.get("max_sentences", 12))
    return {
        "answer": " ".join(all_sents[:max_s]),
        "common_ground": common_ground,
        "differences": differences,
        "source_count": len(answers),
        "sources": sources,
    }


def narrative_order(values: list[Any], config: dict[str, Any]) -> list[Any]:
    """Sort document chunks into natural reading order.

    Reads ``chunk_index`` → ``position`` → ``sequence`` from each item's
    ``metadata`` dict (same keys written by the chunker). Chunks without
    these keys sort to the front.
    """
    items = _flatten(values)

    def _key(item: Any) -> tuple:
        if isinstance(item, dict):
            meta = item.get("metadata", {})
            return (
                int(meta.get("chunk_index", meta.get("index", 0)) or 0),
                int(meta.get("position", 0) or 0),
                int(meta.get("sequence", 0) or 0),
                str(meta.get("heading", "") or ""),
            )
        return (0, 0, 0, "")

    return sorted(items, key=_key)


STRATEGIES: dict[str, Callable[[list[Any], dict[str, Any]], Any]] = {
    "concat": concat,
    "voting": voting,
    "ranking": ranking,
    "weighted": weighted,
    "dedup": dedup,
    "consensus": consensus,
    "rrf": rrf,
    "answer_merge": answer_merge,
    "narrative_order": narrative_order,
}


def merge(strategy: str, values: list[Any], config: dict[str, Any]) -> Any:
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown merge strategy {strategy!r}. Available: {sorted(STRATEGIES)}"
        )
    return STRATEGIES[strategy](values, config)
