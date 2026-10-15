"""Incremental BM25 index (P1): same results as rebuilding BM25 per query."""
from __future__ import annotations

import random

import pytest

from app.rag.search.bm25 import BM25, tokenize
from app.rag.vectorstore.base import VectorRecord
from app.rag.vectorstore.memory import InMemoryVectorStore

WORDS = ("parking garage opens badge vpn error token expired leave policy days team mobile beacon "
         "office lisbon budget travel meal limit e-4012 sku lap-14").split()


def _rebuilt(store: InMemoryVectorStore, query: str, top_k: int, namespace: str):
    """The old per-query path: BM25 over list_records, sorted, score > 0."""
    records = store.list_records(namespace)
    scores = BM25([tokenize(str(m.get("text", ""))) for _, m in records]).scores(tokenize(query))
    ranked = sorted(range(len(records)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(records[i][0], scores[i]) for i in ranked if scores[i] > 0]


@pytest.mark.parametrize("seed", range(5))
def test_matches_a_fresh_bm25_after_upserts_updates_and_deletes(seed):
    rng = random.Random(seed)
    store = InMemoryVectorStore("kw", 2)
    ids = [f"d{i}" for i in range(80)]
    for _ in range(300):
        rid = rng.choice(ids)
        if rng.random() < 0.15:
            store.delete([rid], "t")
        else:
            text = " ".join(rng.choices(WORDS, k=rng.randint(0, 25)))
            store.upsert([VectorRecord(id=rid, vector=[1.0, 0.0], metadata={"text": text})], "t")
    for _ in range(40):
        query = " ".join(rng.choices(WORDS, k=rng.randint(1, 4)))
        got = [(h.id, h.score) for h in store.keyword_search(query, 10, "t")]
        want = _rebuilt(store, query, 10, "t")
        assert [i for i, _ in got] == [i for i, _ in want], query
        assert got == pytest.approx(want), query


def test_ties_keep_insertion_order_and_zero_scores_are_dropped():
    store = InMemoryVectorStore("kw", 2)
    for rid in ("c", "a", "b"):
        store.upsert([VectorRecord(id=rid, vector=[1.0, 0.0], metadata={"text": "same words here"})], "t")
    store.upsert([VectorRecord(id="z", vector=[1.0, 0.0], metadata={"text": "unrelated"})], "t")
    assert [h.id for h in store.keyword_search("same", 10, "t")] == ["c", "a", "b"]
    assert store.keyword_search("absent", 10, "t") == []


def test_query_cost_does_not_scale_with_unrelated_chunks():
    """Only postings of the query's terms are scored."""
    store = InMemoryVectorStore("kw", 2)
    rows = [VectorRecord(id=f"n{i}", vector=[1.0, 0.0], metadata={"text": f"noise{i} filler text"})
            for i in range(5000)]
    rows.append(VectorRecord(id="hit", vector=[1.0, 0.0], metadata={"text": "garage opens at 7am"}))
    store.upsert(rows, "t")
    ns = store._namespaces["t"]
    assert len(ns.keywords.postings["garage"]) == 1
    assert [h.id for h in store.keyword_search("garage", 5, "t")] == ["hit"]
