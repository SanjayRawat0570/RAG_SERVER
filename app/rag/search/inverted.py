"""Incremental BM25 index for one vector-store namespace (F13).

Keyword search used to tokenize every chunk of the namespace and build a fresh
:class:`~app.rag.search.bm25.BM25` on every query — linear in the corpus (2.4 s
per query at 100k chunks). This index is kept up to date as chunks are upserted
and deleted, so a query only touches the postings of its own terms.

Scores are the same Okapi BM25 as ``bm25.BM25`` (same k1/b, same +1-smoothed
idf, same tokenizer), and ties keep insertion order, so rankings are identical.
"""
from __future__ import annotations

import heapq
import math
from typing import Iterable

from app.rag.search.bm25 import tokenize


class InvertedIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[str, int]] = {}   # term -> {doc id: term frequency}
        self.doc_len: dict[str, int] = {}
        self._terms: dict[str, list[str]] = {}          # doc id -> its distinct terms
        self.total_len = 0

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.doc_len:
            self.remove(doc_id)
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            self.postings.setdefault(t, {})[doc_id] = c
        self._terms[doc_id] = list(counts)
        self.doc_len[doc_id] = len(tokens)
        self.total_len += len(tokens)

    def remove(self, doc_id: str) -> None:
        if doc_id not in self.doc_len:
            return
        for t in self._terms.pop(doc_id):
            plist = self.postings[t]
            plist.pop(doc_id, None)
            if not plist:
                del self.postings[t]
        self.total_len -= self.doc_len.pop(doc_id)

    def search(self, query: str | Iterable[str], top_k: int, order: dict[str, int]) -> list[tuple[str, float]]:
        """Top *top_k* (doc id, score) with score > 0; *order* breaks ties (insertion position)."""
        n = len(self.doc_len)
        if not n:
            return []
        terms = tokenize(query) if isinstance(query, str) else list(query)
        avgdl = self.total_len / n
        scores: dict[str, float] = {}
        for term in terms:  # repeated query terms count again, as in BM25.scores
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc_id, freq in plist.items():
                dl = self.doc_len[doc_id] / avgdl if avgdl else 0
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl))
        best = heapq.nsmallest(top_k, ((-s, order[d], d) for d, s in scores.items() if s > 0))
        return [(d, -neg) for neg, _, d in best]
