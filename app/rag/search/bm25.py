"""BM25 Okapi ranking — sparse keyword search (F13).

Pure-Python, dependency-free. Good for exact-term / keyword queries where dense
embeddings underperform; combined with dense retrieval it powers hybrid search
(F13/F20). A production deployment would swap this for Elasticsearch/OpenSearch
behind the same node interface.
"""
from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = corpus_tokens
        self.n = len(corpus_tokens)
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

        df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in corpus_tokens:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        # Standard BM25 idf with +1 smoothing to keep it non-negative.
        self.idf = {
            t: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n):
                freq = self.tf[i].get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / self.avgdl if self.avgdl else 0)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores
