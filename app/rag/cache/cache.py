"""In-process TTL + LRU cache with a named registry (F17).

Backs the query/document/LLM-response caches. Process-wide named instances let
different concerns (e.g. ``llm-response``) keep separate stats and policies. In a
distributed deployment these are the seams where Redis/Memcached slot in.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

_MISS = object()


class TTLCache:
    def __init__(self, maxsize: int = 1024, ttl: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        item = self._data.get(key, _MISS)
        if item is _MISS:
            self.misses += 1
            return _MISS
        ts, value = item  # type: ignore[misc]
        if self.ttl and (time.monotonic() - ts) > self.ttl:
            del self._data[key]
            self.misses += 1
            return _MISS
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


MISS = _MISS
_caches: dict[str, TTLCache] = {}


def get_cache(name: str, maxsize: int = 1024, ttl: float = 300.0) -> TTLCache:
    cache = _caches.get(name)
    if cache is None:
        cache = TTLCache(maxsize=maxsize, ttl=ttl)
        _caches[name] = cache
    return cache


def cache_stats() -> dict[str, Any]:
    return {name: cache.stats() for name, cache in _caches.items()}


def reset_caches(name: str | None = None) -> None:
    if name is None:
        _caches.clear()
    else:
        _caches.pop(name, None)
