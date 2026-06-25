"""Circuit breaker pattern (F7).

A breaker guards a flaky dependency (an external service node, a model
endpoint, etc.). After ``failure_threshold`` consecutive failures it *opens* and
short-circuits further calls for ``recovery_timeout`` seconds, after which it
moves to *half-open* and lets a single trial call through. A success closes it
again; a failure re-opens it.

Breakers are keyed by a ``breaker_key`` and shared process-wide, so they track
health *across* workflow runs — that's what makes them useful at scale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    key: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    state: str = CLOSED
    failures: int = 0
    opened_at: float = field(default=0.0)

    def allow(self) -> bool:
        """Whether a call should be attempted right now."""
        if self.state == OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_timeout:
                self.state = HALF_OPEN
                return True
            return False
        return True  # CLOSED or HALF_OPEN both allow a call

    def record_success(self) -> None:
        self.failures = 0
        self.state = CLOSED

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = OPEN
            self.opened_at = time.monotonic()


class CircuitBreakerRegistry:
    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(
        self, key: str, *, failure_threshold: int = 5, recovery_timeout: float = 30.0
    ) -> CircuitBreaker:
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(
                key=key,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
            self._breakers[key] = breaker
        return breaker

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._breakers.clear()
        else:
            self._breakers.pop(key, None)


# Process-wide registry.
registry = CircuitBreakerRegistry()
