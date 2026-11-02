"""Three-tier fallback with a circuit breaker.

Promotion is beta, so the promoted replica must never be a hard dependency:

  tier 1  the promoted streaming replica
  tier 2  an OpenAI-shaped shim over chat_complete_from_checkpoint
          (slower, non-streaming, but needs no replica at all)
  tier 3  the incumbent provider

Three consecutive failures drop a tier for a cool-off window. Recovery is
automatic rather than manual, so a transient replica outage does not leave the
pipeline permanently downgraded.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

TIERS = ("replica", "checkpoint_shim", "fallback_provider")
TRIP_AFTER = 3
COOLOFF_S = 300.0


@dataclass
class Breaker:
    trip_after: int = TRIP_AFTER
    cooloff_s: float = COOLOFF_S
    tier: int = 0
    failures: int = 0
    demoted_at: float = 0.0
    events: list[tuple[str, str]] = field(default_factory=list)
    _now: Callable[[], float] = time.monotonic

    @property
    def current(self) -> str:
        return TIERS[self.tier]

    def _maybe_recover(self) -> None:
        if self.tier > 0 and self._now() - self.demoted_at >= self.cooloff_s:
            self.tier -= 1
            self.failures = 0
            self.demoted_at = self._now()
            self.events.append(("recover", TIERS[self.tier]))

    def success(self) -> None:
        self.failures = 0
        self._maybe_recover()

    def failure(self) -> str:
        self.failures += 1
        if self.failures >= self.trip_after and self.tier < len(TIERS) - 1:
            self.tier += 1
            self.failures = 0
            self.demoted_at = self._now()
            self.events.append(("demote", TIERS[self.tier]))
        return self.current

    def call(self, handlers: dict[str, Callable]) -> object:
        """Serve one request, walking DOWN the tiers until one succeeds.

        Per-request fallthrough and persistent demotion are separate concerns.
        A single request must not be retried three times against a dead replica
        just because demotion needs three consecutive failures -- that would make
        every request during an outage pay three timeouts before falling through.
        So this walks tiers immediately, while `failure()` independently tracks
        consecutive failures for the persistent demotion that spares later
        requests from starting at the dead tier at all.
        """
        last: Exception | None = None
        start = self.tier
        for idx in range(start, len(TIERS)):
            fn = handlers.get(TIERS[idx])
            if fn is None:
                continue
            try:
                out = fn()
            except Exception as exc:                     # noqa: BLE001
                last = exc
                if idx == self.tier:
                    self.failure()                       # only the active tier demotes
                continue
            if idx == self.tier:
                self.success()
            return out
        raise RuntimeError(f"all serving tiers failed; last error: {last}")
