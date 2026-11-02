"""Live cost ledger and abort tripwires.

Fireworks' Training API bills GPU-time, not tokens: 4xB300 for the trainer plus
1xB300 for the sampling deployment, at $15/GPU-hour. That is $1.25 per minute
whether the loop is learning or wedged, which is how an earlier run reached a
spending cap with six idle jobs holding GPUs.

So the budget is enforced, not just reported. Every phase writes a ledger line
to disk after each step, so the spend is observable from outside the process
while it runs, and the tripwires abort on the failure modes that would otherwise
burn the whole balance producing nothing:

  * a loss that is NaN/inf -- the run is dead, every further step is waste
  * a loss that has not improved on its own opening average -- training is not
    working and 200 more steps will not fix it
  * a sampler returning empty strings -- the single most expensive silent
    failure available, because a full eval is ~35 GPU-minutes and scores zero
    whether the model is bad or the sampler is broken
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ACCOUNT = "tikka-0hyait3m5ddc"
GPUS = 5
RATE_PER_GPU_HOUR = 15.0
RATE_PER_MIN = GPUS * RATE_PER_GPU_HOUR / 60.0     # $1.25/min


class Aborted(RuntimeError):
    """A tripwire fired. Stop spending."""


def read_meter(account: str = ACCOUNT, timeout: float = 15.0) -> float | None:
    """Month-to-date spend in USD from the `monthly-spend-usd` quota.

    This is the authoritative number. Estimating spend as wall-clock x an
    assumed GPU rate is how a run gets a nasty surprise: it cannot see spend
    that predates the process, it misses provisioning time before step 0, and it
    is wrong outright if the shape differs from what was assumed. Poll the meter
    and only fall back to arithmetic if the call fails.
    """
    import urllib.request

    try:
        from .trainer import api_key
        req = urllib.request.Request(
            f"https://api.fireworks.ai/v1/accounts/{account}/quotas",
            headers={"Authorization": f"Bearer {api_key()}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for q in json.loads(r.read()).get("quotas", []):
                if q["name"].rsplit("/", 1)[-1] == "monthly-spend-usd":
                    return float(q.get("usage") or 0.0)
    except Exception:  # noqa: BLE001
        return None
    return None


class Budget:
    """Spend guard driven by the real meter, with an arithmetic fallback.

    `limit_usd` caps the spend THIS SESSION adds, measured as the meter's delta
    from startup. `absolute_usd`, if given, is a hard ceiling on total
    month-to-date spend -- the number that actually corresponds to "credits
    left".
    """

    def __init__(self, limit_usd: float, ledger: Path | None = None,
                 gpus: int = GPUS, absolute_usd: float | None = None,
                 poll_seconds: float = 20.0):
        self.t0 = time.time()
        self.limit = limit_usd
        self.absolute = absolute_usd
        self.ledger = ledger
        self.rate = gpus * RATE_PER_GPU_HOUR / 60.0
        self.phase = "startup"
        self.notes: list[str] = []
        self.poll_seconds = poll_seconds
        self.start_meter = read_meter()
        self.meter = self.start_meter
        self._last_poll = time.time()
        self.metered = self.start_meter is not None

    @property
    def minutes(self) -> float:
        return (time.time() - self.t0) / 60

    def _refresh(self) -> None:
        if time.time() - self._last_poll < self.poll_seconds:
            return
        self._last_poll = time.time()
        m = read_meter()
        if m is not None:
            self.meter = m
            self.metered = True

    @property
    def usd(self) -> float:
        """Spend added by this session."""
        self._refresh()
        floor = self.minutes * self.rate
        if self.metered and self.meter is not None and self.start_meter is not None:
            # The meter lags by minutes, so a pure delta reads $0.00 while GPUs
            # bill. Take whichever is higher: a lagging meter must never read as
            # free, and once it catches up it becomes the authority.
            return max(self.meter - self.start_meter, floor)
        return floor

    def remaining(self) -> float:
        return self.limit - self.usd

    def line(self) -> str:
        src = "metered" if self.metered else "ESTIMATED"
        tot = f" total=${self.meter:.2f}" if self.meter is not None else ""
        return (f"{self.minutes:5.1f}min +${self.usd:6.2f}/${self.limit:.0f} "
                f"({src}{tot}, ${self.remaining():.2f} left)")

    def tick(self, phase: str, **extra) -> None:
        """Record a ledger entry; raise if the budget is spent."""
        self.phase = phase
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "t": round(self.minutes, 2), "usd": round(self.usd, 2),
                    "month_to_date": self.meter, "metered": self.metered,
                    "phase": phase, **extra,
                }) + "\n")
        if self.usd >= self.limit:
            raise Aborted(f"session budget exhausted: {self.line()} in {phase}")
        if (self.absolute is not None and self.meter is not None
                and self.meter >= self.absolute):
            raise Aborted(
                f"absolute spend ceiling hit: month-to-date ${self.meter:.2f} "
                f">= ${self.absolute:.2f} in {phase}"
            )

    def afford(self, usd: float) -> bool:
        return self.remaining() >= usd


class LossTripwire:
    """Abort a run whose loss is dead -- or, across epochs, going nowhere.

    Comparing a trailing mean against an opening mean DOES NOT WORK here, and
    the first real run showed why. Batches are emitted bucket by bucket, so
    consecutive steps differ in task mix, sequence length and batch size; the
    per-example loss wandered between 0.37 and 1.52 with no trend while the
    model was in fact training well. A trailing-vs-opening test on that series
    would have aborted a healthy run, which is worse than not testing at all.

    What IS comparable is the same position in a later epoch: identical
    examples, identical bucket, identical batch size, so training is the only
    variable. On the first run those matched pairs came out -47.6% on 5/5
    batches while the raw series looked flat.

    So: a non-finite loss aborts immediately (that is unambiguous), and
    stagnation is only judged once a second epoch supplies matched pairs.
    """

    def __init__(self, steps_per_epoch: int | None = None,
                 min_rel: float = 0.02, min_pairs: int = 5):
        self.steps_per_epoch = steps_per_epoch
        self.min_rel = min_rel
        self.min_pairs = min_pairs
        self.hist: list[float] = []

    def observe(self, per_example_loss: float | None) -> None:
        if per_example_loss is None:
            return
        v = float(per_example_loss)
        if v != v or v in (float("inf"), float("-inf")):
            raise Aborted(f"loss is not finite ({v}); the run is dead")
        self.hist.append(v)

        E = self.steps_per_epoch
        if not E or len(self.hist) < E + self.min_pairs:
            return
        pairs = [(self.hist[i], self.hist[i + E])
                 for i in range(len(self.hist) - E)]
        if len(pairs) < self.min_pairs:
            return
        rel = [(b - a) / a for a, b in pairs if a > 0]
        if not rel:
            return
        mean = sum(rel) / len(rel)
        if mean > -self.min_rel:
            raise Aborted(
                f"no epoch-over-epoch improvement across {len(rel)} matched "
                f"batches: mean change {mean:+.1%} (need better than "
                f"{-self.min_rel:.0%})"
            )

    def matched_report(self) -> str:
        """Human-readable matched-position summary, for the run log."""
        E = self.steps_per_epoch
        if not E or len(self.hist) <= E:
            return "no matched pairs yet"
        rel = [(self.hist[i + E] - self.hist[i]) / self.hist[i]
               for i in range(len(self.hist) - E) if self.hist[i] > 0]
        if not rel:
            return "no matched pairs yet"
        better = sum(1 for x in rel if x < 0)
        return (f"{len(rel)} matched batches, mean {sum(rel) / len(rel):+.1%}, "
                f"improved on {better}/{len(rel)}")


def check_sampler_alive(preds: list[str], probe: int = 25,
                        max_empty: float = 0.8) -> None:
    """Fail fast if the first `probe` completions are mostly empty."""
    head = preds[:probe]
    if not head:
        return
    empty = sum(1 for p in head if not p.strip())
    if empty / len(head) > max_empty:
        raise Aborted(
            f"{empty}/{len(head)} of the first completions are empty; the "
            f"sampler is broken, not the model -- a full eval would cost ~35 "
            f"GPU-minutes and score zero either way"
        )
