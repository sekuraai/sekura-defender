"""Failure policy for the river client.

CORRECTED against the SKILL.md bundled with river-client 0.9.1, which documents
three things the public docs do not and which invalidated the first version of
this module:

  1. `CapacityError` is NEVER RAISED by the client. Capacity squeezes arrive as
     `RiverConnectionError` carrying "Server capacity exceeded". Catching
     CapacityError and treating RiverConnectionError as a bounded connection
     retry meant a capacity squeeze -- the expected steady-state failure -- would
     have killed a run after five attempts instead of backing off.

  2. `SessionHeartbeatError` is a SUBCLASS of `RiverConnectionError`. An
     `except RiverConnectionError` clause listed first silently swallows it, so
     a lost session gets retried in place instead of rebuilt. Order matters.

  3. `RiverTimeoutError` carries `.request_id` and the server-side future STAYS
     RETRIEVABLE. If you still hold the `PendingOp`, calling `.result()` again
     keeps waiting -- far cheaper than replaying from a checkpoint, which is what
     the first version did unconditionally.

The one rule that survives unchanged: never blindly re-SUBMIT a
`forward_backward` after a timeout. The gradient may already be accumulated
server-side, so re-submitting can double-apply it. Re-awaiting the same pending
operation is safe; re-issuing it is not.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

try:                                            # keep importable without the SDK
    import river_client as _river
    AuthenticationError = _river.AuthenticationError
    CapacityError = _river.CapacityError
    ModelNotFoundError = _river.ModelNotFoundError
    RiverConnectionError = _river.RiverConnectionError
    RiverTimeoutError = _river.RiverTimeoutError
    SessionHeartbeatError = _river.SessionHeartbeatError
    RiverError = _river.RiverError
except Exception:                               # pragma: no cover
    class _E(Exception):
        pass
    AuthenticationError = CapacityError = ModelNotFoundError = _E
    RiverConnectionError = RiverTimeoutError = SessionHeartbeatError = RiverError = _E

FATAL = (AuthenticationError, ModelNotFoundError)

# How a capacity squeeze actually presents.
CAPACITY_RE = re.compile(r"capacity|resource exhausted|too many requests", re.I)


def is_capacity(exc: BaseException) -> bool:
    return isinstance(exc, CapacityError) or (
        isinstance(exc, RiverConnectionError) and bool(CAPACITY_RE.search(str(exc))))


def needs_session_rebuild(exc: BaseException) -> bool:
    """A lost session cannot be retried in place. Checked BEFORE the generic
    connection branch, since it is a subclass."""
    return isinstance(exc, SessionHeartbeatError)


@dataclass
class Backoff:
    base: float = 10.0
    cap: float = 300.0
    factor: float = 2.0
    attempts: int = field(default=0, init=False)

    def reset(self) -> None:
        self.attempts = 0

    def next_delay(self) -> float:
        d = min(self.cap, self.base * (self.factor ** self.attempts))
        self.attempts += 1
        return d * (0.5 + random.random() / 2)


class SessionLost(RuntimeError):
    """Raised so the caller rebuilds the session and resumes from a checkpoint."""


def call_with_retry(fn: Callable, *, backoff: Backoff | None = None,
                    sleep: Callable[[float], None] = time.sleep,
                    max_connection_retries: int = 8):
    """Run `fn`, absorbing capacity squeezes and transient connection failures.

    Capacity retries indefinitely (river publishes no rate limits, so it is the
    expected steady-state failure). Plain connection errors retry a bounded
    number of times. Session loss and timeouts are re-raised: the first needs a
    session rebuild, the second needs the caller's PendingOp.
    """
    bo = backoff or Backoff()
    conn = 0
    while True:
        try:
            out = fn()
            bo.reset()
            return out
        except FATAL:
            raise
        except RiverTimeoutError:
            raise                                   # caller re-awaits or resumes
        except BaseException as exc:                # noqa: BLE001
            if needs_session_rebuild(exc):
                raise SessionLost(str(exc)) from exc
            if is_capacity(exc):
                sleep(bo.next_delay())              # unbounded: capacity is normal
                continue
            if isinstance(exc, RiverConnectionError):
                conn += 1
                if conn > max_connection_retries:
                    raise
                sleep(bo.next_delay())
                continue
            raise


def await_pending(pending, *, retries: int = 3,
                  sleep: Callable[[float], None] = time.sleep,
                  backoff: Backoff | None = None):
    """Wait on a PendingOp, re-awaiting on timeout rather than re-submitting.

    The server-side future survives a client-side timeout, so `.result()` can be
    called again. Re-issuing the operation instead could double-apply a gradient
    that was already accumulated.
    """
    bo = backoff or Backoff(base=5.0, cap=60.0)
    for attempt in range(retries + 1):
        try:
            return pending.result()
        except RiverTimeoutError:
            if attempt == retries:
                raise
            sleep(bo.next_delay())


# Back-compat alias: the loop and trainer were written against the old name.
call_with_capacity_retry = call_with_retry
