"""Teacher/judge access via the local Claude Code CLI.

Zero API cost: this drives the authenticated `claude` binary on subscription
auth, which is how the 258-doc extraction backlog in this repo was drained. The
constraint is WALL CLOCK, not money -- each call pays 1-2s of CLI startup, so
throughput comes from concurrency plus a disk cache, not from batching.

Reuses the seam in SocraticAI/AliceBobCindy.py rather than reimplementing it:
`--tools ""` blocks agentic tool use, `--no-session-persistence` avoids a session
file per call, and running in a scratch cwd keeps the repo's CLAUDE.md out of
every prompt.

TEACHER CHOICE MATTERS. 45% of knowledge.jsonl was extracted by
claude-haiku-4-5. Using Haiku as the teacher over Haiku-extracted claims is
self-distillation: it compounds one model's biases and caps the student at
Haiku's ceiling. Use a stronger model for the generative slices and reserve
Haiku for cheap structured ones. Cost is zero either way; only wall clock moves.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import re
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", shutil.which("claude") or "claude")
DEFAULT_TIMEOUT = float(os.environ.get("TEACHER_TIMEOUT", "300"))

# Strong model for generative slices; Haiku only where output is short/structured.
TEACHER_STRONG = os.environ.get("TEACHER_STRONG", "claude-sonnet-5")
TEACHER_CHEAP = os.environ.get("TEACHER_CHEAP", "claude-haiku-4-5")


class TeacherError(RuntimeError):
    pass


class RateLimited(TeacherError):
    """The subscription window is exhausted. Distinguished from other failures
    because the right response is to WAIT, not to move on to the next job."""


# Phrases the CLI surfaces when the subscription window is exhausted or the
# upstream is shedding load. Matched case-insensitively against stderr.
RATE_LIMIT_RE = re.compile(
    r"rate.?limit|usage limit|quota|too many requests|429|"
    r"overloaded|capacity|try again later|resets? at", re.I)


# Variable parts of an error message -- ids, paths, timings, counts -- so the
# same underlying failure collapses to one bucket instead of N unique strings.
_NOISE = [
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "<hex>"),
    (re.compile(r"\b\d+(\.\d+)?s\b"), "<dur>"),
    (re.compile(r"/[\w./-]{8,}"), "<path>"),
    (re.compile(r"\b\d{3,}\b"), "<n>"),
    (re.compile(r"\s+"), " "),
]


def normalize_error(exc: BaseException, limit: int = 140) -> str:
    """Collapse an exception to a stable bucket key.

    The first two campaigns reported only `{'TeacherError': 6092}` and
    `{'TeacherError': 66}` -- the class name and nothing else -- which made it
    impossible to say afterwards whether those were timeouts, auth failures or
    quota. The message is the diagnosis; keep it.
    """
    msg = f"{type(exc).__name__}: {exc}"
    for pat, sub in _NOISE:
        msg = pat.sub(sub, msg)
    return msg.strip()[:limit]


class Breaker:
    """Shared pause gate across the worker pool.

    The first campaign had none, and it cost the whole run: calls started
    failing around job 3,350 and the pool burned through the remaining 6,092
    jobs in seconds, every one an error, producing nothing. Failures are not
    cached, so no work was lost -- but five hours of wall clock were.

    Consecutive failures now pause every worker rather than advancing the job
    list. Backoff is long by design: a subscription window takes hours to reset,
    so retrying in seconds just converts quota into errors.
    """

    def __init__(self, trip_after: int = 8,
                 waits: tuple[float, ...] = (300, 900, 1800, 3600, 3600, 7200)):
        self.trip_after = trip_after
        self.waits = waits
        self._lock = threading.Lock()
        self._consecutive = 0
        self._trips = 0
        self._resume_at = 0.0
        self.total_paused = 0.0

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def record_failure(self, exc: BaseException) -> None:
        hard = isinstance(exc, RateLimited)
        with self._lock:
            self._consecutive += 1
            if hard or self._consecutive >= self.trip_after:
                wait = self.waits[min(self._trips, len(self.waits) - 1)]
                self._trips += 1
                self._consecutive = 0
                self._resume_at = max(self._resume_at, time.time() + wait)

    def wait_if_open(self, sleep=time.sleep) -> float:
        """Block until the gate clears. Returns seconds actually waited."""
        waited = 0.0
        while True:
            with self._lock:
                remaining = self._resume_at - time.time()
            if remaining <= 0:
                return waited
            chunk = min(30.0, remaining)
            sleep(chunk)
            waited += chunk
            with self._lock:
                self.total_paused += chunk

    @property
    def trips(self) -> int:
        return self._trips


def cache_key(system: str, user: str, model: str, version: str = "v1") -> str:
    return hashlib.sha256(
        "\x00".join([version, model, system, user]).encode()).hexdigest()


@dataclass
class TeacherClient:
    cache_dir: Path
    model: str = TEACHER_STRONG
    timeout: float = DEFAULT_TIMEOUT
    version: str = "v1"
    _lock: threading.Lock = None

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, key: str) -> str | None:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))["content"]
            except Exception:
                return None
        return None

    def _store(self, key: str, system: str, user: str, content: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": self.model, "version": self.version,
                                   "system": system, "user": user,
                                   "content": content}, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(p)

    breaker: "Breaker | None" = None

    def ask(self, system: str, user: str, *, model: str | None = None) -> str:
        model = model or self.model
        if self.breaker is not None:
            self.breaker.wait_if_open()
        key = cache_key(system, user, model, self.version)
        hit = self.cached(key)
        if hit is not None:
            return hit
        cmd = [CLAUDE_BIN, "-p", "--output-format", "text", "--model", model,
               "--system-prompt", system, "--tools", "", "--no-session-persistence"]
        try:
            proc = subprocess.run(cmd, input=user, capture_output=True, text=True,
                                  timeout=self.timeout, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired as exc:
            raise TeacherError(f"claude-cli timed out after {self.timeout}s") from exc
        except FileNotFoundError as exc:
            raise TeacherError(
                f"claude binary not found at {CLAUDE_BIN!r}; set CLAUDE_BIN") from exc
        if proc.returncode != 0:
            err = proc.stderr.strip()
            if RATE_LIMIT_RE.search(err) or RATE_LIMIT_RE.search(proc.stdout or ""):
                raise RateLimited(f"claude-cli exit {proc.returncode}: {err[:300]}")
            raise TeacherError(f"claude-cli exit {proc.returncode}: {err[:300]}")
        out = proc.stdout.strip()
        if not out:
            raise TeacherError("claude-cli returned empty output")
        self._store(key, system, user, out)
        if self.breaker is not None:
            self.breaker.record_success()
        return out

    def available(self) -> bool:
        return bool(shutil.which(CLAUDE_BIN) or Path(CLAUDE_BIN).exists())


def map_concurrent(client: TeacherClient, jobs: Iterable[tuple],
                   fn: Callable[[TeacherClient, tuple], object],
                   *, workers: int = 6,
                   on_error: Callable[[tuple, Exception], None] | None = None
                   ) -> Iterator[object]:
    """Run `fn(client, job)` across a thread pool.

    A subprocess per call means wall clock is the constraint; 6 workers keeps the
    machine responsive while cutting a multi-hour campaign to a fraction. Results
    stream as they land, so a long run can be checkpointed by the caller.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, client, job): job for job in jobs}
        # as_completed, not submission order: on a multi-hour campaign results
        # must reach the caller (and the checkpoint file) as they land, so a
        # killed run loses minutes rather than everything since the last slow job.
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                out = fut.result()
            except Exception as exc:                    # noqa: BLE001
                if client.breaker is not None:
                    client.breaker.record_failure(exc)
                if on_error:
                    on_error(job, exc)
                continue
            yield out
