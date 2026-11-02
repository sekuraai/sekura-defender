"""Supervisor: run the teacher campaign overnight, across usage windows.

Detaches from the calling shell so it survives the session ending -- the first
campaign did not, and died with 6,092 jobs unrun. Holds the Mac awake, retries
across subscription-window resets, and stops as soon as the cache says there is
nothing left to do.

  python scripts/overnight_teacher.py --start            # detach and go
  python scripts/overnight_teacher.py --start --at 23:30 # begin at a set time
  python scripts/overnight_teacher.py --status           # progress so far
  python scripts/overnight_teacher.py --stop
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = ROOT / ".venv" / "bin" / "python"
LOG = ROOT / "data" / "overnight.log"
PIDFILE = ROOT / "data" / "overnight.pid"
OUT = ROOT / "data" / "examples" / "teacher.jsonl"

# A subscription window takes hours to reset, so retrying in minutes just turns
# quota into errors. Gaps grow when an attempt makes no progress at all.
GAP_AFTER_PROGRESS = 5 * 60
GAP_AFTER_STALL = (30 * 60, 60 * 60, 90 * 60, 120 * 60)


def _log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
        fh.flush()


CACHE_DIR = ROOT / "data" / ".teacher-cache"


def _kept() -> int:
    """Unique examples, not rows. Rows can repeat; unique work cannot."""
    if not OUT.exists():
        return 0
    seen = set()
    with open(OUT, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["datum_id"])
            except Exception:
                continue
    return len(seen)


def _cached_calls() -> int:
    return sum(1 for _ in CACHE_DIR.rglob("*.json")) if CACHE_DIR.exists() else 0


def supervise(passes: list[list[str]], max_attempts: int, workers: int) -> int:
    _log(f"supervisor start pid={os.getpid()} passes={len(passes)} workers={workers}")
    stalls = 0
    for attempt in range(1, max_attempts + 1):
        before = _kept()
        calls_before = _cached_calls()
        done_all = True
        for extra in passes:
            cmd = [str(PY), str(ROOT / "scripts" / "run_teacher.py"),
                   "--workers", str(workers), *extra]
            _log(f"attempt {attempt}: {' '.join(extra) or 'full campaign'}")
            with open(LOG, "a", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            if rc == 2:
                _log(f"attempt {attempt}: misconfigured (exit 2) -- stopping")
                return 2
            if rc != 0:
                done_all = False
        after = _kept()
        gained = after - before
        new_calls = _cached_calls() - calls_before
        _log(f"attempt {attempt}: +{gained} unique examples (total {after}), "
             f"+{new_calls} new calls")

        # An attempt that made no new CALLS did no work: everything it touched
        # came from cache, so another attempt will do exactly the same. This is
        # the stop condition that matters -- exit codes alone let a finished
        # campaign replay itself eleven times.
        if new_calls == 0 and gained == 0:
            _log(f"attempt {attempt}: no new work (all cache hits) -- campaign complete")
            return 0

        if done_all:
            _log(f"campaign complete after {attempt} attempt(s), {after} examples")
            return 0
        if gained > 0:
            stalls = 0
            gap = GAP_AFTER_PROGRESS
        else:
            gap = GAP_AFTER_STALL[min(stalls, len(GAP_AFTER_STALL) - 1)]
            stalls += 1
            _log(f"no progress -- likely a usage window; waiting {gap // 60} min")
        time.sleep(gap)
    _log(f"stopped after {max_attempts} attempts, {_kept()} examples")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--at", default="", help="HH:MM local time to begin")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--safety-first", action="store_true", default=True)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.status:
        print(f"examples: {_kept()}")
        if PIDFILE.exists():
            pid = int(PIDFILE.read_text().strip())
            alive = subprocess.call(["kill", "-0", str(pid)],
                                    stderr=subprocess.DEVNULL) == 0
            print(f"supervisor pid {pid}: {'running' if alive else 'not running'}")
        else:
            print("supervisor: not started")
        if LOG.exists():
            print("\n".join(LOG.read_text().splitlines()[-12:]))
        return 0

    if args.stop:
        if not PIDFILE.exists():
            print("not running")
            return 0
        pid = int(PIDFILE.read_text().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        PIDFILE.unlink(missing_ok=True)
        print(f"stopped {pid}")
        return 0

    if args._child:
        PIDFILE.write_text(str(os.getpid()))
        if args.at:
            hh, mm = (int(x) for x in args.at.split(":"))
            now = datetime.now()
            when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if when <= now:
                when += timedelta(days=1)
            _log(f"waiting until {when:%Y-%m-%d %H:%M}")
            time.sleep((when - now).total_seconds())
        # Safety slice first: it is small and it unblocks the two failing tests.
        passes = ([["--slice", "safety"], []] if args.safety_first else [[]])
        try:
            return supervise(passes, args.max_attempts, args.workers)
        finally:
            PIDFILE.unlink(missing_ok=True)

    if args.start:
        if PIDFILE.exists():
            print(f"already running (pid {PIDFILE.read_text().strip()})")
            return 1
        child = [str(PY), __file__, "--_child", "--workers", str(args.workers),
                 "--max-attempts", str(args.max_attempts)]
        if args.at:
            child += ["--at", args.at]
        # caffeinate -i holds off idle sleep for as long as the child lives;
        # start_new_session detaches it so it outlives this shell.
        proc = subprocess.Popen(["/usr/bin/caffeinate", "-i", *child],
                                cwd=ROOT, start_new_session=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        print(f"started pid {proc.pid}" + (f", begins at {args.at}" if args.at else ""))
        print(f"  log:    {LOG}")
        print(f"  status: python scripts/overnight_teacher.py --status")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
