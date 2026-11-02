"""Run the teacher slices. Resumable, cached, and free (subscription auth).

  python scripts/run_teacher.py --limit 20          # sample, verify quality first
  python scripts/run_teacher.py --slice remediation # one slice
  python scripts/run_teacher.py                     # full campaign (~3h at 6 workers)

Wall clock, not money, is the constraint: ~5s of CLI startup per call. Results
append to data/examples/teacher.jsonl after each batch, so a killed run resumes
from the cache with no repeated calls.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekura_sft.config import BUNDLES_DIR, EXAMPLES_DIR, MANIFESTS_DIR
from sekura_sft.corpus.bundles import iter_pool
from sekura_sft.sources import intel
from sekura_sft.tasks import teacher_slices as T
from sekura_sft.teacher.client import (Breaker, RateLimited, TeacherClient,
                                       map_concurrent, normalize_error)

CACHE = Path(__file__).resolve().parent.parent / "data" / ".teacher-cache"
OUT = EXAMPLES_DIR / "teacher.jsonl"


def collect_jobs(args, rng) -> list[tuple]:
    bundles = []
    bpath = BUNDLES_DIR / "bundles.jsonl"
    if bpath.exists():
        bundles = list(iter_pool(bpath))
        rng.shuffle(bundles)
    findings = list(intel.load_findings())
    rng.shuffle(findings)

    jobs: list[tuple] = []
    want = args.slice
    if not want or want == "threat_model":
        jobs += list(T.t1_jobs(bundles[:args.bundles]))
    if not want or want == "detection":
        jobs += list(T.t2_jobs(bundles[:args.bundles * 2]))
    if not want or want == "grounded_qa":
        jobs += list(T.t3_jobs(bundles[args.bundles:args.bundles * 2]))
    if not want or want == "remediation":
        jobs += list(T.t4_jobs(findings[:args.findings]))
    if not want or want == "safety":
        jobs += list(T.t7_jobs(rng, findings))
    rng.shuffle(jobs)
    return jobs[:args.limit] if args.limit else jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--slice", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bundles", type=int, default=3000)
    ap.add_argument("--findings", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trip-after", type=int, default=8,
                    help="consecutive failures before the pool pauses")
    args = ap.parse_args()

    # Without this the pool advances through the job list on every failure. The
    # first campaign did exactly that: calls began failing around job 3,350 and
    # it consumed the remaining 6,092 jobs in seconds, all errors, all wasted.
    client = TeacherClient(cache_dir=CACHE)
    client.breaker = Breaker(trip_after=args.trip_after)
    if not client.available():
        print("claude CLI not found; set CLAUDE_BIN", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    jobs = collect_jobs(args, rng)
    from sekura_sft.teacher.client import cache_key

    def _key(kind: str, user: str) -> str:
        spec = T.SPECS[kind]
        return cache_key(f"{spec.system}\n\nYou are producing a training example. "
                         "Answer directly and completely.", user, spec.model,
                         client.version)

    cached = sum(1 for k, _, u in jobs if client.cached(_key(k, u)) is not None)
    remaining = len(jobs) - cached
    eta_h = remaining * 41 / max(1, args.workers) / 3600
    print(f"jobs: {len(jobs)}  cached: {cached}  remaining: {remaining}  "
          f"workers: {args.workers}  ETA ~{eta_h:.1f}h")
    if not jobs:
        return 0

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Never re-append an example already on disk.
    #
    # Cached calls return instantly, so a re-run replays the ENTIRE corpus in
    # seconds -- and without this it appends all of it again. Twelve supervisor
    # attempts turned 9,654 unique examples into 130,553 rows and a 1.0 GB file.
    # No quota was burned (every call was a cache hit) but the output became
    # 13.5x redundant.
    already: set[str] = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    already.add(json.loads(line)["datum_id"])
                except Exception:
                    continue
    if already:
        print(f"  {len(already)} examples already on disk; will not re-append them")
    kept = Counter()
    reused = 0
    rejected = Counter()
    errors = Counter()          # normalized message -> count
    error_samples: dict[str, str] = {}
    rate_limited = Counter()
    t0 = time.time()
    n_done = 0

    def on_error(job, exc):
        key = normalize_error(exc)
        errors[key] += 1
        if key not in error_samples:
            # One verbatim sample per bucket: the normalized key loses exactly
            # the identifiers needed to chase a specific failure.
            error_samples[key] = f"[{job[0]}] {type(exc).__name__}: {exc}"[:400]
        if isinstance(exc, RateLimited):
            rate_limited[key] += 1

    with open(OUT, "a", encoding="utf-8") as fh:
        for ex in map_concurrent(client, jobs, T.run_job, workers=args.workers,
                                 on_error=on_error):
            n_done += 1
            if ex is not None:
                kept[ex.meta["slice"]] += 1
                row = ex.to_dict()
                if row["datum_id"] not in already:
                    already.add(row["datum_id"])
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                else:
                    reused += 1
            if n_done % 50 == 0:
                el = time.time() - t0
                rate = el / n_done
                left = (len(jobs) - n_done) * rate / 3600
                n_err = sum(errors.values())
                top = errors.most_common(1)[0][0][:52] if errors else ""
                print(f"  [{n_done}/{len(jobs)}] kept={sum(kept.values())} "
                      f"rejected={n_done - sum(kept.values()) - n_err} "
                      f"errors={n_err} {rate:.1f}s/job ETA {left:.1f}h"
                      + (f"  top-err: {top}" if top else ""), flush=True)

    total_kept = sum(kept.values())
    elapsed = time.time() - t0
    paused = client.breaker.total_paused if client.breaker else 0.0
    trips = client.breaker.trips if client.breaker else 0
    if reused:
        print(f"  {reused} results already on disk, not re-written")
    print(f"\nkept {total_kept}/{len(jobs)} in {elapsed:.0f}s "
          f"({elapsed / max(1, len(jobs)):.1f}s/job)")
    if trips:
        print(f"  breaker tripped {trips}x, paused {paused / 60:.0f} min total")
    print("  by slice:", dict(kept))
    if errors:
        print(f"  errors: {sum(errors.values())} across {len(errors)} distinct causes")
        for key, count in errors.most_common(8):
            tag = " [RATE-LIMIT]" if rate_limited.get(key) else ""
            print(f"     {count:5d}x{tag} {key}")
            print(f"           e.g. {error_samples.get(key, '')[:180]}")
    print(f"  verification rejects: {len(jobs) - total_kept - sum(errors.values())}")
    (MANIFESTS_DIR / "teacher.json").write_text(json.dumps(
        {"jobs": len(jobs), "kept": total_kept, "by_slice": dict(kept),
         "errors": dict(errors.most_common(30)),
         "error_samples": {k: error_samples[k] for k, _ in errors.most_common(30)},
         "rate_limited": dict(rate_limited),
         "errors_total": sum(errors.values()), "seconds": round(elapsed),
         "breaker_trips": trips, "paused_seconds": round(paused)}, indent=2))
    print(f"\nwrote {OUT}")
    # Exit 3 means "meaningful work remains, retry after a gap"; 0 means done.
    #
    # Retrying on ANY error was wrong: three transient timeouts out of 9,784
    # jobs triggered eleven pointless supervisor attempts. Verification rejects
    # are permanent -- the teacher answered and the answer failed a gate, so
    # asking again produces the same answer from cache -- and a handful of
    # timeouts is not a reason to replay the whole campaign. Retry only when
    # enough failed to be worth another window.
    failed = sum(errors.values())
    retry_worthy = failed >= max(25, int(0.02 * len(jobs)))
    if failed:
        print(f"  {failed} error(s) = {failed / max(1, len(jobs)):.1%} of jobs; "
              f"{'retrying' if retry_worthy else 'below retry threshold, done'}")
    return 3 if retry_worthy else 0


if __name__ == "__main__":
    raise SystemExit(main())
