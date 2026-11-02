"""Train and evaluate in ONE Fireworks session.

Why one session: checkpoints are session-scoped and `cleanup_trainer_on_close`
is now True, so the trainer job dies with this process and takes its
checkpoints with it. A second process would find NOT_FOUND. One session is the
only shape that is both cheap and correct.

Billing is GPU-time, not tokens: 4xB300 for the trainer plus 1xB300 for the
sampling deployment, at $15/GPU-hour = $1.25/minute. An earlier run was killed
by a spending cap mid-flight, so every phase here is wall-clock bounded and the
cost so far is printed at each transition.

Order is deliberate: tuned is evaluated BEFORE base. If the budget runs out
mid-eval we still have the tuned numbers, which the gate can score against the
stored oracle/degenerate baselines; the reverse would leave us with nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_eval.gate import check, load as load_baselines
from sekura_eval.suite import load_eval, render, run
from sekura_fireworks.budget import Aborted, Budget, LossTripwire
from sekura_fireworks.infer import SessionSampler
from sekura_fireworks.trainer import connect, train
from sekura_sft.config import MANIFESTS_DIR

GPUS = 5                 # 4 trainer + 1 sampling deployment
RATE_PER_GPU_HOUR = 15.0
RATE_PER_MIN = GPUS * RATE_PER_GPU_HOUR / 60.0   # $1.25/min

OUT = Path("data/fireworks")


def spent(t0: float) -> str:
    m = (time.time() - t0) / 60
    return f"{m:.1f} min / ${m * RATE_PER_MIN:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-limit", type=int)
    ap.add_argument("--train-minutes", type=float, default=50.0)
    ap.add_argument("--total-minutes", type=float, default=80.0,
                    help="hard stop; 80 min x $1.25 = $100")
    ap.add_argument("--budget-usd", type=float, default=45.0,
                    help="max USD this session may add (metered)")
    ap.add_argument("--absolute-usd", type=float, default=None,
                    help="hard ceiling on month-to-date spend")
    ap.add_argument("--mock", action="store_true",
                    help="run the whole campaign offline against doubles")
    ap.add_argument("--cache", default="data/fireworks/cache")
    ap.add_argument("--reuse-job", help="attach to an already-provisioned trainer job")
    ap.add_argument("--token-budget", type=int,
                    help="supervised tokens/step. MUST match prep_fireworks, or "
                         "train() defaults to 120 steps/epoch and triples cost")
    ap.add_argument("--ckpt", default="sekura-sft-v1")
    args = ap.parse_args()

    cache = Path(args.cache)
    train_data = None
    if (cache / "train.jsonl").exists():
        train_data = [json.loads(x) for x in
                      (cache / "train.jsonl").read_text().splitlines() if x.strip()]
        print(f"[cache] {len(train_data)} pre-tokenized train examples")
    if (cache / "eval.jsonl").exists():
        rows = [json.loads(x) for x in
                (cache / "eval.jsonl").read_text().splitlines() if x.strip()]
        print(f"[cache] {len(rows)} pre-rendered eval rows")
    else:
        rows = load_eval(args.split)
    if args.eval_limit:
        rows = rows[: args.eval_limit]

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    ledger = OUT / "ledger.jsonl"
    ledger.unlink(missing_ok=True)
    budget = Budget(args.budget_usd, ledger=ledger,
                    absolute_usd=args.absolute_usd)
    tripwire = LossTripwire()
    print(f"[budget] ceiling ${args.budget_usd:.0f} at ${budget.rate:.2f}/min "
          f"({GPUS}xB300); live ledger -> {ledger}")

    # One session, deployment included so we can sample without a second job.
    if args.mock:
        from sekura_fireworks.mock import MockServiceClient, MockTrainingClient
        by_prompt = {tuple(r["_prompt_ids"]): r["target"] for r in rows
                     if "_prompt_ids" in r}
        sc, tc = MockServiceClient(), MockTrainingClient(by_prompt)
        print(f"[mock] offline doubles; sampler replays {len(by_prompt)} targets "
              f"-> the suite should score at the oracle ceiling")
    else:
        sc, tc = connect(create_deployment=True, trainer_job_id=args.reuse_job)
    try:
        res = None
        res = train(
            epochs=args.epochs, ckpt=args.ckpt, base_ref="base-ref",
            session=(sc, tc), max_minutes=args.train_minutes,
            log=OUT / "sft-v1.jsonl", cost_budget=budget, tripwire=tripwire,
            data=train_data, budget=args.token_budget,
        )
        print(f"\n[phase] trained {res['steps_done']}/{res['planned']} steps "
              f"-- {spent(t0)}", flush=True)

        for label, ckpt in (("tuned", res["tuned"]), ("base", res["base"])):
            if not ckpt:
                print(f"[skip] no {label} checkpoint")
                continue
            left = args.total_minutes - (time.time() - t0) / 60
            if left < 5:
                print(f"[budget] {left:.1f} min left; skipping {label} eval")
                continue
            print(f"\n[eval:{label}] {ckpt} over {len(rows)} rows "
                  f"({left:.0f} min of budget left)", flush=True)
            s = SessionSampler.from_training_client(tc, ckpt)
            preds = s.predictions(rows, budget=budget)
            it = iter(preds)
            rep = run(lambda sysmsg, turns: next(it), rows=rows)
            # Keep run metadata OUT of the report dict. render() and the gate
            # both walk its keys as task names, so an extra key is a KeyError
            # that only fires after the eval has been paid for. It goes in the
            # file, not in the object those two see.
            print(render(rep))
            reports[label] = rep
            meta = {"label": label, "checkpoint": ckpt, "split": args.split,
                    "n": len(rows), "steps": res["steps_done"]}
            (OUT / f"report-{label}.json").write_text(
                json.dumps({**rep, "_meta": meta}, indent=2))
            print(f"[phase] {label} eval done -- {spent(t0)}", flush=True)
    except Aborted as e:
        print(f"\n[ABORT] {e}")
        print(f"[ABORT] stopped at {budget.line()}")
    finally:
        try:
            sc.close()
        except Exception:  # noqa: BLE001, S110
            pass
        print(f"\n[cost] session closed -- {spent(t0)}")

    if "tuned" in reports:
        store = {}
        p = MANIFESTS_DIR / "eval_baselines.json"
        if p.exists():
            store = json.loads(p.read_text())
        g = check(reports["tuned"], store.get("production"))
        print("\n=== gate ===")
        print(g.render())
    if "base" in reports and "tuned" in reports:
        # The report is a FLAT dict keyed by task name, plus "_overall" --
        # not a list under "tasks". Iterating it as a list crashed only AFTER
        # both evals had been paid for, which is the worst place to find out.
        print("\n=== base vs tuned (mean score by task) ===")
        base, tuned = reports["base"], reports["tuned"]
        for task in sorted(k for k in tuned if not k.startswith("_")):
            bb = base.get(task, {}).get("mean_score", 0.0)
            tt = tuned[task].get("mean_score", 0.0)
            print(f"  {task:22s} base={bb:.3f} tuned={tt:.3f}  {tt - bb:+.3f}")
        bo = base.get("_overall", {}).get("mean_score", 0.0)
        to = tuned.get("_overall", {}).get("mean_score", 0.0)
        print(f"  {'OVERALL':22s} base={bo:.3f} tuned={to:.3f}  {to - bo:+.3f}")
        (OUT / "comparison.json").write_text(json.dumps(
            {"base": base, "tuned": tuned}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
