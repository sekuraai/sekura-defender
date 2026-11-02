"""One reattach, then do whatever is needed to get tuned numbers.

Session checkpoints are scoped to the training session, and ours closed when
the campaign aborted on the sampler bug. The checkpoint may therefore be gone.
Reattaching costs ~10 minutes of deployment provisioning either way, so this
script pays that once and then branches: if the old checkpoint still samples,
evaluate it; if not, retrain and evaluate. Running eval-only and discovering
the checkpoint is missing would cost a second 10-minute reattach.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_eval.gate import check, load as load_baselines
from sekura_eval.suite import render, run
from sekura_fireworks.budget import Aborted, Budget, LossTripwire
from sekura_fireworks.infer import SessionSampler
from sekura_fireworks.trainer import connect, train

OUT = Path("data/fireworks")
CACHE = Path("data/fireworks/cache")
JOB = "training-api-service-28ba9178"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=JOB)
    ap.add_argument("--ckpt", default="sekura-sft-v1-1aaad1de",
                    help="existing checkpoint to try before retraining")
    ap.add_argument("--base-ckpt", default="base-ref-1aaad1de")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--token-budget", type=int, default=40955)
    ap.add_argument("--budget-usd", type=float, default=45.0)
    ap.add_argument("--absolute-usd", type=float, default=195.0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in (CACHE / "eval.jsonl").read_text().splitlines()
            if x.strip()]
    train_data = [json.loads(x) for x in
                  (CACHE / "train.jsonl").read_text().splitlines() if x.strip()]
    print(f"[cache] {len(train_data)} train, {len(rows)} eval rows")

    ledger = OUT / "ledger2.jsonl"
    ledger.unlink(missing_ok=True)
    budget = Budget(args.budget_usd, ledger=ledger, absolute_usd=args.absolute_usd)
    print(f"[budget] +${args.budget_usd:.0f} session / ${args.absolute_usd:.0f} absolute; "
          f"{budget.line()}")

    reports: dict[str, dict] = {}
    t0 = time.time()
    sc, tc = connect(create_deployment=True, trainer_job_id=args.job)
    print(f"[session] attached after {(time.time() - t0) / 60:.1f} min", flush=True)
    try:
        tuned, base = args.ckpt, args.base_ckpt
        # Does the old checkpoint still sample? One row answers it.
        try:
            s = SessionSampler.from_training_client(tc, tuned)
            probe = s.predictions(rows[:2], workers=2, progress=False, probe=2)
            if not any(p.strip() for p in probe):
                raise RuntimeError("checkpoint sampled empty")
            print(f"[reuse] checkpoint {tuned} still samples; skipping training")
        except Exception as e:  # noqa: BLE001
            print(f"[retrain] {tuned} unusable ({e!r}); training fresh", flush=True)
            res = train(epochs=args.epochs, ckpt="sekura-sft-v2",
                        base_ref="base-ref-v2", session=(sc, tc),
                        budget=args.token_budget, data=train_data,
                        cost_budget=budget,
                        tripwire=LossTripwire(steps_per_epoch=47),
                        log=OUT / "sft-v2.jsonl")
            tuned, base = res["tuned"], res["base"]
            print(f"[retrain] done: tuned={tuned} base={base}", flush=True)

        for label, ck in (("tuned", tuned), ("base", base)):
            if not ck:
                print(f"[skip] no {label} checkpoint")
                continue
            print(f"\n[eval:{label}] {ck} over {len(rows)} rows", flush=True)
            s = SessionSampler.from_training_client(tc, ck)
            preds = s.predictions(rows, budget=budget)
            it = iter(preds)
            rep = run(lambda sm, tn: next(it), rows=rows)
            print(render(rep))
            reports[label] = rep
            (OUT / f"report-{label}.json").write_text(json.dumps(
                {**rep, "_meta": {"label": label, "checkpoint": ck,
                                  "n": len(rows)}}, indent=2))
            print(f"[phase] {label} done -- {budget.line()}", flush=True)
    except Aborted as e:
        print(f"\n[ABORT] {e}")
    finally:
        try:
            sc.close()
        except Exception:  # noqa: BLE001, S110
            pass
        print(f"\n[cost] closed -- {budget.line()}")

    if "tuned" in reports:
        store = load_baselines(Path("data/manifests/eval_baselines.json")) or {}
        g = check(reports["tuned"], store.get("production"))
        print("\n=== gate ===")
        print(g.render())
        if "base" in reports:
            (OUT / "comparison.json").write_text(json.dumps(
                {"base": reports["base"], "tuned": reports["tuned"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
