"""Do every non-GPU thing BEFORE provisioning. Free; run on the subscription.

Paid minutes bill at 5xB300 = $1.25/minute from the moment a job is created, so
anything that can happen on a laptop must happen here, not inside the campaign:

  * Tokenization. The campaign used to call connect() and THEN tokenize 14,172
    examples, holding five B300s hot for ~90s of local CPU work.
  * Prompt rendering for eval, same reason.
  * Subset selection, which is the biggest lever on paid time: the mixture is
    badly skewed (grounded_qa 2,985 vs attacktree 171), so a balanced,
    length-capped subset both costs less and -- at rank 32, where every task
    competes for one low-rank subspace -- is a better use of the capacity.

Writes a cache the campaign loads in seconds, and prints the projected paid
cost so the scope decision happens before any money is spent.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft.batching import batches, tokenize_split
from sekura_sft.config import DATA, MAX_COMPLETION_BY_TASK, supervised_budget
from sekura_sft.tokenization import load, render_prompt, supervised_tokens

CACHE = Path("data/fireworks/cache")
SEC_PER_STEP = 7.9          # measured
PROVISION_MIN = 10.0        # measured
RATE_PER_MIN = 1.25         # 5 x B300 at $15/GPU-hour


def spread(rows: list[dict], cap: int) -> list[dict]:
    """Take `cap` rows spanning the length range, not just the cheapest.

    Keeping only short examples would bias the mixture toward whichever tasks
    happen to be terse, which is the same mistake as not normalizing the loss.
    """
    if len(rows) <= cap:
        return rows
    rows = sorted(rows, key=lambda d: len(d["input_ids"]))
    step = len(rows) / cap
    return [rows[int(i * step)] for i in range(cap)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=2048,
                    help="drop the long tail; cost scales with sequence length")
    ap.add_argument("--steps-per-epoch", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-per-task", type=int, default=20)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    tok = load()

    # ---- train subset -------------------------------------------------
    data, skipped = tokenize_split(DATA / "dataset" / "train")
    print(f"[train] tokenized {len(data)} ({len(skipped)} skipped)")
    by_task: dict[str, list] = defaultdict(list)
    dropped_long = 0
    for d in data:
        if len(d["input_ids"]) > args.max_len:
            dropped_long += 1
            continue
        by_task[d["task"]].append(d)
    print(f"[train] dropped {dropped_long} over {args.max_len} tokens")

    # SAFETY_MIX is enforced at build time (52% answer_fully, 13% dual_use,
    # 20% decline, 9% scoped, 6% injection). Subsampling by length would
    # reshape that mixture silently, so this slice is never trimmed.
    KEEP_WHOLE = {"safety"}
    subset: list[dict] = []
    for task in sorted(by_task):
        cap = len(by_task[task]) if task in KEEP_WHOLE else args.per_task
        picked = spread(by_task[task], cap)
        subset.extend(picked)
        print(f"  {task:22s} {len(by_task[task]):5d} -> {len(picked):4d}")

    sup = sum(supervised_tokens(d) for d in subset)
    budget = supervised_budget(sup, target_steps=args.steps_per_epoch)
    n_steps = len(list(batches(subset, budget))) * args.epochs
    train_min = n_steps * SEC_PER_STEP / 60

    with open(CACHE / "train.jsonl", "w", encoding="utf-8") as fh:
        for d in subset:
            fh.write(json.dumps({
                "input_ids": d["input_ids"], "target_tokens": d["target_tokens"],
                "weights": d["weights"], "datum_id": d["datum_id"],
                "task": d["task"],
            }) + "\n")

    # ---- eval subset, prompts pre-rendered ----------------------------
    rows = []
    for shard in sorted((DATA / "dataset" / "test").glob("*.jsonl")):
        for line in open(shard, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    ev: dict[str, list] = defaultdict(list)
    for r in rows:
        ev[r.get("task", "?")].append(r)
    esub = []
    for task in sorted(ev):
        esub.extend(ev[task][: args.eval_per_task])
    with open(CACHE / "eval.jsonl", "w", encoding="utf-8") as fh:
        for r in esub:
            text = render_prompt(tok, r["system"], r["turns"],
                                 thinking=bool(r.get("thinking")))
            r = dict(r)
            r["_prompt_ids"] = tok(text, add_special_tokens=False)["input_ids"]
            r["_max_tokens"] = MAX_COMPLETION_BY_TASK.get(r.get("task", ""), 2048)
            fh.write(json.dumps(r) + "\n")

    # Measured: ~10s/row at 12-way concurrency, two models.
    eval_min = 2 * len(esub) * 10 / 12 / 60
    total_min = PROVISION_MIN + train_min + eval_min

    print(f"\n[cache] {CACHE}/train.jsonl  {len(subset)} examples, "
          f"{sup:,} supervised tokens, budget {budget:,}/step")
    print(f"[cache] {CACHE}/eval.jsonl   {len(esub)} rows "
          f"({args.eval_per_task}/task over {len(ev)} tasks)")
    print(f"\n--- projected PAID cost at ${RATE_PER_MIN:.2f}/min (5xB300) ---")
    print(f"  provisioning   {PROVISION_MIN:5.1f} min   ${PROVISION_MIN * RATE_PER_MIN:6.2f}")
    print(f"  train {n_steps:3d} steps {train_min:5.1f} min   ${train_min * RATE_PER_MIN:6.2f}")
    print(f"  eval 2x{len(esub):4d} rows {eval_min:5.1f} min   ${eval_min * RATE_PER_MIN:6.2f}")
    print(f"  {'TOTAL':14s} {total_min:5.1f} min   ${total_min * RATE_PER_MIN:6.2f}")
    print(json.dumps(dict(Counter(d["task"] for d in subset)), indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
