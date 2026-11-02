"""Tokenize the assembled dataset; report token volume, batching and cost."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft.batching import batches, tokenize_split
from sekura_sft.config import (BASE_MODEL, DATASET_DIR, MANIFESTS_DIR,
                               supervised_budget)
from sekura_sft.tokenization import preflight, supervised_tokens

# river training price, USD per 1M tokens. ALL tokens in a datum are billed,
# not only the supervised ones.
TRAIN_PRICE = {"Qwen/Qwen3.6-35B-A3B-FP8": 1.00, "Qwen/Qwen3.5-9B": 1.46,
               "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": 1.00,
               "Qwen/Qwen3.8-27B-FP8": 4.10}


def main() -> int:
    info = preflight()
    print(f"preflight OK: {info['tokenizer']} vocab={info['vocab_size']} "
          f"eos={info['eos_token']!r}\n")

    report = {"base_model": BASE_MODEL, "splits": {}}
    for split in ("train", "val", "test"):
        d = DATASET_DIR / split
        if not d.exists():
            continue
        kept, skipped = tokenize_split(d)
        billed = sum(len(x["input_ids"]) for x in kept)
        sup = sum(supervised_tokens(x) for x in kept)
        lens = sorted(len(x["input_ids"]) for x in kept)
        p = lambda q: lens[int(len(lens) * q)] if lens else 0
        budget = supervised_budget(sup)
        steps = len(list(batches(kept, budget))) if split == "train" else 0
        report["splits"][split] = {
            "examples": len(kept), "skipped": len(skipped),
            "billed_tokens": billed, "supervised_tokens": sup,
            "len_p50": p(.50), "len_p90": p(.90), "len_max": lens[-1] if lens else 0,
            "steps_per_epoch": steps, "supervised_budget_per_step": budget,
            "skipped_by_task": dict(Counter(s["task"] for s in skipped)),
        }
        print(f"{split:5s} {len(kept):5d} examples  billed={billed:9,d}  "
              f"supervised={sup:8,d}  len p50={p(.50)} p90={p(.90)} max={lens[-1] if lens else 0}"
              + (f"  steps/epoch={steps} @ {budget:,} sup-tok/step" if steps else ""))
        if skipped:
            print(f"      skipped {len(skipped)}: {dict(Counter(s['task'] for s in skipped))}")

    price = TRAIN_PRICE.get(BASE_MODEL, 1.00)
    train = report["splits"].get("train", {})
    billed = train.get("billed_tokens", 0)
    print(f"\ncost @ ${price:.2f}/1M billed training tokens")
    for ep in (1, 2, 3):
        print(f"   {ep} epoch{'s' if ep > 1 else ' '}: {billed * ep / 1e6:7.2f}M tokens "
              f"= ${billed * ep / 1e6 * price:6.2f}")
    report["price_per_1m"] = price
    report["cost_2_epochs_usd"] = round(billed * 2 / 1e6 * price, 2)
    (MANIFESTS_DIR / "tokenization.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {MANIFESTS_DIR / 'tokenization.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
