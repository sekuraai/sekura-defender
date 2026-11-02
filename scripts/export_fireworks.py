"""Convert the assembled corpus to Fireworks chat-JSONL."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_fireworks.export import balanced_subset, load_split, to_chat, write
from sekura_sft.config import DATA, MANIFESTS_DIR

OUT = DATA / "fireworks"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", action="store_true",
                    help="duplicate high-TASK_WEIGHT tasks (no per-token weights on Fireworks)")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--fast", type=int, default=0,
                    help="also emit a task-balanced subset of N train records")
    args = ap.parse_args()

    report = {}
    for split, fname in (("train", "train.jsonl"), ("val", "eval.jsonl"),
                         ("test", "test.jsonl")):
        r = write(split, OUT / fname,
                  resample=args.resample and split == "train",
                  max_records=args.max_train if split == "train" and args.max_train else None)
        report[split] = r
        print(f"  {split:5s} {r['records']:6d} records  {r['bytes']/1e6:7.1f} MB  "
              f"~{r['approx_tokens']:,} tokens" + (f"  skipped {r['skipped']}" if r['skipped'] else ""))
    if args.fast:
        rows = balanced_subset(list(load_split("train")), args.fast)
        dest = OUT / f"train-fast-{args.fast}.jsonl"
        n = chars = 0
        with open(dest, "w", encoding="utf-8") as fh:
            for row in rows:
                line = json.dumps(to_chat(row), ensure_ascii=False)
                fh.write(line + "\n"); n += 1; chars += len(line)
        import collections
        mix = collections.Counter(r.get("task") for r in rows)
        report["fast"] = {"records": n, "bytes": dest.stat().st_size,
                          "approx_tokens": chars // 4, "path": str(dest),
                          "tasks": dict(mix.most_common())}
        print(f"\n  fast  {n:6d} records  {dest.stat().st_size/1e6:7.1f} MB  "
              f"~{chars//4:,} tokens  across {len(mix)} tasks")
        for k, v in mix.most_common():
            print(f"          {k:20s} {v}")
    (MANIFESTS_DIR / "fireworks_export.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}/  and  {MANIFESTS_DIR/'fireworks_export.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
