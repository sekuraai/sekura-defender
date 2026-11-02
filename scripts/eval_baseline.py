"""Record or check an eval baseline.

  python scripts/eval_baseline.py --oracle          # write the ceiling (targets replayed)
  python scripts/eval_baseline.py --degenerate      # write the floor (empty output)
  python scripts/eval_baseline.py --check FILE      # gate a candidate report
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_eval.gate import check, load
from sekura_eval.suite import load_eval, render, run
from sekura_sft.config import MANIFESTS_DIR

BASE = MANIFESTS_DIR / "eval_baselines.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--degenerate", action="store_true")
    ap.add_argument("--check")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    rows = load_eval(args.split)
    if not rows:
        print("no eval rows; run scripts/build_dataset.py first", file=sys.stderr)
        return 2

    store = json.loads(BASE.read_text()) if BASE.exists() else {}

    if args.check:
        cand = json.loads(Path(args.check).read_text())
        g = check(cand, store.get("production"))
        print(render(cand))
        print()
        print(g.render())
        return 0 if g.passed else 1

    if args.oracle:
        it = iter([r["target"] for r in rows])
        rep = run(lambda s, t: next(it), rows=rows)
        store["oracle_ceiling"] = rep
        print("=== oracle ceiling (targets replayed) ===")
        print(render(rep))
    if args.degenerate:
        rep = run(lambda s, t: "", rows=rows)
        store["degenerate_floor"] = rep
        print("\n=== degenerate floor (empty output) ===")
        print(render(rep))

    if args.oracle or args.degenerate:
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        BASE.write_text(json.dumps(store, indent=2))
        print(f"\nwrote {BASE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
