"""Score a Fireworks checkpoint -- or the base model -- through the eval gate.

  python scripts/eval_fireworks.py --base                      # untrained adapter = base
  python scripts/eval_fireworks.py --checkpoint PATH --job ID   # a tuned checkpoint
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_eval.suite import load_eval, render, run
from sekura_fireworks.infer import SessionSampler
from sekura_sft.config import MANIFESTS_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--job")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_eval(args.split)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[eval] {len(rows)} rows from {args.split}")

    s = SessionSampler(checkpoint=args.checkpoint, trainer_job_id=args.job,
                       name="sekura-eval-base" if args.base else "sekura-eval-sft")
    try:
        preds = s.predictions(rows)
    finally:
        s.close()

    it = iter(preds)
    rep = run(lambda sysmsg, turns: next(it), rows=rows)
    rep["_meta"] = {"checkpoint": s.checkpoint, "split": args.split, "n": len(rows)}
    print(render(rep))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
