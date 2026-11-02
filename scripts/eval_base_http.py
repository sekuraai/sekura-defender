"""Base-model eval over the HTTP inference API, parallel to training.

The orphaned deployment left behind by a killed run is READY and addressable as
`<model>#<deployment>`, so the base reference can be measured without spending
any session GPU time -- and while training occupies the session.

This is INSURANCE AND A CROSS-CHECK, not the headline comparison. The campaign
still evaluates base through the session sampler, which is the figure that
belongs beside the tuned number because it shares a serving path with it. If
these two base measurements disagree materially, that difference IS train/serve
skew and has to be reported rather than averaged away.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_eval.suite import render, run
from sekura_fireworks.infer import predictions

DEP = ("accounts/fireworks/models/qwen3p6-35b-a3b"
       "#accounts/tikka-0hyait3m5ddc/deployments/qwen3p6-35b-a3b-1793655179")
CACHE = Path("data/fireworks/cache/eval.jsonl")
OUT = Path("data/fireworks/report-base-http.json")


def main() -> int:
    rows = [json.loads(x) for x in CACHE.read_text().splitlines() if x.strip()]
    print(f"[base-http] {len(rows)} rows via {DEP.split('#')[1].split('/')[-1]}")
    preds = predictions(rows, DEP, workers=8)
    empty = sum(1 for p in preds if not p.strip())
    print(f"[base-http] {empty}/{len(preds)} empty completions")
    it = iter(preds)
    rep = run(lambda s, t: next(it), rows=rows)
    print(render(rep))
    OUT.write_text(json.dumps({**rep, "_meta": {
        "label": "base-http", "model": DEP, "n": len(rows)}}, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
