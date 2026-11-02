"""Group the curated pool into grounded context bundles."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft.config import BUNDLES_DIR, MANIFESTS_DIR, POOL_DIR
from sekura_sft.corpus.bundles import build, iter_pool


def main() -> int:
    pool = POOL_DIR / "pool.jsonl"
    if not pool.exists():
        print(f"{pool} missing; run scripts/build_pool.py first", file=sys.stderr)
        return 2
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    out = BUNDLES_DIR / "bundles.jsonl"

    tiers, srcs = collections.Counter(), collections.Counter()
    sizes, toks = [], []
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for b in build(iter_pool(pool)):
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
            n += 1
            tiers[b["license_tier"]] += 1
            srcs[b["source_id"]] += 1
            sizes.append(b["n_claims"])
            toks.append(b["est_prompt_tokens"])

    sizes.sort(); toks.sort()
    p = lambda v, q: v[int(len(v) * q)] if v else 0
    rep = {
        "bundles": n,
        "claims_total": sum(sizes),
        "claims_p10": p(sizes, .10), "claims_p50": p(sizes, .50), "claims_p90": p(sizes, .90),
        "tokens_p10": p(toks, .10), "tokens_p50": p(toks, .50), "tokens_p90": p(toks, .90),
        "tokens_max": toks[-1] if toks else 0,
        "by_tier": dict(tiers.most_common()),
        "by_source": dict(srcs.most_common(20)),
    }
    (MANIFESTS_DIR / "bundles.json").write_text(json.dumps(rep, indent=2))
    print(f"bundles: {n}   claims covered: {sum(sizes)}")
    print(f"claims/bundle  p10={rep['claims_p10']} p50={rep['claims_p50']} p90={rep['claims_p90']}")
    print(f"prompt tokens  p10={rep['tokens_p10']} p50={rep['tokens_p50']} "
          f"p90={rep['tokens_p90']} max={rep['tokens_max']}")
    print("by tier: " + "  ".join(f"{k}={v}" for k, v in rep["by_tier"].items()))
    print("\ntop sources: " + "  ".join(f"{k}={v}" for k, v in list(rep["by_source"].items())[:8]))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
