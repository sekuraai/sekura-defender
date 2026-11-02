"""Run the curation funnel, write the surviving record pool, and report the funnel.

  python scripts/build_pool.py                      # all tiers (default)
  python scripts/build_pool.py --max-tier=T1-sharealike   # licence-restricted rebuild
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft import stats
from sekura_sft.config import MANIFESTS_DIR, POOL_DIR, TIER_ORDER
from sekura_sft.corpus.filters import Funnel


def main() -> int:
    max_tier = "T3-reference"
    for a in sys.argv[1:]:
        if a.startswith("--max-tier="):
            max_tier = a.split("=", 1)[1]
    if max_tier not in TIER_ORDER:
        print(f"unknown tier {max_tier!r}; expected one of {TIER_ORDER}", file=sys.stderr)
        return 2

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = POOL_DIR / "pool.jsonl"

    f = Funnel(max_tier=max_tier)
    tiers, types, sources = collections.Counter(), collections.Counter(), collections.Counter()
    groups, kept, claim_chars = set(), 0, 0
    with open(out, "w", encoding="utf-8") as fh:
        for rec in f.run():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            tiers[rec["license_tier"]] += 1
            types[rec["type"]] += 1
            sources[rec.get("source_id", "")] += 1
            groups.add((rec.get("source_path"), rec.get("heading", "")))
            claim_chars += len(rec.get("claim", ""))

    rep = {
        "max_tier": max_tier, "raw": f.raw, "kept": kept,
        "rejected": dict(f.rejected),
        "reconciles": kept + sum(f.rejected.values()) == f.raw,
        "groups": len(groups),
        "by_tier": dict(tiers.most_common()),
        "by_type": dict(types.most_common()),
        "by_source": dict(sources.most_common(25)),
        "claim_chars": claim_chars,
        "approx_claim_tokens": claim_chars // 4,
    }
    (MANIFESTS_DIR / "funnel.json").write_text(json.dumps(rep, indent=2))
    print(stats.render(rep))
    print(f"\nwrote {out}  ({kept} records, {out.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {MANIFESTS_DIR / 'funnel.json'}")
    if not rep["reconciles"]:
        print("FUNNEL DOES NOT RECONCILE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
