"""Funnel report: the auditable record of what was kept, dropped, and why."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .corpus.filters import Funnel


def build(max_tier: str = "T3-reference", limit: int | None = None) -> dict:
    f = Funnel(max_tier=max_tier, limit=limit)
    tiers, types, sources = collections.Counter(), collections.Counter(), collections.Counter()
    groups, kept = set(), 0
    claim_chars = 0
    for r in f.run():
        kept += 1
        tiers[r["license_tier"]] += 1
        types[r["type"]] += 1
        sources[r.get("source_id", "")] += 1
        groups.add((r.get("source_path"), r.get("heading", "")))
        claim_chars += len(r.get("claim", ""))
    return {
        "max_tier": max_tier,
        "raw": f.raw,
        "kept": kept,
        "rejected": dict(f.rejected),
        "reconciles": kept + sum(f.rejected.values()) == f.raw,
        "groups": len(groups),
        "by_tier": dict(tiers.most_common()),
        "by_type": dict(types.most_common()),
        "by_source": dict(sources.most_common(25)),
        "claim_chars": claim_chars,
        "approx_claim_tokens": claim_chars // 4,
    }


def render(rep: dict) -> str:
    from .corpus.filters import REASONS
    out = [f"raw records: {rep['raw']}", ""]
    run = rep["raw"]
    for r in REASONS:
        n = rep["rejected"].get(r, 0)
        if n:
            run -= n
            out.append(f"  -{n:>7d}  {r:<22s} -> {run}")
    out += ["",
            f"kept: {rep['kept']} ({rep['kept'] / rep['raw']:.1%})   groups: {rep['groups']}",
            f"reconciles: {rep['reconciles']}",
            f"approx claim tokens: {rep['approx_claim_tokens']:,}",
            "", "by tier:  " + "  ".join(f"{k}={v}" for k, v in rep["by_tier"].items()),
            "by type:  " + "  ".join(f"{k}={v}" for k, v in rep["by_type"].items())]
    return "\n".join(out)
