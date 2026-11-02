"""ENV-B: recon -> coverage planning. Run this environment FIRST.

Cheapest completions, a fully deterministic reward, no teacher anywhere in the
loop, and a reward CLIFF that shows up immediately if credit assignment is
miswired. It also fixes a documented production bug, which makes it the best
early demonstration that any of this works.

    R = 0.55*coverage + 0.20*wellformed + 0.15*priority + 0.10*efficiency

Two hard zeros rather than gradients, because they are correctness violations
and not quality gradations:
  * an empty plan against an inventory that requires coverage
  * any hallucinated surface_ref (inventing surfaces is the obvious way to
    inflate coverage, so it must never pay)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sekura_sft.sources import recon as R
from sekura_sft.sources.recon import TargetInventory

# Deterministic reference ordering: the one judgment the contract delegates.
PRIORITY = {"auth_surface": 0, "upload_surface": 1, "graphql_endpoint": 2,
            "websocket_endpoint": 3, "form": 4, "openapi_spec": 5,
            "endpoint": 6, "cookie": 7, "header_set": 8, "js_bundle": 9, "page": 10}

WEIGHTS = {"coverage": 0.55, "wellformed": 0.20, "priority": 0.15, "efficiency": 0.10}
HALLUCINATION_PENALTY = 0.30


@dataclass
class Score:
    reward: float
    components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_plan(text: str) -> list[dict] | None:
    """Tolerant of a fenced block, intolerant of anything that is not a task list."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    out = []
    for item in obj:
        if not isinstance(item, dict):
            return None
        ref, kind = item.get("surface_ref"), item.get("kind")
        if not isinstance(ref, str) or not isinstance(kind, str):
            return None
        out.append({"surface_ref": ref, "kind": kind})
    return out


def _spearman(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    d2 = sum((x - y) ** 2 for x, y in zip(a, b))
    rho = 1 - (6 * d2) / (n * (n * n - 1))
    return max(0.0, min(1.0, (rho + 1) / 2))


def score(text: str, inv: TargetInventory, ruleset: str) -> Score:
    required = R.required_pairs(inv, ruleset)
    required_set = {(str(s), k) for s, k in required}
    real_refs = {str(ref) for rule in R.RULESETS[ruleset] for ref in rule.applies_to(inv)}

    plan = parse_plan(text)
    if plan is None:
        return Score(0.0, {"wellformed": 0.0}, ["unparseable plan"])

    # Cliff 1: the production bug. An empty plan against required coverage is
    # always an error -- encoded as a cliff, not a gradient.
    if not plan and required_set:
        return Score(0.0, {"coverage": 0.0}, ["empty plan despite required coverage"])
    if not plan and not required_set:
        return Score(1.0, {k: 1.0 for k in WEIGHTS}, ["correctly empty"])

    # Cliff 2: hallucinated surfaces. Inventing surfaces is how a model would
    # otherwise inflate coverage, so it must never pay.
    hallucinated = sorted({t["surface_ref"] for t in plan
                           if t["surface_ref"] not in real_refs})
    if hallucinated:
        return Score(0.0, {"wellformed": 0.0},
                     [f"hallucinated surfaces: {hallucinated[:3]}"])

    bad_kind = [t for t in plan if t["kind"] not in R.TASK_KINDS]
    wellformed = 1.0 - min(1.0, len(bad_kind) / max(1, len(plan)))

    emitted = {(t["surface_ref"], t["kind"]) for t in plan}
    coverage = len(required_set & emitted) / len(required_set) if required_set else 1.0

    # Deduped for coverage, counted in full against efficiency: padding with
    # duplicates must not buy coverage but must still cost.
    efficiency = min(1.0, len(required_set) / max(1, len(plan))) if required_set else 1.0

    got = [PRIORITY.get(t["surface_ref"].split(":")[0], 99) for t in plan]
    priority = _spearman(list(range(len(got))), sorted(range(len(got)), key=lambda i: got[i]))

    reward = (WEIGHTS["coverage"] * coverage + WEIGHTS["wellformed"] * wellformed
              + WEIGHTS["priority"] * priority + WEIGHTS["efficiency"] * efficiency)
    notes = []
    if coverage < 1.0:
        missing = sorted(required_set - emitted)[:3]
        notes.append(f"missing {len(required_set - emitted)} pairs e.g. {missing}")
    return Score(round(max(0.0, min(1.0, reward)), 4),
                 {"coverage": round(coverage, 4), "wellformed": round(wellformed, 4),
                  "priority": round(priority, 4), "efficiency": round(efficiency, 4)},
                 notes)
