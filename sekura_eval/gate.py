"""Regression gate. Exits non-zero when a candidate is worse than the baseline.

Hard failures are absolute thresholds on the metrics that decide whether the
model is safe to put in front of a customer. Relative failures catch a
capability quietly regressing while the headline average holds up -- which is
exactly why the suite never reports one blended number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Absolute floors/ceilings. Both rates are the ones that decide shippability:
# a plan that covers nothing means the scan returns nothing, and accepting a
# finding that should be rejected ships a bad finding to a customer.
HARD: dict[tuple[str, str], tuple[str, float]] = {
    ("planner", "zero_coverage_rate"): ("<=", 0.01),
    ("triage", "dangerous_rate"): ("<=", 0.03),
    ("attacktree_repair", "exact"): (">=", 0.60),
    ("cvss", "exact"): (">=", 0.90),
}

# A metric may not fall more than this below the recorded baseline.
MAX_REGRESSION = 0.03

# Safety-critical rates get a tighter tolerance than general quality metrics.
# Going from 0% to 2% dangerous errors is a real regression even though it sits
# under the absolute cap of 3% -- the cap says "never ship worse than this",
# while the tolerance says "do not silently get worse than you were".
TIGHT_REGRESSION = 0.005
TIGHT_METRICS = frozenset({
    ("triage", "dangerous_rate"),
    ("planner", "zero_coverage_rate"),
})


@dataclass
class GateResult:
    passed: bool
    hard_failures: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["GATE: " + ("PASS" if self.passed else "FAIL")]
        for label, items in (("hard failures", self.hard_failures),
                             ("regressions", self.regressions),
                             ("missing from candidate", self.missing),
                             ("improvements", self.improvements)):
            for item in items:
                lines.append(f"  [{label}] {item}")
        return "\n".join(lines)


def check(candidate: dict, baseline: dict | None = None,
          max_regression: float = MAX_REGRESSION) -> GateResult:
    res = GateResult(True)

    for (task, metric), (op, bound) in HARD.items():
        got = (candidate.get(task) or {}).get(metric)
        if got is None:
            # An absent hard metric is a failure, not a pass: a capability with
            # no test examples cannot be gated, and silently skipping it is how
            # a regression ships.
            res.missing.append(f"{task}.{metric} absent; cannot gate it")
            res.passed = False
            continue
        ok = got <= bound if op == "<=" else got >= bound
        if not ok:
            res.hard_failures.append(f"{task}.{metric}={got} violates {op}{bound}")
            res.passed = False

    if baseline:
        for task, metrics in baseline.items():
            if task == "_overall":
                continue
            cand = candidate.get(task)
            if cand is None:
                res.missing.append(f"{task} missing from candidate")
                res.passed = False
                continue
            for metric, base_val in metrics.items():
                if metric == "n" or not isinstance(base_val, (int, float)):
                    continue
                got = cand.get(metric)
                if got is None:
                    continue
                # Rates where lower is better invert the comparison.
                lower_better = metric.endswith("_rate") and metric != "reason_exact_rate"
                delta = (base_val - got) if not lower_better else (got - base_val)
                tol = (TIGHT_REGRESSION if (task, metric) in TIGHT_METRICS
                       else max_regression)
                if delta > tol:
                    res.regressions.append(
                        f"{task}.{metric} {base_val} -> {got} "
                        f"({'worse' if not lower_better else 'higher'} by {delta:.3f}, "
                        f"tolerance {tol})")
                    res.passed = False
                elif -delta > max_regression:
                    res.improvements.append(f"{task}.{metric} {base_val} -> {got}")
    return res


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
