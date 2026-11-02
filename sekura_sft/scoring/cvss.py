"""CVSS v3.0 / v3.1 base score, implemented from the specification.

Zero dependencies -- the `cvss` package is not installed and reproducibility is
better served by having the arithmetic in-repo anyway. This is used three ways:

  * a deterministic SFT slice (vector -> score), no teacher, no ambiguity
  * an RL sub-environment with an exactly checkable reward
  * the C2 corruption in the triage engine, which inflates a finding's severity
    past what its own vector supports -- which requires computing what the vector
    actually supports

Spec: FIRST CVSS v3.1 section 7.1. v3.0 differs only in the final roundup, which
is why `version` is honoured rather than assumed.
"""

from __future__ import annotations

import math

WEIGHTS = {
    "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
    "AC": {"L": 0.77, "H": 0.44},
    "UI": {"N": 0.85, "R": 0.62},
    "C": {"H": 0.56, "L": 0.22, "N": 0.0},
    "I": {"H": 0.56, "L": 0.22, "N": 0.0},
    "A": {"H": 0.56, "L": 0.22, "N": 0.0},
}
# Privileges Required is scope-dependent: a changed scope makes low/high
# privileges cheaper for the attacker.
PR = {"U": {"N": 0.85, "L": 0.62, "H": 0.27},
      "C": {"N": 0.85, "L": 0.68, "H": 0.5}}

REQUIRED = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")

# Temporal metrics. Present in some advisory vectors; they never affect the base
# score, but a published score that looks "wrong" against the base is often the
# temporal score instead -- so we must be able to tell the two apart rather than
# silently reporting a mismatch.
TEMPORAL = {
    "E": {"X": 1.0, "H": 1.0, "F": 0.97, "P": 0.94, "U": 0.91},
    "RL": {"X": 1.0, "U": 1.0, "W": 0.97, "T": 0.96, "O": 0.95},
    "RC": {"X": 1.0, "C": 1.0, "R": 0.96, "U": 0.92},
}


class CvssError(ValueError):
    pass


def parse_vector(vector: str) -> dict[str, str]:
    """Parse a CVSS vector string into its base metrics."""
    if not vector or "/" not in vector:
        raise CvssError(f"not a CVSS vector: {vector!r}")
    parts = [p for p in vector.strip().split("/") if p]
    out: dict[str, str] = {}
    version = "3.1"
    for part in parts:
        if ":" not in part:
            raise CvssError(f"malformed component {part!r} in {vector!r}")
        k, v = part.split(":", 1)
        k, v = k.strip().upper(), v.strip().upper()
        if k == "CVSS":
            version = v
            continue
        out.setdefault(k, v)
    missing = [m for m in REQUIRED if m not in out]
    if missing:
        raise CvssError(f"missing base metrics {missing} in {vector!r}")
    out["_version"] = version
    return out


def _roundup(x: float, version: str) -> float:
    """v3.1 defines roundup over integer arithmetic specifically to avoid the
    float representation errors that made some v3.0 implementations disagree."""
    if version.startswith("3.0"):
        return math.ceil(x * 10) / 10
    i = int(round(x * 100000))
    if i % 10000 == 0:
        return i / 100000.0
    return (math.floor(i / 10000) + 1) / 10.0


def base_score(vector: str) -> float:
    m = parse_vector(vector)
    scope_changed = m["S"] == "C"

    try:
        iss = 1 - ((1 - WEIGHTS["C"][m["C"]]) *
                   (1 - WEIGHTS["I"][m["I"]]) *
                   (1 - WEIGHTS["A"][m["A"]]))
        if scope_changed:
            impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
        else:
            impact = 6.42 * iss
        exploitability = (8.22
                          * WEIGHTS["AV"][m["AV"]]
                          * WEIGHTS["AC"][m["AC"]]
                          * PR["C" if scope_changed else "U"][m["PR"]]
                          * WEIGHTS["UI"][m["UI"]])
    except KeyError as exc:
        raise CvssError(f"unknown metric value {exc} in {vector!r}") from exc

    if impact <= 0:
        return 0.0
    raw = min((1.08 if scope_changed else 1.0) * (impact + exploitability), 10.0)
    return _roundup(raw, m["_version"])


def has_temporal(vector: str) -> bool:
    m = parse_vector(vector)
    return any(m.get(k, "X") != "X" for k in TEMPORAL)


def temporal_score(vector: str) -> float:
    """Temporal score. Per spec this multiplies the ROUNDED base score, not the
    raw one -- which is why 8.1 x E:U x RL:O rounds up to 7.1 rather than 7.0."""
    m = parse_vector(vector)
    base = base_score(vector)
    factor = 1.0
    for k, table in TEMPORAL.items():
        v = m.get(k, "X")
        if v not in table:
            raise CvssError(f"unknown {k} value {v!r} in {vector!r}")
        factor *= table[v]
    return _roundup(base * factor, m["_version"])


def score(vector: str) -> float:
    """The score a published advisory most likely means: temporal when temporal
    metrics are supplied, base otherwise."""
    return temporal_score(vector) if has_temporal(vector) else base_score(vector)


def severity_band(score: float) -> str:
    """FIRST qualitative severity rating scale."""
    if score == 0.0:
        return "none"
    if score < 4.0:
        return "low"
    if score < 7.0:
        return "medium"
    if score < 9.0:
        return "high"
    return "critical"
