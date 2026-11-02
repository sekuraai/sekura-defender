"""ENV-C: the scan-critic. Cheap completions, directly consumed by sekura-rs.

    R = 0.45*verdict + 0.25*reason + 0.20*confidence + 0.10*format

Two design choices carry the environment:

VERDICT SCORING IS ASYMMETRIC, not binary. Accepting a finding that should have
been rejected ships a bad finding to a customer; rejecting a good one only wastes
a finding. A symmetric matrix would treat those as equally bad, and the model
would land wherever the base rate is cheapest.

REASON IS 25% AND IT IS WHAT STOPS GUESSING. A model that emits the majority
verdict collects partial verdict credit but ZERO reason credit, because
reason_code must equal the corruption that was actually injected. You cannot
guess why without reading the finding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sekura_sft.tasks.det_triage import CORRUPTIONS, VERDICTS

WEIGHTS = {"verdict": 0.45, "reason": 0.25, "confidence": 0.20, "format": 0.10}

# (truth, predicted) -> credit. Rows are the true verdict.
VERDICT_MATRIX = {
    ("accept", "accept"): 1.0, ("accept", "challenge"): 0.5,
    ("accept", "reject"): 0.2, ("accept", "reroute"): 0.3,
    ("challenge", "challenge"): 1.0, ("challenge", "reject"): 0.6,
    ("challenge", "accept"): 0.1, ("challenge", "reroute"): 0.3,
    # The dangerous error: shipping a finding that should have been rejected.
    ("reject", "reject"): 1.0, ("reject", "challenge"): 0.5,
    ("reject", "accept"): 0.0, ("reject", "reroute"): 0.3,
    ("reroute", "reroute"): 1.0, ("reroute", "challenge"): 0.4,
    ("reroute", "accept"): 0.1, ("reroute", "reject"): 0.3,
}
DANGEROUS = (("reject", "accept"),)
DANGEROUS_PENALTY = 0.30

VALID_CODES = set(CORRUPTIONS) | {"none"}
JUSTIFICATION_MAX = 600


@dataclass
class Score:
    reward: float
    components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dangerous: bool = False


def parse(text: str) -> dict | None:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def score(text: str, truth: dict) -> Score:
    """`truth` is the example's own target: verdict, reason_code, offending_field,
    confidence_multiplier."""
    obj = parse(text)
    if obj is None:
        return Score(0.0, {"format": 0.0}, ["unparseable"])

    required = {"verdict", "confidence_multiplier", "reason_code",
                "offending_field", "justification"}
    present = required & set(obj)
    fmt = len(present) / len(required)
    pred_v = obj.get("verdict")
    if pred_v not in VERDICTS:
        fmt *= 0.5
    conf = obj.get("confidence_multiplier")
    if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        fmt *= 0.5
        conf = None
    just = str(obj.get("justification", ""))
    if len(just) > JUSTIFICATION_MAX:
        fmt *= 0.8
    if obj.get("reason_code") not in VALID_CODES:
        fmt *= 0.8

    true_v = truth["verdict"]
    verdict = VERDICT_MATRIX.get((true_v, pred_v), 0.0) if pred_v in VERDICTS else 0.0
    dangerous = (true_v, pred_v) in DANGEROUS

    # Reason: exact match against the injected corruption. Partial credit only
    # for correctly localizing the field while naming the wrong code.
    true_code = truth.get("reason_code", "none")
    pred_code = obj.get("reason_code")
    if pred_code == true_code:
        reason = 1.0
    elif true_code == "none":
        reason = 0.0                      # a spurious reason on a clean finding
    elif obj.get("offending_field") == truth.get("offending_field"):
        reason = 0.4                      # right field, wrong diagnosis
    else:
        reason = 0.0

    # Brier-style calibration on the multiplier sekura-rs actually consumes.
    if conf is None:
        confidence = 0.0
    else:
        target = float(truth.get("confidence_multiplier", 1.0))
        confidence = max(0.0, 1.0 - (float(conf) - target) ** 2)

    reward = (WEIGHTS["verdict"] * verdict + WEIGHTS["reason"] * reason
              + WEIGHTS["confidence"] * confidence + WEIGHTS["format"] * fmt)
    if dangerous:
        reward -= DANGEROUS_PENALTY
    notes = []
    if dangerous:
        notes.append("DANGEROUS: accepted a finding that should be rejected")
    return Score(round(max(0.0, min(1.0, reward)), 4),
                 {"verdict": round(verdict, 3), "reason": round(reason, 3),
                  "confidence": round(confidence, 3), "format": round(fmt, 3)},
                 notes, dangerous)
