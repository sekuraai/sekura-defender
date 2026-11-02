"""Per-capability scorers.

Deliberately reuses the RL reward functions where the question is the same, so
"good" means one thing in training and in evaluation. Where evaluation wants a
sharper answer than a shaped reward (exact CVSS, exact mapping, does the patch
actually fix the tree), it gets its own exact scorer.

Judged metrics are reported SEPARATELY from deterministic ones and never
averaged into a single headline number: a judge is noisy and a deterministic
scorer is not, and blending them hides which one moved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from sekura_rl.envs import attacktree as A
from sekura_rl.envs import planner as P
from sekura_rl.envs import triage as C
from sekura_sft.scoring import cvss
from sekura_sft.sources import attack_trees as at


@dataclass
class Result:
    score: float                       # 0..1, higher is better
    exact: bool = False
    detail: dict = field(default_factory=dict)


def _json(text: str) -> object | None:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    for lo, hi in (("{", "}"), ("[", "]")):
        a, b = t.find(lo), t.rfind(hi)
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except Exception:
                continue
    return None


# --- exact scorers --------------------------------------------------------

def score_cvss(pred: str, target: str, meta: dict) -> Result:
    p, t = _json(pred), _json(target)
    if not isinstance(p, dict) or not isinstance(t, dict):
        return Result(0.0)
    if "score" in t:
        got, want = p.get("score"), t.get("score")
        if not isinstance(got, (int, float)):
            return Result(0.0)
        exact = abs(float(got) - float(want)) < 0.05
        band_ok = p.get("severity") == t.get("severity")
        return Result(1.0 if (exact and band_ok) else (0.5 if exact else 0.0),
                      exact, {"delta": round(float(got) - float(want), 2)})
    # severity-consistency audit
    ok = p.get("verdict") == t.get("verdict")
    return Result(1.0 if ok else 0.0, ok)


def score_mapping(pred: str, target: str, meta: dict) -> Result:
    p, t = _json(pred), _json(target)
    if p is None or t is None:
        return Result(0.0)
    if isinstance(t, list):
        got = set(map(str, p)) if isinstance(p, list) else set()
        want = set(map(str, t))
        return Result(_jaccard(got, want), got == want)
    if not isinstance(p, dict):
        return Result(0.0)
    # The unmapped case is the whole point of that slice: 128 of 161 corpus CWEs
    # have no ATT&CK mapping, so a model that always emits a technique is wrong
    # 79% of the time. Any non-empty prediction against an empty target is a
    # hallucination and scores zero -- averaging the two empty-tactic lists into
    # it would hand out 0.5 for inventing a technique.
    want_empty = not (t.get("techniques") or t.get("tactics"))
    got_any = bool(p.get("techniques") or p.get("tactics"))
    if want_empty:
        return Result(0.0 if got_any else 1.0, not got_any, {"unmapped": True})
    scores, exact = [], True
    for key in ("techniques", "tactics"):
        want = {str(x) for x in (t.get(key) or [])}
        got = {str(x) for x in (p.get(key) or [])}
        scores.append(_jaccard(got, want))
        exact = exact and got == want
    return Result(sum(scores) / len(scores), exact)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0                      # both correctly empty (the unmapped case)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def score_exact_label(pred: str, target: str, meta: dict) -> Result:
    got = (pred or "").strip().strip('"`').split("\n")[0].strip()
    want = (target or "").strip()
    ok = got.upper() == want.upper()
    return Result(1.0 if ok else 0.0, ok, {"pred": got[:40]})


def score_attacktree_repair(pred: str, target: str, meta: dict) -> Result:
    """Graded: naming the defect, locating it, and supplying the fix are
    separately checkable, so a near-miss is distinguishable from a miss."""
    def parse(t: str) -> dict:
        out = {}
        for line in (t or "").splitlines():
            for key in ("DEFECT", "PATH", "FIX"):
                if line.startswith(key + ":"):
                    out[key] = line.split(":", 1)[1].strip()
        return out
    p, t = parse(pred), parse(target)
    if not t:
        return Result(0.0)
    parts = []
    parts.append(1.0 if p.get("PATH") == t.get("PATH") else 0.0)
    pd, td = p.get("DEFECT", "").lower(), t.get("DEFECT", "").lower()
    parts.append(1.0 if pd == td else (0.5 if pd and _overlap(pd, td) > 0.5 else 0.0))
    parts.append(1.0 if p.get("FIX", "").strip() == t.get("FIX", "").strip() else 0.0)
    exact = all(x == 1.0 for x in parts)
    return Result(sum(parts) / len(parts), exact,
                  {"path_ok": parts[0] == 1.0, "defect_ok": parts[1] == 1.0})


def _overlap(a: str, b: str) -> float:
    aw, bw = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
    return len(aw & bw) / len(aw | bw) if (aw | bw) else 0.0


# --- shaped scorers reused from RL ---------------------------------------

def score_planner(pred: str, target: str, meta: dict) -> Result:
    from sekura_sft.sources.recon import TargetInventory
    inv = meta.get("_inventory")
    ruleset = meta.get("ruleset", "v1")
    if inv is None:
        want = P.parse_plan(target) or []
        got = P.parse_plan(pred)
        if got is None:
            # Operationally identical to an empty plan: the pipeline gets no
            # tasks and the scan returns nothing. Not flagging it let a model
            # that emits prose slip past the zero-coverage gate.
            return Result(0.0, False, {"zero_coverage": bool(want),
                                       "unparseable": True})
        gs = {(t["surface_ref"], t["kind"]) for t in got}
        ws = {(t["surface_ref"], t["kind"]) for t in want}
        if not ws:
            return Result(1.0 if not gs else 0.0, not gs)
        coverage = len(ws & gs) / len(ws)
        return Result(coverage, gs == ws,
                      {"coverage": round(coverage, 3),
                       "zero_coverage": bool(ws) and not gs})
    s = P.score(pred, inv, ruleset)
    return Result(s.reward, s.components.get("coverage") == 1.0,
                  {**s.components, "zero_coverage": s.components.get("coverage", 0) == 0.0})


def score_triage(pred: str, target: str, meta: dict) -> Result:
    truth = _json(target)
    if not isinstance(truth, dict):
        return Result(0.0)
    s = C.score(pred, truth)
    got = C.parse(pred) or {}
    return Result(s.reward, got.get("verdict") == truth.get("verdict"),
                  {**s.components, "dangerous": s.dangerous,
                   "pred_verdict": got.get("verdict"),
                   "true_verdict": truth.get("verdict"),
                   "reason_exact": got.get("reason_code") == truth.get("reason_code")})


def _fields(text: str, keys: tuple[str, ...]) -> dict:
    out = {}
    for line in (text or "").splitlines():
        for key in keys:
            if line.upper().startswith(key + ":"):
                out[key] = line.split(":", 1)[1].strip()
    return out


def score_attacktree(pred: str, target: str, meta: dict) -> Result:
    """Node-level tasks, NOT full-tree authoring.

    Two shapes reach this: mask-a-branch reconstruction (TYPE + DESCRIPTION) and
    AND/OR inference (a single word). Routing them through the YAML authoring
    reward scored every one of them 0.0 -- caught by replaying targets as
    predictions, which is what that oracle floor is for.
    """
    if target.strip().upper().startswith("TYPE:"):
        t = _fields(target, ("TYPE", "DESCRIPTION"))
        p = _fields(pred, ("TYPE", "DESCRIPTION"))
        if not p:
            return Result(0.0)
        type_ok = p.get("TYPE", "").upper() == t.get("TYPE", "").upper()
        desc = _overlap(p.get("DESCRIPTION", "").lower(),
                        t.get("DESCRIPTION", "").lower())
        # Node type is the checkable half; the description is graded by overlap
        # because there are many correct ways to phrase the same step.
        return Result(0.6 * float(type_ok) + 0.4 * desc,
                      type_ok and desc > 0.9,
                      {"type_ok": type_ok, "desc_overlap": round(desc, 3)})
    return score_exact_label(pred, target, meta)


def score_remediation(pred: str, target: str, meta: dict) -> Result:
    """Either a chokepoint node id (exact) or COUNTERMEASURE + VERIFICATION
    prose (graded by overlap -- there is no single correct wording)."""
    t = target.strip()
    if "\n" not in t and len(t.split()) <= 3:
        return score_exact_label(pred, target, meta)
    want = _fields(target, ("COUNTERMEASURE", "VERIFICATION"))
    got = _fields(pred, ("COUNTERMEASURE", "VERIFICATION"))
    if not want:
        # Teacher-written remediation is markdown sections, not COUNTERMEASURE:/
        # VERIFICATION: fields. Returning the overlap without setting `exact`
        # reported 0.37 exact on a perfect replay of the targets themselves.
        cover = _overlap(pred.lower(), target.lower())
        return Result(cover, cover > 0.9, {"coverage": round(cover, 3)})
    if not got:
        return Result(0.0)
    parts = [_overlap(got.get(k, "").lower(), want.get(k, "").lower()) for k in want]
    mean = sum(parts) / len(parts)
    return Result(mean, mean > 0.9,
                  {k: round(v, 3) for k, v in zip(want, parts)})


def score_grounded_text(pred: str, target: str, meta: dict) -> Result:
    """Free-text answers that must be grounded in supplied evidence.

    There is no single correct wording for a threat model or an analyst answer,
    so this scores the properties that ARE checkable without a judge:

      citation validity  every [c#] the answer cites must exist in the evidence.
                         A fabricated citation is worse than none -- it teaches
                         that evidence ids are decorative.
      groundedness       security entities named in the answer (CVE/CWE/ATT&CK/
                         control ids) should appear in the evidence rather than
                         being recalled from parametric memory.
      coverage           overlap with the reference answer, as a weak signal that
                         the same ground was covered.

    Deliberately NOT a quality judgement -- that needs the LLM judge, reported
    separately so judged and deterministic metrics never blend into one number.
    """
    from sekura_sft.teacher.verify import RE_CID, entities

    evidence = meta.get("_evidence", "") if meta else ""
    text = (pred or "").strip()
    if not text:
        return Result(0.0)

    cited = {m.group(1) for m in RE_CID.finditer(text)}
    available = {m.group(1) for m in RE_CID.finditer(evidence)} if evidence else set()
    if available:
        bogus = cited - available
        citation = 0.0 if bogus else (1.0 if cited else 0.3)
    else:
        # Nothing to cite: the safety slice grounds in a question, not an
        # evidence bundle. Penalizing an answer for not citing sources that do
        # not exist capped that whole task at 0.82 on the oracle floor.
        citation = 0.0 if cited else 1.0

    ents = entities(text)
    if ents and evidence:
        grounded = len(ents & entities(evidence)) / len(ents)
    else:
        grounded = 1.0

    cover = _overlap(text.lower(), (target or "").lower())
    score = 0.35 * citation + 0.30 * grounded + 0.35 * min(1.0, cover / 0.6)
    return Result(min(1.0, score), cover > 0.9,
                  {"citation": round(citation, 3), "grounded": round(grounded, 3),
                   "coverage": round(cover, 3),
                   "fabricated_citations": sorted(cited - available)[:3]})


def score_detection(pred: str, target: str, meta: dict) -> Result:
    """Two shapes reach this task.

    Deterministic examples emit JSON (telemetry channels, Sigma template lists);
    teacher-written ones are prose containing a Sigma rule plus data sources and
    false positives. Routing both through the JSON scorer scored the prose ones
    0.25 -- the same mis-routing that scored attacktree node tasks 0.0, caught
    the same way, by replaying targets as predictions.
    """
    t = (target or "").strip()
    if t.startswith(("{", "[")) or "```json" in t:
        return score_mapping(pred, target, meta)
    return score_grounded_text(pred, target, meta)


SCORERS: dict[str, Callable[[str, str, dict], Result]] = {
    "cvss": score_cvss,
    "mapping": score_mapping,
    "detection": score_detection,
    "threat_model": score_grounded_text,
    "grounded_qa": score_grounded_text,
    "safety": score_grounded_text,
    "classify_cwe": score_exact_label,
    "classify_category": score_exact_label,
    "attacktree_repair": score_attacktree_repair,
    "attacktree": score_attacktree,
    "remediation": score_remediation,
    "planner": score_planner,
    "triage": score_triage,
}


def score_one(task: str, pred: str, target: str, meta: dict | None = None) -> Result:
    fn = SCORERS.get(task)
    if fn is None:
        # No scorer: fall back to exact match rather than silently scoring 1.0.
        return score_exact_label(pred, target, meta or {})
    return fn(pred, target, meta or {})
