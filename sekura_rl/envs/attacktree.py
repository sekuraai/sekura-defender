"""ENV-A: attack-subtree authoring. Highest value, hardest reward.

    R = 0.30*parse_schema + 0.20*invariants + 0.15*grounding + 0.15*coverage
      + 0.10*detection_quality + 0.10*composability - penalties

R_GROUNDING IS THE POINT. validate.py happily accepts
`windows_event_ids: [1,2,3]` and `tool: "foo"` -- it checks SHAPE, not TRUTH. A
model optimizing only the validator learns to emit a minimal schema-valid tree
full of invented techniques, invented tools and invented event ids. Grounding is
what makes that not pay.

Grounding is GRADED, not binary, because the harvested vocabulary is a subset:
94 ATT&CK techniques against the real ~600. A well-formed technique id that we
simply have not harvested scores partial credit rather than zero, so the reward
does not punish a model for being right about something our sources missed.

R_COVERAGE PLATEAUS. Structural richness is compared against the profile
measured from the 26 real trees (3.9 leaves each); there is no bonus above
target, so emitting 50 junk leaves buys nothing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from sekura_sft.sources import attack_trees as at
from sekura_sft.sources.yamlemit import dumps

from ..vocab import load as load_vocab

# Structure GATES content; it does not substitute for it.
#
# A flat weighted sum is farmable by construction. With parse_schema and
# invariants worth 0.50 between them, a minimal schema-valid tree full of
# invented techniques, tools and event ids scored 0.67 against a real tree's
# 0.93 -- a gap narrow enough for a model to happily plateau on fabrication,
# since validate.py checks SHAPE and never TRUTH.
#
# So validity is a multiplier and content is what is actually being scored.
# Perfect structure with fabricated content cannot exceed CONTENT_FLOOR plus a
# small share, while genuine content is rewarded in full. This is the same
# anti-farming argument the composition environment needs, applied here.
CONTENT_WEIGHTS = {"grounding": 0.40, "coverage": 0.25,
                   "detection_quality": 0.20, "composability": 0.15}
CONTENT_FLOOR = 0.15

# Measured from the catalog: 26 trees, 101 leaves, 52 chokepoints, 149 nodes.
TARGET_LEAVES = 3
TARGET_TECHNIQUES = 2
TARGET_CHOKEPOINTS = 2

PENALTY_DUPLICATE = 0.50      # copying a catalog file is the default hack at 26 seeds
PENALTY_REPETITION = 0.30
PENALTY_TRUNCATED = 0.20
DUP_JACCARD = 0.80

RE_TECH_ID = re.compile(r"^(T\d{4}(?:\.\d{3})?):\s*.+")
RE_TACTIC_ID = re.compile(r"^(TA\d{4}):\s*.+")
RE_HTTP = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH|HEAD)\b|/[\w.\-]+", re.I)


@dataclass
class Score:
    reward: float
    components: dict = field(default_factory=dict)
    penalties: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"\w+", (text or "").lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _leaves(doc: dict) -> list[dict]:
    out: list[dict] = []

    def w(n):
        if n.get("type") == "LEAF":
            out.append(n)
        for c in n.get("children") or []:
            w(c)
    w(doc.get("logic_tree") or {})
    return out


def _all_nodes(doc: dict) -> list[dict]:
    out: list[dict] = []

    def w(n):
        out.append(n)
        for c in n.get("children") or []:
            w(c)
    w(doc.get("logic_tree") or {})
    return out


def strip_fence(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", t, re.S)
    return m.group(1).strip() if m else t


def score(text: str, *, stop_reason: str = "stop",
          catalog_shingles: list[set] | None = None,
          group_texts: list[str] | None = None) -> Score:
    vocab = load_vocab()
    body = strip_fence(text)

    # Hard zero: no partial credit for prose with a YAML-ish preamble.
    try:
        doc = at.yamllite.load(body)
    except Exception as exc:                                    # noqa: BLE001
        return Score(0.0, {"parse_schema": 0.0}, {}, [f"unparseable YAML: {exc}"])
    if not isinstance(doc, dict):
        return Score(0.0, {"parse_schema": 0.0}, {}, ["not a mapping"])

    subtree_id = doc.get("subtree_id", "generated")
    category = doc.get("category", "appsec")
    fake_path = f"catalog/{category}/{subtree_id}.yaml"
    errors = at.validate.validate_doc(fake_path, doc).errors
    parse_schema = max(0.0, 1.0 - len(errors) / 8.0)
    invariants = 1.0 if not errors else max(0.0, 1.0 - len(errors) / 12.0)

    nodes = _all_nodes(doc)
    leaves = _leaves(doc)

    # --- grounding ---------------------------------------------------------
    techs = doc.get("mitre_attack_mappings", {}).get("techniques", []) or []
    tacs = doc.get("mitre_attack_mappings", {}).get("tactics", []) or []
    tech_scores = []
    for t in techs:
        m = RE_TECH_ID.match(str(t))
        if not m:
            tech_scores.append(0.0)
        elif m.group(1) in vocab.techniques:
            tech_scores.append(1.0)
        else:
            tech_scores.append(0.5)     # well-formed but outside our subset
    tac_scores = [1.0 if (m := RE_TACTIC_ID.match(str(t))) and m.group(1) in vocab.tactics
                  else (0.5 if RE_TACTIC_ID.match(str(t)) else 0.0) for t in tacs]

    tools = [tc.get("tool", "") for n in nodes for tc in (n.get("tooling_and_commands") or [])]
    tool_scores = [1.0 if str(t).strip().lower() in vocab.tools else 0.3 for t in tools]

    evt_ids = [e for n in leaves
               for e in (n.get("telemetry_signatures", {}) or {}).get("windows_event_ids", [])
               if isinstance(e, int)]
    evt_scores = [1.0 if e in vocab.windows_event_ids else 0.0 for e in evt_ids]

    parts = tech_scores + tac_scores + tool_scores + evt_scores
    grounding = sum(parts) / len(parts) if parts else 0.0

    # --- coverage (plateaus: no bonus above target) ------------------------
    n_leaves = len(leaves)
    distinct_tech = len({RE_TECH_ID.match(str(t)).group(1) for t in techs
                         if RE_TECH_ID.match(str(t))})
    node_types = {n.get("type") or n.get("node_type") for n in nodes}
    chokepoints = (doc.get("chokepoints_and_countermeasures", {})
                   .get("critical_nodes", []) or [])
    coverage = sum([
        min(1.0, n_leaves / TARGET_LEAVES),
        min(1.0, distinct_tech / TARGET_TECHNIQUES),
        1.0 if {"AND", "OR"} & node_types and len(node_types) > 1 else 0.0,
        min(1.0, len(chokepoints) / TARGET_CHOKEPOINTS),
    ]) / 4.0

    # --- detection quality -------------------------------------------------
    dq = []
    for leaf in leaves:
        tel = leaf.get("telemetry_signatures") or {}
        channels = sum(1 for k in ("windows_event_ids", "linux_auditd",
                                   "cloudtrail_events", "web_logs") if tel.get(k))
        web = str(tel.get("web_logs") or "")
        good_web = 1.0 if (len(web) >= 20 and RE_HTTP.search(web)) else 0.0
        cmds = leaf.get("tooling_and_commands") or []
        good_cmd = 1.0 if any(len(str(c.get("command_example", "")).split()) >= 2
                              for c in cmds) else 0.0
        dq.append((min(1.0, channels / 2) + good_web + good_cmd) / 3)
    detection_quality = sum(dq) / len(dq) if dq else 0.0

    # --- composability -----------------------------------------------------
    index = at.load_index()
    pre_all, post_all = set(), set()
    for s in index.get("subtrees", []):
        pre_all |= {p.lower() for p in s.get("preconditions", [])}
        post_all |= {p.lower() for p in s.get("postconditions", [])}
    contract = doc.get("contract", {}) or {}
    mine_pre = [str(p).lower() for p in contract.get("preconditions", []) or []]
    mine_post = [str(p).lower() for p in contract.get("postconditions", []) or []]

    def overlaps(a: str, pool: set) -> bool:
        aw = set(re.findall(r"\w+", a))
        return any(len(aw & set(re.findall(r"\w+", b))) >= 3 for b in pool)

    composability = (
        (0.5 if any(overlaps(p, post_all) for p in mine_pre) else 0.0)
        + (0.5 if any(overlaps(p, pre_all) for p in mine_post) else 0.0))

    # --- penalties ---------------------------------------------------------
    pen: dict[str, float] = {}
    if catalog_shingles is not None:
        mine = _shingles(body)
        if any(_jaccard(mine, c) > DUP_JACCARD for c in catalog_shingles):
            pen["near_duplicate_of_catalog"] = PENALTY_DUPLICATE
    words = re.findall(r"\w+", body.lower())
    grams = Counter(" ".join(words[i:i + 20]) for i in range(max(0, len(words) - 19)))
    if grams and grams.most_common(1)[0][1] > 3:
        pen["degenerate_repetition"] = PENALTY_REPETITION
    if stop_reason != "stop":
        pen["truncated"] = PENALTY_TRUNCATED

    structural = parse_schema * invariants          # gate, not an addend
    content = sum(CONTENT_WEIGHTS[k] * v for k, v in (
        ("grounding", grounding), ("coverage", coverage),
        ("detection_quality", detection_quality), ("composability", composability)))
    base = structural * (CONTENT_FLOOR + (1 - CONTENT_FLOOR) * content)
    reward = max(0.0, min(1.0, base - sum(pen.values())))

    notes = []
    if errors:
        notes.append(f"{len(errors)} validator error(s): {errors[0][:70]}")
    return Score(round(reward, 4),
                 {"structural": round(structural, 3),
                  "parse_schema": round(parse_schema, 3), "invariants": round(invariants, 3),
                  "grounding": round(grounding, 3), "coverage": round(coverage, 3),
                  "detection_quality": round(detection_quality, 3),
                  "composability": round(composability, 3)},
                 pen, notes)


def catalog_shingles() -> list[set]:
    return [_shingles(dumps(s.doc)) for s in at.load_all()]
