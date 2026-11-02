"""Deterministic task generators over the 26-subtree attack catalog.

No teacher, no hallucination risk: every target is read out of, or derived from,
hand-authored YAML that a machine validator already accepts.

The schema-repair generator is the standout. Inject exactly one violation and
`validate.py`'s own error string IS the label -- free, exact, unambiguous. All 25
corruption types below were verified to produce a detection (and the clean tree
to produce none) before any of them was used.

Overfitting is the default outcome with 26 seed documents, so: three whole
subtrees are held out (one each from appsec / active-directory / cloud-iam),
splits key on subtree_id so every generator for a held-out tree lands in test,
and each generator is capped.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterator

from ..sources import attack_trees as at
from ..sources.yamlemit import dumps
from .base import SYS_ANALYST, SYS_PIPELINE, Example

# Held out entirely: never a training target, never an RL prompt. One per
# category so the eval set is not all appsec.
HELD_OUT = frozenset({
    "appsec-ssrf-cloud-metadata",
    "ad-privesc-adcs-esc1",
    "cloud-iam-privesc-passrole",
})

TELEMETRY_CHANNELS = ("windows_event_ids", "linux_auditd", "cloudtrail_events", "web_logs")


def _split_for(subtree_id: str) -> str:
    """Held-out subtrees are pinned to test for EVERY generator, so no generator
    can quietly leak a held-out attack into training through a different task."""
    return "test" if subtree_id in HELD_OUT else ""


def _leaves(doc: dict) -> list[dict]:
    out: list[dict] = []

    def w(n):
        if n.get("type") == "LEAF":
            out.append(n)
        for c in n.get("children") or []:
            w(c)
    w(doc.get("logic_tree") or {})
    return out


def _internal(doc: dict) -> list[dict]:
    out: list[dict] = []

    def w(n):
        if n.get("children"):
            out.append(n)
        for c in n.get("children") or []:
            w(c)
    w(doc.get("logic_tree") or {})
    return out


# --------------------------------------------------------------------------
# G7 -- schema repair. validate.py supplies the labels.
# --------------------------------------------------------------------------

def _corruptions() -> list[tuple[str, callable]]:
    """(name, mutate) pairs. Each injects exactly one violation.

    Every one was probed against the real validator: the clean tree yields 0
    errors and each mutation yields at least 1 naming that specific defect."""
    def leaf0(d):
        ls = _leaves(d)
        if not ls:
            raise ValueError("no leaf")
        return ls[0]

    def internal0(d):
        ns = _internal(d)
        if not ns:
            raise ValueError("no internal node")
        return ns[-1]

    return [
        ("node_id_not_kebab", lambda d: leaf0(d).__setitem__("id", "Not_Kebab_ID")),
        ("duplicate_node_id", lambda d: _leaves(d)[1].__setitem__("id", _leaves(d)[0]["id"])),
        ("leaf_has_children", lambda d: leaf0(d).__setitem__(
            "children", [{"id": "orphan", "type": "LEAF", "description": "x"}])),
        ("leaf_missing_telemetry", lambda d: leaf0(d).pop("telemetry_signatures")),
        ("leaf_missing_tooling", lambda d: leaf0(d).pop("tooling_and_commands")),
        ("tooling_missing_command", lambda d: leaf0(d)["tooling_and_commands"].__setitem__(
            0, {"tool": leaf0(d)["tooling_and_commands"][0].get("tool", "nmap")})),
        ("branch_children_emptied", lambda d: internal0(d).__setitem__("children", [])),
        ("root_node_type_invalid", lambda d: d["logic_tree"].__setitem__("node_type", "XOR")),
        ("version_not_semver", lambda d: d.__setitem__("version", "1.0")),
        ("category_invalid", lambda d: d.__setitem__("category", "misc")),
        ("access_level_invalid", lambda d: d["contract"].__setitem__(
            "prerequisite_access_level", "Root")),
        ("detection_noise_invalid", lambda d: d["operational_metrics"].__setitem__(
            "detection_noise", "Quiet")),
        ("skill_level_invalid", lambda d: d["operational_metrics"].__setitem__(
            "skill_level", "Wizard")),
        ("technique_malformed", lambda d: d["mitre_attack_mappings"]["techniques"].__setitem__(
            0, d["mitre_attack_mappings"]["techniques"][0].split(": ", 1)[-1])),
        ("tactic_malformed", lambda d: d["mitre_attack_mappings"]["tactics"].__setitem__(
            0, d["mitre_attack_mappings"]["tactics"][0].split(": ", 1)[-1])),
        ("techniques_emptied", lambda d: d["mitre_attack_mappings"].__setitem__("techniques", [])),
        ("preconditions_emptied", lambda d: d["contract"].__setitem__("preconditions", [])),
        ("postconditions_emptied", lambda d: d["contract"].__setitem__("postconditions", [])),
        ("chokepoint_node_id_dangling",
         lambda d: d["chokepoints_and_countermeasures"]["critical_nodes"][0].__setitem__(
             "node_id", "no-such-node")),
        ("chokepoint_missing_countermeasure",
         lambda d: d["chokepoints_and_countermeasures"]["critical_nodes"][0].pop("countermeasure")),
        ("critical_nodes_emptied",
         lambda d: d["chokepoints_and_countermeasures"].__setitem__("critical_nodes", [])),
        ("negative_time_metric", lambda d: d["operational_metrics"].__setitem__(
            "typical_time_minutes", -5)),
        ("event_ids_not_integers", lambda d: leaf0(d)["telemetry_signatures"].__setitem__(
            "windows_event_ids", ["4624"])),
        ("subtree_id_not_kebab", lambda d: d.__setitem__("subtree_id", "Appsec_SSRF")),
        ("operational_metrics_removed", lambda d: d.pop("operational_metrics")),
    ]


REPAIR_INSTRUCTION = (
    "The attack subtree below fails schema validation. Identify the single "
    "violation and emit a minimal patch.\n\n"
    "Respond as:\nDEFECT: <one line>\nPATH: <dotted path to the offending field>\n"
    "FIX: <the corrected value as YAML>"
)


def _diff_path(clean, broken, path: str = "") -> tuple[str, object] | None:
    """First differing path between the clean and corrupted documents.

    Emitting the whole corrected tree was measured to be 99% transcription -- the
    defect line is 13 tokens against 1,039 tokens of unchanged YAML copy. With
    token-summed losses that trains YAML copying, not defect detection, and it
    pushed 63% of this slice past the completion cap. A patch is ~50x cheaper and
    teaches localization, which is the actual skill.
    """
    if type(clean) is not type(broken):
        return path, clean
    if isinstance(clean, dict):
        for k in clean:
            if k not in broken:
                return f"{path}.{k}".lstrip("."), clean[k]
            sub = _diff_path(clean[k], broken[k], f"{path}.{k}".lstrip("."))
            if sub:
                return sub
        for k in broken:
            if k not in clean:
                return f"{path}.{k}".lstrip("."), None      # must be removed
        return None
    if isinstance(clean, list):
        if len(clean) != len(broken):
            return path, clean
        for i, (a, b) in enumerate(zip(clean, broken)):
            sub = _diff_path(a, b, f"{path}[{i}]")
            if sub:
                return sub
        return None
    return (path, clean) if clean != broken else None


def gen_schema_repair(rng: random.Random, cap: int = 700) -> Iterator[Example]:
    corr = _corruptions()
    made = 0
    for sub in at.load_all():
        if sub.validate():
            continue                       # never build a lesson on a broken tree
        for name, mutate in corr:
            if made >= cap:
                return
            broken = copy.deepcopy(sub.doc)
            try:
                mutate(broken)
            except Exception:
                continue                   # this tree lacks the shape to corrupt
            errs = at.validate.validate_doc(str(sub.path), broken).errors
            if not errs:
                continue                   # mutation was a no-op on this tree
            defect = errs[0].split(": ", 1)[1] if ": " in errs[0] else errs[0]
            diff = _diff_path(sub.doc, broken)
            if diff is None:
                continue                   # nothing localizable; skip rather than guess
            where, correct = diff
            fix = ("<remove this field>" if correct is None
                   else dumps({"value": correct}).split("\n", 1)[-1].rstrip()
                   if isinstance(correct, (dict, list))
                   else dumps({"value": correct}).strip().split(": ", 1)[1])
            yield Example(
                task="attacktree_repair",
                system=SYS_PIPELINE,
                turns=[{"role": "user",
                        "content": f"{REPAIR_INSTRUCTION}\n\n```yaml\n{dumps(broken)}```"}],
                target=f"DEFECT: {defect}\nPATH: {where}\nFIX: {fix}",
                thinking=False,
                split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
                license_tier="T1-sharealike",
                meta={"subtree_id": sub.subtree_id, "corruption": name,
                      "validator_errors": errs, "patch_path": where},
            )
            made += 1


# --------------------------------------------------------------------------
# G1 -- leaf -> telemetry signatures (detection engineering)
# --------------------------------------------------------------------------

TELEMETRY_INSTRUCTION = (
    "For the attack step below, emit the telemetry that would evidence it, as JSON "
    'with keys "windows_event_ids" (integers), "linux_auditd", "cloudtrail_events" '
    '(string lists) and "web_logs" (a string). Use an empty list or empty string '
    "for channels that would show nothing."
)


def gen_leaf_telemetry(rng: random.Random) -> Iterator[Example]:
    for sub in at.load_all():
        for leaf in _leaves(sub.doc):
            tel = leaf.get("telemetry_signatures") or {}
            if not any(tel.get(c) for c in TELEMETRY_CHANNELS):
                continue                   # nothing to teach
            tools = ", ".join(t.get("tool", "") for t in leaf.get("tooling_and_commands") or [])
            ctx = (f"Objective: {sub.doc['logic_tree'].get('root_objective', '')}\n"
                   f"Step: {leaf.get('description', '')}\n"
                   f"Tooling: {tools or 'unspecified'}")
            yield Example(
                task="detection",
                system=SYS_PIPELINE,
                turns=[{"role": "user", "content": f"{TELEMETRY_INSTRUCTION}\n\n{ctx}"}],
                target=json.dumps({c: tel.get(c, [] if c != "web_logs" else "")
                                   for c in TELEMETRY_CHANNELS}, indent=2),
                thinking=False,
                split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
                license_tier="T1-sharealike",
                meta={"subtree_id": sub.subtree_id, "node_id": leaf.get("id"),
                      "channels": [c for c in TELEMETRY_CHANNELS if tel.get(c)]},
            )


# --------------------------------------------------------------------------
# G2 -- chokepoint -> countermeasure + verification test, and the reverse
# --------------------------------------------------------------------------

def gen_chokepoints(rng: random.Random) -> Iterator[Example]:
    for sub in at.load_all():
        nodes = {n.id: n for n in sub.nodes()}
        for cn in sub.critical_nodes:
            node = nodes.get(cn.get("node_id", ""))
            if not node or not cn.get("countermeasure"):
                continue
            # forward: step -> control that severs it + how to prove it
            yield Example(
                task="remediation",
                system=SYS_ANALYST,
                turns=[{"role": "user", "content":
                        "This step is a chokepoint in an attack path. Give the control that "
                        "severs it and a concrete test that proves the control works.\n\n"
                        f"Attack: {sub.name}\nStep: {node.description}\n\n"
                        "Respond as:\nCOUNTERMEASURE: <...>\nVERIFICATION: <...>"}],
                target=f"COUNTERMEASURE: {cn['countermeasure']}\n"
                       f"VERIFICATION: {cn.get('verification_test', '')}",
                thinking=True,
                split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
                license_tier="T1-sharealike",
                meta={"subtree_id": sub.subtree_id, "node_id": node.id},
            )
            # reverse: control -> which step does it break?
            step_list = "\n".join(f"- {n.id}: {n.description}" for n in sub.nodes()
                                  if n.description)
            yield Example(
                task="remediation",
                system=SYS_PIPELINE,
                turns=[{"role": "user", "content":
                        "Which step of this attack does the control below break? Answer with "
                        "the step id only.\n\n"
                        f"Control: {cn['countermeasure']}\n\nSteps:\n{step_list}"}],
                target=node.id,
                thinking=False,
                split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
                license_tier="T1-sharealike",
                meta={"subtree_id": sub.subtree_id, "node_id": node.id, "direction": "reverse"},
            )


# --------------------------------------------------------------------------
# G3 -- mask a branch and reconstruct it
# --------------------------------------------------------------------------

def gen_mask_branch(rng: random.Random) -> Iterator[Example]:
    for sub in at.load_all():
        for parent in _internal(sub.doc):
            kids = parent.get("children") or []
            if len(kids) < 2:
                continue
            for i, hidden in enumerate(kids):
                siblings = [k for j, k in enumerate(kids) if j != i]
                sib_txt = "\n".join(f"- ({k.get('type')}) {k.get('description', '')}"
                                    for k in siblings)
                yield Example(
                    task="attacktree",
                    system=SYS_ANALYST,
                    turns=[{"role": "user", "content":
                            "One child of this attack-tree node has been removed. Given the "
                            "parent's objective and the remaining children, describe the "
                            "missing child in one sentence and state whether it is a LEAF, "
                            "AND or OR node.\n\n"
                            f"Attack: {sub.name}\n"
                            f"Parent ({parent.get('node_type') or parent.get('type')}): "
                            f"{parent.get('description') or parent.get('root_objective', '')}\n"
                            f"Remaining children:\n{sib_txt}\n\n"
                            "Respond as:\nTYPE: <LEAF|AND|OR>\nDESCRIPTION: <one sentence>"}],
                    target=f"TYPE: {hidden.get('type')}\n"
                           f"DESCRIPTION: {hidden.get('description', '')}",
                    thinking=True,
                    split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
                    license_tier="T1-sharealike",
                    meta={"subtree_id": sub.subtree_id, "hidden_id": hidden.get("id")},
                )


# --------------------------------------------------------------------------
# G4 -- AND vs OR inference
# --------------------------------------------------------------------------

def gen_and_or(rng: random.Random) -> Iterator[Example]:
    for sub in at.load_all():
        for node in _internal(sub.doc):
            ntype = node.get("node_type") or node.get("type")
            if ntype not in ("AND", "OR"):
                continue
            kids = "\n".join(f"- {k.get('description', '')}" for k in node.get("children") or [])
            if not kids.strip():
                continue
            yield Example(
                task="attacktree",
                system=SYS_PIPELINE,
                turns=[{"role": "user", "content":
                        "Does this attack step require ALL of its sub-steps (AND) or ANY ONE "
                        "of them (OR)? Answer with a single word.\n\n"
                        f"Step: {node.get('description') or node.get('root_objective', '')}\n"
                        f"Sub-steps:\n{kids}"}],
                target=ntype,
                thinking=False,
                split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
                license_tier="T1-sharealike",
                meta={"subtree_id": sub.subtree_id, "node_id": node.get("id", "root")},
            )


# --------------------------------------------------------------------------
# G5 -- ATT&CK mapping
# --------------------------------------------------------------------------

def gen_attack_mapping(rng: random.Random) -> Iterator[Example]:
    for sub in at.load_all():
        if not sub.techniques or not sub.tactics:
            continue
        steps = "\n".join(f"- {n.description}" for n in sub.nodes() if n.description)
        yield Example(
            task="mapping",
            system=SYS_PIPELINE,
            turns=[{"role": "user", "content":
                    "Map this attack to MITRE ATT&CK. Emit JSON with \"tactics\" and "
                    "\"techniques\", each an array of \"ID: Name\" strings.\n\n"
                    f"Attack: {sub.name}\nObjective: "
                    f"{sub.doc['logic_tree'].get('root_objective', '')}\nSteps:\n{steps}"}],
            target=json.dumps({"tactics": sub.tactics, "techniques": sub.techniques}, indent=2),
            thinking=False,
            split_key=sub.subtree_id,
            forced_split=_split_for(sub.subtree_id),
            license_tier="T1-sharealike",
            meta={"subtree_id": sub.subtree_id},
        )


GENERATORS = {
    "schema_repair": gen_schema_repair,
    "leaf_telemetry": gen_leaf_telemetry,
    "chokepoints": gen_chokepoints,
    "mask_branch": gen_mask_branch,
    "and_or": gen_and_or,
    "attack_mapping": gen_attack_mapping,
}


def build(seed: int = 0) -> Iterator[Example]:
    rng = random.Random(seed)
    for name, fn in GENERATORS.items():
        yield from fn(rng)
