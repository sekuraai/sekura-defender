"""Grounding vocabularies for the attack-tree reward.

THE CIRCULAR-GROUNDING HAZARD, and why this module is strict:

If a teacher writes an attack tree, the tree is committed to the catalog, and the
catalog then feeds the vocabulary that R_grounding scores against, the teacher's
hallucinations become ground truth. A fabricated technique id like T1548.006
would be laundered into "verified vocabulary", and R_grounding -- the component
built specifically to reject garbage content -- would start REWARDING it.

Human prose review does not catch this. A reviewer skimming a tree for technical
plausibility will not notice that a technique id does not exist.

So every entry carries a provenance tag, and entries whose only provenance is
generated catalog content are refused. Generated trees are validated AGAINST the
vocabulary; they never extend it. Extending it is a separate, deliberate act
requiring an external citation.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from sekura_sft.config import CWE_TO_ATTACK, POOL_DIR, REPO

# Provenance values that may ground a reward. "catalog-generated" is absent by
# design and is rejected at load.
EXTERNAL = frozenset({"cwe_to_attack", "pentest-kg", "kali-kg", "knowledge-jsonl",
                      "mitre", "curated-external"})
FORBIDDEN = frozenset({"catalog-generated", "teacher", "model-output"})

RE_TECHNIQUE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
RE_TACTIC = re.compile(r"\bTA\d{4}\b")

# A tool name, not a command flag. Harvesting `subject` and `name` fields
# indiscriminately pulled in 9,968 "tools" including "--exclude-ports",
# "!analyze -v" and "--max-hostgroup", which would make the grounding check
# match almost any string and stop being a check at all.
RE_TOOL_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_. ][a-z0-9]+){0,3}$")
TOOL_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "with", "tool", "tools", "command",
    "commands", "option", "options", "flag", "flags", "file", "files", "user",
    "users", "host", "hosts", "port", "ports", "target", "output", "input",
    "example", "note", "step", "steps", "value", "values", "name", "type",
})


def _is_tool_name(s: str) -> bool:
    s = s.strip().lower()
    if not (1 < len(s) <= 30):
        return False
    if s[0] in "-!<>/\\.@#$%*([" or s in TOOL_STOPWORDS:
        return False
    if not RE_TOOL_NAME.match(s):
        return False
    return not all(w in TOOL_STOPWORDS for w in re.split(r"[-_. ]", s))

# Real Windows Security / Sysmon event ids. Curated externally; a tree that cites
# an id outside this set is asserting telemetry that does not exist.
WINDOWS_EVENT_IDS = frozenset({
    1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25,
    1102, 4103, 4104, 4624, 4625, 4634, 4648, 4657, 4662, 4670, 4672, 4673, 4688,
    4697, 4698, 4699, 4700, 4702, 4719, 4720, 4722, 4724, 4728, 4732, 4738, 4740,
    4756, 4768, 4769, 4770, 4771, 4776, 4778, 4779, 4798, 4799, 5136, 5140, 5145,
    5379, 5861, 7034, 7036, 7045, 8004, 8222,
})


class ProvenanceError(RuntimeError):
    pass


@dataclass
class Vocab:
    techniques: set[str] = field(default_factory=set)
    tactics: set[str] = field(default_factory=set)
    tools: set[str] = field(default_factory=set)
    windows_event_ids: set[int] = field(default_factory=set)
    provenance: dict = field(default_factory=dict)

    def check_provenance(self) -> None:
        bad = {k: v for k, v in self.provenance.items() if v in FORBIDDEN}
        if bad:
            raise ProvenanceError(
                f"vocabulary entries grounded only in generated content: {list(bad)[:5]}. "
                "Generated trees are validated against the vocabulary; they never "
                "extend it.")

    def summary(self) -> dict:
        return {"techniques": len(self.techniques), "tactics": len(self.tactics),
                "tools": len(self.tools),
                "windows_event_ids": len(self.windows_event_ids),
                "sources": dict(Counter(self.provenance.values()))}


def _from_cwe_attack(v: Vocab) -> None:
    data = json.loads(CWE_TO_ATTACK.read_text(encoding="utf-8"))
    for entry in data.values():
        for t in entry.get("techniques", []) or []:
            if tid := t.get("id"):
                v.techniques.add(tid)
                v.provenance[f"tech:{tid}"] = "cwe_to_attack"
            if tac := t.get("tactic_id"):
                v.tactics.add(tac)
                v.provenance[f"tac:{tac}"] = "cwe_to_attack"


def _from_graphs(v: Vocab) -> None:
    for name, tag in (("pentest-knowledge-graph.json", "pentest-kg"),
                      ("kali_linux_knowledge_graph.json", "kali-kg"),
                      ("network_pentest_knowledge_graph.json", "pentest-kg"),
                      ("appsec_knowledge_graph.json", "pentest-kg")):
        path = REPO / name
        if not path.exists():
            continue
        blob = path.read_text(encoding="utf-8")
        for tid in RE_TECHNIQUE.findall(blob):
            v.techniques.add(tid)
            v.provenance.setdefault(f"tech:{tid}", tag)
        for tac in RE_TACTIC.findall(blob):
            v.tactics.add(tac)
            v.provenance.setdefault(f"tac:{tac}", tag)
        try:
            data = json.loads(blob)
        except Exception:
            continue
        _harvest_tools(data, v, tag)


def _harvest_tools(obj, v: Vocab, tag: str) -> None:
    if isinstance(obj, dict):
        for key in ("tool", "tools", "name"):
            val = obj.get(key)
            for candidate in ([val] if isinstance(val, str) else
                              val if isinstance(val, list) else []):
                if isinstance(candidate, str) and _is_tool_name(candidate):
                    name = candidate.strip().lower()
                    v.tools.add(name)
                    v.provenance.setdefault(f"tool:{name}", tag)
        for value in obj.values():
            _harvest_tools(value, v, tag)
    elif isinstance(obj, list):
        for item in obj:
            _harvest_tools(item, v, tag)


def _from_pool(v: Vocab, limit: int = 400_000) -> None:
    """Tool names from the curated corpus -- records typed `tool`, which came
    from harvested third-party sources, never from generated catalog content."""
    pool = POOL_DIR / "pool.jsonl"
    if not pool.exists():
        return
    with open(pool, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "tool":
                continue
            subj = (rec.get("subject") or "").strip().lower()
            if _is_tool_name(subj):
                v.tools.add(subj)
                v.provenance.setdefault(f"tool:{subj}", "knowledge-jsonl")


@lru_cache(maxsize=1)
def load() -> Vocab:
    v = Vocab()
    _from_cwe_attack(v)
    _from_graphs(v)
    _from_pool(v)
    v.windows_event_ids = set(WINDOWS_EVENT_IDS)
    for eid in v.windows_event_ids:
        v.provenance[f"evt:{eid}"] = "curated-external"
    v.check_provenance()
    return v
