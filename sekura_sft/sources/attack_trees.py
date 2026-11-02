"""Loader and structural walker for the sekura-attack-trees catalog.

This is the highest-quality data in the repo and the only source with a machine
validator, so it is treated as a first-class structured source rather than text.
`tools/validate.py` gives free, exact, unambiguous labels for the schema-repair
slice: inject a violation, and the validator's own error list IS the ground truth.

Deliberately imports the catalog's own tooling (yamllite, validate) rather than
reimplementing either. Two consequences worth knowing:
  * yamllite is a dependency-free YAML *subset* parser, which is why the catalog
    stays parseable with no third-party install.
  * validate_doc() walks one already-parsed document; it does not touch disk, so
    it is safe to call in a tight loop over generated candidates (an RL reward
    can call it thousands of times per step).

Measured structure of the catalog as committed:
  26 trees, 149 nodes (incl. the 26 logic_tree roots), 101 LEAF, 52 chokepoints
  (all 52 carry both countermeasure and verification_test), 35 unique ATT&CK
  techniques, 10 tactics, 8 categories (appsec 12, everything else 2).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import ATTACK_CATALOG, ATTACK_INDEX, ATTACK_TOOLS


def _load_catalog_tool(name: str):
    """Import a module from sekura-attack-trees/tools/ without requiring it to be
    an installed package."""
    path = ATTACK_TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_attack_tools_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


yamllite = _load_catalog_tool("yamllite")
validate = _load_catalog_tool("validate")

# Re-exported so generators and rewards share one definition of "valid".
CATEGORIES = validate.CATEGORIES
ACCESS_LEVELS = validate.ACCESS_LEVELS
NOISE = validate.NOISE
RELIABILITY = validate.RELIABILITY
SKILL = validate.SKILL
NODE_TYPES = validate.NODE_TYPES
KEBAB = validate.KEBAB
SEMVER = validate.SEMVER
TACTIC_RE = validate.TACTIC
TECHNIQUE_RE = validate.TECHNIQUE


@dataclass
class Node:
    """One node of a logic tree, with its path back to the root."""
    id: str
    type: str
    description: str
    parent: str | None
    depth: int
    subtree_id: str
    tooling: list[dict] = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)
    children: list[str] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.type == "LEAF"

    def telemetry_channels(self) -> list[str]:
        """Which of the four channels this leaf actually populates. Across the
        whole catalog only ~70 host/cloud telemetry facts exist, so this is the
        thinnest ground truth backing the detection-engineering capability."""
        return [k for k, v in (self.telemetry or {}).items() if v]


@dataclass
class Subtree:
    subtree_id: str
    name: str
    category: str
    path: Path
    doc: dict

    @property
    def preconditions(self) -> list[str]:
        return self.doc.get("contract", {}).get("preconditions", []) or []

    @property
    def postconditions(self) -> list[str]:
        return self.doc.get("contract", {}).get("postconditions", []) or []

    @property
    def access_level(self) -> str:
        return self.doc.get("contract", {}).get("prerequisite_access_level", "")

    @property
    def tactics(self) -> list[str]:
        return self.doc.get("mitre_attack_mappings", {}).get("tactics", []) or []

    @property
    def techniques(self) -> list[str]:
        return self.doc.get("mitre_attack_mappings", {}).get("techniques", []) or []

    @property
    def critical_nodes(self) -> list[dict]:
        return (self.doc.get("chokepoints_and_countermeasures", {})
                .get("critical_nodes", []) or [])

    def nodes(self) -> list[Node]:
        out: list[Node] = []
        root = self.doc.get("logic_tree") or {}

        def walk(n: dict, parent: str | None, depth: int):
            nid = n.get("id") or f"{self.subtree_id}-root"
            kids = n.get("children") or []
            node = Node(
                id=nid,
                type=n.get("type") or n.get("node_type") or "AND",
                description=n.get("description") or n.get("root_objective", ""),
                parent=parent, depth=depth, subtree_id=self.subtree_id,
                tooling=n.get("tooling_and_commands") or [],
                telemetry=n.get("telemetry_signatures") or {},
                children=[c.get("id", "") for c in kids],
            )
            out.append(node)
            for c in kids:
                walk(c, nid, depth + 1)

        walk(root, None, 0)
        return out

    def leaves(self) -> list[Node]:
        return [n for n in self.nodes() if n.is_leaf]

    def validate(self) -> list[str]:
        """The validator's error list. Empty means schema- and invariant-clean."""
        return validate.validate_doc(str(self.path), self.doc).errors


@lru_cache(maxsize=1)
def load_all() -> tuple[Subtree, ...]:
    subs = []
    for path in sorted(ATTACK_CATALOG.glob("*/*.yaml")):
        doc = yamllite.load(path.read_text(encoding="utf-8"))
        subs.append(Subtree(
            subtree_id=doc.get("subtree_id", path.stem),
            name=doc.get("name", ""),
            category=doc.get("category", path.parent.name),
            path=path, doc=doc,
        ))
    return tuple(subs)


@lru_cache(maxsize=1)
def load_index() -> dict:
    """catalog/index.json -- the composition domain. Note COMPOSITION.md states
    postcondition/precondition matching is SEMANTIC, not string-equal, so this
    enumerates candidate pairs; it does not decide them."""
    return json.loads(ATTACK_INDEX.read_text(encoding="utf-8"))


def by_id() -> dict[str, Subtree]:
    return {s.subtree_id: s for s in load_all()}


def all_nodes() -> Iterator[Node]:
    for s in load_all():
        yield from s.nodes()


def catalog_stats() -> dict:
    subs = load_all()
    nodes = list(all_nodes())
    leaves = [n for n in nodes if n.is_leaf]
    crit = [c for s in subs for c in s.critical_nodes]
    tech = {t for s in subs for t in s.techniques}
    tac = {t for s in subs for t in s.tactics}
    cats: dict[str, int] = {}
    for s in subs:
        cats[s.category] = cats.get(s.category, 0) + 1
    return {
        "subtrees": len(subs),
        "nodes": len(nodes),
        "leaves": len(leaves),
        "internal": len(nodes) - len(leaves),
        "critical_nodes": len(crit),
        "critical_with_verification": sum(1 for c in crit if c.get("verification_test")),
        "tooling_entries": sum(len(n.tooling) for n in nodes),
        "telemetry_channels_populated": sum(len(n.telemetry_channels()) for n in leaves),
        "unique_techniques": len(tech),
        "unique_tactics": len(tac),
        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "access_levels": {a: sum(1 for s in subs if s.access_level == a)
                          for a in sorted({s.access_level for s in subs})},
        "invalid": [s.subtree_id for s in subs if s.validate()],
    }
