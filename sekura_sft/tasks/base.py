"""The Example type every generator emits, plus split assignment.

Splits key on an ENTITY, never on the row. The corpus repeats itself in ways
that make row-wise splitting pure leakage: one advisory appears across up to 60
packages, and claims from one book section are mutually near-duplicative. The
split key is therefore the source_path / cve_id / subtree_id / rule-set version,
and it is hashed rather than shuffled so it survives dataset regeneration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from ..config import HELDOUT_CUT, HELDOUT_MOD

SYS_ANALYST = (
    "You are Sekura Defender, a defensive security analyst. You reason from the "
    "evidence you are given, cite it by its bracketed id, and say plainly when the "
    "evidence does not support a conclusion. You give full technical depth on "
    "attacker tradecraft because detection and remediation depend on it."
)

SYS_PIPELINE = (
    "You are Sekura Defender running inside an automated security pipeline. "
    "Emit only the requested structured output. No preamble, no commentary, no "
    "markdown fences unless the schema calls for them."
)


@dataclass
class Example:
    """One training example, before tokenization."""
    task: str
    system: str
    turns: list[dict]            # alternating user/assistant, ending on user
    target: str
    thinking: bool = False
    split_key: str = ""          # the ENTITY the split is keyed on
    forced_split: str = ""       # pin an entity to a split, e.g. a held-out subtree
    license_tier: str = "T0-permissive"
    meta: dict = field(default_factory=dict)

    def datum_id(self) -> str:
        payload = json.dumps(
            {"task": self.task, "turns": self.turns, "target": self.target},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def split(self, *, forced: str | None = None) -> str:
        """train / val / test, decided by a hash of the split key.

        Frozen by construction: regenerating the dataset cannot move an example
        across the boundary, so a held-out entity stays held out."""
        forced = forced or self.forced_split
        if forced:
            if forced not in ("train", "val", "test"):
                raise ValueError(f"bad forced split {forced!r}")
            return forced
        h = int(hashlib.sha256((self.split_key or self.datum_id()).encode()).hexdigest(), 16)
        bucket = h % HELDOUT_MOD
        if bucket < HELDOUT_CUT:
            return "test"
        if bucket < HELDOUT_CUT * 2:
            return "val"
        return "train"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datum_id"] = self.datum_id()
        d["split"] = self.split()
        return d
