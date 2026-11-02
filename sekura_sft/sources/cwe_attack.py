"""CWE -> ATT&CK technique/tactic + Sigma template mapping.

Exact-match labels, but with a coverage hole that must be taught rather than
hidden: the map covers 51 CWEs while the findings corpus names 162 distinct
ones, so it joins only ~55% of findings-with-CWE. A model trained only on the
joinable half learns to always produce a technique, and will hallucinate one for
the other 45%. The generators therefore emit explicit "no mapping" targets.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ..config import CWE_TO_ATTACK


@lru_cache(maxsize=1)
def load() -> dict[str, dict]:
    return json.loads(CWE_TO_ATTACK.read_text(encoding="utf-8"))


def techniques_for(cwe: str) -> list[dict]:
    return (load().get(cwe) or {}).get("techniques", []) or []


def sigma_for(cwe: str) -> list[str]:
    return (load().get(cwe) or {}).get("sigma_templates", []) or []


def covered() -> set[str]:
    return set(load())


def coverage_against(cwes: list[str]) -> dict:
    known = covered()
    present = {c for c in cwes if c}
    return {"mapped_cwes": len(known),
            "corpus_cwes": len(present),
            "joinable": len(present & known),
            "unjoinable": len(present - known)}
