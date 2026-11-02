"""Quality gates every teacher output must pass before entering the dataset.

Teacher output is the only part of this corpus that can hallucinate, so it is the
only part that needs adversarial checking. Five gates, cheapest first:

  schema      -- parses as the declared output shape
  citation    -- every [cid] referenced exists in the bundle. A fabricated
                 citation is worse than no citation: it teaches the model that
                 evidence ids are decorative.
  grounding   -- a floor fraction of the response's security entities appear in
                 the evidence. This is what catches the model answering from
                 parametric memory while ignoring the bundle it was given.
  leakage     -- no assistant-persona tells or refusal preambles on a
                 non-refusal task
  length      -- inside the task's completion budget, since a target that would
                 be truncated must be dropped rather than cut
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..corpus.secvocab import (RE_ATTACK, RE_CONTROL, RE_CVE, RE_CWE, RE_TACTIC,
                               split_identifiers)

RE_CID = re.compile(r"\[(c\d+)\]")
GROUNDING_FLOOR = 0.60

LEAKS = re.compile(
    r"\b(as an ai|i'?m an ai|as a language model|i cannot assist|i can'?t help with that|"
    r"i'?m sorry, but i (?:cannot|can'?t)|anthropic|claude)\b", re.I)

# Entities whose presence in a response should be traceable to the evidence.
_ENTITY_RES = (RE_CVE, RE_CWE, RE_ATTACK, RE_TACTIC, RE_CONTROL)


@dataclass
class Verdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def fail(self, why: str) -> "Verdict":
        self.ok = False
        self.reasons.append(why)
        return self


def entities(text: str) -> set[str]:
    out: set[str] = set()
    for pat in _ENTITY_RES:
        out |= {m.group(0).upper() for m in pat.finditer(text)}
    return out


def check(response: str, *, evidence: str = "", cids: set[str] | None = None,
          shape: str = "text", max_chars: int = 8000,
          grounding_floor: float = GROUNDING_FLOOR,
          expect_refusal: bool = False) -> Verdict:
    v = Verdict(True)
    text = (response or "").strip()
    if not text:
        return v.fail("empty response")
    if len(text) > max_chars:
        v.fail(f"too long: {len(text)} > {max_chars} chars")

    if shape == "json":
        try:
            json.loads(_strip_fence(text))
        except Exception as exc:                        # noqa: BLE001
            v.fail(f"not valid JSON: {exc}")
    elif shape == "yaml":
        if not text.lstrip().startswith(("-", "subtree_id", "```")):
            v.fail("does not look like YAML")

    if cids is not None:
        used = {m.group(1) for m in RE_CID.finditer(text)}
        bogus = used - cids
        if bogus:
            v.fail(f"cites non-existent evidence ids: {sorted(bogus)[:4]}")
        if not used and cids:
            v.fail("cites no evidence")

    if evidence and grounding_floor > 0:
        resp_ents = entities(text)
        if resp_ents:
            have = entities(evidence)
            grounded = len(resp_ents & have) / len(resp_ents)
            if grounded < grounding_floor:
                v.fail(f"grounding {grounded:.0%} < {grounding_floor:.0%}: "
                       f"{sorted(resp_ents - have)[:4]} not in evidence")

    if not expect_refusal and LEAKS.search(split_identifiers(text)):
        v.fail("assistant-persona or refusal leakage")
    return v


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def strip_fence(text: str) -> str:
    return _strip_fence(text)
