"""Deterministic generators over the findings corpus.

Covers the plan's D1-D6: CVSS scoring, severity-consistency audit, CWE<->ATT&CK
mapping in both directions, and finding classification. Every target is either
computed arithmetically or read from a lookup table, so none needs a teacher and
none can hallucinate.
"""

from __future__ import annotations

import collections
import json
import random
from collections.abc import Iterator

from ..scoring import cvss
from ..sources import cwe_attack, intel
from ..sources.intel import Finding, render
from .base import SYS_ANALYST, SYS_PIPELINE, Example

VULN_CATEGORIES = ("SQLI", "XSS", "AUTH", "SSRF", "AUTHZ", "CMDI", "SSTI",
                   "PATH_TRAVERSAL", "DESERIALIZATION", "CRYPTO", "SECRETS",
                   "LOGIC", "XXE", "FILE_UPLOAD", "CORS", "AI_RED_TEAM",
                   "INFRASTRUCTURE")


def _ex(task, system, user, target, *, thinking, key, tier="T2-internal", **meta):
    return Example(task=task, system=system,
                   turns=[{"role": "user", "content": user}], target=target,
                   thinking=thinking, split_key=key, license_tier=tier, meta=meta)


# --- D1: CVSS vector -> score + band --------------------------------------

def gen_cvss_score(rng: random.Random) -> Iterator[Example]:
    seen: set[str] = set()
    for f in intel.load_findings():
        v = f.cvss_vector
        if not v or v in seen:
            continue
        try:
            score = cvss.score(v)
        except cvss.CvssError:
            continue
        seen.add(v)
        band = cvss.severity_band(score)
        yield _ex("cvss", SYS_PIPELINE,
                  "Compute the CVSS base score for this vector and give its qualitative "
                  f'band. Answer as JSON: {{"score": <number>, "severity": "<band>"}}.\n\n{v}',
                  json.dumps({"score": score, "severity": band}),
                  thinking=False, key=f"cvss:{v}", vector=v)


# --- D6: severity-consistency audit ---------------------------------------

def gen_severity_audit(rng: random.Random) -> Iterator[Example]:
    """Does the stated severity follow from the stated vector?

    Includes the 46 real records whose upstream score disagrees with its own
    vector by +0.1 -- genuine inconsistencies, not synthetic ones.
    """
    for f in intel.load_findings():
        computed = f.computed_score()
        if computed is None or not f.severity:
            continue
        band = cvss.severity_band(computed)
        consistent = band == f.severity
        stated_ok = f.cvss_score is None or abs(f.cvss_score - computed) < 0.05
        verdict = "consistent" if (consistent and stated_ok) else "inconsistent"
        detail = []
        if not consistent:
            detail.append(f"severity says {f.severity} but the vector scores "
                          f"{computed} ({band})")
        if not stated_ok:
            detail.append(f"stated score {f.cvss_score} does not match the vector's "
                          f"{computed}")
        yield _ex("cvss", SYS_PIPELINE,
                  "Audit this finding's severity against its own CVSS vector. Answer as "
                  'JSON: {"verdict": "consistent|inconsistent", "computed_score": <number>, '
                  '"computed_severity": "<band>", "detail": "<one line>"}.\n\n'
                  f"Severity: {f.severity}\nStated CVSS score: {f.cvss_score}\n"
                  f"CVSS vector: {f.cvss_vector}",
                  json.dumps({"verdict": verdict, "computed_score": computed,
                              "computed_severity": band,
                              "detail": "; ".join(detail) or
                                        "stated severity and score both follow from the vector"},
                             indent=2),
                  thinking=False, key=f.split_key, severity=f.severity, consistent=consistent)


# --- D2/D3: CWE <-> ATT&CK ------------------------------------------------

def gen_cwe_to_attack(rng: random.Random) -> Iterator[Example]:
    """Both directions, and -- critically -- the unmapped case.

    The map covers 51 CWEs against 162 in the corpus. Training only on the
    joinable half teaches the model to always emit a technique, so ~45% of real
    inputs would get a hallucinated one."""
    m = cwe_attack.load()
    for cwe, entry in m.items():
        techs = entry.get("techniques", []) or []
        if not techs:
            continue
        yield _ex("mapping", SYS_PIPELINE,
                  "Map this CWE to MITRE ATT&CK. Answer as JSON with \"techniques\" "
                  "(array of \"ID: Name\") and \"tactics\" (array of \"ID: Name\"). "
                  "If there is no established mapping, return empty arrays.\n\n"
                  f"{cwe}",
                  json.dumps({"techniques": [f"{t['id']}: {t['name']}" for t in techs],
                              "tactics": sorted({f"{t['tactic_id']}: {t['tactic']}"
                                                 for t in techs})}, indent=2),
                  thinking=False, key=f"cwe:{cwe}", tier="T0-permissive", cwe=cwe)

        if entry.get("sigma_templates"):
            yield _ex("detection", SYS_PIPELINE,
                      "Name the Sigma detection templates that cover this weakness class. "
                      'Answer as a JSON array of filenames.\n\n' + cwe,
                      json.dumps(entry["sigma_templates"], indent=2),
                      thinking=False, key=f"cwe:{cwe}", tier="T0-permissive", cwe=cwe)

    # The unmapped half: the model must learn to say so.
    known = cwe_attack.covered()
    unmapped = sorted({f.cwe_id for f in intel.load_findings()
                       if f.cwe_id and f.cwe_id not in known})
    for cwe in unmapped:
        yield _ex("mapping", SYS_PIPELINE,
                  "Map this CWE to MITRE ATT&CK. Answer as JSON with \"techniques\" "
                  "(array of \"ID: Name\") and \"tactics\" (array of \"ID: Name\"). "
                  "If there is no established mapping, return empty arrays.\n\n"
                  f"{cwe}",
                  json.dumps({"techniques": [], "tactics": []}, indent=2),
                  thinking=False, key=f"cwe:{cwe}", tier="T0-permissive",
                  cwe=cwe, unmapped=True)


# --- D4/D5: finding -> CWE and -> category --------------------------------

def _balanced(items: list, key, cap_frac: float = 0.20) -> list:
    counts = collections.Counter(key(i) for i in items)
    cap = max(3, int(len(items) * cap_frac))
    kept, seen = [], collections.Counter()
    for i in items:
        k = key(i)
        if seen[k] >= cap:
            continue
        seen[k] += 1
        kept.append(i)
    return kept


def gen_classify(rng: random.Random) -> Iterator[Example]:
    fs = [f for f in intel.load_findings() if f.cwe_id and f.description]
    # Only CWE classes with enough support to be learnable.
    counts = collections.Counter(f.cwe_id for f in fs)
    learnable = {c for c, n in counts.items() if n >= 5}
    pool = _balanced([f for f in fs if f.cwe_id in learnable], lambda f: f.cwe_id)
    for f in pool:
        body = render(_stripped(f, drop=("cwe_id", "category", "severity",
                                         "cvss_score", "cvss_vector")))
        yield _ex("classify_cwe", SYS_PIPELINE,
                  "Assign the CWE that best describes this finding. Answer with the CWE "
                  f"identifier only.\n\n---\n{body}\n---",
                  f.cwe_id, thinking=False, key=f.split_key, cwe=f.cwe_id)

    cat_pool = _balanced([f for f in intel.load_findings() if f.category and f.description],
                         lambda f: f.category, cap_frac=0.25)
    for f in cat_pool:
        body = render(_stripped(f, drop=("category", "cwe_id")))
        yield _ex("classify_category", SYS_PIPELINE,
                  "Classify this finding into one of: " + ", ".join(VULN_CATEGORIES) +
                  f".\nAnswer with the category only.\n\n---\n{body}\n---",
                  f.category, thinking=False, key=f.split_key, category=f.category)


def _stripped(f: Finding, drop: tuple[str, ...]) -> Finding:
    import copy
    g = copy.deepcopy(f)
    for d in drop:
        setattr(g, d, "" if isinstance(getattr(g, d), str) else None)
    return g


GENERATORS = {
    "cvss_score": gen_cvss_score,
    "severity_audit": gen_severity_audit,
    "cwe_to_attack": gen_cwe_to_attack,
    "classify": gen_classify,
}


def build(seed: int = 0) -> Iterator[Example]:
    """Deduplicate globally by datum_id.

    Several generators legitimately collide: 885 findings share only 195 distinct
    CVSS vectors, and many share a (severity, stated score, vector) triple, so the
    same prompt is reachable from different findings. Emitting both would weight
    those prompts twice for no added signal."""
    rng = random.Random(seed)
    seen: set[str] = set()
    for fn in GENERATORS.values():
        for ex in fn(rng):
            did = ex.datum_id()
            if did in seen:
                continue
            seen.add(did)
            yield ex
