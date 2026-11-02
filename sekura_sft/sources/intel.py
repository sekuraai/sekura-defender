"""Findings loader and the ONE canonical renderer.

`sekura_rs_intel.json` holds 3,612 rows but only 885 unique findings -- the same
advisory repeats across up to 60 packages. Splits therefore key on CVE, or the
test set is pure leakage.

The renderer is load-bearing for the triage corruption engine. Anti-artifact
control #1 is that a corrupted finding must be byte-identical in layout to a
clean one, so every finding -- perturbed or not -- is serialized from a dataclass
through this single function. Nothing ever string-splices a finding.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

from ..config import INTEL_JSON, INTEL_TRAINING
from ..corpus.bundles import tidy
from ..scoring import cvss

RE_CVE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.I)
RE_FIXED_IN = re.compile(r"fixed\s*in[:\s]+v?([0-9][0-9A-Za-z.\-]*)", re.I)

ECOSYSTEMS = ("npm", "pypi", "pip", "crates", "cargo", "maven", "go", "nuget", "rubygems")


@dataclass
class Finding:
    title: str
    severity: str
    category: str
    description: str
    evidence: str
    recommendation: str
    tool: str
    technique: str
    cwe_id: str
    cvss_score: float | None
    cvss_vector: str
    package: str = ""
    ecosystem: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def cve(self) -> str:
        m = RE_CVE.search(self.title) or RE_CVE.search(self.description)
        return m.group(0).upper() if m else ""

    @property
    def split_key(self) -> str:
        """CVE first: one advisory across 60 packages must not straddle splits."""
        return self.cve or f"title:{self.title[:80]}"

    def computed_score(self) -> float | None:
        if not self.cvss_vector:
            return None
        try:
            return cvss.score(self.cvss_vector)
        except cvss.CvssError:
            return None

    def fixed_in(self) -> str:
        m = RE_FIXED_IN.search(self.evidence) or RE_FIXED_IN.search(self.recommendation)
        return m.group(1) if m else ""

    def path_hint(self) -> str:
        """Where the evidence points. Non-production paths are a real reroute
        signal rather than a synthetic one: 62 findings sit under .github/."""
        m = re.search(r"(?:^|[\s(])((?:[\w.\-]+/){1,6}[\w.\-]+\.\w{1,6})", self.evidence)
        return m.group(1) if m else ""


def _canon(s: str) -> str:
    return tidy(str(s or ""))


FIELD_ORDER = ("title", "severity", "category", "cwe_id", "cvss_score", "cvss_vector",
               "package", "ecosystem", "tool", "technique", "description",
               "evidence", "recommendation")

_LABELS = {
    "title": "Title", "severity": "Severity", "category": "Category", "cwe_id": "CWE",
    "cvss_score": "CVSS score", "cvss_vector": "CVSS vector", "package": "Package",
    "ecosystem": "Ecosystem", "tool": "Detector", "technique": "Technique",
    "description": "Description", "evidence": "Evidence",
    "recommendation": "Recommendation",
}


def render(f: Finding, *, max_chars: int = 1600) -> str:
    """The single canonical rendering. Field order, labels and whitespace are
    fixed here so a corruption can never be spotted from layout."""
    out = []
    for key in FIELD_ORDER:
        val = getattr(f, key)
        if val in ("", None):
            continue
        text = _canon(val) if isinstance(val, str) else str(val)
        if key in ("description", "evidence") and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        out.append(f"{_LABELS[key]}: {text}")
    return "\n".join(out)


def _walk(o, out: list):
    if isinstance(o, dict):
        if "verdict" in o and "severity" in o:
            out.append(o)
        else:
            for v in o.values():
                _walk(v, out)
    elif isinstance(o, list):
        for v in o:
            _walk(v, out)


def _ecosystem_of(technique: str, package: str) -> tuple[str, str]:
    m = re.search(r"Trivy SCA:\s*(\w+)", technique or "")
    mgr = (m.group(1) if m else "").lower()
    eco = {"yarn": "npm", "pnpm": "npm", "npm": "npm", "bun": "npm",
           "pip": "pypi", "uv": "pypi", "poetry": "pypi",
           "cargo": "crates", "go": "go", "maven": "maven"}.get(mgr, mgr)
    return eco, package


@lru_cache(maxsize=1)
def load_findings() -> tuple[Finding, ...]:
    """Unique findings only. 3,612 rows collapse to 885 distinct advisories."""
    raw: list[dict] = []
    _walk(json.loads(INTEL_JSON.read_text(encoding="utf-8")), raw)

    seen: set[tuple] = set()
    out: list[Finding] = []
    for r in raw:
        key = (r.get("title"), r.get("cwe_id"), r.get("severity"),
               (r.get("description") or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        title = r.get("title", "")
        pkg = ""
        m = re.search(r":\s*([\w@/.\-]+)\s+vulnerability", title)
        if m:
            pkg = m.group(1)
        eco, pkg = _ecosystem_of(r.get("technique", ""), pkg)
        out.append(Finding(
            title=title,
            severity=(r.get("severity") or "").lower(),
            category=r.get("category") or "",
            description=_canon(r.get("description")),
            evidence=_canon(r.get("evidence")),
            recommendation=_canon(r.get("recommendation")),
            tool=r.get("tool") or "",
            technique=r.get("technique") or "",
            cwe_id=r.get("cwe_id") or "",
            cvss_score=r.get("cvss_score"),
            cvss_vector=r.get("cvss_vector") or "",
            package=pkg, ecosystem=eco,
            meta={"source": "sekura_rs_intel"},
        ))
    return tuple(out)


@lru_cache(maxsize=1)
def load_intel_training() -> tuple[dict, ...]:
    """sekura-intel's labelled rows -- used for BREADTH, not as triage truth.

    Measured: label_source is cve_match 3,540 (always label 1, and circular for a
    triage model since "it matched a CVE" is already in the input), default 1,246
    (always label 0 at confidence 0.3, i.e. effectively unlabelled), heuristic
    1,304 (confidence 0.35-0.6). `is_reachable` is 0.0 for all 6,090 rows, so it
    teaches nothing about exploitability. Only the heuristic rows carry judgment.
    """
    if not INTEL_TRAINING.exists():
        return ()
    return tuple(json.loads(INTEL_TRAINING.read_text(encoding="utf-8")))


def judged_rows() -> tuple[dict, ...]:
    """The subset of sekura-intel rows that reflect an actual judgment."""
    return tuple(r for r in load_intel_training() if r.get("label_source") == "heuristic")


def by_cwe() -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {}
    for f in load_findings():
        if f.cwe_id:
            out.setdefault(f.cwe_id, []).append(f)
    return out


def stats() -> dict:
    fs = load_findings()
    import collections
    return {
        "unique_findings": len(fs),
        "unique_cves": len({f.cve for f in fs if f.cve}),
        "with_vector": sum(1 for f in fs if f.cvss_vector),
        "with_cwe": sum(1 for f in fs if f.cwe_id),
        "with_recommendation": sum(1 for f in fs if f.recommendation),
        "by_severity": dict(collections.Counter(f.severity for f in fs).most_common()),
        "by_category": dict(collections.Counter(f.category for f in fs).most_common(8)),
        "by_ecosystem": dict(collections.Counter(f.ecosystem for f in fs if f.ecosystem).most_common()),
        "cwe_classes_ge10": sum(1 for v in by_cwe().values() if len(v) >= 10),
        "non_production_paths": sum(
            1 for f in fs if re.match(r"^(\.github|docs|examples?|tests?|website|perf)/",
                                      f.path_hint())),
        "with_fixed_in": sum(1 for f in fs if f.fixed_in()),
        "intel_training_rows": len(load_intel_training()),
        "intel_judged_rows": len(judged_rows()),
    }
