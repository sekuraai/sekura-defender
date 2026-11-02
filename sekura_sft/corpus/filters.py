"""The six-stage curation funnel: 370,231 raw records -> a high-signal pool.

Each stage is a pure generator over the record stream and every rejection carries
a reason code, so the funnel is auditable end to end and the report reconciles
exactly against the raw count.

Stage order matters. Document-level decisions (1-2) run before claim-level ones
(3-5) because duplicate book editions share only ~2% of exact claims -- claim
dedup literally cannot see them, so it must not be asked to.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator

from ..config import (GROUP_MIN_SEC_FRAC, GROUP_MIN_SIZE, MIN_CLAIM_CHARS,
                      SIMHASH_HAMMING, tier_allowed, tier_for)
from . import secvocab as sv
from .docset import load as load_docset
from .nearsim import DupIndex, normalize, simhash
from .records import doc_key, group_key, is_legacy, iter_records

# --- Stage 5 patterns ------------------------------------------------------
# Claims that describe the *document* rather than the security domain.
BOILERPLATE = [
    re.compile(p, re.I) for p in (
        r"\bis a tool in\b",
        r"\bis a (document|section|chapter|page|repository|file|book|guide|project)\b",
        r"\bis (part of|described in|mentioned in|listed in|covered in|discussed in)\b",
        r"^(the |this )?(document|table of contents|readme|index|appendix|foreword|preface)\b",
        r"\b(free and open|licensed under|copyright|all rights reserved|"
        r"contributions are welcome|available at www\.|ebook versions)\b",
        r"\bthe (book|chapter|author|reader|following section)\b",
        r"\b(compiles source content|output document formats)\b",
        # Graph-conversion filler: "X is a <role> in <Y> Graph." Contentless by
        # construction -- 99 such claims in the first 60k records alone.
        r"\bis an? [\w\s-]{2,30} in .{0,60}\bgraph\.?\s*$",
        r"\b(packt|o'reilly|wiley|apress|no starch)\b.*\b(publish|offer|book)",
    )
]

REASONS = ("legacy_schema_1_0", "doc_dropped", "tier_excluded", "not_english",
           "exact_dup", "near_dup", "boilerplate", "subject_restatement",
           "low_information", "group_too_small", "group_not_security")


class Funnel:
    """Runs the stages and records why every rejected record was rejected."""

    def __init__(self, max_tier: str = "T3-reference", limit: int | None = None):
        self.max_tier = max_tier
        self.limit = limit
        self.rejected: Counter[str] = Counter()
        self.raw = 0
        self.docset = load_docset()

    # -- Stage 5 predicates -------------------------------------------------
    @staticmethod
    def _is_subject_restatement(subject: str, claim: str) -> bool:
        """The claim adds nothing beyond restating its own subject, e.g.
        'Fingerprint Web Server is an information gathering activity that
        identifies the web server software and version used by the target.'"""
        s_toks = set(normalize(subject).split())
        c_toks = normalize(claim).split()
        if not s_toks or not c_toks:
            return False
        overlap = sum(1 for t in c_toks if t in s_toks)
        if overlap / len(c_toks) <= 0.55:
            return False
        return not sv.has_entity(claim)

    def _stage5_reason(self, rec: dict) -> str | None:
        claim = rec.get("claim", "")
        if not sv.is_probably_english(claim):
            return "not_english"
        if len(claim) < MIN_CLAIM_CHARS:
            return "low_information"
        if any(p.search(claim) for p in BOILERPLATE):
            return "boilerplate"
        if self._is_subject_restatement(rec.get("subject", ""), claim):
            return "subject_restatement"
        # A claim with no security entity AND no security wording is inert.
        if not sv.has_entity(claim) and not sv.is_security(rec):
            return "low_information"
        return None

    # -- Driver -------------------------------------------------------------
    def run(self) -> Iterator[dict]:
        """Yield surviving records, annotated with `license_tier` and `_group`.

        Stages 1-5 stream. Stage 6 needs whole-group statistics, so groups are
        buffered and flushed on group change -- records for one (source_path,
        heading) arrive contiguously in knowledge.jsonl, which keeps this O(1)
        in memory rather than buffering the corpus.
        """
        seen_exact: set[str] = set()
        dup_idx: dict[tuple, DupIndex] = defaultdict(lambda: DupIndex(SIMHASH_HAMMING))
        cur_key: tuple | None = None
        buf: list[dict] = []

        def flush(records: list[dict]) -> Iterator[dict]:
            if not records:
                return
            if len(records) < GROUP_MIN_SIZE:
                self.rejected["group_too_small"] += len(records)
                return
            sec = sum(1 for r in records if sv.is_security(r))
            if sec / len(records) < GROUP_MIN_SEC_FRAC:
                self.rejected["group_not_security"] += len(records)
                return
            yield from records

        for rec in iter_records(limit=self.limit):
            self.raw += 1

            # Stage 1 -- legacy schema-1.0 shadow copy.
            if is_legacy(rec):
                self.rejected["legacy_schema_1_0"] += 1
                continue

            # Stage 2 -- document curation (purity, family dedup, provenance).
            row = self.docset.get(doc_key(rec))
            if row is None or not row["keep"]:
                self.rejected["doc_dropped"] += 1
                continue

            # Licence tier gate (`--max-tier`).
            tier = tier_for(rec.get("license", ""))
            if not tier_allowed(tier, self.max_tier):
                self.rejected["tier_excluded"] += 1
                continue

            claim = rec.get("claim", "")

            # Stage 3 -- exact claim dedup, corpus-wide.
            norm = normalize(claim)
            if norm in seen_exact:
                self.rejected["exact_dup"] += 1
                continue
            seen_exact.add(norm)

            # Stage 5 -- boilerplate / tautology / low information.
            # Runs before the near-dup index so junk never becomes an anchor
            # that a real claim is then measured against.
            reason = self._stage5_reason(rec)
            if reason:
                self.rejected[reason] += 1
                continue

            # Stage 4 -- near-dup within (dedup_family or doc, heading).
            # Bucket by entity identity as well as location, so two claims are
            # only ever compared by surface form once they are already known to
            # concern the same subject and the same identifiers. SimHash alone
            # collapses templated-but-distinct facts (different CVEs, different
            # packages, one sentence shape) and silently deletes real data.
            fam = row["dedup_family"] or doc_key(rec)
            bucket = (fam, rec.get("heading", ""),
                      sv.entity_key(rec.get("subject", ""), claim))
            if not dup_idx[bucket].add_if_new(simhash(claim)):
                self.rejected["near_dup"] += 1
                continue

            rec["license_tier"] = tier
            gk = group_key(rec)
            if gk != cur_key:
                yield from flush(buf)
                buf, cur_key = [], gk
            buf.append(rec)

        yield from flush(buf)

    def report(self) -> str:
        kept = self.raw - sum(self.rejected.values())
        lines = [f"raw records: {self.raw}", ""]
        run = self.raw
        for r in REASONS:
            if self.rejected[r]:
                run -= self.rejected[r]
                lines.append(f"  -{self.rejected[r]:>7d}  {r:<22s} -> {run}")
        lines += ["", f"kept: {kept} ({kept / self.raw:.1%} of raw)",
                  f"reconciles: {kept + sum(self.rejected.values()) == self.raw}"]
        return "\n".join(lines)
