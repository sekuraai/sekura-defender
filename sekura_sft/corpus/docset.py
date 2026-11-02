"""Document-level curation: the `allow.tsv` proposer and loader.

This is the highest-leverage artifact in the pipeline (plan §2.1). One pass over
knowledge.jsonl computes per-document security purity and near-duplicate family
membership, then proposes keep/drop. A human reviews ~418 book titles and flips
the rows the heuristic gets wrong -- it is a proposal generator, NOT an
unattended gate. Measured failure modes it gets wrong in both directions:
  correctly drops  Linux Programming Interface (9,100 records, tag_frac 0.06)
  wrongly drops    Kali_Linux_Social_Engineering.md (1,357, tag_frac 0.01)

Near-duplicate families matter because different extractor models paraphrased the
same source text differently: Web Hacking Arsenal x3 shares only ~2% exact-claim
overlap, so claim-level dedup cannot see it. Only document-level can.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from ..config import DOC_KEEP_TAG_FRAC, DOC_KEEP_TEXT_FRAC, DOCSET_TSV, tier_for
from . import secvocab as sv
from .secvocab import RE_SEC_TEXT
from .records import doc_key, iter_records

COLUMNS = ("source_path", "keep", "domain", "dedup_family", "license_tier",
           "n_records", "tag_frac", "text_frac", "title_verified", "title_signal",
           "drop_reason", "note")

# Untitled PDF dumps: no identifiable provenance, therefore unattestable. They
# cannot be listed honestly in a TrainingDataArtifact manifest, so they default out.
RE_UNTITLED = re.compile(r"/(\d{1,3})\.md$")

# Precedence for the single primary drop reason. Provenance beats structure,
# structure beats purity: an untitled duplicate is untitled first.
DROP_PRECEDENCE = ("legacy-schema-1.0-shadow", "untitled-unattestable",
                   "dup-of", "low-purity")

_STOP = {"the", "a", "an", "of", "and", "for", "to", "in", "with", "edition", "ed",
         "vol", "volume", "second", "third", "fourth", "fifth", "sixth", "seventh",
         "eighth", "ninth", "tenth", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th",
         "8th", "9th", "10th", "book", "books", "pdf", "epub", "zlibrary", "z", "library",
         "final", "draft", "copy", "new", "full", "complete", "compressed"}


# Family detection applies ONLY to the book corpus, where the filename *is* the
# title and duplicate editions genuinely exist. Repo-sourced docs (owasp/*,
# standards/*, references/*) are disambiguated by their directory path: a naive
# stem-only key clusters 94 unrelated README.md files into one "family" and
# wrongly discards ~71k records.
BOOK_PREFIX = "local/reference-corpus/"
FAMILY_MIN_TOKENS = 2

# A title match only makes a family a *candidate*. It is confirmed by content:
# the subject-set Jaccard between two copies of one book. Measured separation is
# wide and unambiguous --
#   Linux-Bible vs linux-bible-by-christopher-negus  subj 0.228 (claim only 0.013)
#   Computer Networking 6th vs 8th edition           subj 0.224 (claim only 0.032)
#   Bash_level1 vs Bash_level2 (a series, not a dup) subj 0.000
#   Linux-Bible vs Computer Networking (unrelated)   subj 0.008
# Claim-level overlap is ~17x lower than subject overlap because different
# extractor models paraphrased the same source text differently, which is exactly
# why claim-level dedup cannot see these and document-level must.
FAMILY_CONFIRM_JACCARD = 0.10


def family_key(source_path: str) -> str:
    """Normalize a book title into a family key so different editions and
    differently-suffixed copies collapse together. Returns "" for anything that
    is not a book, or whose title is too generic to identify."""
    if not source_path.startswith(BOOK_PREFIX):
        return ""
    stem = Path(source_path).stem.lower()
    stem = re.sub(r"\(.*?\)|\[.*?\]", " ", stem)          # (Z-Library), [PDF]
    stem = re.sub(r"v?\d+(\.\d+)*", " ", stem)             # version-ish numbers
    toks = [t for t in re.split(r"[^a-z]+", stem) if t and t not in _STOP and len(t) > 1]
    if len(toks) < FAMILY_MIN_TOKENS:
        return ""                                          # too generic to be a family
    return " ".join(toks[:6])


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def scan() -> tuple[dict[str, dict], dict[str, set]]:
    """One streaming pass: per-document record count, tag purity, text purity,
    tier, plus the normalized subject set for every book doc (used to confirm
    candidate families by content)."""
    stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "tag": 0, "text": 0, "tiers": defaultdict(int),
                 "source_id": "", "legacy": 0})
    subjects: dict[str, set] = defaultdict(set)
    for rec in iter_records():
        if rec.get("source_path", "").startswith(BOOK_PREFIX):
            subjects[rec["source_path"]].add(_norm(rec.get("subject")))
        dk = doc_key(rec)
        s = stats[dk]
        s["n"] += 1
        if sv.has_security_tag(rec):
            s["tag"] += 1
        if sv.has_security_text(rec):
            s["text"] += 1
        s["tiers"][tier_for(rec.get("license", ""))] += 1
        if not s["source_id"]:
            s["source_id"] = rec.get("source_id", "")
        if not rec.get("source_path"):
            s["legacy"] += 1
    return stats, subjects


def _confirm_families(cands: dict[str, list[str]], stats: dict[str, dict],
                      subjects: dict[str, set]) -> dict[str, list[str]]:
    """Keep only family members whose content actually overlaps the largest member.
    Rejects title collisions such as Bash_level1/2/3, which are a progressive
    series rather than duplicate copies."""
    confirmed: dict[str, list[str]] = {}
    for fam, members in cands.items():
        ordered = sorted(members, key=lambda p: (-stats[p]["n"], p))
        anchor, rest = ordered[0], ordered[1:]
        a = subjects.get(anchor, set())
        keep = [anchor]
        for m in rest:
            b = subjects.get(m, set())
            union = a | b
            jac = len(a & b) / len(union) if union else 0.0
            if jac >= FAMILY_CONFIRM_JACCARD:
                keep.append(m)
        if len(keep) > 1:
            confirmed[fam] = keep
    return confirmed


def _domain(tag_frac: float, text_frac: float) -> str:
    if tag_frac >= DOC_KEEP_TAG_FRAC or text_frac >= 0.55:
        return "security"
    if tag_frac >= 0.08 or text_frac >= DOC_KEEP_TEXT_FRAC:
        return "mixed"
    return "general-cs"


def propose(out: Path | None = None) -> Path:
    """Write the reviewable proposal. Within a dedup_family exactly one member is
    kept -- most records first, as a proxy for the most complete conversion."""
    stats, subjects = scan()
    cands: dict[str, list[str]] = defaultdict(list)
    for dk in stats:
        fk = family_key(dk)
        if fk:
            cands[fk].append(dk)
    fams = _confirm_families({k: v for k, v in cands.items() if len(v) > 1},
                             stats, subjects)

    # Family winner = most records, tie-broken by path for determinism.
    winner = {fam: members[0] for fam, members in fams.items()}
    in_family = {p: fam for fam, members in fams.items() for p in members}

    rows = []
    for dk, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        tag_frac = s["tag"] / s["n"] if s["n"] else 0.0
        text_frac = s["text"] / s["n"] if s["n"] else 0.0
        fam = in_family.get(dk, "")
        multi = bool(fam)
        tier = max(s["tiers"], key=lambda t: s["tiers"][t]) if s["tiers"] else "DROP"

        notes = []
        is_book = dk.startswith(BOOK_PREFIX)
        if s["legacy"] == s["n"]:
            # Decided by provenance alone; no purity or family reasoning applies.
            rows.append({
                "source_path": dk, "keep": "0", "domain": "mixed", "dedup_family": "",
                "license_tier": tier, "n_records": s["n"],
                "tag_frac": f"{tag_frac:.3f}", "text_frac": f"{text_frac:.3f}",
                "title_verified": "0", "title_signal": "0",
                "drop_reason": "legacy-schema-1.0-shadow",
                "note": "legacy-schema-1.0-shadow",
            })
            continue
        # Purity filtering exists to weed out mis-filed general-CS *books*. The
        # repo-sourced corpora (owasp/*, standards/nist/*, references/*) are
        # curated, licensed and provenance-tracked security material by
        # construction -- judging an OWASP MASTG test by book-corpus tag density
        # is a category error that drops 20+ legitimate MASTG files. Off-topic
        # sections inside them are handled later, at group granularity (Stage 6).
        if is_book:
            keep = tag_frac >= DOC_KEEP_TAG_FRAC or text_frac >= DOC_KEEP_TEXT_FRAC
            if not keep:
                notes.append("low-purity")
        else:
            keep = True
            notes.append("curated-source")
        if RE_UNTITLED.search(dk) or re.fullmatch(r"\d{1,3}\.md", dk):
            keep, _ = False, notes.append("untitled-unattestable")
        if multi and dk != winner[fam]:
            keep, _ = False, notes.append(f"dup-of:{Path(winner[fam]).name}")

        rows.append({
            "source_path": dk,
            "keep": "1" if keep else "0",
            "domain": _domain(tag_frac, text_frac),
            "dedup_family": fam if multi else "",
            "license_tier": tier,
            "n_records": s["n"],
            "tag_frac": f"{tag_frac:.3f}",
            "text_frac": f"{text_frac:.3f}",
            "title_verified": "0",
            # Advisory only, never a gate. Measured unreliable in both directions:
            # "[PDF] Kali Linux Social Engineering" is actually an ExtJS book
            # (top tags extjs 626, ext-js 289), and "Python for Secret Agents"
            # matches merely on the word "secret".
            "title_signal": "1" if RE_SEC_TEXT.search(
                re.sub(r"[_\-]+", " ", Path(dk).stem)) else "0",
            # One primary reason per doc, by precedence, so the funnel report
            # reconciles exactly. `note` keeps every contributing reason: a book
            # can be both low-purity and a duplicate.
            "drop_reason": "" if keep else next(
                (n.split(":")[0] for r_ in DROP_PRECEDENCE
                 for n in notes if n.startswith(r_)), "other"),
            "note": ";".join(notes),
        })

    dest = out or DOCSET_TSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return dest


def load(path: Path | None = None) -> dict[str, dict]:
    """Load the reviewed table. Missing file is a hard error -- silently defaulting
    to keep-everything would ship 30,745 records of Linux internals and algorithms."""
    p = path or DOCSET_TSV
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run: python -m scripts.propose_docset  then review it.")
    out = {}
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row["keep"] = row["keep"].strip() == "1"
            row["n_records"] = int(row["n_records"])
            out[row["source_path"]] = row
    return out
