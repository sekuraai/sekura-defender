"""Prioritized worklist for reviewing data/docset/allow.tsv.

The proposer is a heuristic; this ranks rows by how much a wrong call costs, so
the reviewer spends the 2 hours where it matters instead of reading 1,796 rows.

Risk = records at stake x how close the doc sits to the keep/drop boundary.
A 9,100-record doc at tag_frac 0.06 is a confident drop (low risk). A
1,357-record doc at tag_frac 0.24 sitting just under the 0.25 threshold is the
expensive kind of mistake, and it is a real one: Kali_Linux_Social_Engineering.

Usage:
  python scripts/review_docset.py            # top 40 riskiest calls
  python scripts/review_docset.py --all      # every borderline row
  python scripts/review_docset.py --kept     # riskiest KEEPs (false accepts)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft.config import DOC_KEEP_TAG_FRAC, DOC_KEEP_TEXT_FRAC, DOCSET_TSV


def risk(row: dict) -> float:
    """Distance-to-boundary, inverted, scaled by records at stake."""
    tag, text = float(row["tag_frac"]), float(row["text_frac"])
    # How far past (or short of) the nearer threshold is this doc?
    margin = min(abs(tag - DOC_KEEP_TAG_FRAC), abs(text - DOC_KEEP_TEXT_FRAC))
    # Structural drops (untitled, dup, legacy) are decided by provenance, not
    # purity, so a purity margin says nothing about them.
    if any(k in row["note"] for k in ("untitled", "dup-of", "legacy")):
        margin = max(margin, 0.25)
    return int(row["n_records"]) / (margin + 0.01)


def main() -> int:
    rows = list(csv.DictReader(open(DOCSET_TSV, encoding="utf-8"), delimiter="\t"))
    args = set(sys.argv[1:])
    if "--kept" in args:
        rows = [r for r in rows if r["keep"] == "1"]
    rows.sort(key=risk, reverse=True)
    limit = len(rows) if "--all" in args else 40

    print(f"{'risk':>8} {'recs':>6} {'keep':>4} {'tag':>5} {'text':>5}  note / document")
    print("-" * 108)
    for r in rows[:limit]:
        flag = "KEEP" if r["keep"] == "1" else "drop"
        name = r["source_path"]
        if name.startswith("local/reference-corpus/"):
            name = "book: " + name.split("/")[-1]
        print(f"{risk(r):8.0f} {int(r['n_records']):6d} {flag:>4} "
              f"{r['tag_frac']:>5} {r['text_frac']:>5}  {(r['note'] or '-')[:22]:22s} {name[:56]}")
    print("-" * 108)
    print("To override: edit the `keep` column (1/0) and set `title_verified` to 1.")
    print("`title_verified=1` marks a row a human has actually looked at.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
