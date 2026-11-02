"""Grounded context bundles: the unit that actually gets trained on.

The single most important design decision in this pipeline: **a claim is never a
training target.** Claims appear only inside the prompt, as retrieved evidence.
Targets are either constructed deterministically from structured data or written
by one teacher in one voice.

That matters because the corpus was extracted by three different models
(DeepSeek 175,711 / Claude Haiku 167,830 / Gemini 8,713) with visibly different
styles. Training on claims directly would teach an average of three voices. Once
claims live only on the input side, extractor style stops mattering entirely --
and the median 144-character claim stops being a trivia flashcard and becomes
one line of evidence in a real analyst answer.

The aggregation unit is the (source_path, heading) group, because the corpus is
already organized into coherent human-authored sections. A real example, one
group from DOM_based_XSS_Prevention_Cheat_Sheet.md "GUIDELINE #6", holds 24
claims spanning object[x] accessors, indirection, eval() vs JSON.parse(), and the
DOMPurify / sanitize-html / HTML Sanitizer API tool set. That is a teachable
answer; the individual claims are not.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator

from ..config import (BUNDLE_OVERLAP, BUNDLE_SHARD_ABOVE, BUNDLE_WINDOW,
                      GROUP_MIN_SIZE)
from . import secvocab as sv

ACTIONABLE_TYPES = frozenset({"procedure", "vulnerability", "tool"})

# One heading in sekura_rs_intel.json holds 4,497 records. Windowing that by
# position would produce 450 meaningless bundles; it shards by CVE instead, which
# is its real semantic unit.
RE_CVE_IN = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.I)


def _bundle_id(source_path: str, heading: str, shard: str) -> str:
    return "b:" + hashlib.sha1(
        f"{source_path}|{heading}|{shard}".encode()).hexdigest()[:16]


def tidy(text: str) -> str:
    """Collapse the whitespace the PDF/markdown conversions left behind.

    Advisory text in sekura_rs_intel.json carries runs of 3-4 blank lines between
    every line ("### Summary\n\n\n\n"). Those are pure token cost -- they carry
    no information, and at $1.00/1M training tokens a corpus-wide 20% waste is
    real money. Fenced code blocks keep their internal single newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)          # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)           # >2 blank lines -> 1
    text = re.sub(r"[ \t]{2,}", " ", text)           # runs of spaces
    return text.strip()


def _est_tokens(claims: list[dict]) -> int:
    """~4 chars/token, plus a little for the rendered scaffolding per claim."""
    return sum(len(c["claim"]) + len(c["subject"]) + 24 for c in claims) // 4


def _topic_tags(recs: list[dict], k: int = 6) -> list[str]:
    c = Counter(t for r in recs for t in (r.get("tags") or [])
                if str(t).strip().lower() in sv.SECURITY_TAGS)
    if not c:
        c = Counter(t for r in recs for t in (r.get("tags") or []))
    return [t for t, _ in c.most_common(k)]


def _windows(recs: list[dict]) -> Iterator[tuple[str, list[dict]]]:
    """Split an oversized group into overlapping windows, preserving file order.
    Overlap keeps a claim that closes one window and opens the next from losing
    its context on both sides."""
    if len(recs) <= BUNDLE_SHARD_ABOVE:
        yield "0", recs
        return
    step = BUNDLE_WINDOW - BUNDLE_OVERLAP
    for i in range(0, len(recs), step):
        chunk = recs[i:i + BUNDLE_WINDOW]
        if len(chunk) < GROUP_MIN_SIZE:
            break                      # tail too small to stand alone
        yield f"w{i // step}", chunk


def _shard_by_cve(recs: list[dict]) -> Iterator[tuple[str, list[dict]]]:
    by_cve: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        m = RE_CVE_IN.search(r.get("claim", "")) or RE_CVE_IN.search(r.get("subject", ""))
        by_cve[m.group(0).upper() if m else "_nocve"].append(r)
    for cve, group in by_cve.items():
        if cve == "_nocve":
            yield from _windows(group)
        elif len(group) >= 2:
            yield cve, group


def make_bundle(source_path: str, heading: str, shard: str,
                recs: list[dict]) -> dict | None:
    """Assemble one bundle, or None if it fails the anchor/actionable test.

    Both requirements exist so a bundle can actually ground an answer: an anchor
    means the section is about security at all, and an actionable claim means
    there is something to *do* rather than only something to know."""
    if len(recs) < GROUP_MIN_SIZE:
        return None
    if not any(sv.is_security(r) for r in recs):
        return None
    if not any(r.get("type") in ACTIONABLE_TYPES for r in recs):
        return None

    claims = [{
        "cid": f"c{i + 1}",
        "type": r.get("type", "fact"),
        "subject": tidy(r.get("subject", "")),
        "claim": tidy(r.get("claim", "")),
        "confidence": r.get("confidence"),
        "record_id": r.get("id", ""),
    } for i, r in enumerate(recs)]

    head = recs[0]
    # A bundle inherits the most restrictive tier present, so a mixed bundle can
    # never launder T3 evidence into a T0 rebuild.
    from ..config import TIER_ORDER
    tier = max((r.get("license_tier", "T3-reference") for r in recs),
               key=lambda t: TIER_ORDER.index(t))
    return {
        "bundle_id": _bundle_id(source_path, heading, shard),
        "source_path": source_path,
        "source_id": head.get("source_id", ""),
        "source_url": head.get("source_url", ""),
        "source_sha": head.get("source_sha", ""),
        "heading": heading,
        "shard": shard,
        "license": head.get("license", ""),
        "license_tier": tier,
        "topic_tags": _topic_tags(recs),
        "n_claims": len(claims),
        "est_prompt_tokens": _est_tokens(claims),
        "claims": claims,
    }


def build(records: Iterable[dict]) -> Iterator[dict]:
    """Group the curated pool into bundles.

    Records for one (source_path, heading) arrive contiguously in the pool, but
    small sibling headings under one document are merged rather than discarded,
    so a short section is not lost just for being short.
    """
    by_doc: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    cur: tuple | None = None
    buf: list[dict] = []

    def close():
        if cur is not None and buf:
            by_doc[cur[0]].append((cur[1], list(buf)))

    for r in records:
        key = (r.get("source_path", ""), r.get("heading", ""))
        if key != cur:
            close()
            cur, buf = key, []
        buf.append(r)
    close()

    for source_path, sections in by_doc.items():
        pending: list[dict] = []
        pending_head = ""
        for heading, recs in sections:
            if len(recs) < GROUP_MIN_SIZE:
                # Merge short sibling sections instead of dropping them.
                if not pending:
                    pending_head = heading
                pending.extend(recs)
                if len(pending) >= GROUP_MIN_SIZE:
                    for shard, chunk in _windows(pending):
                        b = make_bundle(source_path, pending_head, f"m{shard}", chunk)
                        if b:
                            yield b
                    pending, pending_head = [], ""
                continue

            sharder = _shard_by_cve if len(recs) > 200 else _windows
            for shard, chunk in sharder(recs):
                b = make_bundle(source_path, heading, shard, chunk)
                if b:
                    yield b

        if len(pending) >= GROUP_MIN_SIZE:
            for shard, chunk in _windows(pending):
                b = make_bundle(source_path, pending_head, f"m{shard}", chunk)
                if b:
                    yield b


def render(bundle: dict) -> str:
    """The evidence block as the model sees it. Stable and citable: every claim
    carries a cid the teacher must reference, which is what makes the citation
    check in teacher/verify.py enforceable."""
    lines = [f"## Source: {bundle['source_id']} - {bundle['heading'] or '(document root)'}"]
    for c in bundle["claims"]:
        subj = f"{c['subject']}: " if c["subject"] else ""
        lines.append(f"[{c['cid']}] ({c['type']}) {subj}{c['claim']}")
    return "\n".join(lines)


def iter_pool(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
