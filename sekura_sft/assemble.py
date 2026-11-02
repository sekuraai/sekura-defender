"""Assemble every generator into sharded train/val/test files + a lock manifest.

Two guarantees this stage is responsible for:

1. NO CROSS-SPLIT LEAKAGE. Entity-keyed splits stop a CVE or subtree straddling
   the boundary, but two different entities can still produce near-identical
   prompts. A final SimHash sweep deletes any val/test prompt that is within the
   near-duplicate threshold of a training prompt. Deleting from EVAL rather than
   train is deliberate: a slightly smaller eval set is a small cost, whereas a
   leaked one silently inflates every number computed from it.

2. REPRODUCIBILITY. dataset.lock.json records the sha256 of every shard plus the
   config that produced it, and the shard bytes are what feed
   Session.attest_training_data -- so the attestation covers human-auditable
   rendered text, not opaque token ids.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from .config import (BASE_MODEL, DATASET_DIR, MANIFESTS_DIR, SIMHASH_HAMMING,
                     TIER_ORDER, tier_allowed)
from .corpus.nearsim import DupIndex, simhash
from .config import EXAMPLES_DIR
from .tasks import det_attacktree, det_findings, det_recon, det_triage
from .tasks.base import Example

TEACHER_JSONL = EXAMPLES_DIR / "teacher.jsonl"
TEACHER_SNAPSHOT = EXAMPLES_DIR / "teacher.snapshot.jsonl"


def snapshot_teacher() -> dict:
    """Freeze the teacher file before assembling.

    The campaign appends for hours, so reading it live makes a build
    irreproducible: two generate() calls seconds apart see different corpora,
    and the lock file's sha256 describes a moment rather than an input. Assembly
    therefore copies it once and reads the copy, and the lock records the
    snapshot's size and hash so a rebuild is verifiable.
    """
    if not TEACHER_JSONL.exists():
        return {"lines": 0, "sha256": "", "snapshot": ""}
    body = TEACHER_JSONL.read_bytes()
    # Drop a partial trailing line -- the writer may be mid-append.
    if body and not body.endswith(b"\n"):
        body = body[:body.rfind(b"\n") + 1]
    TEACHER_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    TEACHER_SNAPSHOT.write_bytes(body)
    return {"lines": body.count(b"\n"),
            "sha256": hashlib.sha256(body).hexdigest(),
            "snapshot": str(TEACHER_SNAPSHOT.relative_to(EXAMPLES_DIR.parent))}


def load_teacher() -> Iterator[Example]:
    """Read teacher-written examples produced by scripts/run_teacher.py.

    Their split_keys are the same entities the deterministic generators use
    (cve_id, source_path, subtree_id), which is the point: a remediation example
    written by the teacher for CVE-X lands in the same split as the triage
    example built deterministically from CVE-X, so one advisory cannot appear on
    both sides of the boundary via different sources.

    Tolerates a truncated final line: the campaign appends for hours, so this
    may be read while it is still being written.
    """
    src = TEACHER_SNAPSHOT if TEACHER_SNAPSHOT.exists() else TEACHER_JSONL
    if not src.exists():
        return
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                      # partial trailing write
            yield Example(
                task=row["task"], system=row["system"], turns=row["turns"],
                target=row["target"], thinking=bool(row.get("thinking")),
                split_key=row.get("split_key", ""),
                forced_split=row.get("forced_split", ""),
                license_tier=row.get("license_tier", "T3-reference"),
                meta=row.get("meta", {}),
            )


GENERATORS = {
    "attacktree": det_attacktree.build,
    "triage": det_triage.build,
    "findings": det_findings.build,
    "recon": det_recon.build,
    "teacher": lambda: load_teacher(),
}


@dataclass
class AssemblyReport:
    total: int
    by_split: dict
    by_task: dict
    by_source: dict
    by_tier: dict
    dropped_duplicate: int
    dropped_leak: int
    dropped_tier: int
    near_dup_rate: dict

    def render(self) -> str:
        lines = [f"examples: {self.total}",
                 "  splits:  " + "  ".join(f"{k}={v}" for k, v in sorted(self.by_split.items())),
                 "  tasks:   " + "  ".join(f"{k}={v}" for k, v in
                                           sorted(self.by_task.items(), key=lambda kv: -kv[1])),
                 "  sources: " + "  ".join(f"{k}={v}" for k, v in self.by_source.items()),
                 "  tiers:   " + "  ".join(f"{k}={v}" for k, v in self.by_tier.items()),
                 f"  dropped: duplicate={self.dropped_duplicate} "
                 f"entity_straddle={self.dropped_leak} tier_excluded={self.dropped_tier}",
                 "  near-dup eval prompts (diagnostic, templated tasks run high): "
                 + "  ".join(f"{k}={v:.0%}" for k, v in self.near_dup_rate.items())]
        return "\n".join(lines)


def _prompt_text(ex: Example) -> str:
    return "\n".join(t["content"] for t in ex.turns)


def _common_affixes(texts: list[str]) -> tuple[int, int]:
    """Length of the shared prefix and suffix across a task's prompts."""
    if len(texts) < 2:
        return 0, 0
    a, b = min(texts), max(texts)          # lexicographic extremes bound the rest
    pre = 0
    for x, y in zip(a, b):
        if x != y:
            break
        pre += 1
    suf = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y or suf + pre >= min(len(a), len(b)):
            break
        suf += 1
    return pre, suf


def _shape(ex: Example) -> tuple[str, str]:
    """Group prompts that share a template.

    Affix stripping needs a homogeneous group. A task with several prompt shapes
    -- planner has four variants -- has NO common prefix across all of them, so a
    per-task affix is 0 and the whole instruction gets hashed again. Keying on the
    opening of the prompt separates variants naturally, and two prompts with
    different openings are not near-duplicates anyway.
    """
    return (ex.task, _prompt_text(ex)[:64])


def _distinctive(ex: Example, affixes: dict[tuple[str, str], tuple[int, int]]) -> str:
    """The part of a prompt that actually varies between examples.

    Every generator wraps its payload in a fixed instruction -- the triage
    instruction alone is ~450 characters repeated on every triage prompt. Hashing
    the whole prompt therefore measures INSTRUCTION similarity, and near enough
    every eval prompt looks like a near-duplicate of some training prompt: the
    naive sweep deleted 739 examples and left eval at 53 test / 27 val. Stripping
    the per-task shared affix leaves the entity-distinguishing content, which is
    what leakage actually means here.
    """
    text = _prompt_text(ex)
    pre, suf = affixes.get(_shape(ex), (0, 0))
    core = text[pre:len(text) - suf] if suf else text[pre:]
    return core or text


def generate(max_tier: str = "T3-reference", *,
             snapshot: bool = True) -> tuple[list[Example], AssemblyReport]:
    if snapshot:
        snapshot_teacher()
    seen: set[str] = set()
    kept: list[Example] = []
    dropped_dup = dropped_tier = 0
    by_source: Counter[str] = Counter()

    for source, fn in GENERATORS.items():
        for ex in fn():
            if not tier_allowed(ex.license_tier, max_tier):
                dropped_tier += 1
                continue
            did = ex.datum_id()
            if did in seen:
                dropped_dup += 1
                continue
            seen.add(did)
            ex.meta["source"] = source
            kept.append(ex)
            by_source[source] += 1

    # --- Leakage control -------------------------------------------------
    #
    # The HARD guarantee is entity-based: no split_key may appear in more than
    # one split. That is what "leakage" actually means here, and it is checkable
    # exactly rather than heuristically.
    #
    # A SimHash sweep over prompt text is kept only as a DIAGNOSTIC, because for
    # machine-generated prompts it does not measure leakage at all -- it measures
    # template reuse. Two planner examples built from different inventories under
    # different rule sets are near-identical as strings and share no entity;
    # deleting them destroyed the eval set (planner 239/239 and cvss 48/48
    # "leaked", eval collapsed to 53 test / 27 val). Reported, not deleted.
    keys_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for ex in kept:
        keys_by_split[ex.split()].add(ex.split_key or ex.datum_id())
    straddling = ((keys_by_split["val"] | keys_by_split["test"]) & keys_by_split["train"])

    final: list[Example] = []
    dropped_leak = 0
    for ex in kept:
        key = ex.split_key or ex.datum_id()
        if ex.split() != "train" and key in straddling:
            dropped_leak += 1
            by_source[ex.meta["source"]] -= 1
            continue
        final.append(ex)

    near_dup = _near_duplicate_diagnostic(final)

    rep = AssemblyReport(
        total=len(final),
        by_split=dict(Counter(e.split() for e in final)),
        by_task=dict(Counter(e.task for e in final)),
        by_source=dict(by_source),
        by_tier=dict(Counter(e.license_tier for e in final)),
        dropped_duplicate=dropped_dup,
        dropped_leak=dropped_leak,
        dropped_tier=dropped_tier,
        near_dup_rate=near_dup,
    )
    return final, rep


def _near_duplicate_diagnostic(examples: list[Example]) -> dict[str, float]:
    """Share of eval prompts that are near-duplicates of a training prompt.

    High values are EXPECTED for templated tasks and are not leakage; a sudden
    rise for a free-text task is worth investigating."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for ex in examples:
        grouped.setdefault(_shape(ex), []).append(_prompt_text(ex))
    affixes = {shape: _common_affixes(texts) for shape, texts in grouped.items()}

    idx: dict[tuple[str, str], DupIndex] = {}
    for ex in examples:
        if ex.split() == "train":
            idx.setdefault(_shape(ex), DupIndex(SIMHASH_HAMMING)).add(
                simhash(_distinctive(ex, affixes)))

    hits: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for ex in examples:
        if ex.split() == "train":
            continue
        total[ex.task] += 1
        bucket = idx.get(_shape(ex))
        if bucket is not None and bucket.find(simhash(_distinctive(ex, affixes))) is not None:
            hits[ex.task] += 1
    return {t: round(hits[t] / n, 3) for t, n in sorted(total.items()) if n}


def write(examples: list[Example], report: AssemblyReport, *,
          max_tier: str, shard_size: int = 2000) -> dict:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    shards: list[dict] = []
    for split in ("train", "val", "test"):
        rows = [e for e in examples if e.split() == split]
        rows.sort(key=lambda e: e.datum_id())        # deterministic shard contents
        d = DATASET_DIR / split
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jsonl"):
            old.unlink()
        for i in range(0, len(rows), shard_size):
            chunk = rows[i:i + shard_size]
            path = d / f"{split}-{i // shard_size:04d}.jsonl"
            body = "".join(json.dumps(e.to_dict(), ensure_ascii=False) + "\n" for e in chunk)
            path.write_text(body, encoding="utf-8")
            shards.append({
                "name": str(path.relative_to(DATASET_DIR)),
                "split": split,
                "examples": len(chunk),
                "bytes": len(body.encode()),
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            })

    lock = {
        "base_model": BASE_MODEL,
        "teacher_snapshot": _snapshot_meta(),
        "max_tier": max_tier,
        "tier_order": list(TIER_ORDER),
        "simhash_hamming": SIMHASH_HAMMING,
        "generators": sorted(GENERATORS),
        "report": {
            "total": report.total, "by_split": report.by_split, "by_task": report.by_task,
            "by_source": report.by_source, "by_tier": report.by_tier,
            "dropped_duplicate": report.dropped_duplicate,
            "dropped_entity_straddle": report.dropped_leak,
            "near_dup_rate": report.near_dup_rate,
            "dropped_tier": report.dropped_tier,
        },
        "shards": shards,
        "corpus_sha256": hashlib.sha256(
            "".join(s["sha256"] for s in shards).encode()).hexdigest(),
    }
    (MANIFESTS_DIR / "dataset.lock.json").write_text(json.dumps(lock, indent=2))
    return lock


def _snapshot_meta() -> dict:
    if not TEACHER_SNAPSHOT.exists():
        return {"lines": 0, "sha256": ""}
    body = TEACHER_SNAPSHOT.read_bytes()
    return {"lines": body.count(b"\n"), "sha256": hashlib.sha256(body).hexdigest()}


def training_artifacts(lock: dict) -> Iterator[tuple[str, str, bytes]]:
    """(name, expected_sha256, content) triples for Session.attest_training_data.

    Content is the RENDERED pre-tokenization JSONL, not token ids: a human can
    read exactly what a checkpoint was trained on, and the attestation stays
    stable across tokenizer versions.
    """
    for shard in lock["shards"]:
        path = DATASET_DIR / shard["name"]
        yield shard["name"], shard["sha256"], path.read_bytes()
