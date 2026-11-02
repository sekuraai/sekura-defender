"""Assembly, split integrity, and the lock manifest."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft import assemble as A
from sekura_sft.config import DATASET_DIR, MANIFESTS_DIR


@pytest.fixture(scope="module")
def built():
    examples, report = A.generate()
    return examples, report


def test_no_entity_appears_in_two_splits(built):
    """The hard leakage guarantee. Everything else about splitting is heuristic;
    this one is exact."""
    examples, _ = built
    by_key = {}
    for e in examples:
        key = e.split_key or e.datum_id()
        by_key.setdefault(key, set()).add(e.split())
    straddling = {k: v for k, v in by_key.items() if len(v) > 1}
    assert not straddling, f"{len(straddling)} entities straddle splits: {list(straddling)[:3]}"


def test_every_split_is_populated(built):
    examples, _ = built
    counts = Counter(e.split() for e in examples)
    assert counts["train"] > 3000
    assert counts["test"] > 150, "eval set too small to measure anything"
    assert counts["val"] > 150


def test_all_tasks_reach_the_test_split(built):
    """A capability with no test examples cannot be regression-gated."""
    examples, _ = built
    tasks = {e.task for e in examples}
    tested = {e.task for e in examples if e.split() == "test"}
    assert tasks - tested == set(), f"untested tasks: {tasks - tested}"


def test_datum_ids_are_unique(built):
    examples, _ = built
    ids = [e.datum_id() for e in examples]
    assert len(ids) == len(set(ids))


def test_near_dup_diagnostic_is_advisory_not_destructive(built):
    """Templated prompts are near-identical by construction; deleting on that
    basis destroyed the eval set (planner 239/239 'leaked'). The metric is
    reported so a regression in a FREE-TEXT task is visible."""
    _, rep = built
    assert rep.dropped_leak == 0
    assert rep.near_dup_rate
    # Free-text tasks whose PROMPT is not itself templated. remediation and
    # detection are excluded: their prompts embed a rendered finding or evidence
    # bundle, so they are templated by construction like the others.
    for free_text in ("mapping", "safety"):
        if free_text in rep.near_dup_rate:
            assert rep.near_dup_rate[free_text] < 0.25


def test_tier_gate_restricts_the_corpus():
    permissive, rep = A.generate(max_tier="T0-permissive")
    assert rep.dropped_tier > 0
    assert {e.license_tier for e in permissive} == {"T0-permissive"}
    assert len(permissive) < 1000


def test_lock_manifest_is_complete_and_hashes_match():
    examples, rep = A.generate()
    lock = A.write(examples, rep, max_tier="T3-reference")
    assert lock["shards"]
    import hashlib
    for shard in lock["shards"]:
        body = (DATASET_DIR / shard["name"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == shard["sha256"]
        assert len(body) == shard["bytes"]
    assert sum(s["examples"] for s in lock["shards"]) == len(examples)


def test_attestation_artifacts_carry_readable_text():
    """Attestation must cover human-auditable rendered text, not token ids."""
    lock = json.loads((MANIFESTS_DIR / "dataset.lock.json").read_text())
    arts = list(A.training_artifacts(lock))
    assert arts
    name, sha, content = arts[0]
    assert isinstance(content, bytes)
    first = json.loads(content.decode().splitlines()[0])
    assert {"task", "turns", "target", "split", "datum_id"} <= set(first)


def test_shards_are_deterministic():
    """Reproducible against a FROZEN input. The teacher campaign appends for
    hours, so assembly snapshots it first -- without that, two builds seconds
    apart legitimately differ and the lock's sha256 describes a moment rather
    than an input."""
    A.snapshot_teacher()
    e1, r1 = A.generate(snapshot=False)
    l1 = A.write(e1, r1, max_tier="T3-reference")
    e2, r2 = A.generate(snapshot=False)
    l2 = A.write(e2, r2, max_tier="T3-reference")
    assert l1["corpus_sha256"] == l2["corpus_sha256"]
    assert l1["teacher_snapshot"]["sha256"] == l2["teacher_snapshot"]["sha256"]
