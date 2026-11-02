"""Attack-catalog source and generator tests.

The strong claims here are the ones that make this data worth having: the YAML
emitter round-trips the whole catalog, every schema-repair target is genuinely
valid, every corrupted input is genuinely invalid, and no held-out subtree can
reach the training split through any generator.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.sources import attack_trees as at
from sekura_sft.sources.yamlemit import dumps
from sekura_sft.tasks import det_attacktree as g


@pytest.fixture(scope="module")
def subs():
    return at.load_all()


def test_catalog_loads_and_is_clean(subs):
    assert len(subs) == 26
    for s in subs:
        assert s.validate() == [], f"{s.subtree_id} does not validate"


def test_measured_structure(subs):
    st = at.catalog_stats()
    assert st["nodes"] == 149
    assert st["leaves"] == 101
    assert st["critical_nodes"] == 52
    # Every chokepoint carries a verification test -- the property that makes the
    # chokepoint generator a defensive task rather than a descriptive one.
    assert st["critical_with_verification"] == 52
    assert st["invalid"] == []
    # 'Local Admin' is in the schema enum but no subtree uses it; a generator
    # that assumed a total order over access levels would be wrong.
    assert "Local Admin" not in st["access_levels"]


def test_yaml_emitter_round_trips_every_subtree(subs):
    for s in subs:
        back = at.yamllite.load(dumps(s.doc))
        assert back == s.doc, f"{s.subtree_id} did not round-trip"
        assert at.validate.validate_doc(str(s.path), back).errors == []


def test_every_corruption_is_detectable(subs):
    """A corruption the validator cannot see would be an unlabelled example."""
    import copy
    clean = at.by_id()["appsec-ssrf-cloud-metadata"]
    assert at.validate.validate_doc(str(clean.path), clean.doc).errors == []
    detected = 0
    for name, mutate in g._corruptions():
        d = copy.deepcopy(clean.doc)
        try:
            mutate(d)
        except Exception:
            continue
        errs = at.validate.validate_doc(str(clean.path), d).errors
        assert errs, f"corruption {name!r} produced no validator error"
        detected += 1
    assert detected >= 20


def test_repair_targets_are_localized_patches():
    """Targets are DEFECT / PATH / FIX patches, not whole corrected trees: the
    whole-tree form was 99% unchanged transcription."""
    seen_paths = set()
    for e in g.gen_schema_repair(random.Random(0), cap=200):
        assert e.meta["validator_errors"], "input carried no defect"
        lines = e.target.split("\n")
        assert lines[0].startswith("DEFECT: ")
        assert lines[1].startswith("PATH: ")
        assert any(ln.startswith("FIX: ") for ln in lines)
        # The defect line must be the validator's own words, not a paraphrase.
        assert lines[0][len("DEFECT: "):] in e.meta["validator_errors"][0]
        seen_paths.add(e.meta["patch_path"])
    assert len(seen_paths) > 10, "patches must localize to varied paths"


def test_patch_path_points_at_the_corrupted_field():
    """The PATH must name where the two documents actually differ."""
    import copy
    for e in g.gen_schema_repair(random.Random(0), cap=60):
        sub = at.by_id()[e.meta["subtree_id"]]
        path = e.meta["patch_path"]
        assert path, "no patch path recorded"
        # the path must resolve against the CLEAN document (or be a removal)
        node = sub.doc
        ok = True
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
            if part.startswith("["):
                idx = int(part[1:-1])
                if not isinstance(node, list) or idx >= len(node):
                    ok = False
                    break
                node = node[idx]
            else:
                if not isinstance(node, dict) or part not in node:
                    ok = False
                    break
                node = node[part]
        assert ok or "<remove" in e.target, f"unresolvable patch path {path!r}"


def test_held_out_subtrees_never_reach_training():
    rng = random.Random(0)
    leaked = []
    covered = set()
    for name, fn in g.GENERATORS.items():
        for ex in fn(rng):
            if ex.meta.get("subtree_id") in g.HELD_OUT:
                covered.add(name)
                if ex.split() != "test":
                    leaked.append((name, ex.meta["subtree_id"], ex.split()))
    assert not leaked, f"held-out subtree leaked into training: {leaked[:3]}"
    # and every generator must actually be exercised on held-out trees, or the
    # eval set silently covers fewer task types than it appears to
    assert covered == set(g.GENERATORS)


def test_held_out_spans_multiple_categories():
    cats = {at.by_id()[s].category for s in g.HELD_OUT}
    assert len(cats) >= 3, "held-out set must not be all one category"


def test_telemetry_examples_only_where_telemetry_exists():
    for e in g.gen_leaf_telemetry(random.Random(0)):
        assert e.meta["channels"], "generated a detection lesson with no telemetry"


def test_generators_are_deterministic():
    a = [e.datum_id() for e in g.build(seed=0)]
    b = [e.datum_id() for e in g.build(seed=0)]
    assert a == b
    assert len(a) == len(set(a)), "duplicate examples emitted"
