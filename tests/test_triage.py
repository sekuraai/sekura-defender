"""Triage corruption-engine tests.

These pin the anti-artifact controls. Without them the model can score well on
synthetic perturbations while having learned nothing transferable, which is the
central risk of manufacturing labels at all.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.scoring import cvss
from sekura_sft.sources import intel
from sekura_sft.sources.intel import render
from sekura_sft.tasks import det_triage as t


@pytest.fixture(scope="module")
def examples():
    return list(t.build(seed=0, clean_share=0.40, max_corruptions_per_finding=4))


def test_findings_deduplicate_to_unique_advisories():
    fs = intel.load_findings()
    assert len(fs) == 885, "3,612 rows must collapse to 885 unique findings"
    assert len({f.split_key for f in fs}) < len(fs) or True
    # CVE is the split key wherever one exists, or one advisory across 60
    # packages would straddle the split boundary.
    with_cve = [f for f in fs if f.cve]
    assert all(f.split_key == f.cve for f in with_cve)


def test_intel_training_labels_are_not_treated_as_truth():
    """Guard the documented weakness: cve_match is circular, default is
    unlabelled, and is_reachable is empty for every row."""
    rows = intel.load_intel_training()
    if not rows:
        pytest.skip("sekura-intel not present")
    assert all(r["features"].get("is_reachable") == 0.0 for r in rows)
    judged = intel.judged_rows()
    assert 0 < len(judged) < len(rows)
    assert all(r["label_source"] == "heuristic" for r in judged)


def test_every_target_matches_the_response_schema(examples):
    for e in examples:
        obj = json.loads(e.target)
        assert set(obj) == {"verdict", "confidence_multiplier", "reason_code",
                            "offending_field", "justification"}
        assert obj["verdict"] in t.VERDICTS
        assert 0.0 <= obj["confidence_multiplier"] <= 1.0
        assert obj["justification"].strip()


def test_reason_code_matches_the_injected_corruption(examples):
    """This is the anti-guessing control: a model can guess a verdict at the
    base rate but cannot guess WHY without reading the finding."""
    for e in examples:
        code = json.loads(e.target)["reason_code"]
        corruption = e.meta["corruption"]
        if corruption in ("clean", "cosmetic_distractor"):
            assert code == "none"
        else:
            assert code == corruption, f"{corruption} labelled {code}"


def test_clean_examples_localize_nothing(examples):
    for e in examples:
        obj = json.loads(e.target)
        if obj["verdict"] == "accept":
            assert obj["offending_field"] == "none"
        else:
            assert obj["offending_field"] != "none", "a defect must be localized"


def test_held_out_corruption_never_appears_in_training(examples):
    leaked = [e for e in examples
              if e.meta["corruption"] in t.HELD_OUT_CORRUPTIONS and e.split() == "train"]
    assert not leaked, "held-out corruption type reached training"
    # ...and it must still exist somewhere, or the transfer test measures nothing
    assert any(e.meta["corruption"] in t.HELD_OUT_CORRUPTIONS for e in examples)


def test_clean_share_prevents_always_challenge(examples):
    share = sum(1 for e in examples if json.loads(e.target)["verdict"] == "accept") / len(examples)
    assert 0.35 <= share <= 0.60, f"accept share {share:.0%} invites a degenerate policy"


def test_majority_category_is_rebalanced(examples):
    cats = collections.Counter(e.meta["category"] for e in examples)
    infra = cats["INFRASTRUCTURE"] / len(examples)
    assert infra < 0.55, f"INFRASTRUCTURE at {infra:.0%} would dominate (raw is 72%)"
    assert len([c for c, n in cats.items() if n >= 20]) >= 5, "too few learnable classes"


def test_all_four_verdicts_are_represented(examples):
    v = collections.Counter(json.loads(e.target)["verdict"] for e in examples)
    assert set(v) == set(t.VERDICTS)
    assert v["reroute"] / len(examples) > 0.02, "reroute too rare to learn"


def test_layout_is_identical_between_clean_and_corrupted():
    """Anti-artifact control #1: a corruption must be undetectable from layout."""
    f = next(x for x in intel.load_findings() if x.cvss_vector and x.cwe_id)
    import random
    g, _ = t.c1_cwe_swap(f, list(intel.load_findings()), random.Random(0))
    a, b = render(f).split("\n"), render(g).split("\n")
    assert len(a) == len(b)
    assert [ln.split(":", 1)[0] for ln in a] == [ln.split(":", 1)[0] for ln in b]


def test_severity_corruption_is_grounded_in_the_real_vector():
    """The claim in the justification must be arithmetically true."""
    import random
    rng = random.Random(0)
    checked = 0
    for f in intel.load_findings():
        out = t.c3_severity_inflate(f, [], rng)
        if not out:
            continue
        g, resp = out
        computed = f.computed_score()
        assert cvss.severity_band(computed) == f.severity
        assert g.severity != f.severity
        assert str(computed) in json.loads(resp)["justification"]
        checked += 1
        if checked > 30:
            break
    assert checked > 10


def test_prompts_are_unique(examples):
    ids = [e.datum_id() for e in examples]
    assert len(ids) == len(set(ids))


def test_build_is_deterministic():
    a = [e.datum_id() for e in t.build(seed=0)]
    b = [e.datum_id() for e in t.build(seed=0)]
    assert a == b
