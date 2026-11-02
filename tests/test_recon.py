"""Recon planner: contract port, oracle, and generator.

The headline property is the one the contract was written for: an empty plan
against a non-empty inventory must always fail validation, no exceptions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.sources import recon as R
from sekura_sft.sources.recon import PlanValidationError, TargetInventory
from sekura_sft.tasks import det_recon as d


@pytest.fixture(scope="module")
def inv():
    return TargetInventory(
        forms=[{"id": "login"}, {"id": "contact"}],
        endpoints=[{"id": "e1"}],
        auth_surfaces=[{"id": "a1"}],
        architecture_hints={"likely_static": True})


@pytest.fixture(scope="module")
def examples():
    return list(d.build(seed=0))


def test_fallback_plan_is_valid_by_construction(inv):
    for rs in R.RULESETS:
        plan = R.deterministic_fallback_plan(inv, rs)
        R.validate_plan(inv, plan, rs)          # must not raise


def test_empty_plan_against_non_empty_inventory_always_fails(inv):
    """The production bug: planner called a site 'static', emitted nothing."""
    for rs in R.RULESETS:
        with pytest.raises(PlanValidationError) as e:
            R.validate_plan(inv, [], rs)
        assert e.value.empty


def test_empty_plan_is_valid_only_for_an_empty_inventory():
    empty = TargetInventory(architecture_hints={"likely_static": True})
    assert empty.is_empty()
    R.validate_plan(empty, [], "v1")


def test_architecture_hints_never_affect_coverage(inv):
    """Hints are advisory. Flipping every hint must not change what is required."""
    a = R.required_pairs(inv, "v1")
    flipped = TargetInventory(**{**inv.__dict__,
                                 "architecture_hints": {"likely_static": False,
                                                        "likely_spa": True}})
    assert R.required_pairs(flipped, "v1") == a


def test_missing_coverage_is_reported_precisely(inv):
    plan = R.deterministic_fallback_plan(inv, "v1")
    dropped = plan.pop()
    with pytest.raises(PlanValidationError) as e:
        R.validate_plan(inv, plan, "v1")
    assert len(e.value.missing) == 1
    surface, kind = e.value.missing[0]
    assert str(surface) == dropped["surface_ref"] and kind == dropped["kind"]


def test_hallucinated_surface_is_rejected(inv):
    plan = R.deterministic_fallback_plan(inv, "v1")
    plan.append({"surface_ref": "form:does-not-exist", "kind": "FormFuzz"})
    with pytest.raises(PlanValidationError) as e:
        R.validate_plan(inv, plan, "v1")
    assert "form:does-not-exist" in e.value.hallucinated


def test_rulesets_actually_differ(inv):
    sizes = {rs: len(R.deterministic_fallback_plan(inv, rs)) for rs in R.RULESETS}
    assert len(set(sizes.values())) > 1, "rule sets must not be interchangeable"


def test_held_out_ruleset_never_appears_in_training(examples):
    leaked = [e for e in examples
              if e.split() == "train" and e.meta["ruleset"] == R.HELD_OUT_RULESET]
    assert not leaked
    # and the test split must use it, or the anti-memorization check is vacuous
    assert all(e.meta["ruleset"] == R.HELD_OUT_RULESET
               for e in examples if e.split() == "test")


def test_adversarial_static_hint_is_over_represented(examples):
    adv = [e for e in examples if e.meta["adversarial"]]
    assert 0.25 <= len(adv) / len(examples) <= 0.45
    # the trap must be real: hint says static AND surfaces exist
    for e in adv[:20]:
        # the trap is "hint lies WHILE coverage is required"
        assert not e.meta["no_required_coverage"]
        assert e.meta["required_tasks"] > 0


def test_targets_needing_no_plan_are_present(examples):
    """The counterweight to the production bug. Without these the model learns
    'always emit tasks'; with only these it learns 'emit nothing'."""
    share = sum(1 for e in examples if e.meta["no_required_coverage"]) / len(examples)
    assert 0.03 < share < 0.20, f"no-coverage share {share:.1%} is unbalanced"
    for e in examples:
        if e.meta["no_required_coverage"] and e.meta["variant"] == "plan":
            assert json.loads(e.target) == []


def test_plan_targets_match_the_oracle(examples):
    for e in examples:
        if e.meta["variant"] not in ("plan", "repair", "prioritize"):
            continue
        assert len(json.loads(e.target)) == e.meta["required_tasks"]


def test_validate_variant_targets_are_self_consistent(examples):
    for e in examples:
        if e.meta["variant"] != "validate":
            continue
        obj = json.loads(e.target)
        assert obj["valid"] is not bool(obj["missing"])


def test_prioritization_preserves_coverage(examples):
    for e in examples:
        if e.meta["variant"] != "prioritize":
            continue
        tasks = json.loads(e.target)
        assert len(tasks) == e.meta["required_tasks"]
        kinds = [t["surface_ref"].split(":")[0] for t in tasks]
        if "auth_surface" in kinds and "endpoint" in kinds:
            assert kinds.index("auth_surface") < kinds.index("endpoint")


def test_examples_are_unique_and_deterministic():
    a = [e.datum_id() for e in d.build(seed=0)]
    b = [e.datum_id() for e in d.build(seed=0)]
    assert a == b
    assert len(a) == len(set(a))
