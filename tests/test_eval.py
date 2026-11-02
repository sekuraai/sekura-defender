"""Eval scorers, suite and regression gate.

The load-bearing test is the ORACLE FLOOR: replaying each target as the
prediction must score 1.0 on every task. A scorer that cannot score its own
target is broken, and that check caught two mis-routed scorers.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_eval.gate import HARD, check
from sekura_eval.scorers import score_one
from sekura_eval.suite import ALWAYS_REPORT, load_eval, run

pytestmark = pytest.mark.skipif(not load_eval("test"),
                                reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def rows():
    return load_eval("test")


@pytest.fixture(scope="module")
def oracle(rows):
    it = iter([r["target"] for r in rows])
    return run(lambda s, t: next(it), rows=rows)


def test_oracle_floor_is_perfect_on_every_task(oracle):
    """Every scorer must score its own target 1.0. Two scorers were mis-routed
    (attacktree node tasks through the YAML reward, remediation with no scorer
    at all) and only this check found them."""
    for task, m in oracle.items():
        if task == "_overall":
            continue
        assert m["mean_score"] == pytest.approx(1.0), f"{task} scored {m['mean_score']}"
        assert m["exact"] == pytest.approx(1.0), f"{task} exact {m['exact']}"


def test_every_task_has_a_real_scorer(rows):
    """A missing scorer silently falls back to first-line exact match, which
    quietly scores multi-line targets as failures."""
    from sekura_eval.scorers import SCORERS
    tasks = {r["task"] for r in rows}
    assert tasks <= set(SCORERS), f"no scorer for {tasks - set(SCORERS)}"


def test_degenerate_model_scores_near_zero(rows):
    rep = run(lambda s, t: "", rows=rows)
    assert rep["_overall"]["mean_score"] < 0.15
    assert rep["planner"]["zero_coverage_rate"] > 0.5


def test_unparseable_plan_counts_as_zero_coverage(rows):
    """Operationally identical to an empty plan: the pipeline gets no tasks."""
    rep = run(lambda s, t: "I would test the forms.", rows=rows)
    assert rep["planner"]["zero_coverage_rate"] > 0.5


def test_majority_class_triage_produces_dangerous_errors(rows):
    def lazy(system, turns):
        if "accept|challenge|reject|reroute" in turns[-1]["content"]:
            return ('{"verdict":"accept","confidence_multiplier":1.0,'
                    '"reason_code":"none","offending_field":"none",'
                    '"justification":"looks fine"}')
        return ""
    rep = run(lazy, rows=rows)
    assert rep["triage"]["dangerous_rate"] > 0.0
    assert rep["triage"]["reason_exact_rate"] < 0.9


def test_failure_rates_are_reported_even_when_zero(oracle):
    """A metric that vanishes exactly when the model is perfect made the gate
    fail the oracle against itself."""
    for task, keys in ALWAYS_REPORT.items():
        for key in keys:
            assert key in oracle[task], f"{task}.{key} missing on a perfect run"


def test_gate_passes_the_oracle_against_itself(oracle):
    assert check(oracle, oracle).passed


def test_gate_fails_a_degenerate_candidate(rows, oracle):
    rep = run(lambda s, t: "", rows=rows)
    g = check(rep, oracle)
    assert not g.passed
    assert len(g.hard_failures) >= 3
    assert g.regressions


def test_gate_treats_a_missing_hard_metric_as_failure():
    """Silently skipping an ungateable capability is how a regression ships."""
    g = check({"planner": {"n": 1, "mean_score": 1.0, "exact": 1.0}})
    assert not g.passed
    assert any("absent" in m for m in g.missing)


def test_gate_detects_a_small_targeted_regression(oracle):
    import copy
    cand = copy.deepcopy(oracle)
    cand["classify_cwe"]["mean_score"] = round(
        oracle["classify_cwe"]["mean_score"] - 0.10, 4)
    g = check(cand, oracle)
    assert not g.passed
    assert any("classify_cwe" in r for r in g.regressions)


def test_gate_allows_noise_below_the_threshold(oracle):
    import copy
    cand = copy.deepcopy(oracle)
    cand["classify_cwe"]["mean_score"] = round(
        oracle["classify_cwe"]["mean_score"] - 0.01, 4)
    assert check(cand, oracle).passed


def test_lower_is_better_rates_are_compared_in_the_right_direction(oracle):
    import copy
    cand = copy.deepcopy(oracle)
    cand["triage"]["dangerous_rate"] = 0.02      # worse than 0.0, still under the cap
    g = check(cand, oracle)
    assert not g.passed
    assert any("dangerous_rate" in r for r in g.regressions)


def test_hard_thresholds_cover_the_shippability_metrics():
    keys = set(HARD)
    assert ("planner", "zero_coverage_rate") in keys
    assert ("triage", "dangerous_rate") in keys
