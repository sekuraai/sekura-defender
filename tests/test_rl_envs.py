"""RL reward functions. The tests that matter are the anti-hacking ones."""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_rl import vocab
from sekura_rl.envs import attacktree as A
from sekura_rl.envs import planner as P
from sekura_rl.envs import triage as C
from sekura_sft.sources import attack_trees as at
from sekura_sft.sources import recon as R
from sekura_sft.sources.yamlemit import dumps


# --- vocabulary -----------------------------------------------------------

def test_vocabulary_refuses_generated_provenance():
    """The circular-grounding hazard: teacher output must never become the
    oracle that scores teacher output."""
    v = vocab.Vocab(provenance={"tech:T1548.006": "catalog-generated"})
    with pytest.raises(vocab.ProvenanceError):
        v.check_provenance()


def test_vocabulary_knows_real_things_and_not_invented_ones():
    v = vocab.load()
    assert "T1190" in v.techniques and "T9999" not in v.techniques
    assert 4769 in v.windows_event_ids and 99999 not in v.windows_event_ids
    for tool in ("nmap", "sqlmap", "hashcat", "mimikatz"):
        assert tool in v.tools


def test_command_flags_are_not_tool_names():
    """Harvesting name-ish fields indiscriminately made almost any string a
    'known tool', which stops the grounding check being a check."""
    for flag in ("--exclude-ports", "!analyze -v", "<target>", "the", "command"):
        assert not vocab._is_tool_name(flag)
    for tool in ("nmap", "burp suite", "impacket-secretsdump"):
        assert vocab._is_tool_name(tool)


# --- ENV-B planner --------------------------------------------------------

@pytest.fixture(scope="module")
def inv():
    return R.TargetInventory(forms=[{"id": "login"}, {"id": "contact"}],
                             endpoints=[{"id": "e1"}], auth_surfaces=[{"id": "a1"}],
                             architecture_hints={"likely_static": True})


def test_empty_plan_is_a_cliff_not_a_gradient(inv):
    """The production bug. A partial plan must beat an empty one by a lot."""
    assert P.score("[]", inv, "v1").reward == 0.0
    oracle = R.deterministic_fallback_plan(inv, "v1")
    assert P.score(json.dumps(oracle[:2]), inv, "v1").reward > 0.3


def test_hallucinated_surfaces_never_pay(inv):
    oracle = R.deterministic_fallback_plan(inv, "v1")
    good = P.score(json.dumps(oracle), inv, "v1").reward
    bad = P.score(json.dumps(oracle + [{"surface_ref": "form:ghost",
                                        "kind": "FormFuzz"}]), inv, "v1")
    assert good > 0.8 and bad.reward == 0.0


def test_empty_plan_is_correct_when_nothing_is_required():
    static = R.TargetInventory(pages=["/a", "/b"], architecture_hints={"likely_static": True})
    assert P.score("[]", static, "v1").reward == 1.0
    # ...and inventing work for it must not pay
    assert P.score('[{"surface_ref":"form:x","kind":"FormFuzz"}]', static, "v1").reward == 0.0


def test_padding_with_duplicates_costs_efficiency(inv):
    oracle = R.deterministic_fallback_plan(inv, "v1")
    padded = P.score(json.dumps(oracle + [oracle[0]] * 40), inv, "v1")
    assert padded.components["coverage"] == 1.0
    assert padded.components["efficiency"] < 0.3
    assert padded.reward < P.score(json.dumps(oracle), inv, "v1").reward


def test_priority_ordering_is_rewarded(inv):
    oracle = R.deterministic_fallback_plan(inv, "v1")
    ranked = sorted(oracle, key=lambda t: (P.PRIORITY.get(t["surface_ref"].split(":")[0], 99),
                                           t["surface_ref"], t["kind"]))
    assert P.score(json.dumps(ranked), inv, "v1").reward == 1.0
    assert P.score(json.dumps(ranked[::-1]), inv, "v1").reward < 0.95


def test_unparseable_output_scores_zero(inv):
    assert P.score("I would test the login form.", inv, "v1").reward == 0.0


# --- ENV-C triage ---------------------------------------------------------

TRUTH_REJECT = {"verdict": "reject", "confidence_multiplier": 0.0,
                "reason_code": "wrong_cwe", "offending_field": "cwe_id"}
TRUTH_CLEAN = {"verdict": "accept", "confidence_multiplier": 1.0,
               "reason_code": "none", "offending_field": "none"}


def _resp(**kw):
    d = {"verdict": "reject", "confidence_multiplier": 0.0, "reason_code": "wrong_cwe",
         "offending_field": "cwe_id", "justification": "because"}
    d.update(kw)
    return json.dumps(d)


def test_dangerous_error_is_punished_hardest():
    """Accepting a finding that should be rejected ships a bad finding to a
    customer; over-rejecting only wastes one."""
    dangerous = C.score(_resp(verdict="accept", confidence_multiplier=1.0,
                              reason_code="none", offending_field="none"), TRUTH_REJECT)
    over_reject = C.score(_resp(verdict="reject", confidence_multiplier=0.0,
                                justification="x"), TRUTH_CLEAN)
    assert dangerous.dangerous and dangerous.reward == 0.0
    assert over_reject.reward > dangerous.reward


def test_reason_code_is_what_stops_guessing():
    right = C.score(_resp(), TRUTH_REJECT).reward
    wrong_reason = C.score(_resp(reason_code="already_patched",
                                 offending_field="evidence"), TRUTH_REJECT).reward
    assert right - wrong_reason == pytest.approx(C.WEIGHTS["reason"], abs=1e-6)


def test_partial_credit_for_right_field_wrong_code():
    v = C.score(_resp(reason_code="already_patched"), TRUTH_REJECT)
    assert 0 < v.components["reason"] < 1.0


def test_spurious_reason_on_a_clean_finding_costs():
    clean = C.score(json.dumps({**TRUTH_CLEAN, "justification": "ok"}), TRUTH_CLEAN)
    spurious = C.score(json.dumps({**TRUTH_CLEAN, "reason_code": "wrong_cwe",
                                   "offending_field": "cwe_id",
                                   "justification": "ok"}), TRUTH_CLEAN)
    assert spurious.reward < clean.reward


def test_confidence_is_calibration_scored():
    good = C.score(_resp(), TRUTH_REJECT)
    bad = C.score(_resp(confidence_multiplier=1.0), TRUTH_REJECT)
    assert good.components["confidence"] == 1.0
    assert bad.components["confidence"] == 0.0


def test_unparseable_triage_scores_zero():
    assert C.score("I think this is wrong.", TRUTH_REJECT).reward == 0.0


# --- ENV-A attack trees ---------------------------------------------------

FAKE = {
    "subtree_id": "appsec-fake", "name": "Fake", "category": "appsec",
    "author_tier": "Senior Red Team / Threat Research", "version": "1.0.0",
    "contract": {"preconditions": ["x"], "postconditions": ["y"],
                 "prerequisite_access_level": "Unauthenticated"},
    "operational_metrics": {"typical_time_minutes": 5, "detection_noise": "Low",
                            "exploit_reliability": "Reliable (Logic/Auth)",
                            "skill_level": "Novice"},
    "mitre_attack_mappings": {"tactics": ["TA9999: Made Up"],
                              "techniques": ["T9999.001: Invented"]},
    "logic_tree": {"root_objective": "do a thing", "node_type": "AND", "children": [
        {"id": "a", "type": "LEAF", "description": "d",
         "tooling_and_commands": [{"tool": "foo", "command_example": "bar"}],
         "telemetry_signatures": {"windows_event_ids": [99999], "linux_auditd": [],
                                  "cloudtrail_events": [], "web_logs": "x"}}]},
    "chokepoints_and_countermeasures": {
        "critical_nodes": [{"node_id": "a", "countermeasure": "c",
                            "verification_test": "v"}]},
}


def test_real_tree_beats_a_schema_valid_fabrication_by_a_wide_margin():
    """validate.py checks SHAPE, never TRUTH. A model optimizing only the
    validator learns to emit valid trees full of invented content."""
    real = A.score(dumps(at.by_id()["appsec-ssrf-cloud-metadata"].doc))
    fake = A.score(dumps(FAKE))
    assert fake.components["structural"] == 1.0, "the fabrication IS schema-valid"
    assert real.reward - fake.reward > 0.35


def test_structure_gates_content_rather_than_substituting_for_it():
    """Same shape, real content vs invented content."""
    good = copy.deepcopy(FAKE)
    good["mitre_attack_mappings"] = {"tactics": ["TA0006: Credential Access"],
                                     "techniques": ["T1552.005: IMDS"]}
    leaf = good["logic_tree"]["children"][0]
    leaf["tooling_and_commands"] = [{"tool": "nmap", "command_example": "nmap -sV 10.0.0.1"}]
    leaf["telemetry_signatures"] = {
        "windows_event_ids": [4769], "linux_auditd": ["execve of curl"],
        "cloudtrail_events": [],
        "web_logs": "GET /latest/meta-data/iam/security-credentials/"}
    a, b = A.score(dumps(good)), A.score(dumps(FAKE))
    assert a.components["structural"] == b.components["structural"] == 1.0
    assert a.reward - b.reward > 0.25


def test_unparseable_yaml_is_a_hard_zero():
    assert A.score("I would confirm the SSRF then hit IMDS.").reward == 0.0


def test_copying_a_catalog_file_is_penalised():
    """With 26 seed documents, memorise-and-copy is the default hack."""
    cs = A.catalog_shingles()
    doc = dumps(at.by_id()["appsec-ssrf-cloud-metadata"].doc)
    assert A.score(doc, catalog_shingles=cs).reward < A.score(doc).reward - 0.4


def test_truncation_is_penalised():
    doc = dumps(at.by_id()["appsec-ssrf-cloud-metadata"].doc)
    assert A.score(doc, stop_reason="length").reward < A.score(doc).reward


def test_coverage_plateaus_so_junk_leaves_buy_nothing():
    doc = copy.deepcopy(FAKE)
    base = A.score(dumps(doc)).components["coverage"]
    doc["logic_tree"]["children"] += [
        {"id": f"j{i}", "type": "LEAF", "description": "junk",
         "tooling_and_commands": [{"tool": "foo", "command_example": "bar"}],
         "telemetry_signatures": {"windows_event_ids": [], "linux_auditd": [],
                                  "cloudtrail_events": [], "web_logs": ""}}
        for i in range(40)]
    padded = A.score(dumps(doc))
    assert padded.components["coverage"] <= base + 0.30
    assert padded.reward < A.score(dumps(FAKE)).reward + 0.15
