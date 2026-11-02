"""Teacher client, cache, and verification gates.

No live CLI calls: the subprocess is stubbed so the suite stays fast and
deterministic. The live path was exercised separately (24/24 accepted).
"""
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.tasks import teacher_slices as T
from sekura_sft.teacher import verify
from sekura_sft.teacher.client import TeacherClient, cache_key

EVIDENCE = ("[c1] SSRF against CVE-2021-1234 reaches IMDS.\n"
            "[c2] Mitigate with IMDSv2; detection maps to T1552.005.")
CIDS = {"c1", "c2"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    calls = {"n": 0}

    class Proc:
        returncode = 0
        stdout = "stubbed answer citing [c1]"
        stderr = ""

    def fake_run(cmd, **kw):
        calls["n"] += 1
        return Proc()

    monkeypatch.setattr("sekura_sft.teacher.client.subprocess.run", fake_run)
    c = TeacherClient(cache_dir=tmp_path / "cache", model="stub-model")
    c._calls = calls
    return c


def test_cache_prevents_repeat_calls(client):
    a = client.ask("sys", "user")
    b = client.ask("sys", "user")
    assert a == b
    assert client._calls["n"] == 1, "second identical ask must hit the cache"
    client.ask("sys", "different user")
    assert client._calls["n"] == 2


def test_cache_key_is_sensitive_to_everything_that_matters():
    base = cache_key("s", "u", "m", "v1")
    assert base == cache_key("s", "u", "m", "v1")
    for changed in (cache_key("s2", "u", "m", "v1"), cache_key("s", "u2", "m", "v1"),
                    cache_key("s", "u", "m2", "v1"), cache_key("s", "u", "m", "v2")):
        assert changed != base
    # a version bump must invalidate, so a rubric change cannot serve stale output
    assert cache_key("s", "u", "m", "v2") != base


def test_cli_failure_surfaces(monkeypatch, tmp_path):
    class Proc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("sekura_sft.teacher.client.subprocess.run",
                        lambda *a, **k: Proc())
    c = TeacherClient(cache_dir=tmp_path / "c")
    with pytest.raises(T.TeacherError if hasattr(T, "TeacherError") else Exception):
        c.ask("s", "u")


# --- verification gates ---------------------------------------------------

def test_fabricated_citation_is_rejected():
    v = verify.check("As shown in [c9] this is exploitable.",
                     evidence=EVIDENCE, cids=CIDS)
    assert not v.ok and "non-existent" in v.reasons[0]


def test_uncited_answer_is_rejected():
    v = verify.check("SSRF reaches the metadata service.", evidence=EVIDENCE, cids=CIDS)
    assert not v.ok


def test_parametric_memory_is_caught_by_the_grounding_floor():
    """The gate that matters: the teacher answering from recall while ignoring
    the evidence it was handed."""
    v = verify.check("[c1] This maps to T9999.001 and CVE-2099-0001.",
                     evidence=EVIDENCE, cids=CIDS)
    assert not v.ok and "grounding" in " ".join(v.reasons)


def test_grounded_answer_passes():
    v = verify.check("Per [c1] SSRF reaches IMDS (CVE-2021-1234); see [c2] for "
                     "T1552.005 detection.", evidence=EVIDENCE, cids=CIDS)
    assert v.ok, v.reasons


def test_persona_leakage_is_rejected_except_on_refusal_tasks():
    bad = verify.check("[c1] As an AI, I cannot assist.", evidence=EVIDENCE, cids=CIDS)
    assert not bad.ok
    ok = verify.check("I won't provide a working exploit. Here is the detection "
                      "instead: monitor for [c1] IMDS access.",
                      evidence=EVIDENCE, cids=CIDS, expect_refusal=True)
    assert ok.ok, ok.reasons


def test_json_shape_is_enforced_and_fences_tolerated():
    assert not verify.check("{not json", shape="json").ok
    assert verify.check('{"verdict":"accept"}', shape="json").ok
    assert verify.check('```json\n{"verdict":"accept"}\n```', shape="json").ok


def test_char_caps_track_the_token_budget():
    """A char cap stricter than the token budget rejects responses that would
    have tokenized fine -- that cost 4 of 20 good samples before it was fixed."""
    from sekura_sft.config import max_completion_for
    for task, spec in T.SPECS.items():
        budget = max_completion_for(task)
        assert abs(spec.max_chars / T.CHARS_PER_TOKEN - budget) < 2, task


def test_brevity_hint_sits_below_the_hard_cap():
    """Models overshoot a stated word budget, so aiming at the cap guarantees a
    tail of marginal rejections."""
    for task, spec in T.SPECS.items():
        words = int(T.brevity_for(task).split("under ")[1].split(" words")[0])
        assert words * T.CHARS_PER_WORD < spec.max_chars * 0.9, task


def _safety_jobs():
    import random
    from sekura_sft.sources import intel
    return list(T.t7_jobs(random.Random(0), list(intel.load_findings())))


def test_safety_slice_covers_every_posture():
    kinds = {ctx["kind"] for _, ctx, _ in _safety_jobs()}
    assert kinds == set(T.SAFETY_MIX)


def test_answering_dominates_declining():
    """Over-refusal is both the likelier failure and the more damaging one: a
    model that declines to explain attacker tradecraft cannot do threat
    modelling or detection engineering at all. A first pass let the generators
    run free and produced MORE decline examples than answer-fully ones."""
    jobs = _safety_jobs()
    kinds = Counter(ctx["kind"] for _, ctx, _ in jobs)
    answering = (kinds["answer_fully"] + kinds["dual_use"]) / len(jobs)
    declining = kinds["decline_redirect"] / len(jobs)
    assert answering > 0.55
    assert answering > 2 * declining


def test_safety_mix_is_enforced_not_hoped_for():
    jobs = _safety_jobs()
    kinds = Counter(ctx["kind"] for _, ctx, _ in jobs)
    for kind, target in T.SAFETY_MIX.items():
        assert abs(kinds[kind] / len(jobs) - target) < 0.03, kind


def test_safety_volume_is_enough_to_move_a_release_gate():
    """22 examples cannot shift E8. Volume comes from the catalog -- 26 trees,
    their leaves and chokepoints, and real findings -- not hand-written seeds."""
    assert len(_safety_jobs()) > 250


def test_decline_prompts_name_only_fictional_targets():
    """Nothing in this slice should name a real organization."""
    jobs = [j for j in _safety_jobs() if j[1]["kind"] == "decline_redirect"]
    assert jobs
    for _, ctx, _ in jobs:
        q = ctx["q"]
        if "targeting " in q or "against " in q:
            assert any(t in q for t in T.FAKE_TARGETS) or "acme" in q.lower() \
                or "contoso" in q.lower(), q[:90]


def test_scope_conditioned_prompts_state_authorization():
    jobs = [j for j in _safety_jobs() if j[1]["kind"] == "scope_conditioned"]
    assert jobs
    for _, ctx, _ in jobs:
        assert any(w in ctx["q"].lower()
                   for w in ("sow", "engagement", "rules of engagement",
                             "own staging", "i am the owner", "purple-team"))


def test_injection_prompts_wrap_untrusted_content():
    jobs = [j for j in _safety_jobs() if j[1]["kind"] == "injection"]
    assert jobs
    for _, _, user in jobs:
        assert "--- untrusted content ---" in user
        assert "REAL TASK" in user


# --- circuit breaker ------------------------------------------------------

def test_rate_limit_phrases_are_recognised():
    from sekura_sft.teacher.client import RATE_LIMIT_RE
    for msg in ("Claude usage limit reached, resets at 3pm",
                "429 Too Many Requests", "API Error: Overloaded",
                "rate limit exceeded", "please try again later"):
        assert RATE_LIMIT_RE.search(msg), msg
    for msg in ("connection reset by peer", "file not found",
                "invalid model name"):
        assert not RATE_LIMIT_RE.search(msg), msg


def test_breaker_pauses_instead_of_burning_the_job_list():
    """The failure this exists to prevent: the first campaign had no gate, so
    when calls began failing it consumed the remaining 6,092 jobs in seconds --
    every one an error, producing nothing."""
    from sekura_sft.teacher.client import Breaker
    b = Breaker(trip_after=3, waits=(0.05,))
    for _ in range(2):
        b.record_failure(Exception("x"))
    assert b.trips == 0, "must tolerate isolated failures"
    b.record_failure(Exception("x"))
    assert b.trips == 1
    slept = []
    assert b.wait_if_open(sleep=slept.append) > 0
    assert slept


def test_a_single_rate_limit_trips_immediately():
    """A usage window will not clear by trying the next job, so it should not
    take eight failures to notice."""
    from sekura_sft.teacher.client import Breaker, RateLimited
    b = Breaker(trip_after=99, waits=(0.05,))
    b.record_failure(RateLimited("usage limit reached"))
    assert b.trips == 1


def test_success_resets_the_consecutive_counter():
    from sekura_sft.teacher.client import Breaker
    b = Breaker(trip_after=3, waits=(0.01,))
    b.record_failure(Exception("x"))
    b.record_failure(Exception("x"))
    b.record_success()
    b.record_failure(Exception("x"))
    assert b.trips == 0


def test_backoff_grows_across_trips():
    """Subscription windows take hours; retrying in seconds converts quota into
    errors."""
    from sekura_sft.teacher.client import Breaker
    b = Breaker(trip_after=1, waits=(10, 60, 300))
    gaps = []
    for _ in range(3):
        before = b._resume_at
        b.record_failure(Exception("x"))
        gaps.append(b._resume_at - max(before, time.time()))
    assert gaps[0] < gaps[1] < gaps[2]


def test_error_normalization_buckets_the_same_failure():
    """Two campaigns reported only {'TeacherError': 6092} and
    {'TeacherError': 66} -- a class name and nothing else -- so afterwards it was
    impossible to say whether those were timeouts, auth failures or quota. The
    message is the diagnosis."""
    from sekura_sft.teacher.client import RateLimited, TeacherError, normalize_error
    a = normalize_error(TeacherError("exit 1: id 7f3a9b2c failed after 300.0s"))
    b = normalize_error(TeacherError("exit 1: id a1b2c3d4 failed after 12.5s"))
    assert a == b, "variable ids and durations must collapse"

    distinct = {
        normalize_error(TeacherError("claude-cli returned empty output")),
        normalize_error(TeacherError("claude-cli exit 2: model not found")),
        normalize_error(RateLimited("usage limit reached")),
    }
    assert len(distinct) == 3, "genuinely different failures must stay distinct"


def test_error_normalization_keeps_the_class_name():
    from sekura_sft.teacher.client import RateLimited, normalize_error
    assert normalize_error(RateLimited("usage limit")).startswith("RateLimited:")


def test_error_normalization_is_bounded():
    from sekura_sft.teacher.client import TeacherError, normalize_error
    assert len(normalize_error(TeacherError("x" * 5000))) <= 140
