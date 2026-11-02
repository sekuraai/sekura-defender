"""Serving: promotion polling, env emission, and the fallback breaker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_serve.breaker import TIERS, Breaker
from sekura_serve.promote import PromotionFailed, Replica, emit_env, promote


class FakeRep:
    def __init__(self, status, url=None, reason=""):
        self.status, self.base_url, self.status_reason = status, url, reason
        self.checkpoint, self.base_model, self.replica_id = "ck", "B", "r1"


class FakeClient:
    def __init__(self, seq):
        self.seq = list(seq)
        self.promoted = None

    def promote_streaming_replica(self, checkpoint, model):
        self.promoted = (checkpoint, model)

    def get_streaming_replica(self, model):
        return self.seq.pop(0) if self.seq else None


def test_env_uses_the_alias_not_the_checkpoint_path():
    r = Replica(model="sekura-defender-analyst-v1", base_url="https://x/v1",
                checkpoint="river://u/sampler_weights/best")
    env = r.env()
    assert env["LLM_MODEL"] == "sekura-defender-analyst-v1"
    assert "river://" not in env["LLM_MODEL"]


def test_trailing_slash_is_stripped():
    """The OpenAI client builds '{api_base}/chat/completions'."""
    r = Replica(model="m", base_url="https://x.river.ai/v1/", checkpoint="c")
    assert r.env()["LLM_API_BASE"] == "https://x.river.ai/v1"
    assert "export LLM_API_BASE=https://x.river.ai/v1\n" in emit_env(r) + "\n"


def test_promotion_polls_until_ready():
    c = FakeClient([FakeRep("pending", None, "warming"),
                    FakeRep("pending", None, "loading"),
                    FakeRep("ready", "https://r.river.ai/v1")])
    rep = promote(c, "river://u/sampler_weights/best", "alias-v1",
                  poll_s=0, sleep=lambda _: None)
    assert rep.base_url == "https://r.river.ai/v1"
    assert c.promoted == ("river://u/sampler_weights/best", "alias-v1")


def test_promotion_failure_surfaces_status_reason():
    """status_reason is the only diagnostic promotion offers."""
    c = FakeClient([FakeRep("failed", None, "quota exceeded")])
    with pytest.raises(PromotionFailed, match="quota exceeded"):
        promote(c, "ck", "alias", poll_s=0, sleep=lambda _: None)


def test_promotion_times_out_rather_than_hanging():
    c = FakeClient([FakeRep("pending", None, "warming")] * 3)
    with pytest.raises(PromotionFailed, match="not ready"):
        promote(c, "ck", "alias", timeout_s=0.01, poll_s=0, sleep=lambda _: None)


def test_a_ready_replica_without_a_url_is_not_accepted():
    c = FakeClient([FakeRep("ready", None), FakeRep("ready", "https://r/v1")])
    rep = promote(c, "ck", "a", poll_s=0, sleep=lambda _: None)
    assert rep.base_url == "https://r/v1"


# --- breaker --------------------------------------------------------------

def test_one_request_falls_through_immediately():
    """Per-request fallthrough is separate from persistent demotion: a request
    must not pay three timeouts against a dead replica just because demotion
    needs three consecutive failures."""
    calls = []
    b = Breaker()

    def bad():
        calls.append("replica")
        raise RuntimeError("503")

    def good():
        calls.append("shim")
        return "ok"

    assert b.call({"replica": bad, "checkpoint_shim": good,
                   "fallback_provider": good}) == "ok"
    assert calls == ["replica", "shim"]
    assert b.current == "replica", "one failure must not demote"


def test_three_failures_demote_persistently():
    b = Breaker()

    def bad():
        raise RuntimeError("503")

    for _ in range(3):
        b.call({"replica": bad, "checkpoint_shim": lambda: "ok",
                "fallback_provider": lambda: "ok"})
    assert b.current == "checkpoint_shim"
    calls = []
    b.call({"replica": bad, "checkpoint_shim": lambda: calls.append("shim") or "ok"})
    assert calls == ["shim"], "later requests must start at the healthy tier"


def test_breaker_recovers_after_cooloff():
    b = Breaker(cooloff_s=0)
    for _ in range(3):
        b.failure()
    assert b.current == TIERS[1]
    b.success()
    assert b.current == TIERS[0], "a transient outage must not downgrade forever"


def test_breaker_stops_at_the_last_tier():
    b = Breaker()
    for _ in range(20):
        b.failure()
    assert b.current == TIERS[-1]


def test_all_tiers_failing_raises():
    b = Breaker()

    def bad():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="all serving tiers failed"):
        b.call({t: bad for t in TIERS})
