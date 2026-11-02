"""Training harness: schedule, cost gate, error policy, loop, and resume.

Exercised against a fake model so every path is covered without an API key.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_defender import checkpoints as ck
from sekura_defender.costs import BudgetExceeded, estimate, gate
from sekura_defender.errors import Backoff, call_with_capacity_retry
from sekura_defender.schedule import WarmupCosine
from sekura_defender.trainer import Trainer, TrainConfig


class FakeModel:
    def __init__(self, fail_on: set[int] | None = None, exc: type = RuntimeError):
        self.step = 0
        self.lrs: list[float] = []
        self.opt_kwargs: list[dict] = []
        self.saved: list[tuple] = []
        self.fb_calls = 0
        self.fwd_calls = 0
        self.fail_on = fail_on or set()
        self.exc = exc

    def forward_backward(self, data, loss_fn="cross_entropy", **kw):
        self.fb_calls += 1
        if self.fb_calls in self.fail_on:
            raise self.exc("injected")
        class R:
            metrics = {"loss": 2.0, "loss_mean": 2.0}
        return R()

    def optim_step(self, lr, **kw):
        self.step += 1
        self.lrs.append(lr)
        self.opt_kwargs.append(kw)
        class R:
            pass
        return R()

    def train_step(self, data, lr, **kw):
        """river's preferred call: submits fb + optim together."""
        self.fb_calls += 1
        for d in data:
            assert set(d) == {"input_ids", "target_tokens", "weights"}
        self.step += 1                      # advances at SUBMIT time
        self.lrs.append(lr)
        self.opt_kwargs.append(kw)
        class FB:
            metrics = {"loss": 2.0, "loss_mean": 2.0}
        class OPT:
            metrics = {"grad_norm": 1.0}
        return FB(), OPT()

    def forward(self, data, loss_fn="cross_entropy", **kw):
        self.fwd_calls += 1
        class R:
            metrics = {"loss": 50.0}
        return R()

    def save_weights(self, name, mode="training", ttl=None):
        self.saved.append((name, mode, ttl))
        class C:
            path = f"river://x/{mode}/{name}"
            step = 0
            checkpoint_type = mode
        return C()


def _data(n=1200, sup=40):
    return [{"input_ids": list(range(60)), "target_tokens": list(range(60)),
             "weights": [0.0] * (60 - sup) + [1.0 / sup] * sup,
             "datum_id": f"d{i:05d}", "task": "triage" if i % 2 else "planner",
             "split": "train"} for i in range(n)]


def _run(model, cfg, data, heldout=None, run_dir=None):
    run_dir = run_dir or Path(tempfile.mkdtemp()) / "run"
    t = Trainer(model=model,
                manifest=ck.RunManifest(run_id="t", base_model="B", lora_rank=32),
                run_dir=run_dir, cfg=cfg)
    return t.train(data, heldout=heldout), run_dir, t


# --- schedule -------------------------------------------------------------

def test_schedule_warms_up_then_decays():
    s = WarmupCosine(total_steps=264)
    assert s.lr(0) == 0.0
    assert s.lr(1) < s.lr(s.warmup)
    assert s.lr(s.warmup) == pytest.approx(s.peak)
    assert s.lr(264) < s.lr(s.warmup)
    assert s.lr(264) == pytest.approx(s.floor, rel=1e-6)
    lrs = [s.lr(i) for i in range(s.warmup, 265)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:])), "decay must be monotonic"


def test_schedule_handles_tiny_runs():
    s = WarmupCosine(total_steps=3)
    assert s.warmup < 3
    assert s.lr(3) > 0


# --- cost gate ------------------------------------------------------------

def test_cost_gate_blocks_and_can_be_overridden():
    e = estimate("Qwen/Qwen3.6-35B-A3B-FP8", 4_730_714, 2, eval_tokens=331_077)
    assert e.total_usd == pytest.approx(9.79, abs=0.02)
    with pytest.raises(BudgetExceeded):
        gate(e, budget_usd=5.0)
    gate(e, budget_usd=5.0, override=True)
    gate(e, budget_usd=50.0)


def test_unknown_model_is_priced_pessimistically():
    e = estimate("some/unknown", 1_000_000, 1)
    assert e.price_per_1m == 8.00, "an unrecognised id must not understate cost"


# --- error policy ---------------------------------------------------------

def test_capacity_squeezes_arrive_as_connection_errors_and_retry_forever():
    """river-client's SKILL.md: CapacityError is never actually raised --
    capacity squeezes present as RiverConnectionError carrying "Server capacity
    exceeded". Treating those as bounded connection retries would kill a run on
    the expected steady-state failure."""
    import river_client as river
    calls = {"n": 0}
    slept: list[float] = []

    def fn():
        calls["n"] += 1
        if calls["n"] < 12:            # more than max_connection_retries
            raise river.RiverConnectionError("Server capacity exceeded, retry")
        return "ok"

    assert call_with_capacity_retry(fn, backoff=Backoff(base=0.001, cap=0.002),
                                    sleep=slept.append) == "ok"
    assert calls["n"] == 12


def test_plain_connection_errors_are_bounded():
    import river_client as river

    def fn():
        raise river.RiverConnectionError("connection reset by peer")

    with pytest.raises(river.RiverConnectionError):
        call_with_capacity_retry(fn, backoff=Backoff(base=0.001),
                                 sleep=lambda _: None, max_connection_retries=3)


def test_session_loss_is_not_retried_in_place():
    """SessionHeartbeatError subclasses RiverConnectionError, so a generic
    connection clause listed first swallows it and retries a dead session."""
    import river_client as river
    from sekura_defender.errors import SessionLost, needs_session_rebuild
    assert issubclass(river.SessionHeartbeatError, river.RiverConnectionError)
    assert needs_session_rebuild(river.SessionHeartbeatError("gone"))

    def fn():
        raise river.SessionHeartbeatError("heartbeat rejected")

    with pytest.raises(SessionLost):
        call_with_capacity_retry(fn, sleep=lambda _: None)


def test_pending_op_is_re_awaited_not_resubmitted():
    """The server-side future survives a client timeout, so .result() can be
    called again -- much cheaper than checkpoint replay, and unlike re-issuing
    the op it cannot double-apply an already-accumulated gradient."""
    import river_client as river
    from sekura_defender.errors import await_pending

    class Pending:
        def __init__(self, fails):
            self.fails, self.calls = fails, 0

        def result(self):
            self.calls += 1
            if self.calls <= self.fails:
                raise river.RiverTimeoutError("still running")
            return "done"

    p = Pending(2)
    assert await_pending(p, sleep=lambda _: None) == "done"
    assert p.calls == 3

    with pytest.raises(river.RiverTimeoutError):
        await_pending(Pending(99), retries=2, sleep=lambda _: None)


def test_timeouts_are_not_retried_blindly():
    """A retried forward_backward can double-apply a gradient already
    accumulated server-side. Recovery is checkpoint replay, not retry."""
    import river_client as river

    def fn():
        raise river.RiverTimeoutError("slow")

    with pytest.raises(river.RiverTimeoutError):
        call_with_capacity_retry(fn, backoff=Backoff(base=0.001), sleep=lambda _: None)


def test_backoff_is_bounded_and_jittered():
    b = Backoff(base=1.0, cap=4.0)
    ds = [b.next_delay() for _ in range(8)]
    assert all(0 < d <= 4.0 for d in ds)
    assert ds[0] < ds[-1]


# --- loop -----------------------------------------------------------------

def test_loop_runs_expected_steps_and_logs():
    m = FakeModel()
    cfg = TrainConfig(epochs=2, eval_every=10, checkpoint_every=20,
                      target_steps_per_epoch=20)
    mf, run_dir, _ = _run(m, cfg, _data(), heldout=_data(40))
    assert m.step > 20, "too few optimizer updates to learn anything"
    assert m.step == mf.step
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == m.step
    assert rows[0]["lr"] < max(r["lr"] for r in rows)


def test_uses_train_step_the_documented_default():
    """The package's pitfalls list names forward_backward + optim_step in a loop
    as a mistake when a plain complete step is meant: it costs an extra client
    round trip per step."""
    m = FakeModel()
    _run(m, TrainConfig(epochs=1, target_steps_per_epoch=6), _data(300))
    assert m.lrs, "no step ran"
    assert len({round(x, 12) for x in m.lrs}) > 1, "schedule must vary per step"


def test_optimizer_args_are_explicit():
    """weight_decay's guide prose says 0.01 while the signature says 0.0, so it
    must never be left to the default."""
    m = FakeModel()
    _run(m, TrainConfig(epochs=1, target_steps_per_epoch=8), _data(400))
    for kw in m.opt_kwargs:
        assert kw["weight_decay"] == 0.01
        assert kw["grad_clip_norm"] == 1.0
        assert kw["beta1"] == 0.9 and kw["beta2"] == 0.999
        assert kw["timeout"] == 1800.0


def test_serving_checkpoint_has_no_ttl():
    m = FakeModel()
    _run(m, TrainConfig(epochs=1, target_steps_per_epoch=6), _data(300))
    inference = [s for s in m.saved if s[1] == "inference"]
    assert inference and all(s[2] is None for s in inference), \
        "a TTL'd checkpoint expiring under a live replica is an outage"


def test_heldout_loss_is_per_token_and_per_task():
    m = FakeModel()
    _, run_dir, t = _run(m, TrainConfig(epochs=1, eval_every=2, target_steps_per_epoch=8),
                         _data(400), heldout=_data(40))
    assert m.fwd_calls > 0
    rows = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines()]
    assert rows
    per = rows[0]["nll_per_token"]
    assert set(per) == {"triage", "planner"}, "aggregate loss hides a regressing slice"
    assert all(0 < v < 50 for v in per.values()), "loss must be divided by token count"


def test_resume_continues_from_the_cursor():
    """The manifest carries epoch+cursor so a replayed run does not redo work."""
    data = _data(800)
    cfg = TrainConfig(epochs=2, target_steps_per_epoch=10, checkpoint_every=1000)
    m1 = FakeModel()
    run_dir = Path(tempfile.mkdtemp()) / "run"
    t1 = Trainer(model=m1, manifest=ck.RunManifest(run_id="t", base_model="B", lora_rank=32),
                 run_dir=run_dir, cfg=TrainConfig(epochs=1, target_steps_per_epoch=10,
                                                  checkpoint_every=1000))
    t1.train(data)
    first_pass = m1.step

    resumed = ck.RunManifest.load(run_dir / "manifest.json")
    assert resumed.epoch == 0 and resumed.cursor == 0   # epoch completed, cursor reset
    m2 = FakeModel()
    t2 = Trainer(model=m2, manifest=resumed, run_dir=run_dir, cfg=cfg)
    t2.train(data)
    assert m2.step > 0
    assert resumed.step > first_pass, "resume must advance, not restart"


def test_checkpoint_record_preserves_step_across_reload():
    m = ck.RunManifest(run_id="r", base_model="B", lora_rank=32, step=57, cursor=41)
    m.checkpoints.append(ck.CheckpointRecord("river://u/weights/s100", 100,
                                             "training", "training"))
    p = Path(tempfile.mkdtemp()) / "m.json"
    m.save(p)
    back = ck.RunManifest.load(p)
    assert back.step == 57 and back.cursor == 41
    assert back.latest("training").step == 100, \
        "a bare path would lose the step and restart the LR warmup"
