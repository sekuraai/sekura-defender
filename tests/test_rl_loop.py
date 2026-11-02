"""RL loop against a fake model: loss switching, flip tripwires, hacking canary."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_defender import checkpoints as ck
from sekura_rl.loop import (CISPO_AFTER_STEPS, RLConfig, RLLoop, RoutingUnstable,
                            StepLog)


class FakeSample:
    def __init__(self, text, tokens, logprobs, stop_reason="stop"):
        self.text, self.tokens, self.logprobs = text, tokens, logprobs
        self.stop_reason = stop_reason


class FakeEnv:
    """Rewards vary within a group so advantages are non-degenerate."""
    def __init__(self, rewards=None):
        self.rewards = rewards
        self.n = 0

    def prompts(self, step, n):
        p = [f"prompt {step}-{i}" for i in range(n)]
        return p, [{"i": i} for i in range(n)]

    def prompt_tokens(self, prompts):
        return [[1, 2, 3] for _ in prompts]

    def reward(self, text, ctx, stop_reason="stop"):
        self.n += 1
        if self.rewards is not None:
            return self.rewards[self.n % len(self.rewards)]
        return (self.n % 5) / 4.0


class FakeModel:
    def __init__(self, flip=None):
        self.flip = flip
        self.losses = []
        self.fb_kwargs = []
        self.opt_kwargs = []
        self.saved = []
        self.sample_kwargs = []

    def sample(self, prompts, num_samples, **kw):
        self.sample_kwargs.append(kw)
        return [[FakeSample(f"out{i}{j}", [10 + j, 11 + j], [-0.1, -0.2])
                 for j in range(num_samples)] for i in range(len(prompts))]

    def forward_backward(self, data, loss_fn="importance_sampling", **kw):
        self.losses.append(loss_fn)
        self.fb_kwargs.append(kw)
        for d in data:
            assert set(d) == {"input_ids", "attention_mask", "old_logprobs", "advantages"}
        class R:
            metrics = {"expert_flip_rate": self.flip} if self.flip is not None else {}
        return R()

    def optim_step(self, lr, **kw):
        self.opt_kwargs.append({"lr": lr, **kw})
        class R:
            pass
        return R()

    def save_weights(self, name, mode="training", ttl=None):
        self.saved.append((name, mode, ttl))
        class C:
            path = f"river://x/{mode}/{name}"
            step = 0
            checkpoint_type = mode
        return C()


def _loop(model, cfg, env=None, canary=None):
    run_dir = Path(tempfile.mkdtemp()) / "rl"
    return RLLoop(model=model, env=env or FakeEnv(),
                  manifest=ck.RunManifest(run_id="rl", base_model="B", lora_rank=32),
                  run_dir=run_dir, cfg=cfg, canary=canary), run_dir


def test_logprobs_are_always_requested():
    """importance_sampling needs old_logprobs; sampling without them is fatal
    later, at array-build time, after paying for the rollouts."""
    m = FakeModel()
    loop, _ = _loop(m, RLConfig(steps=2, prompts_per_step=4, group_size=4))
    loop.run()
    assert all(kw.get("logprobs") for kw in m.sample_kwargs)


def test_datum_shape_matches_the_rl_contract():
    m = FakeModel()
    loop, _ = _loop(m, RLConfig(steps=1, prompts_per_step=4, group_size=4))
    loop.run()
    assert m.losses, "no training step ran"


def test_loss_switches_to_cispo_for_long_runs():
    """cispo clips the RATIO, keeping gradient on the low-probability tokens that
    decide schema validity. ppo clips the objective and would zero them."""
    m = FakeModel()
    loop, _ = _loop(m, RLConfig(steps=CISPO_AFTER_STEPS + 2, prompts_per_step=2,
                                group_size=4, checkpoint_every=10_000))
    loop.run()
    assert m.losses[0] == "importance_sampling"
    assert m.losses[-1] == "cispo"
    assert m.fb_kwargs[-1].get("eps_max") == 6.0


def test_expert_flip_is_measured_every_step():
    m = FakeModel(flip=0.01)
    loop, _ = _loop(m, RLConfig(steps=2, prompts_per_step=2, group_size=4))
    loop.run()
    assert all(kw.get("compute_expert_flip_metric") for kw in m.fb_kwargs)


def test_high_flip_enables_routing_replay_and_halves_lr():
    m = FakeModel(flip=0.10)
    loop, _ = _loop(m, RLConfig(steps=3, prompts_per_step=2, group_size=4, lr=4e-5))
    loop.run()
    assert any(kw.get("force_routing_replay") for kw in m.fb_kwargs)
    assert loop.cfg.lr < 4e-5


def test_catastrophic_flip_aborts():
    """Above the abort threshold old_logprobs no longer describe the policy that
    produced the samples, so the update is meaningless regardless of reward."""
    m = FakeModel(flip=0.5)
    loop, _ = _loop(m, RLConfig(steps=3, prompts_per_step=2, group_size=4))
    with pytest.raises(RoutingUnstable):
        loop.run()


def test_rl_optimizer_uses_short_horizon_beta2_and_no_weight_decay():
    m = FakeModel()
    loop, _ = _loop(m, RLConfig(steps=2, prompts_per_step=2, group_size=4))
    loop.run()
    for kw in m.opt_kwargs:
        assert kw["beta2"] == 0.95
        assert kw["weight_decay"] == 0.0
        assert kw["grad_clip_norm"] == 1.0


def test_zero_advantage_groups_are_skipped_and_counted():
    m = FakeModel()
    loop, _ = _loop(m, RLConfig(steps=1, prompts_per_step=4, group_size=4),
                    env=FakeEnv(rewards=[1.0]))     # constant reward -> no signal
    loop.run()
    assert loop.logs[0].skipped_zero == loop.logs[0].groups
    assert loop.logs[0].used_rollouts == 0
    assert not m.losses, "must not submit an empty batch"


def test_metrics_are_written_per_step():
    m = FakeModel()
    loop, run_dir = _loop(m, RLConfig(steps=3, prompts_per_step=2, group_size=4))
    loop.run()
    rows = [json.loads(l) for l in (run_dir / "rl_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {"step", "mean_reward", "used_rollouts", "loss_fn"} <= set(rows[0])


def test_serving_checkpoint_has_no_ttl():
    m = FakeModel()
    loop, _ = _loop(m, RLConfig(steps=2, prompts_per_step=2, group_size=4))
    loop.run()
    inf = [s for s in m.saved if s[1] == "inference"]
    assert inf and all(s[2] is None for s in inf)


def test_reward_hacking_canary_fires_on_divergence():
    """Programmatic reward rising while the judge falls is the single most
    reliable hacking signal available."""
    loop, _ = _loop(FakeModel(), RLConfig(steps=1, prompts_per_step=1, group_size=2))
    for i, (r, j) in enumerate([(0.50, 4.5), (0.55, 4.0), (0.60, 3.5),
                                (0.66, 3.0), (0.72, 2.4)], start=1):
        loop.logs.append(StepLog(step=i, mean_reward=r, max_reward=r, groups=1,
                                 used_rollouts=2, skipped_zero=0,
                                 skipped_degenerate=0, loss_fn="cispo",
                                 seconds=1.0, canary=j))
    assert loop.hacking_detected()


def test_canary_quiet_when_both_rise():
    loop, _ = _loop(FakeModel(), RLConfig(steps=1, prompts_per_step=1, group_size=2))
    for i, (r, j) in enumerate([(0.50, 3.0), (0.55, 3.2), (0.60, 3.4),
                                (0.66, 3.6), (0.72, 3.9)], start=1):
        loop.logs.append(StepLog(step=i, mean_reward=r, max_reward=r, groups=1,
                                 used_rollouts=2, skipped_zero=0,
                                 skipped_degenerate=0, loss_fn="cispo",
                                 seconds=1.0, canary=j))
    assert not loop.hacking_detected()
