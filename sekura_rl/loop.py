"""The RL loop: sample -> reward -> advantages -> forward_backward -> optim_step.

Loss choice, and why it changes with run length:
  `importance_sampling` for run 1, because it is what river demonstrates end to
  end with a verified reference curve. With no baseline of our own, matching a
  documented-good trajectory is worth a lot.
  `cispo` past ~100 steps. CISPO clips the RATIO rather than the objective, so it
  keeps gradient alive for low-probability-but-high-reward tokens -- which is
  precisely the regime here: the tokens deciding schema validity (`LEAF`, `AND`,
  a `T1190:` prefix, a CVSS digit) are individually unlikely under the base
  model. `ppo` clips the objective and would zero exactly those gradients, which
  is the wrong failure mode for structured output.

MoE routing is instrumented from step 1, not after a bad run. Routing drifts
between sampling and update and silently invalidates old_logprobs, which makes
the importance ratio a comparison against a network that no longer exists.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sekura_defender import checkpoints as ck
from sekura_defender.errors import Backoff, call_with_retry
from sekura_defender.river_session import TRAIN_TIMEOUT

from .common import Rollout, build_batch

# Expert-flip tripwire. Escalates rather than failing outright, because a little
# drift is normal and a lot invalidates the whole update.
FLIP_REDUCE_INNER = 0.05
FLIP_FORCE_REPLAY = 0.08
FLIP_ABORT = 0.20

CISPO_AFTER_STEPS = 100


@dataclass
class RLConfig:
    steps: int = 300
    group_size: int = 6
    prompts_per_step: int = 64
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    lr: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.95            # shorter-horizon second moment, per river's RL example
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    stop: tuple[str, ...] = ("<|im_end|>",)
    loss_fn: str = "importance_sampling"
    switch_to_cispo: bool = True
    cispo_eps_max: float = 6.0
    checkpoint_every: int = 50
    canary_every: int = 20


@dataclass
class StepLog:
    step: int
    mean_reward: float
    max_reward: float
    groups: int
    used_rollouts: int
    skipped_zero: int
    skipped_degenerate: int
    loss_fn: str
    seconds: float
    expert_flip: float | None = None
    canary: float | None = None

    def line(self) -> str:
        flip = f" flip={self.expert_flip:.3f}" if self.expert_flip is not None else ""
        can = f" judge={self.canary:.2f}" if self.canary is not None else ""
        return (f"step {self.step:4d} R={self.mean_reward:.4f} max={self.max_reward:.3f} "
                f"used={self.used_rollouts:4d} skip0={self.skipped_zero:2d}/"
                f"{self.skipped_degenerate:2d} {self.loss_fn}{flip}{can} "
                f"{self.seconds:.1f}s")


class RoutingUnstable(RuntimeError):
    pass


@dataclass
class RLLoop:
    """`env` supplies prompts and scores completions; `model` is a river Model."""
    model: object
    env: object                      # .prompts(step, n) -> list[str]; .reward(text, ctx)
    manifest: ck.RunManifest
    run_dir: Path
    cfg: RLConfig = field(default_factory=RLConfig)
    logs: list[StepLog] = field(default_factory=list)
    canary: Callable[[list[tuple[str, float]]], float] | None = None
    _inner_steps: int = 1
    _force_replay: bool = False

    def _loss_for(self, step: int) -> tuple[str, dict]:
        if self.cfg.switch_to_cispo and step > CISPO_AFTER_STEPS:
            return "cispo", {"eps_max": self.cfg.cispo_eps_max}
        return self.cfg.loss_fn, {}

    def _apply_flip_policy(self, flip: float | None) -> None:
        if flip is None:
            return
        if flip > FLIP_ABORT:
            raise RoutingUnstable(
                f"expert flip rate {flip:.3f} > {FLIP_ABORT}: old_logprobs no longer "
                "describe the sampling policy. Switch to a dense base model.")
        if flip > FLIP_FORCE_REPLAY:
            self._force_replay = True
            self.cfg.lr = max(self.cfg.lr / 2, 1e-6)
        elif flip > FLIP_REDUCE_INNER:
            self._inner_steps = 1

    def run(self) -> ck.RunManifest:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        backoff = Backoff()
        step = self.manifest.step

        while step < self.cfg.steps:
            t0 = time.time()
            step += 1
            prompts, contexts = self.env.prompts(step, self.cfg.prompts_per_step)
            if not prompts:
                break

            groups_raw = call_with_retry(
                lambda: self.model.sample(
                    prompts=prompts, num_samples=self.cfg.group_size,
                    max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p, stop=list(self.cfg.stop),
                    logprobs=1,               # required: IS needs old_logprobs
                    seed=step * 1_000_003),
                backoff=backoff)

            groups: list[list[Rollout]] = []
            all_rewards: list[float] = []
            best: list[tuple[str, float]] = []
            for prompt_tokens, samples, ctx in zip(
                    self.env.prompt_tokens(prompts), groups_raw, contexts):
                rollouts = []
                for s in samples:
                    r = self.env.reward(s.text, ctx, stop_reason=getattr(s, "stop_reason", "stop"))
                    rollouts.append(Rollout(
                        prompt_tokens=prompt_tokens, tokens=list(s.tokens),
                        logprobs=list(s.logprobs), reward=r, text=s.text,
                        stop_reason=getattr(s, "stop_reason", "stop")))
                    all_rewards.append(r)
                if rollouts:
                    top = max(rollouts, key=lambda x: x.reward)
                    best.append((top.text, top.reward))
                groups.append(rollouts)

            data, stats = build_batch(groups)
            loss_fn, loss_kw = self._loss_for(step)
            flip = None
            if data:
                fb = call_with_retry(
                    lambda: self.model.forward_backward(
                        data, loss_fn=loss_fn, timeout=TRAIN_TIMEOUT,
                        compute_expert_flip_metric=True,
                        force_routing_replay=self._force_replay, **loss_kw),
                    backoff=backoff)
                flip = (getattr(fb, "metrics", {}) or {}).get("expert_flip_rate")
                self._apply_flip_policy(flip)
                call_with_retry(
                    lambda: self.model.optim_step(
                        lr=self.cfg.lr, beta1=self.cfg.beta1, beta2=self.cfg.beta2,
                        eps=self.cfg.eps, grad_clip_norm=self.cfg.grad_clip_norm,
                        weight_decay=0.0,       # RL: no decay on a policy-gradient run
                        timeout=TRAIN_TIMEOUT),
                    backoff=backoff)

            judged = None
            if self.canary and step % self.cfg.canary_every == 0 and best:
                judged = self.canary(best)

            log = StepLog(
                step=step,
                mean_reward=statistics.fmean(all_rewards) if all_rewards else 0.0,
                max_reward=max(all_rewards) if all_rewards else 0.0,
                groups=stats["groups"], used_rollouts=stats["rollouts"],
                skipped_zero=stats["skipped_zero_advantage"],
                skipped_degenerate=stats["skipped_degenerate"],
                loss_fn=loss_fn, seconds=time.time() - t0,
                expert_flip=flip, canary=judged)
            self.logs.append(log)
            with open(self.run_dir / "rl_metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log.__dict__) + "\n")

            self.manifest.step = step
            if step % self.cfg.checkpoint_every == 0:
                ck.save(self.model, self.manifest, f"rl-s{step:05d}", kind="training")
            self.manifest.save(self.run_dir / "manifest.json")

        ck.save(self.model, self.manifest, "rl-final", kind="inference", ttl_days=None)
        self.manifest.save(self.run_dir / "manifest.json")
        return self.manifest

    def hacking_detected(self, window: int = 5) -> bool:
        """Programmatic reward rising while the judge falls is the most reliable
        reward-hacking signal available -- worth halting on."""
        pts = [(l.step, l.mean_reward, l.canary) for l in self.logs if l.canary is not None]
        if len(pts) < window:
            return False
        recent = pts[-window:]
        rewards = [p[1] for p in recent]
        judged = [p[2] for p in recent]
        return rewards[-1] > rewards[0] + 0.02 and judged[-1] < judged[0] - 0.2
