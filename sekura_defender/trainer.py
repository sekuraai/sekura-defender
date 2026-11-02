"""The SFT loop.

Uses `train_step`, which river documents as the default: it submits
forward-backward and the optimizer step back-to-back so the server pipelines
them without a client round trip in between. The package's own pitfalls list
names the two-call form as a mistake when a plain complete step is meant.

(An earlier version here used forward_backward + optim_step explicitly, on the
reasoning that train_step "takes a single lr for the whole call and would
flatten the schedule". That was simply wrong: a train_step call IS one step and
takes `lr` per call, so a per-step schedule passes through unchanged.)

Two train_step semantics the loop has to respect:
  * `model.step` advances at SUBMIT time, not on success -- a failed or
    timed-out step leaves it advanced, so the local counter is reconciled to the
    server's rather than assumed.
  * A failed forward-backward does NOT cancel the already-queued optimizer step.
    There is no safe "retry this batch": recovery is checkpoint resume.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from sekura_sft.batching import batches, to_river
from sekura_sft.config import (ADAM_BETA1, ADAM_BETA2, ADAM_EPS, GRAD_CLIP_NORM,
                               WEIGHT_DECAY, supervised_budget)
from sekura_sft.tokenization import supervised_tokens

from . import checkpoints as ck
from .errors import Backoff, SessionLost, call_with_retry
from .river_session import TRAIN_TIMEOUT
from .schedule import WarmupCosine


@dataclass
class TrainConfig:
    epochs: int = 2
    eval_every: int = 25
    checkpoint_every: int = 100
    lr_peak: float | None = None
    target_steps_per_epoch: int = 120
    max_resumes: int = 5


@dataclass
class StepLog:
    step: int
    epoch: int
    loss: float
    lr: float
    examples: int
    supervised: int
    billed: int
    seconds: float
    expert_flip: float | None = None

    def line(self) -> str:
        flip = f" flip={self.expert_flip:.3f}" if self.expert_flip is not None else ""
        return (f"e{self.epoch} step {self.step:4d} loss={self.loss:.4f} "
                f"lr={self.lr:.2e} n={self.examples:3d} sup={self.supervised:6d}"
                f"{flip} {self.seconds:.1f}s")


@dataclass
class Trainer:
    model: object
    manifest: ck.RunManifest
    run_dir: Path
    cfg: TrainConfig = field(default_factory=TrainConfig)
    on_log: Callable[[StepLog], None] | None = None
    logs: list[StepLog] = field(default_factory=list)

    def _emit(self, log: StepLog) -> None:
        self.logs.append(log)
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log.__dict__) + "\n")
        if self.on_log:
            self.on_log(log)

    def train(self, data: list[dict], *, heldout: list[dict] | None = None) -> ck.RunManifest:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        budget = supervised_budget(sum(supervised_tokens(d) for d in data),
                                   self.cfg.target_steps_per_epoch)
        plan = list(batches(data, budget))
        total_steps = len(plan) * self.cfg.epochs
        sched = WarmupCosine(total_steps, peak=self.cfg.lr_peak or WarmupCosine(1).peak)
        backoff = Backoff()

        step = self.manifest.step
        for epoch in range(self.manifest.epoch, self.cfg.epochs):
            # Resume mid-epoch from the recorded cursor; the batch plan is
            # deterministic, so replay is exact.
            start = self.manifest.cursor if epoch == self.manifest.epoch else 0
            for i in range(start, len(plan)):
                batch = plan[i]
                t0 = time.time()
                payload = to_river(batch)

                step += 1
                lr = sched.lr(step)

                def _do():
                    return self.model.train_step(
                        payload, lr=lr, loss_fn="cross_entropy",
                        beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS,
                        weight_decay=WEIGHT_DECAY,   # explicit: guide says .01, sig says 0.0
                        grad_clip_norm=GRAD_CLIP_NORM,
                        timeout=TRAIN_TIMEOUT)

                fb, _opt = call_with_retry(_do, backoff=backoff)
                # model.step advances at submit time, so trust the server's
                # counter over a locally incremented one.
                step = int(getattr(self.model, "step", step) or step)

                metrics = getattr(fb, "metrics", {}) or {}
                self._emit(StepLog(
                    step=step, epoch=epoch, loss=float(metrics.get("loss", float("nan"))),
                    lr=lr, examples=len(batch),
                    supervised=sum(supervised_tokens(d) for d in batch),
                    billed=sum(len(d["input_ids"]) for d in batch),
                    seconds=time.time() - t0,
                    expert_flip=metrics.get("expert_flip_rate")))

                self.manifest.step, self.manifest.epoch, self.manifest.cursor = step, epoch, i + 1
                if heldout and step % self.cfg.eval_every == 0:
                    self.evaluate(heldout, step)
                if step % self.cfg.checkpoint_every == 0:
                    ck.save(self.model, self.manifest, f"sft-s{step:05d}",
                            kind="training", note=f"epoch {epoch}")
                self.manifest.save(self.run_dir / "manifest.json")

            self.manifest.cursor = 0
            ck.save(self.model, self.manifest, f"sft-e{epoch}", kind="training",
                    note=f"end of epoch {epoch}")
            self.manifest.save(self.run_dir / "manifest.json")

        ck.save(self.model, self.manifest, "sft-final", kind="inference",
                ttl_days=None, note="serving candidate")
        self.manifest.save(self.run_dir / "manifest.json")
        return self.manifest

    def evaluate(self, heldout: list[dict], step: int) -> dict:
        """Held-out loss via Model.forward (no gradients) -- river has no eval API.

        The server returns a SUM over tokens, so per-token NLL is computed here.
        Reported per task slice: one aggregate number hides a capability
        regressing while another improves."""
        by_task: dict[str, list[dict]] = {}
        for d in heldout:
            by_task.setdefault(d["task"], []).append(d)
        out: dict[str, float] = {}
        for task, rows in by_task.items():
            payload = to_river(rows, normalize_by_batch=False)
            res = call_with_retry(
                lambda: self.model.forward(payload, loss_fn="cross_entropy",
                                           timeout=TRAIN_TIMEOUT))
            m = getattr(res, "metrics", {}) or {}
            # Prefer the server's own per-token mean; fall back to dividing the
            # sum by our own supervised count.
            if "loss_mean" in m:
                out[task] = float(m["loss_mean"])
            else:
                sup = sum(supervised_tokens(d) for d in rows)
                out[task] = float(m.get("loss", float("nan"))) / max(1, sup)
        with open(self.run_dir / "eval.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "nll_per_token": out}) + "\n")
        return out
