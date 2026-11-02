"""Manual LR schedule. river provides none -- optim_step takes a bare `lr`."""

from __future__ import annotations

import math

from sekura_sft.config import LR_FLOOR_FRAC, LR_PEAK, LR_WARMUP_FRAC


class WarmupCosine:
    """Linear warmup to peak, then cosine decay to a floor.

    Evaluated per optimizer step and passed explicitly to optim_step, which is
    why the trainer calls forward_backward + optim_step rather than the fused
    train_step: train_step would take one LR for the whole call.
    """

    def __init__(self, total_steps: int, peak: float = LR_PEAK,
                 warmup_frac: float = LR_WARMUP_FRAC, floor_frac: float = LR_FLOOR_FRAC):
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        self.total = total_steps
        self.peak = peak
        self.floor = peak * floor_frac
        self.warmup = max(10, int(total_steps * warmup_frac))
        if self.warmup >= total_steps:
            self.warmup = max(1, total_steps // 10)

    def lr(self, step: int) -> float:
        """`step` is 1-based, matching river's Model.step after an optim_step."""
        if step <= 0:
            return 0.0
        if step <= self.warmup:
            return self.peak * step / self.warmup
        progress = (step - self.warmup) / max(1, self.total - self.warmup)
        progress = min(1.0, progress)
        return self.floor + 0.5 * (self.peak - self.floor) * (1 + math.cos(math.pi * progress))
