"""Pre-flight cost estimation. Refuses to launch a run over budget."""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1M tokens, river training rate. ALL tokens in a datum are billed, not
# only the supervised ones.
TRAIN_PRICE = {
    "Qwen/Qwen3.6-35B-A3B-FP8": 1.00,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": 1.00,
    "Qwen/Qwen3.5-9B": 1.46,
    "Qwen/Qwen3.5-122B-A10B-FP8": 4.00,
    "Qwen/Qwen3.8-27B-FP8": 4.10,
    "Qwen/Qwen3.6-27B-FP8": 4.10,
    "zai-org/GLM-5.3-Flash": 8.00,
    "deepseek-ai/DeepSeek-V4-Flash-0731": 8.00,
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostEstimate:
    base_model: str
    price_per_1m: float
    train_tokens: int
    eval_tokens: int
    epochs: int
    total_usd: float

    def render(self) -> str:
        t = self.train_tokens * self.epochs / 1e6
        e = self.eval_tokens / 1e6
        return (f"model {self.base_model} @ ${self.price_per_1m:.2f}/1M\n"
                f"  training {t:.2f}M tokens x {self.epochs} epoch(s) = "
                f"${t * self.price_per_1m:.2f}\n"
                f"  held-out forwards {e:.2f}M tokens = ${e * self.price_per_1m:.2f}\n"
                f"  TOTAL ${self.total_usd:.2f}")


def estimate(base_model: str, train_tokens: int, epochs: int,
             eval_tokens: int = 0) -> CostEstimate:
    price = TRAIN_PRICE.get(base_model)
    if price is None:
        # Unknown model: assume the most expensive published rate rather than a
        # cheap default, so an unrecognised id cannot understate a run.
        price = max(TRAIN_PRICE.values())
    total = (train_tokens * epochs + eval_tokens) / 1e6 * price
    return CostEstimate(base_model, price, train_tokens, eval_tokens, epochs,
                        round(total, 2))


def gate(est: CostEstimate, budget_usd: float, *, override: bool = False) -> None:
    if est.total_usd > budget_usd and not override:
        raise BudgetExceeded(
            f"estimated ${est.total_usd:.2f} exceeds budget ${budget_usd:.2f}. "
            "Raise --budget or shrink the run; pass --i-know to override.")
