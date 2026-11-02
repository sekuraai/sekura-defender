"""Whole-project river cost model, built from MEASURED token counts.

Two things about river's billing shape the arithmetic:

  * TRAINING bills every token in a datum, not only the supervised ones. A
    2,000-token prompt with a 40-token answer costs 2,040 tokens to train on.
  * RL cost is dominated by the GRADIENT STEP, not by sampling -- at least in
    this regime, and contrary to the usual intuition. Measured split at our
    parameters: training is 58-70% of a step (ENV-B 66%, ENV-A 58%, ENV-C 70%).
    The reason is that forward_backward bills prompt+completion at the TRAINING
    rate ($1.00/M) while sampling bills the completion at $0.82/M and the prompt
    at $0.33/M discounted to 20% when cached. Long prompts are therefore paid
    for twice, once cheaply at sampling and once at full training rate.

    The practical lever follows: to cut RL cost, shorten the PROMPT. Halving a
    900-token inventory saves more than halving the group size, and it also
    improves the cache-hit rate. Reducing group size cuts sampling and training
    together but also cuts the advantage signal, so it is the worse trade.

Cached prompt tokens bill at 20% of the prompt rate, but only when the prefix is
byte-identical -- a timestamp or scan id in the system block silently costs 5x.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per 1M tokens (river published rates).
PRICES = {
    "Qwen/Qwen3.6-35B-A3B-FP8": {"train": 1.00, "prompt": 0.33, "completion": 0.82},
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4":
        {"train": 1.00, "prompt": 0.30, "completion": 0.80},
    "Qwen/Qwen3.5-9B": {"train": 1.46, "prompt": 0.66, "completion": 1.99},
    "Qwen/Qwen3.8-27B-FP8": {"train": 4.10, "prompt": 1.80, "completion": 5.50},
    "Qwen/Qwen3.5-122B-A10B-FP8": {"train": 4.00, "prompt": 1.00, "completion": 3.00},
}
CACHED_PROMPT_FRACTION = 0.20
CHECKPOINT_GB_MONTH = 0.10


@dataclass
class Line:
    name: str
    usd: float
    detail: str = ""


@dataclass
class Estimate:
    model: str
    lines: list[Line] = field(default_factory=list)
    contingency: float = 0.30

    def add(self, name: str, usd: float, detail: str = "") -> None:
        self.lines.append(Line(name, round(usd, 2), detail))

    @property
    def subtotal(self) -> float:
        return round(sum(l.usd for l in self.lines), 2)

    @property
    def total(self) -> float:
        return round(self.subtotal * (1 + self.contingency), 2)

    def render(self) -> str:
        w = max(len(l.name) for l in self.lines) + 2
        out = [f"river cost estimate -- {self.model}", ""]
        for l in self.lines:
            out.append(f"  {l.name:<{w}} ${l.usd:>9,.2f}   {l.detail}")
        out.append("")
        out.append(f"  {'subtotal':<{w}} ${self.subtotal:>9,.2f}")
        out.append(f"  {'contingency ' + str(int(self.contingency * 100)) + '%':<{w}} "
                   f"${self.subtotal * self.contingency:>9,.2f}")
        out.append(f"  {'TOTAL':<{w}} ${self.total:>9,.2f}")
        return "\n".join(out)


def sft_cost(model: str, billed_tokens_per_epoch: int, epochs: int,
             eval_tokens: int, eval_passes: int) -> tuple[float, str]:
    p = PRICES[model]["train"]
    train = billed_tokens_per_epoch * epochs / 1e6 * p
    evals = eval_tokens * eval_passes / 1e6 * p
    return train + evals, (f"{billed_tokens_per_epoch * epochs / 1e6:.2f}M train + "
                           f"{eval_tokens * eval_passes / 1e6:.2f}M eval @ ${p:.2f}/M")


def rl_cost(model: str, *, steps: int, prompts_per_step: int, group_size: int,
            prompt_tokens: int, completion_tokens: int,
            cached_share: float = 0.75, skip_share: float = 0.25) -> tuple[float, str]:
    """Sampling dominates. `skip_share` is the fraction of groups dropped for
    zero or degenerate advantage -- sampled and paid for, never trained on."""
    p = PRICES[model]
    samples = prompts_per_step * group_size
    prompt_tok = samples * prompt_tokens
    cached = prompt_tok * cached_share
    fresh = prompt_tok - cached
    completion_tok = samples * completion_tokens
    train_tok = samples * (1 - skip_share) * (prompt_tokens + completion_tokens)

    per_step = (cached / 1e6 * p["prompt"] * CACHED_PROMPT_FRACTION
                + fresh / 1e6 * p["prompt"]
                + completion_tok / 1e6 * p["completion"]
                + train_tok / 1e6 * p["train"])
    return per_step * steps, (f"{steps} steps x {samples} samples "
                              f"(~{prompt_tokens}p/{completion_tokens}c) "
                              f"= ${per_step:.2f}/step")


def inference_cost(model: str, *, prompt_tokens: int, completion_tokens: int,
                   calls: int, cached_share: float = 0.80) -> float:
    p = PRICES[model]
    cached = prompt_tokens * cached_share
    fresh = prompt_tokens - cached
    per_call = (cached / 1e6 * p["prompt"] * CACHED_PROMPT_FRACTION
                + fresh / 1e6 * p["prompt"]
                + completion_tokens / 1e6 * p["completion"])
    return per_call * calls
