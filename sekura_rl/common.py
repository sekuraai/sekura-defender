"""Shared RL scaffolding: the per-token array builder and advantage computation.

WRITE THIS ONCE. Inlining the array construction per environment is how the
off-by-one gets introduced in one env and not another.

THE ALIGNMENT RULE, from river's own RL example:
    ob = len(prompt_tokens)
    old_logprobs = [0.0]*(ob-1) + sample.logprobs + [0.0]
    advantages   = [0.0]*(ob-1) + [adv]*len(sample.tokens) + [0.0]
The completion-aligned arrays start at ob-1 because the LAST PROMPT POSITION is
what predicts the FIRST response token, and they end with a trailing 0.0 for the
no-next-token slot. `target_tokens` is omitted: the server shifts input_ids by one.

Getting ob-1 wrong shifts every credit assignment by one token and produces a run
that trains cleanly, costs full price, and never improves. It is the most
expensive bug available here, so every datum is asserted before submission.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

# Groups whose rewards are all equal carry no signal; groups that are nearly
# equal carry mostly noise. river's example skips the former; the latter is
# skipped too because with token-summed losses a tiny spread still produces a
# real gradient built almost entirely of sampling noise.
DEGENERATE_STDEV = 0.02

# Advantages are clamped before submission. `importance_sampling` has unbounded
# ratios, and with LoRA on an MoE a single high-ratio token can dominate a
# token-summed loss.
ADVANTAGE_CLAMP_SIGMA = 2.0


class AlignmentError(AssertionError):
    pass


@dataclass
class Rollout:
    """One sampled completion with everything the loss needs."""
    prompt_tokens: list[int]
    tokens: list[int]
    logprobs: list[float]
    reward: float
    stop_reason: str = "stop"
    text: str = ""
    meta: dict | None = None


def build_datum(rollout: Rollout, advantage: float) -> dict:
    """One `importance_sampling` datum, asserted for alignment."""
    ob = len(rollout.prompt_tokens)
    if ob < 1:
        raise AlignmentError("empty prompt")
    n = len(rollout.tokens)
    if n == 0:
        raise AlignmentError("empty completion")
    if len(rollout.logprobs) != n:
        raise AlignmentError(
            f"logprobs {len(rollout.logprobs)} != completion tokens {n}; "
            "sampling must request logprobs")

    input_ids = list(rollout.prompt_tokens) + list(rollout.tokens)
    attention_mask = [1] * (ob + n)
    old_logprobs = [0.0] * (ob - 1) + list(rollout.logprobs) + [0.0]
    advantages = [0.0] * (ob - 1) + [advantage] * n + [0.0]

    if not (len(input_ids) == len(attention_mask) == len(old_logprobs) == len(advantages)):
        raise AlignmentError("parallel arrays differ in length")
    first = next((i for i, a in enumerate(advantages) if a != 0.0), None)
    if advantage != 0.0 and first != ob - 1:
        raise AlignmentError(
            f"first non-zero advantage at {first}, expected ob-1 = {ob - 1}")
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "old_logprobs": old_logprobs, "advantages": advantages}


def group_advantages(rewards: Sequence[float], *,
                     clamp_sigma: float = ADVANTAGE_CLAMP_SIGMA) -> list[float] | None:
    """GRPO-style group-relative advantage, or None if the group carries no signal.

    Deliberately NOT divided by the standard deviation. With token-summed losses
    and variable completion lengths, std-normalization couples reward scale to
    length; if it is added later, completion length must be divided out too.
    """
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    if all(r == mean for r in rewards):
        return None                                   # river's zero-advantage skip
    if statistics.pstdev(rewards) < DEGENERATE_STDEV:
        return None                                   # near-degenerate: mostly noise
    advs = [r - mean for r in rewards]
    if clamp_sigma:
        sigma = statistics.pstdev(advs) or 1.0
        lim = clamp_sigma * sigma
        advs = [max(-lim, min(lim, a)) for a in advs]
    return advs


def build_batch(groups: Sequence[Sequence[Rollout]]) -> tuple[list[dict], dict]:
    """Turn scored rollout groups into a training batch, with skip accounting."""
    data: list[dict] = []
    stats = {"groups": len(groups), "skipped_zero_advantage": 0,
             "skipped_degenerate": 0, "rollouts": 0}
    for group in groups:
        rewards = [r.reward for r in group]
        advs = group_advantages(rewards)
        if advs is None:
            if len(set(rewards)) == 1:
                stats["skipped_zero_advantage"] += 1
            else:
                stats["skipped_degenerate"] += 1
            continue
        for rollout, adv in zip(group, advs):
            data.append(build_datum(rollout, adv))
            stats["rollouts"] += 1
    return data, stats
