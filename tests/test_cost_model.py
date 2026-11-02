"""Cost model. The arithmetic decides whether a run is launched, so it is
tested against the published rates rather than trusted."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_defender.cost_model import (CACHED_PROMPT_FRACTION, PRICES, Estimate,
                                        inference_cost, rl_cost, sft_cost)

MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def test_training_bills_every_token_not_only_supervised():
    """A 2,000-token prompt with a 40-token answer costs 2,040 tokens."""
    usd, _ = sft_cost(MODEL, 1_000_000, epochs=1, eval_tokens=0, eval_passes=0)
    assert usd == pytest.approx(PRICES[MODEL]["train"])


def test_epochs_scale_linearly():
    one, _ = sft_cost(MODEL, 1_000_000, 1, 0, 0)
    three, _ = sft_cost(MODEL, 1_000_000, 3, 0, 0)
    assert three == pytest.approx(one * 3)


def test_rl_is_dominated_by_the_gradient_step_not_sampling():
    """Contrary to the usual intuition. forward_backward bills prompt+completion
    at the TRAINING rate while sampling bills the prompt at a cached discount,
    so a long prompt is paid for twice and the update is the larger half."""
    p = PRICES[MODEL]
    for prompts, group, ptok, ctok in [(64, 6, 800, 450), (96, 6, 900, 1400),
                                       (128, 8, 550, 220)]:
        usd, _ = rl_cost(MODEL, steps=1, prompts_per_step=prompts, group_size=group,
                         prompt_tokens=ptok, completion_tokens=ctok)
        samples = prompts * group
        train_only = samples * 0.75 * (ptok + ctok) / 1e6 * p["train"]
        assert train_only / usd > 0.55, (
            f"training should dominate at {ptok}p/{ctok}c, got {train_only / usd:.0%}")


def test_shortening_the_prompt_saves_more_than_shrinking_the_group():
    """The practical lever, and it is not the obvious one."""
    base, _ = rl_cost(MODEL, steps=10, prompts_per_step=64, group_size=6,
                      prompt_tokens=900, completion_tokens=450)
    half_prompt, _ = rl_cost(MODEL, steps=10, prompts_per_step=64, group_size=6,
                             prompt_tokens=450, completion_tokens=450)
    smaller_group, _ = rl_cost(MODEL, steps=10, prompts_per_step=64, group_size=5,
                               prompt_tokens=900, completion_tokens=450)
    assert base - half_prompt > base - smaller_group


def test_group_size_scales_rl_cost():
    a, _ = rl_cost(MODEL, steps=10, prompts_per_step=64, group_size=4,
                   prompt_tokens=800, completion_tokens=500)
    b, _ = rl_cost(MODEL, steps=10, prompts_per_step=64, group_size=8,
                   prompt_tokens=800, completion_tokens=500)
    assert b == pytest.approx(a * 2, rel=0.01)


def test_skipped_groups_are_still_paid_for():
    """Zero-advantage groups are sampled and billed, then thrown away."""
    trained, _ = rl_cost(MODEL, steps=10, prompts_per_step=32, group_size=4,
                         prompt_tokens=500, completion_tokens=300, skip_share=0.0)
    skipped, _ = rl_cost(MODEL, steps=10, prompts_per_step=32, group_size=4,
                         prompt_tokens=500, completion_tokens=300, skip_share=1.0)
    assert skipped < trained
    # Sampling alone is ~28% of a fully-trained step: real money for zero
    # gradient, which is why the degenerate-group skip threshold matters.
    assert 0.2 * trained < skipped < 0.4 * trained


def test_caching_discounts_the_prompt_but_only_the_prompt():
    full, _ = rl_cost(MODEL, steps=1, prompts_per_step=10, group_size=1,
                      prompt_tokens=1000, completion_tokens=0, cached_share=0.0,
                      skip_share=1.0)
    cached, _ = rl_cost(MODEL, steps=1, prompts_per_step=10, group_size=1,
                        prompt_tokens=1000, completion_tokens=0, cached_share=1.0,
                        skip_share=1.0)
    assert cached == pytest.approx(full * CACHED_PROMPT_FRACTION, rel=1e-6)


def test_serving_cost_is_per_call_and_small():
    one = inference_cost(MODEL, prompt_tokens=4000, completion_tokens=600, calls=1)
    assert 0.0001 < one < 0.01
    assert inference_cost(MODEL, prompt_tokens=4000, completion_tokens=600,
                          calls=200) == pytest.approx(one * 200)


def test_a_more_expensive_model_costs_more_everywhere():
    cheap, _ = sft_cost(MODEL, 1_000_000, 2, 0, 0)
    dear, _ = sft_cost("Qwen/Qwen3.8-27B-FP8", 1_000_000, 2, 0, 0)
    assert dear > cheap * 4


def test_estimate_applies_contingency():
    e = Estimate(model=MODEL, contingency=0.30)
    e.add("a", 100.0)
    e.add("b", 100.0)
    assert e.subtotal == 200.0
    assert e.total == 260.0
    assert "TOTAL" in e.render()
