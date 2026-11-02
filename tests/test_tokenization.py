"""Tokenization tests against the real Qwen tokenizer.

These need the tokenizer files (cached after first run). They exist because a
chat-template or tokenizer change corrupts every training example identically
and invisibly: the loss curve looks fine and the model is merely worse.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.config import TASK_WEIGHT
from sekura_sft.tokenization import (encode, load, preflight, render_prompt,
                                     supervised_tokens)

SYSTEM = "You are Sekura Defender."
TURNS = [{"role": "user", "content": "Explain SSRF to IMDS in one sentence."}]
TARGET = "SSRF lets an attacker coerce the server into reading IMDS credentials."


@pytest.fixture(scope="module")
def tok():
    return load()


def test_preflight_passes():
    info = preflight()
    assert info["eos_token"] == "<|im_end|>"
    assert info["vocab_size"] > 200_000


def test_arrays_are_parallel_and_aligned(tok):
    p = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    d = encode(tok, p, TARGET)
    n = len(d["input_ids"])
    assert len(d["target_tokens"]) == n
    assert len(d["weights"]) == n
    assert n == d["_n_prompt"] + d["_n_completion"]
    # position i predicts ids[i+1]
    assert d["target_tokens"][:-1] == d["input_ids"][1:]


def test_prompt_fully_masked_completion_fully_supervised(tok):
    p = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    d = encode(tok, p, TARGET)
    first = next(i for i, w in enumerate(d["weights"]) if w > 0)
    assert first == d["_n_prompt"] - 1          # last prompt position predicts first completion token
    assert all(w == 0.0 for w in d["weights"][:first])
    # ...supervised through T-2 only: the server force-zeros weights[T-1]
    # because there is no next-token target there.
    assert all(w > 0.0 for w in d["weights"][first:-1])
    assert d["weights"][-1] == 0.0
    assert supervised_tokens(d) == d["_n_completion"]


def test_target_carries_no_leading_space(tok):
    """river's SFT example writes tok(" " + target) because it renders a bare
    text prompt. apply_chat_template already ends with a newline, so a leading
    space would train a spurious token at the head of every completion."""
    p = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    plain = encode(tok, p, TARGET)
    spaced = encode(tok, p, " " + TARGET)
    first_plain = tok.decode([plain["input_ids"][plain["_n_prompt"]]])
    first_spaced = tok.decode([spaced["input_ids"][spaced["_n_prompt"]]])
    # The harm is not extra tokens -- both encode to the same count. It is that
    # the model would learn a space-prefixed variant as the first token of every
    # completion, which then leaks into generation.
    assert first_plain != first_spaced
    assert not first_plain.startswith(" ")
    assert first_spaced.startswith(" ")


def test_weights_normalize_by_completion_length(tok):
    """Losses are token-summed, so without this a long example silently
    outweighs a short one and the task mixture stops meaning anything."""
    p = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    short = encode(tok, p, "SSRF reads IMDS credentials.")
    long = encode(tok, p, TARGET * 12)
    assert long["_n_completion"] > 6 * short["_n_completion"]
    # Total supervised mass is EXACTLY equal, not merely close: dividing by the
    # supervised-position count (C+1) rather than by C is what makes it exact.
    assert sum(short["weights"]) == pytest.approx(1.0, abs=1e-9)
    assert sum(long["weights"]) == pytest.approx(1.0, abs=1e-9)


def test_mass_is_exactly_task_weight_at_every_length(tok):
    """Regression guard for the (C+1)/C inflation bug: a 3-token completion
    once carried 33% more gradient mass than a long one."""
    p = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    for target in ("Yes.", "SSRF reads IMDS credentials.", TARGET, TARGET * 12):
        d = encode(tok, p, target, task="triage")
        assert sum(d["weights"]) == pytest.approx(TASK_WEIGHT["triage"], abs=1e-9)


def test_task_weight_scales_total_mass(tok):
    p = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    base = encode(tok, p, TARGET, task="__unknown__")
    triage = encode(tok, p, TARGET, task="triage")
    assert TASK_WEIGHT["triage"] > 1.0
    assert sum(triage["weights"]) == pytest.approx(
        sum(base["weights"]) * TASK_WEIGHT["triage"], rel=1e-6)


def test_thinking_flag_changes_the_prompt(tok):
    off = render_prompt(tok, SYSTEM, TURNS, thinking=False)
    on = render_prompt(tok, SYSTEM, TURNS, thinking=True)
    assert off != on
    # thinking=False closes an empty think block so the answer starts immediately
    assert off.endswith("</think>\n\n")
    # thinking=True leaves the block OPEN: targets must close it, not reopen it
    assert on.endswith("<think>\n")
    assert not on.endswith("</think>\n\n")


def test_turns_must_end_on_user(tok):
    with pytest.raises(ValueError):
        render_prompt(tok, SYSTEM, [{"role": "assistant", "content": "hi"}], thinking=False)
    with pytest.raises(ValueError):
        render_prompt(tok, SYSTEM, [], thinking=False)


def test_multi_turn_prompt_includes_history(tok):
    turns = [{"role": "user", "content": "What is IMDSv2?"},
             {"role": "assistant", "content": "A session-token-protected metadata service."},
             {"role": "user", "content": "How does SSRF defeat it?"}]
    p = render_prompt(tok, SYSTEM, turns, thinking=False)
    assert "IMDSv2" in p and "session-token-protected" in p
    assert p.count("<|im_start|>assistant") == 2   # the history turn + generation prompt


def test_training_and_sampling_prompts_are_byte_identical(tok):
    """The property that actually removes train/serve skew.

    river samples from the raw string we hand it, so consistency is ours to
    hold: the string used to build a training example must be the same string
    used to build the sampling prompt. Note river's own renderer does NOT have
    this property -- its sample prompt and training example diverge at the
    think boundary on a BPE merge.
    """
    system, turns = SYSTEM, TURNS
    for thinking in (False, True):
        train_prompt = render_prompt(tok, system, turns, thinking=thinking)
        serve_prompt = render_prompt(tok, system, turns, thinking=thinking)
        assert train_prompt == serve_prompt
        d = encode(tok, train_prompt, TARGET)
        served = tok(serve_prompt, add_special_tokens=False)["input_ids"]
        assert d["input_ids"][:len(served)] == served, (
            "the sampling prompt must be an exact token prefix of the training "
            "sequence, or the model never sees its serving context during training")
        assert d["_n_prompt"] == len(served)


def test_the_rl_env_prompt_matches_the_sft_prompt(tok):
    """ENV-B builds its prompts through the same renderer as the SFT slice; if
    those drifted apart, RL would optimize a distribution the model was never
    trained on."""
    from sekura_rl.envs.planner_env import PlannerEnv
    env = PlannerEnv(seed=0)
    prompts, _ = env.prompts(step=1, n=2)
    for p in prompts:
        assert p.startswith("<|im_start|>system")
        assert p.endswith("</think>\n\n") or p.endswith("<think>\n")
        toks = env.prompt_tokens([p])[0]
        assert toks == tok(p, add_special_tokens=False)["input_ids"]
