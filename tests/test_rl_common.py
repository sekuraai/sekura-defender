"""RL array alignment. The ob-1 rule is the most expensive thing to get wrong."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_rl.common import (AlignmentError, Rollout, build_batch, build_datum,
                              group_advantages)


def _r(reward=1.0, prompt=3, comp=2):
    return Rollout(prompt_tokens=list(range(prompt)),
                   tokens=list(range(100, 100 + comp)),
                   logprobs=[-0.5] * comp, reward=reward)


def test_hand_computed_five_token_case():
    """prompt=[1,2,3] completion=[4,5]; ob=3 so index 2 is the first supervised
    position -- the last prompt token is what predicts the first response token."""
    r = Rollout(prompt_tokens=[1, 2, 3], tokens=[4, 5], logprobs=[-0.1, -0.2], reward=1.0)
    d = build_datum(r, advantage=0.5)
    assert d["input_ids"] == [1, 2, 3, 4, 5]
    assert d["attention_mask"] == [1, 1, 1, 1, 1]
    assert d["old_logprobs"] == [0.0, 0.0, -0.1, -0.2, 0.0]
    assert d["advantages"] == [0.0, 0.0, 0.5, 0.5, 0.0]
    assert "target_tokens" not in d, "server shifts input_ids; do not send targets"


def test_first_nonzero_advantage_is_at_ob_minus_one():
    for prompt in (1, 2, 7, 50):
        d = build_datum(_r(prompt=prompt, comp=4), advantage=0.25)
        first = next(i for i, a in enumerate(d["advantages"]) if a != 0.0)
        assert first == prompt - 1


def test_arrays_stay_parallel():
    d = build_datum(_r(prompt=9, comp=13), advantage=-0.3)
    n = len(d["input_ids"])
    assert len(d["attention_mask"]) == n
    assert len(d["old_logprobs"]) == n
    assert len(d["advantages"]) == n


def test_missing_logprobs_is_fatal():
    r = Rollout(prompt_tokens=[1, 2], tokens=[3, 4], logprobs=[], reward=1.0)
    with pytest.raises(AlignmentError):
        build_datum(r, 0.5)


def test_empty_completion_or_prompt_is_fatal():
    with pytest.raises(AlignmentError):
        build_datum(Rollout([1], [], [], 1.0), 0.5)
    with pytest.raises(AlignmentError):
        build_datum(Rollout([], [1], [-0.1], 1.0), 0.5)


def test_zero_advantage_groups_are_skipped():
    assert group_advantages([1.0, 1.0, 1.0]) is None
    assert group_advantages([0.0, 0.0]) is None


def test_near_degenerate_groups_are_skipped():
    """A tiny spread still produces a gradient, built almost entirely of noise."""
    assert group_advantages([0.500, 0.501, 0.502]) is None
    assert group_advantages([0.0, 1.0]) is not None


def test_advantages_are_centred_not_normalized():
    advs = group_advantages([0.0, 1.0], clamp_sigma=0)
    assert advs == pytest.approx([-0.5, 0.5])
    assert sum(advs) == pytest.approx(0.0)


def test_advantages_are_clamped_against_outliers():
    advs = group_advantages([0.0, 0.0, 0.0, 100.0])
    assert max(abs(a) for a in advs) < 100.0


def test_batch_accounts_for_every_skip():
    groups = [[_r(1.0), _r(1.0)],            # zero advantage
              [_r(0.0), _r(1.0)],            # usable
              [_r(0.50), _r(0.505)]]         # degenerate
    data, stats = build_batch(groups)
    assert stats["groups"] == 3
    assert stats["skipped_zero_advantage"] == 1
    assert stats["skipped_degenerate"] == 1
    assert stats["rollouts"] == 2 == len(data)
