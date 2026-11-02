"""The supervised mask must survive the trip into a Fireworks Datum.

This exists because of a live, silent failure: `Datum.loss_fn_inputs` is an
untyped `dict[str, TensorData]`, so naming the weight array `token_weights`
instead of `weights` did not raise -- the key was dropped and `cross_entropy`
scored every position in the sequence, prompt included. The loss came back
bit-identical for as-built, all-zero and doubled weight arrays, which is to say
the run would have trained cleanly and learned to generate prompts.

Nothing local can catch a dropped key by inspecting our own output, so the test
asserts against the SDK's own reader, `_count_response_tokens`, which is the
code path that consumes the array server-side-bound. If a future SDK renames
the key, this fails here instead of after a paid run.
"""
from __future__ import annotations

import pytest

from fireworks.training.sdk.client import _count_response_tokens

from sekura_fireworks.trainer import KEY_TARGETS, KEY_WEIGHTS, to_datum


def _row(prompt: int, completion: int, w: float = 1.0) -> dict:
    n = prompt + completion
    return {
        "input_ids": list(range(10, 10 + n)),
        "target_tokens": list(range(11, 11 + n)),
        # 0.0 across the prompt except its last position, which predicts the
        # first response token -- the off-by-one that trains cleanly and never
        # improves if it is wrong.
        "weights": [0.0] * (prompt - 1) + [w] * (completion + 1),
    }


def test_weight_key_is_the_one_the_sdk_reads():
    assert KEY_WEIGHTS == "weights", (
        "cross_entropy reads loss_fn_inputs['weights']; any other name is "
        "silently ignored and the prompt gets trained on"
    )
    assert KEY_TARGETS == "target_tokens"


def test_sdk_sees_exactly_our_supervised_positions():
    row = _row(prompt=20, completion=5)
    expected = sum(1 for w in row["weights"] if w)
    assert _count_response_tokens([to_datum(row)]) == expected == 6


def test_zero_weights_are_not_silently_promoted_to_full_supervision():
    """The regression itself: an ignored array degrades to 'supervise all'."""
    row = _row(prompt=20, completion=5)
    row["weights"] = [0.0] * len(row["weights"])
    assert _count_response_tokens([to_datum(row)]) == 0.0


def test_arrays_stay_parallel():
    d = to_datum(_row(prompt=7, completion=3))
    lfi = d.loss_fn_inputs
    assert lfi[KEY_TARGETS].to_numpy().shape == lfi[KEY_WEIGHTS].to_numpy().shape
    assert d.model_input.length == 10


def test_dtypes_are_declared():
    """TensorData refuses to infer dtype, so both must be explicit."""
    d = to_datum(_row(prompt=4, completion=2))
    assert d.loss_fn_inputs[KEY_TARGETS].dtype == "int64"
    assert d.loss_fn_inputs[KEY_WEIGHTS].dtype == "float32"


def test_checkpoint_name_cap_is_enforced_before_the_run():
    from sekura_fireworks.trainer import MAX_CKPT_NAME, _check_name

    _check_name("a" * MAX_CKPT_NAME)
    with pytest.raises(ValueError, match="caps it at"):
        _check_name("a" * (MAX_CKPT_NAME + 1))
