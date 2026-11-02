"""Tokenization of real examples, batching, and the river datum shape."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.batching import Skipped, batches, load_split, to_river, tokenize_example
from sekura_sft.config import (DATASET_DIR, MAX_SEQ_OVERFLOW, TASK_WEIGHT,
                               TASK_WEIGHT_DEFAULT, max_completion_for,
                               supervised_budget)
from sekura_sft.tokenization import load, supervised_tokens

pytestmark = pytest.mark.skipif(not (DATASET_DIR / "train").exists(),
                                reason="run scripts/build_dataset.py first")


@pytest.fixture(scope="module")
def tokenized():
    tok = load()
    out, skipped = [], []
    for row in load_split(DATASET_DIR / "train"):
        try:
            out.append(tokenize_example(tok, row))
        except Skipped as e:
            skipped.append((row["task"], str(e)))
    return out, skipped


def test_nothing_is_silently_dropped(tokenized):
    """A skipped example is training data thrown away. The patch-style repair
    target and the per-task planner cap exist so this stays empty."""
    _, skipped = tokenized
    assert not skipped, f"{len(skipped)} examples skipped: {skipped[:3]}"


def test_arrays_are_parallel_and_bounded(tokenized):
    data, _ = tokenized
    for d in data:
        n = len(d["input_ids"])
        assert len(d["target_tokens"]) == n == len(d["weights"])
        assert n <= MAX_SEQ_OVERFLOW
        assert d["_n_completion"] <= max_completion_for(d["task"])


def test_supervised_mass_equals_task_weight(tokenized):
    data, _ = tokenized
    for d in data[:400]:
        want = TASK_WEIGHT.get(d["task"], TASK_WEIGHT_DEFAULT)
        assert sum(d["weights"]) == pytest.approx(want, abs=1e-9)


def test_batch_budget_targets_a_step_count(tokenized):
    """A fixed token budget yielded SIX steps per epoch on this corpus -- twelve
    optimizer updates for a two-epoch run."""
    data, _ = tokenized
    total = sum(supervised_tokens(d) for d in data)
    budget = supervised_budget(total)
    steps = len(list(batches(data, budget)))
    assert 60 <= steps <= 260, f"{steps} steps/epoch is outside a trainable range"


def test_batches_are_length_homogeneous(tokenized):
    data, _ = tokenized
    for batch in batches(data):
        lens = [len(d["input_ids"]) for d in batch]
        if len(lens) > 1:
            assert max(lens) / max(1, min(lens)) <= 16


def test_batches_cover_every_example_exactly_once(tokenized):
    data, _ = tokenized
    seen = [d["datum_id"] for b in batches(data) for d in b]
    assert sorted(seen) == sorted(d["datum_id"] for d in data)


def test_to_river_emits_only_the_contract_fields(tokenized):
    data, _ = tokenized
    batch = next(iter(batches(data)))
    river_batch = to_river(batch)
    assert len(river_batch) == len(batch)
    for d in river_batch:
        assert set(d) == {"input_ids", "target_tokens", "weights"}
        assert len(d["input_ids"]) == len(d["target_tokens"]) == len(d["weights"])
        assert all(isinstance(i, int) for i in d["input_ids"][:8])


def test_batch_loss_scales_with_examples_not_sequence_length(tokenized):
    """river's renderer normalizes each example's weights to sum to 1.0, so
    batch loss scales with the NUMBER OF EXAMPLES and not with sequence length --
    and its published 1e-4 starting LR assumes that regime. Dividing by the batch
    count as well made batch loss scale with nothing and shrank the gradient by
    the batch size."""
    data, _ = tokenized
    for batch in list(batches(data))[:8]:
        mass = sum(sum(d["weights"]) for d in to_river(batch))
        expected = sum(TASK_WEIGHT.get(d["task"], TASK_WEIGHT_DEFAULT) for d in batch)
        assert mass == pytest.approx(expected, rel=1e-6)
        # length must not enter into it
        assert mass > 0.5 * len(batch)


def test_repair_targets_are_patches_not_transcriptions(tokenized):
    """99% of the old whole-tree target was unchanged YAML copy."""
    data, _ = tokenized
    repair = [d for d in data if d["task"] == "attacktree_repair"]
    assert repair
    lens = sorted(d["_n_completion"] for d in repair)
    assert lens[len(lens) // 2] < 200, "repair target regressed to transcription"
