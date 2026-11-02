"""Tokenize examples into river data, and batch them by supervised-token budget.

Batching by EXAMPLE COUNT would be wrong here. river sums losses over tokens, so
a batch of long threat models carries far more gradient than a batch of short
classifications, and the effective learning rate would wander with the shuffle.
Batching to a fixed supervised-token budget keeps step-to-step gradient scale
stable, which is what makes one LR schedule valid for the whole run.

Length-bucketing (not packing): river takes independent variable-length
sequences and charges nothing for unused positions, so packing buys no compute --
and without an exposed block-diagonal mask it would let one example attend to
another, which is invisible in the loss curve and corrupts generalization.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .config import (BASE_MODEL, LENGTH_BUCKETS, MAX_SEQ, MAX_SEQ_OVERFLOW,
                     max_completion_for, supervised_budget)
from .tokenization import encode, load, render_prompt, supervised_tokens


class Skipped(Exception):
    """An example that must not be trained on rather than silently mangled."""


def tokenize_example(tok, row: dict) -> dict:
    prompt = render_prompt(tok, row["system"], row["turns"],
                           thinking=bool(row.get("thinking")))
    d = encode(tok, prompt, row["target"], task=row.get("task", ""))

    # NEVER truncate a target: a cut-off target teaches unterminated generation,
    # which is the worst possible failure for output a Rust pipeline must parse.
    cap = max_completion_for(row.get("task", ""))
    if d["_n_completion"] > cap:
        raise Skipped(f"completion {d['_n_completion']} > {cap}")
    if len(d["input_ids"]) > MAX_SEQ_OVERFLOW:
        raise Skipped(f"sequence {len(d['input_ids'])} > {MAX_SEQ_OVERFLOW}")

    d["datum_id"] = row["datum_id"]
    d["task"] = row.get("task", "")
    d["split"] = row.get("split", "train")
    return d


def bucket_of(n: int) -> int:
    for b in LENGTH_BUCKETS:
        if n <= b:
            return b
    return MAX_SEQ_OVERFLOW


def batches(data: list[dict], budget: int | None = None) -> Iterator[list[dict]]:
    """Group into steps of ~`budget` supervised tokens, length-homogeneous.

    Buckets are emitted in a deterministic order so a run resumes exactly."""
    if budget is None:
        budget = supervised_budget(sum(supervised_tokens(d) for d in data))
    by_bucket: dict[int, list[dict]] = {}
    for d in data:
        by_bucket.setdefault(bucket_of(len(d["input_ids"])), []).append(d)

    for size in sorted(by_bucket):
        rows = sorted(by_bucket[size], key=lambda d: d["datum_id"])
        cur: list[dict] = []
        acc = 0
        for d in rows:
            sup = supervised_tokens(d)
            if cur and acc + sup > budget:
                yield cur
                cur, acc = [], 0
            cur.append(d)
            acc += sup
        if cur:
            yield cur


def to_river(batch: list[dict], *, normalize_by_batch: bool = False) -> list[dict]:
    """Strip bookkeeping and emit river datums.

    DOES NOT divide by batch size, and that default is deliberate. river's own
    renderer normalizes each example's weights to sum to 1.0 -- "per-example
    token-mean, so batch loss scales with the number of examples but not with
    sequence length" -- and its published starting LR of 1e-4 for cross_entropy
    assumes exactly that regime. Dividing by the batch count on top made batch
    loss scale with NOTHING, shrinking the gradient by the batch size (~55x
    here) and putting the effective step size orders of magnitude below the
    reference recipe. Length normalization is kept; batch normalization is not.
    """
    n = len(batch) if normalize_by_batch else 1
    out = []
    for d in batch:
        out.append({
            "input_ids": d["input_ids"],
            "target_tokens": d["target_tokens"],
            "weights": [w / n for w in d["weights"]],
        })
    return out


def load_split(split_dir: Path) -> Iterator[dict]:
    for shard in sorted(split_dir.glob("*.jsonl")):
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def tokenize_split(split_dir: Path, base_model: str = BASE_MODEL) -> tuple[list[dict], list[dict]]:
    tok = load(base_model)
    kept, skipped = [], []
    for row in load_split(split_dir):
        try:
            kept.append(tokenize_example(tok, row))
        except Skipped as e:
            skipped.append({"datum_id": row["datum_id"], "task": row.get("task"),
                            "reason": str(e)})
    return kept, skipped
