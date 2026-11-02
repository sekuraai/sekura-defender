"""Export the assembled corpus to Fireworks chat-JSONL (managed SFT path).

CORRECTION to an earlier note in this file: I wrote that moving to Fireworks
costs us per-token loss masking. That is wrong on both counts.

  * Fireworks' Training API is explicitly Tinker-compatible --
    `forward_backward_custom`, `optim_step`, raw Datums and per-token
    `token_weights` float arrays. Nothing about the low-level path is lost, and
    the river harness ports mostly mechanically.
  * Managed SFT (this file's target) masks at MESSAGE granularity, not token,
    via an optional `weight` on each message: 0 excludes it from the loss. Since
    our examples have exactly one assistant turn, a per-message weight is a
    per-example weight -- which is what TASK_WEIGHT wants.

The docs say only "0 excludes that message from the loss. Any other number
trains it", which does not promise that 1.5 scales the gradient 1.5x. So
`--weights` emits the field for the platform to honour if it does, and
`--resample` remains available as the duplication-based fallback that works
regardless. Do not use both.

The corpus itself -- curation, dedup, entity-keyed splits, teacher slices -- is
platform-agnostic and transfers untouched.

Managed SFT accepts 3 to 3,000,000 JSONL lines, 1 GB max per upload file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from sekura_sft.config import DATASET_DIR, TASK_WEIGHT, TASK_WEIGHT_DEFAULT

# Fireworks documents a conversation-per-line JSONL. Keep the record minimal:
# unknown extra keys are a common source of silent validation failures.
ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT = "system", "user", "assistant"


def to_chat(row: dict, *, include_system: bool = True,
            with_weight: bool = False) -> dict:
    """One dataset row -> one Fireworks chat record.

    `with_weight` stamps TASK_WEIGHT onto the single assistant turn. Harmless if
    the platform treats any non-zero weight as 1.0; a free win if it scales.
    """
    msgs: list[dict] = []
    if include_system and row.get("system"):
        msgs.append({"role": ROLE_SYSTEM, "content": row["system"]})
    for turn in row["turns"]:
        role = turn["role"]
        if role not in (ROLE_USER, ROLE_ASSISTANT):
            raise ValueError(f"unexpected role {role!r}")
        msgs.append({"role": role, "content": turn["content"]})
    if msgs[-1]["role"] != ROLE_USER:
        raise ValueError("turns must end on a user message")
    assistant = {"role": ROLE_ASSISTANT, "content": row["target"]}
    if with_weight:
        assistant["weight"] = TASK_WEIGHT.get(row.get("task", ""), TASK_WEIGHT_DEFAULT)
    msgs.append(assistant)
    return {"messages": msgs}


def load_split(split: str) -> Iterator[dict]:
    d = DATASET_DIR / split
    for shard in sorted(d.glob("*.jsonl")):
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def resample_for_mixture(rows: list[dict], *, cap: float = 2.0) -> list[dict]:
    """Reproduce TASK_WEIGHT by duplication, since Fireworks has no per-token
    weights.

    A task at weight 1.5 appears 1.5x as often; weight 0.5 appears half as
    often. Capped so no task can dominate through duplication alone, and
    deterministic (every nth row) rather than random so a rebuild is stable.
    """
    out: list[dict] = []
    for row in rows:
        w = TASK_WEIGHT.get(row.get("task", ""), TASK_WEIGHT_DEFAULT)
        w = min(w, cap)
        whole, frac = int(w), w - int(w)
        out.extend([row] * whole)
        if frac:
            # deterministic fractional inclusion: keep every 1/frac-th row
            idx = len(out)
            if idx % max(1, round(1 / frac)) == 0:
                out.append(row)
    return out


def write(split: str, dest: Path, *, resample: bool = False,
          max_records: int | None = None, with_weight: bool = False) -> dict:
    if resample and with_weight:
        raise ValueError("use --resample OR --weights, not both: they would "
                         "compound into a squared mixture")
    rows = list(load_split(split))
    if resample:
        rows = resample_for_mixture(rows)
    if max_records:
        rows = rows[:max_records]
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = chars = skipped = 0
    with open(dest, "w", encoding="utf-8") as fh:
        for row in rows:
            try:
                rec = to_chat(row, with_weight=with_weight)
            except ValueError:
                skipped += 1
                continue
            line = json.dumps(rec, ensure_ascii=False)
            fh.write(line + "\n")
            n += 1
            chars += len(line)
    return {"split": split, "records": n, "skipped": skipped,
            "bytes": dest.stat().st_size, "approx_tokens": chars // 4,
            "path": str(dest)}


def balanced_subset(rows: list[dict], n: int, *, min_per_task: int = 40) -> list[dict]:
    """A small, task-balanced subset for a fast first training run.

    A hackathon window does not fit 14,172 examples at ~1.5k tokens each. This
    takes a floor from every task so no capability is absent, then fills the
    remainder proportionally. Deterministic: rows are already sorted by
    datum_id in the shards, so the same subset comes back every time.
    """
    import collections
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_task[r.get("task", "?")].append(r)

    out: list[dict] = []
    # floor first -- a task with no examples cannot be evaluated at all
    for task, rs in sorted(by_task.items()):
        out.extend(rs[:min(min_per_task, len(rs))])
    # then fill proportionally to what each task actually has
    remaining = max(0, n - len(out))
    if remaining:
        total = sum(max(0, len(rs) - min_per_task) for rs in by_task.values()) or 1
        for task, rs in sorted(by_task.items()):
            extra = rs[min_per_task:]
            take = int(remaining * len(extra) / total)
            out.extend(extra[:take])
    return out[:n] if n else out
