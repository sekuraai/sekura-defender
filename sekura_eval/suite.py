"""Run a model over the frozen eval set and report per-capability metrics.

river ships no evaluation framework, so this is entirely ours. `predict` is any
callable taking the rendered messages and returning text, which keeps the suite
usable against a checkpoint, a promoted replica, or a stub.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sekura_sft.config import DATASET_DIR

from .scorers import score_one

Predict = Callable[[str, list], str]

# Rate metrics that must ALWAYS be reported, even at zero.
#
# These are counters over failures, so a perfect model produces none of them and
# the key would simply be absent -- which made the regression gate fail the
# ORACLE against itself ("dangerous_rate absent; cannot gate it"). A metric that
# vanishes exactly when the model is perfect is worse than useless.
ALWAYS_REPORT: dict[str, tuple[str, ...]] = {
    "planner": ("zero_coverage_rate",),
    "triage": ("dangerous_rate", "reason_exact_rate"),
}


@dataclass
class TaskMetrics:
    task: str
    n: int = 0
    mean_score: float = 0.0
    exact: float = 0.0
    extra: dict = field(default_factory=dict)


def load_eval(split: str = "test") -> list[dict]:
    rows = []
    d = DATASET_DIR / split
    for shard in sorted(d.glob("*.jsonl")):
        with open(shard, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def run(predict: Predict, rows: Iterable[dict] | None = None,
        split: str = "test") -> dict:
    rows = list(rows if rows is not None else load_eval(split))
    per: dict[str, list] = defaultdict(list)
    details: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        pred = predict(row["system"], row["turns"])
        # Grounded-text scorers check citations and entities against the evidence,
        # which lives in the prompt itself.
        meta = dict(row.get("meta") or {})
        meta["_evidence"] = "\n".join(t["content"] for t in row["turns"])
        res = score_one(row["task"], pred, row["target"], meta)
        per[row["task"]].append(res)
        d = res.detail or {}
        if d.get("dangerous"):
            details[row["task"]]["dangerous"] += 1
        if d.get("zero_coverage"):
            details[row["task"]]["zero_coverage"] += 1
        if d.get("reason_exact"):
            details[row["task"]]["reason_exact"] += 1

    out: dict[str, dict] = {}
    for task, results in sorted(per.items()):
        n = len(results)
        out[task] = {
            "n": n,
            "mean_score": round(sum(r.score for r in results) / n, 4),
            "exact": round(sum(1 for r in results if r.exact) / n, 4),
        }
        for key, count in details[task].items():
            out[task][f"{key}_rate"] = round(count / n, 4)
        for key in ALWAYS_REPORT.get(task, ()):
            out[task].setdefault(key, 0.0)
    out["_overall"] = {
        "n": len(rows),
        "mean_score": round(
            sum(v["mean_score"] * v["n"] for k, v in out.items() if k != "_overall")
            / max(1, sum(v["n"] for k, v in out.items() if k != "_overall")), 4),
        "tasks": len(out),
    }
    return out


def render(report: dict) -> str:
    lines = [f"{'task':22s} {'n':>5s} {'score':>7s} {'exact':>7s}  flags"]
    for task, m in sorted(report.items()):
        if task == "_overall":
            continue
        flags = "  ".join(f"{k}={v}" for k, v in m.items()
                          if k.endswith("_rate"))
        lines.append(f"{task:22s} {m['n']:5d} {m['mean_score']:7.4f} "
                     f"{m['exact']:7.4f}  {flags}")
    o = report["_overall"]
    lines.append(f"\noverall: {o['n']} examples across {o['tasks'] - 1} tasks, "
                 f"weighted mean {o['mean_score']:.4f}")
    return "\n".join(lines)
