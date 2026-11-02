"""The teacher campaign must be safe to re-run.

Both properties here were violated in a live overnight run: a finished campaign
replayed itself eleven times from cache, and each replay appended the whole
corpus again -- 9,654 unique examples became 130,553 rows and a 1.0 GB file.
No quota was burned (every call was a cache hit) but the output was 13.5x
redundant.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_teacher.py"
SUPERVISOR = ROOT / "scripts" / "overnight_teacher.py"


def _src(path: Path) -> str:
    return path.read_text()


def test_runner_loads_existing_ids_before_appending():
    src = _src(RUNNER)
    assert "already" in src and "datum_id" in src
    assert "will not re-append" in src, "must announce what it is skipping"


def test_runner_retries_only_on_a_meaningful_failure_share():
    """Three transient timeouts out of 9,784 jobs triggered eleven pointless
    supervisor attempts. Verification rejects are permanent -- the teacher
    answered and the answer failed a gate, so asking again returns the same
    cached answer."""
    src = _src(RUNNER)
    assert "retry_worthy" in src
    assert "0.02" in src, "retry threshold should be a share of the job count"


def test_supervisor_stops_when_an_attempt_makes_no_new_calls():
    """The stop condition that matters. Exit codes alone let a finished
    campaign replay itself."""
    src = _src(SUPERVISOR)
    assert "_cached_calls" in src
    assert "no new work" in src


def test_supervisor_counts_unique_examples_not_rows():
    """Rows can repeat; unique work cannot. Counting rows made every replay look
    like progress."""
    src = _src(SUPERVISOR)
    assert "datum_id" in src, "_kept must deduplicate"


@pytest.mark.skipif(not (ROOT / "data" / "examples" / "teacher.jsonl").exists(),
                    reason="no teacher output yet")
def test_teacher_output_holds_no_duplicate_rows():
    path = ROOT / "data" / "examples" / "teacher.jsonl"
    seen, rows = set(), 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows += 1
            seen.add(json.loads(line)["datum_id"])
    assert rows == len(seen), (
        f"{rows - len(seen)} duplicate rows; re-runs must not re-append")
