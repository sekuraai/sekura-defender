"""Promote a checkpoint and wire it into sekura-rs.

Promotion is BETA: there is no published base_url template, no documented rate
limits, and status_reason is the only diagnostic. So the returned base_url is
persisted rather than derived, and a three-tier fallback is mandatory -- a beta
serving path must never be a hard dependency of a customer scan.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

READY_STATES = {"ready", "READY", "running", "RUNNING", "active", "ACTIVE"}
FAILED_STATES = {"failed", "FAILED", "error", "ERROR"}


class PromotionFailed(RuntimeError):
    pass


@dataclass
class Replica:
    model: str
    base_url: str
    checkpoint: str
    base_model: str = ""
    status: str = ""
    replica_id: str = ""

    def env(self, api_key_var: str = "$RIVER_API_KEY") -> dict[str, str]:
        """The three vars sekura-rs already reads -- no Rust changes needed.

        base_url is stripped of a trailing slash: the OpenAI client builds
        "{api_base}/chat/completions", so a trailing slash yields a double slash.
        AliceBobCindy.py documents the same gotcha for the Python client.
        """
        return {
            "LLM_API_BASE": self.base_url.rstrip("/"),
            "LLM_API_KEY": api_key_var,
            # The promote ALIAS, never the checkpoint path.
            "LLM_MODEL": self.model,
        }


def promote(client, checkpoint_path: str, alias: str, *, base_model: str = "",
            timeout_s: float = 900.0, poll_s: float = 5.0,
            sleep=time.sleep) -> Replica:
    """Promote, then poll until ready. Logs status_reason on every non-ready
    poll because it is the only diagnostic available."""
    client.promote_streaming_replica(checkpoint_path, alias)
    deadline = time.time() + timeout_s
    last_reason = ""
    while time.time() < deadline:
        rep = client.get_streaming_replica(alias)
        if rep is not None:
            status = getattr(rep, "status", "") or ""
            reason = getattr(rep, "status_reason", "") or ""
            if reason and reason != last_reason:
                last_reason = reason
            if status in FAILED_STATES:
                raise PromotionFailed(f"{alias}: {status} ({reason})")
            base_url = getattr(rep, "base_url", None)
            if status in READY_STATES and base_url:
                return Replica(model=alias, base_url=base_url,
                               checkpoint=getattr(rep, "checkpoint", checkpoint_path),
                               base_model=getattr(rep, "base_model", base_model),
                               status=status,
                               replica_id=getattr(rep, "replica_id", "") or "")
        sleep(poll_s)
    raise PromotionFailed(
        f"{alias} not ready within {timeout_s}s (last status_reason: {last_reason!r})")


def write_registry(replica: Replica, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(replica), indent=2))


def emit_env(replica: Replica) -> str:
    return "\n".join(f"export {k}={v}" for k, v in replica.env().items())
