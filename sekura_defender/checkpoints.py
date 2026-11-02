"""Checkpoint save/resume with a run manifest.

The footgun this exists to prevent: resuming from a BARE PATH loses the step
counter, which silently restarts the manual LR schedule at warmup in the middle
of a run. Persist the Checkpoint object's fields and reconstruct it on resume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path


@dataclass
class CheckpointRecord:
    path: str
    step: int
    checkpoint_type: str
    kind: str                      # "training" | "inference"
    note: str = ""


@dataclass
class RunManifest:
    run_id: str
    base_model: str
    lora_rank: int
    config: dict = field(default_factory=dict)
    checkpoints: list[CheckpointRecord] = field(default_factory=list)
    cursor: int = 0                # batch index within the epoch, for exact replay
    epoch: int = 0
    step: int = 0
    attestation_id: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        d = json.loads(path.read_text())
        cps = [CheckpointRecord(**c) for c in d.pop("checkpoints", [])]
        return cls(checkpoints=cps, **d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)          # atomic: a crash never leaves a half-written manifest

    def latest(self, kind: str = "training") -> CheckpointRecord | None:
        for c in reversed(self.checkpoints):
            if c.kind == kind:
                return c
        return None


def save(model, manifest: RunManifest, name: str, *, kind: str = "training",
         ttl_days: int | None = 14, note: str = "") -> CheckpointRecord:
    """Save and record. Inference checkpoints that may be served get ttl=None:
    a TTL'd checkpoint expiring under a live replica is an outage."""
    ttl = None if (kind == "inference" or ttl_days is None) else timedelta(days=ttl_days)
    ck = model.save_weights(name, mode=kind, ttl=ttl)
    rec = CheckpointRecord(path=ck.path, step=getattr(ck, "step", manifest.step),
                           checkpoint_type=getattr(ck, "checkpoint_type", kind),
                           kind=kind, note=note)
    manifest.checkpoints.append(rec)
    return rec


def restore(session, base_model: str, lora_cfg, rec: CheckpointRecord | None,
            *, load_optimizer: bool = True):
    """Recreate a model, optionally from a checkpoint.

    load_optimizer=False when transitioning SFT -> RL: AdamW moments accumulated
    under cross-entropy are wrong for a policy-gradient objective and produce a
    large, damaging first step.
    """
    if rec is None:
        return session.create_model(base_model=base_model, lora=lora_cfg)
    model = session.create_model(base_model=base_model, lora=lora_cfg,
                                 checkpoint=rec.path)
    try:
        model.load_weights(rec.path, load_optimizer=load_optimizer)
    except TypeError:
        pass
    return model
