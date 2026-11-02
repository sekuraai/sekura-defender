"""Fireworks Training API SFT loop.

The Training API is Tinker-compatible, so everything upstream of the train call
is reused verbatim: `sekura_sft.tokenization.encode` builds the per-token
arrays, `sekura_sft.batching.batches` groups by supervised-token budget, and
`to_river` applies the per-example length normalization. Only the two train
calls differ from the river harness:

    river:      session.forward_backward(batch, "cross_entropy"); session.optim_step(lr=...)
    fireworks:  tc.forward_backward(data, "cross_entropy");        tc.optim_step(AdamParams(...))

Two documented traps, both guarded here:

  * Checkpoint names are capped at 17 characters and the cap is only enforced at
    promote time -- i.e. AFTER training has been paid for. `_check_name` asserts
    it before the first step instead.
  * Checkpoints are scoped to the training session and become NOT_FOUND when it
    tears down. `train()` promotes before returning, in a finally block, so an
    exception mid-run still leaves the weights reachable.

FireTitan does not resolve tokenizers server-side (`get_tokenizer()` raises
without an explicit `tokenizer_model`), so our tokenizer is the single source of
truth for token ids -- there is no second tokenizer to drift from.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

from fireworks.training.sdk import (
    AdamParams,
    Datum,
    ModelInput,
    ServiceClient,
    TensorData,
)

from sekura_sft.batching import batches, to_river, tokenize_split
from sekura_sft.config import BASE_MODEL, DATA

# Fireworks' model id for the base we tokenized against.
FW_BASE = "accounts/fireworks/models/qwen3p6-35b-a3b"

# create_deployment=True REQUIRES this explicitly: "the deployment accelerator is
# owned by the deployment shape", and the SDK will not read it off the training
# shape even though the training shape names it. Omitting it fails only after
# the trainer job has provisioned, i.e. after ~10 minutes of GPU time is paid.
DEPLOY_SHAPE = ("accounts/fireworks/deploymentShapes/"
                "rft-qwen3p6-35b-a3b-rl-b300-bf16-w1-p1/versions/c41c154l")

LORA_RANK = 32
PEAK_LR = 1e-4
FINAL_LR = 1e-5
WARMUP_FRAC = 0.03
MIN_WARMUP = 10

ADAM = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, grad_clip_norm=1.0)

# Built-in cross_entropy consumes these two per-token arrays alongside the
# ModelInput.
#
# KEY_WEIGHTS MUST BE "weights". `loss_fn_inputs` is an untyped dict, so a wrong
# key is silently DROPPED rather than rejected: with "token_weights" the loss
# came back bit-identical for as-built, all-zero and doubled weight arrays, i.e.
# the prompt was being trained on and the supervised mask did nothing. The
# authority is `_count_response_tokens` in sdk/client.py, which reads
# `loss_fn_inputs["weights"]` and documents them as marking "exactly which
# shifted target positions contribute to the loss" -- the same alignment our
# `encode()` already produces. `tests/test_fw_weights.py` pins this.
KEY_TARGETS = "target_tokens"
KEY_WEIGHTS = "weights"

MAX_CKPT_NAME = 17


def api_key() -> str:
    """Read the key from an INTERACTIVE zsh.

    `zsh -lc` does not source .zshrc for non-interactive shells, so it silently
    returns whatever stale value this process inherited. -i is required.
    """
    k = os.environ.get("FIREWORKS_API_KEY_FORCE")
    if k:
        return k
    out = subprocess.run(
        ["zsh", "-ic", 'printf %s "$FIREWORKS_API_KEY"'],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    if not out:
        raise RuntimeError("FIREWORKS_API_KEY not set in ~/.zshrc")
    return out


def _check_name(name: str) -> str:
    if len(name) > MAX_CKPT_NAME:
        raise ValueError(
            f"checkpoint name {name!r} is {len(name)} chars; Fireworks caps it at "
            f"{MAX_CKPT_NAME} and only checks at promote time, i.e. after you have "
            f"paid for the run"
        )
    return name


def lr_at(step: int, total: int) -> float:
    """Linear warmup to PEAK_LR, then cosine to FINAL_LR."""
    warm = max(MIN_WARMUP, int(WARMUP_FRAC * total))
    if step < warm:
        return PEAK_LR * (step + 1) / warm
    if total <= warm:
        return FINAL_LR
    prog = (step - warm) / max(1, total - warm)
    return FINAL_LR + 0.5 * (PEAK_LR - FINAL_LR) * (1 + math.cos(math.pi * prog))


def to_datum(row: dict) -> Datum:
    """river-shaped per-token arrays -> a Fireworks Datum.

    `weights` is 0.0 across the prompt, so the supervised mask carries over
    exactly; Fireworks costs us nothing in masking fidelity.
    """
    return Datum(
        model_input=ModelInput.from_ints(row["input_ids"]),
        loss_fn_inputs={
            KEY_TARGETS: TensorData(data=row["target_tokens"], dtype="int64"),
            KEY_WEIGHTS: TensorData(
                data=[float(w) for w in row["weights"]], dtype="float32"
            ),
        },
    )


def tokenizer_model() -> str:
    """The tokenizer repo our own encoder resolved, passed through verbatim.

    FP8/NVFP4 repos do not carry tokenizer files, so river maps the quantized
    repo to the tokenizer repo. Handing Fireworks the SAME resolved name is what
    makes a token-id mismatch structurally impossible rather than merely
    checked: there is one tokenizer name, used on both sides.
    """
    import river_client as river

    return river.resolve_tokenizer_name(BASE_MODEL)


def connect(rank: int = LORA_RANK, *, create_deployment: bool = False,
            trainer_job_id: str | None = None):
    """Provision a LoRA trainer.

    `create_deployment=False` by default: SFT needs forward_backward only, and
    an inference deployment we never sample from spends GPU quota and several
    minutes of provisioning for nothing.
    """
    sc = ServiceClient.from_firetitan_config(
        api_key=api_key(),
        base_model=FW_BASE,
        tokenizer_model=tokenizer_model(),
        lora_rank=rank,
        lora_alpha=rank,
        train_attn=True,
        train_mlp=True,
        # A rank-32 adapter should not spend capacity on a 248,044-row unembed
        # matrix; that is format memorization, not capability.
        train_unembed=False,
        create_deployment=create_deployment,
        deployment_shape=DEPLOY_SHAPE if create_deployment else None,
        # Reattaching to an already-provisioned job skips the ~10 minute,
        # ~$12.50 provisioning tax -- worth recovering after a crash that left
        # a warm job behind.
        trainer_job_id=trainer_job_id,
        display_name="sekura-defender-sft",
        # The SDK default is False, i.e. the trainer job KEEPS RUNNING after the
        # process exits. Three smoke/probe runs plus the real run left six GPU
        # jobs billing concurrently and the account hit its spending limit
        # mid-training, which 412'd the checkpoint save and discarded 31 steps.
        # A probe must not outlive its process.
        cleanup_trainer_on_close=True,
    )
    tc = sc.create_lora_training_client(
        FW_BASE,
        rank=rank,
        alpha=rank,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
        user_metadata={"project": "sekura-defender", "stage": "sft"},
    )
    return sc, tc


def train(
    split: str = "train",
    *,
    epochs: int = 1,
    budget: int | None = None,
    ckpt: str = "sekura-sft-v1",
    max_steps: int | None = None,
    log: Path | None = None,
    limit: int | None = None,
    save_every: int = 25,
    session: tuple | None = None,
    base_ref: str | None = None,
    max_minutes: float | None = None,
    cost_budget=None,
    tripwire=None,
    data: list[dict] | None = None,
):
    """Train, optionally inside a session the caller owns.

    `session` lets one process train AND evaluate against a single trainer job.
    That is not only cheaper than two jobs -- checkpoints are session-scoped and
    `cleanup_trainer_on_close=True` tears the job down with the process, so an
    eval in a LATER process would find NOT_FOUND. One session is the only shape
    that is both cheap and correct.

    `base_ref` saves a checkpoint BEFORE the first optim_step. LoRA
    zero-initializes B, so that adapter is the identity and sampling it gives
    base-model behaviour on the exact serving path the tuned checkpoint will
    use.

    `max_minutes` is a hard budget. This bills 4xB300 at $15/GPU-hour plus 1 for
    the sampling deployment -- $1.25/minute -- and an earlier run was killed
    mid-flight by a spending cap, so an unbounded loop is a real liability.
    """
    _check_name(ckpt)
    if base_ref:
        _check_name(base_ref)

    # Pre-tokenized cache from scripts/prep_fireworks.py. Tokenizing inside
    # this function held five B300s hot for ~90s of local CPU work.
    skipped: list = []
    if data is None:
        data, skipped = tokenize_split(DATA / "dataset" / split)
    if limit:
        data = sorted(data, key=lambda d: d["datum_id"])[:limit]
    steps = [b for _ in range(epochs) for b in batches(data, budget)]
    if max_steps:
        steps = steps[:max_steps]
    total = len(steps)
    if not total:
        raise RuntimeError("no batches to train on")

    print(f"[data] {len(data)} examples, {len(skipped)} skipped, "
          f"{total} steps over {epochs} epoch(s)")

    sc, tc = session if session else connect()
    print(f"[session] {sc.training_session_id}  run={tc.run_id}")

    rows: list[dict] = []
    t0 = time.time()
    base_path = None
    tuned_path = None
    try:
        if base_ref:
            # Before ANY optim_step: dW = BA = 0, so this is the base model.
            base_path = tc.save_weights_for_sampler(base_ref).result().path
            print(f"[base-ref] identity adapter: {base_path}", flush=True)
            rows.append({"base_ref": base_path})
        for i, batch in enumerate(steps):
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                print(f"[budget] {max_minutes} min exhausted at step {i}/{total}; "
                      f"stopping and saving", flush=True)
                rows.append({"budget_stop_step": i, "planned": total})
                break
            lr = lr_at(i, total)
            fut = tc.forward_backward([to_datum(r) for r in to_river(batch)],
                                      "cross_entropy")
            out = fut.result()
            st = tc.optim_step(AdamParams(learning_rate=lr, **ADAM))
            if hasattr(st, "result"):
                st.result()

            loss = _loss_of(out)
            # Per-example loss is the comparable quantity: weights are
            # normalized so each example contributes TASK_WEIGHT, so loss:sum
            # scales with the example count and nothing else.
            per_ex = None
            if isinstance(loss, dict) and "loss:sum" in loss and len(batch):
                per_ex = loss["loss:sum"] / len(batch)
            if tripwire is not None:
                tripwire.observe(per_ex)
            if cost_budget is not None:
                cost_budget.tick("train", step=i, of=total, per_example_loss=
                            None if per_ex is None else round(per_ex, 5))
            rec = {"step": i, "lr": lr, "loss": loss, "examples": len(batch),
                   "per_example_loss": None if per_ex is None else round(per_ex, 5),
                   "elapsed": round(time.time() - t0, 1)}
            rows.append(rec)
            cost = f" | {cost_budget.line()}" if cost_budget is not None else ""
            print(f"[{i + 1}/{total}] per_ex={per_ex if per_ex is None else round(per_ex, 4)} "
                  f"lr={lr:.2e} n={len(batch)} t={rec['elapsed']}s{cost}", flush=True)

            # Checkpoint DURING the run, not only at the end. A single save in
            # the finally block is worthless against the failure that actually
            # happened: the account was suspended mid-run, so the save itself
            # 412'd and 31 steps of training were lost with nothing to resume
            # from. Intermediate saves are cheap; a lost run is not.
            if save_every and (i + 1) % save_every == 0 and i + 1 < total:
                try:
                    r = tc.save_weights_for_sampler(ckpt).result()
                    print(f"[ckpt @{i + 1}] {r.path}", flush=True)
                    rows.append({"step": i, "checkpoint": r.path})
                    _write(log, rows)
                except Exception as e:  # noqa: BLE001
                    print(f"[ckpt @{i + 1}] FAILED: {e!r}", flush=True)
    finally:
        # Promote unconditionally: an unpromoted checkpoint dies with the
        # session, so a crash at step N must not throw away N steps of training.
        try:
            res = tc.save_weights_for_sampler(ckpt).result()
            tuned_path = res.path
            print(f"[saved] {tuned_path}")
            rows.append({"checkpoint": tuned_path})
        except Exception as e:  # noqa: BLE001
            print(f"[saved] FAILED: {e!r}")
        _write(log, rows)
        if log:
            print(f"[log] {log}")
    return {"rows": rows, "base": base_path, "tuned": tuned_path,
            "steps_done": sum(1 for r in rows if "loss" in r), "planned": total}


def _write(log: Path | None, rows: list[dict]) -> None:
    if not log:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _loss_of(out) -> float | None:
    for attr in ("loss", "loss_fn_outputs", "metrics"):
        v = getattr(out, attr, None)
        if isinstance(v, (int, float)):
            return round(float(v), 5)
        if isinstance(v, dict):
            for k in ("loss", "cross_entropy", "total_loss"):
                if k in v:
                    try:
                        return round(float(v[k]), 5)
                    except (TypeError, ValueError):
                        return v[k]
            return v
    return None
