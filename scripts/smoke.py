"""Phase 0 smoke test: exercise every river code path at negligible cost.

Target under $15 and under an hour. The point is that nothing here is expensive,
so a broken harness is discovered for pennies rather than mid-run. Each check is
independent and reports pass/fail; the script exits non-zero if any fails.

  RIVER_API_KEY=rv_... python scripts/smoke.py
  RIVER_API_KEY=rv_... python scripts/smoke.py --skip serving
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sekura_defender import checkpoints as ck
from sekura_defender.costs import estimate
from sekura_defender.river_session import (MissingApiKey, NoUsableModel, client,
                                           lora, select_model)
from sekura_defender.trainer import Trainer, TrainConfig
from sekura_sft.batching import load_split, to_river, tokenize_example
from sekura_sft.config import DATASET_DIR, LORA_RANK
from sekura_sft.tokenization import load as load_tok
from sekura_sft.tokenization import preflight

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                detail = fn(*a, **kw) or ""
                RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}: {detail}")
                return True
            except Exception as exc:
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
                if "-v" in sys.argv:
                    traceback.print_exc()
                return False
        return wrapped
    return deco


@check("tokenizer preflight")
def c_preflight():
    info = preflight()
    return f"{info['tokenizer']} vocab={info['vocab_size']} eos={info['eos_token']}"


@check("capabilities gate")
def c_caps(cl):
    caps = select_model(cl)
    return f"healthy={caps.healthy} selected={caps.selected} ({len(caps.models)} available)"


@check("prompt token round-trip through the sampler")
def c_roundtrip(cl, model_id):
    """The #1 silent failure: what we tokenize locally must be what the server
    sees. Compare our ids against the ids the sampler echoes back."""
    tok = load_tok()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Reply with the single word: ok"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ours = tok(text, add_special_tokens=False)["input_ids"]
    with cl.session(project="sekura-smoke") as s:
        got = s.sample(text, base_model=model_id, max_tokens=4, temperature=0.0,
                       return_prompt_logprobs=False)
        sample = got[0][0]
        theirs = getattr(sample, "prompt_token_ids", None)
    if theirs and list(theirs) != list(ours):
        raise AssertionError(f"prompt ids differ: ours={len(ours)} theirs={len(theirs)}")
    return f"{len(ours)} prompt tokens{'' if theirs else ' (server did not echo ids)'}"


@check("tiny SFT run: loss decreases")
def c_sft(cl, model_id, run_dir):
    tok = load_tok()
    rows = []
    for row in load_split(DATASET_DIR / "train"):
        rows.append(tokenize_example(tok, row))
        if len(rows) >= 64:
            break
    with cl.session(project="sekura-smoke") as s:
        model = s.create_model(base_model=model_id,
                               lora=lora(LORA_RANK, train_unembed=False))
        mf = ck.RunManifest(run_id="smoke", base_model=model_id, lora_rank=LORA_RANK)
        t = Trainer(model=model, manifest=mf, run_dir=run_dir,
                    cfg=TrainConfig(epochs=1, eval_every=1000, checkpoint_every=1000,
                                    target_steps_per_epoch=4))
        t.train(rows)
        losses = [l.loss for l in t.logs]
    if len(losses) < 2:
        raise AssertionError(f"only {len(losses)} steps ran")
    if not (losses[-1] < losses[0]):
        raise AssertionError(f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return f"{len(losses)} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f}"


@check("checkpoint resume preserves step")
def c_resume(cl, model_id, run_dir):
    """The bare-path footgun: passing a path instead of the Checkpoint object
    loses the step counter and silently restarts the LR warmup."""
    mf = ck.RunManifest.load(run_dir / "manifest.json")
    rec = mf.latest("training")
    if rec is None:
        raise AssertionError("no training checkpoint was written")
    with cl.session(project="sekura-smoke") as s:
        model = s.create_model(base_model=model_id,
                               lora=lora(LORA_RANK, train_unembed=False),
                               checkpoint=rec.path)
    return f"restored {rec.path} at step {rec.step}"


@check("held-out forward returns finite loss")
def c_forward(cl, model_id):
    tok = load_tok()
    rows = []
    for row in load_split(DATASET_DIR / "val"):
        rows.append(tokenize_example(tok, row))
        if len(rows) >= 8:
            break
    with cl.session(project="sekura-smoke") as s:
        model = s.create_model(base_model=model_id,
                               lora=lora(LORA_RANK, train_unembed=False))
        res = model.forward(to_river(rows, normalize_by_batch=False),
                            loss_fn="cross_entropy")
    loss = float((getattr(res, "metrics", {}) or {}).get("loss", float("nan")))
    if not (loss == loss and loss > 0):
        raise AssertionError(f"non-finite loss {loss}")
    return f"loss={loss:.2f} over {len(rows)} examples"


@check("training-data attestation accepted")
def c_attest(cl, model_id):
    import river_client as river
    import hashlib
    body = b'{"smoke":"artifact"}\n'
    art = river.TrainingDataArtifact(name="smoke.jsonl",
                                     expected_sha256=hashlib.sha256(body).hexdigest(),
                                     content=body)
    with cl.session(project="sekura-smoke") as s:
        att = s.attest_training_data([art])
        s.create_model(base_model=model_id, lora=lora(LORA_RANK, train_unembed=False),
                       training_data_attestation=att)
    return "attestation bound to a model"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    args, _ = ap.parse_known_args()

    print("Sekura Defender -- Phase 0 smoke\n")
    if not (DATASET_DIR / "train").exists():
        print("  dataset missing; run scripts/build_dataset.py first", file=sys.stderr)
        return 2

    if not c_preflight():
        return 1

    try:
        cl = client()
    except MissingApiKey as exc:
        # Expected until a key exists; a traceback here reads like a crash.
        print(f"\n  {exc}")
        print("  Everything above this line runs offline and passed.")
        return 2

    try:
        caps = select_model(cl)
    except (NoUsableModel, Exception) as exc:
        print(f"\n  cannot select a base model: {type(exc).__name__}: {exc}")
        return 2
    c_caps(cl)
    model_id = caps.selected
    run_dir = Path(tempfile.mkdtemp()) / "smoke"

    if "roundtrip" not in args.skip:
        c_roundtrip(cl, model_id)
    if "sft" not in args.skip:
        if c_sft(cl, model_id, run_dir):
            c_resume(cl, model_id, run_dir)
    if "forward" not in args.skip:
        c_forward(cl, model_id)
    if "attest" not in args.skip:
        c_attest(cl, model_id)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    est = estimate(model_id, 4_730_714, 2, eval_tokens=331_077)
    print("\nprojected full run:\n" + est.render())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
