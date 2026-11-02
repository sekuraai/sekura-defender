"""Batched prediction against Fireworks' OpenAI-compatible endpoint.

`sekura_eval.suite.run` calls `predict(system, turns)` row by row, which would
serialize ~1,000 network calls. Rather than thread the eval harness, this
precomputes predictions with a pool and replays them in row order through the
same `next(it)` trick `--oracle` already uses -- so the scorers, the gate and
the baselines see an identical code path for base, tuned and oracle.

The prompt is rendered by `sekura_sft.tokenization.render_prompt`, NOT by
passing `messages` to the chat endpoint. Those two are not the same string: the
eval must exercise the template the model was trained on, or it measures
train/serve skew instead of capability.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from sekura_sft.config import MAX_COMPLETION_BY_TASK
from sekura_sft.tokenization import load, render_prompt

from .trainer import api_key

URL = "https://api.fireworks.ai/inference/v1/completions"
BASE_SERVERLESS = "accounts/fireworks/models/qwen3p6-35b-a3b"

DEFAULT_MAX_TOKENS = 2048
WORKERS = 12
RETRIES = 4

_lock = threading.Lock()


def _post(payload: dict, key: str, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def complete(prompt: str, model: str, key: str, max_tokens: int) -> str:
    """Raw-prompt completion. Greedy: eval must be reproducible run to run."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        # <|im_end|> ends an assistant turn; without it the model runs on into a
        # fabricated next turn and every structured scorer fails on trailing junk.
        "stop": ["<|im_end|>"],
    }
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            d = _post(payload, key)
            return d["choices"][0]["text"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404):
                body = e.read()[:300].decode(errors="replace")
                raise RuntimeError(f"HTTP {e.code}: {body}") from e
            threading.Event().wait(2 ** attempt)
    raise RuntimeError(f"completion failed after {RETRIES} attempts: {last!r}")


def predictions(rows: list[dict], model: str, *, workers: int = WORKERS,
                progress: bool = True) -> list[str]:
    """Predict every row, in row order, with a thread pool."""
    key = api_key()
    tok = load()
    out: list[str | None] = [None] * len(rows)
    done = [0]

    def one(i: int) -> None:
        row = rows[i]
        prompt = render_prompt(tok, row["system"], row["turns"],
                               thinking=bool(row.get("thinking")))
        cap = MAX_COMPLETION_BY_TASK.get(row.get("task", ""), DEFAULT_MAX_TOKENS)
        try:
            out[i] = complete(prompt, model, key, cap)
        except Exception as e:  # noqa: BLE001
            # An exception here must not void the whole suite: an empty string
            # scores as a failure on that row, which is the honest outcome.
            out[i] = ""
            with _lock:
                print(f"  row {i} ({row.get('task')}): {e!r}")
        with _lock:
            done[0] += 1
            if progress and done[0] % 50 == 0:
                print(f"  {done[0]}/{len(rows)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(rows))))
    return [o or "" for o in out]


def predict_fn(rows: list[dict], model: str, **kw):
    """A `Predict` that replays precomputed predictions in row order."""
    it = iter(predictions(rows, model, **kw))
    return lambda system, turns: next(it)


# ---------------------------------------------------------------------------
# Session sampling.
#
# qwen3p6-35b-a3b is NOT on serverless (the /inference completions endpoint
# answers 404 "not deployed"), so both base and tuned must be sampled through a
# training-session deployment.
# ---------------------------------------------------------------------------

from fireworks.training.sdk import (  # noqa: E402
    ModelInput,
    SamplingParams,
    ServiceClient,
)

from .trainer import FW_BASE, LORA_RANK, tokenizer_model  # noqa: E402


class SessionSampler:
    """Sample a checkpoint -- or the base model -- through one deployment.

    The base reference is an UNTRAINED rank-32 adapter. LoRA initializes B to
    zero, so dW = BA = 0 and the adapter is mathematically the identity: the
    samples are the base model's. That matters more than convenience, because it
    puts base and tuned on a byte-identical serving path -- same deployment,
    same template application, same stop handling. A base measured through the
    serverless API and a tuned model measured through a session deployment would
    differ by train/serve skew, and we would book that difference as capability.
    """

    @classmethod
    def from_training_client(cls, tc, checkpoint: str):
        """Wrap a training client the caller already owns.

        Used by the campaign so train and eval share one trainer job: the
        checkpoints are session-scoped, so evaluating them requires the session
        that produced them to still be alive.
        """
        s = cls.__new__(cls)
        s.sc = None
        s.tc = tc
        s.checkpoint = checkpoint
        s.sampler = tc.create_sampling_client(checkpoint)
        s.tok = load()
        return s

    def __init__(self, *, checkpoint: str | None = None,
                 trainer_job_id: str | None = None, rank: int = LORA_RANK,
                 name: str = "sekura-eval"):
        self.sc = ServiceClient.from_firetitan_config(
            api_key=api_key(),
            base_model=FW_BASE,
            tokenizer_model=tokenizer_model(),
            lora_rank=rank,
            lora_alpha=rank,
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
            create_deployment=True,
            trainer_job_id=trainer_job_id,
            display_name=name,
            # Do NOT leave GPUs running after an eval. The SDK default is False,
            # which is how six probe jobs ended up holding quota simultaneously.
            cleanup_trainer_on_close=True,
        )
        self.tc = self.sc.create_lora_training_client(
            FW_BASE, rank=rank, alpha=rank, train_attn=True, train_mlp=True,
            train_unembed=False,
        )
        if checkpoint is None:
            checkpoint = self.tc.save_weights_for_sampler("base-ref").result().path
            print(f"[base] untrained adapter (identity): {checkpoint}")
        self.checkpoint = checkpoint
        self.sampler = self.tc.create_sampling_client(checkpoint)
        self.tok = load()

    def complete(self, prompt_ids: list[int], max_tokens: int) -> str:
        res = self.sampler.sample(
            ModelInput.from_ints(prompt_ids),
            1,
            SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0,
                           stop=["<|im_end|>"]),
        ).result()
        seq = res.sequences[0]
        if getattr(seq, "text", None):      # mock sampler replays text
            return seq.text
        # `tokens` is the accessor; `tokens_np` is None unless the response came
        # back in array form, and list(None) raises TypeError -- which the probe
        # correctly reported as "sampler broken" when the sampler was fine and
        # this line was wrong.
        toks = getattr(seq, "tokens", None)
        if toks is None:
            toks = getattr(seq, "tokens_np", None)
        if toks is None:
            toks = getattr(seq, "_tokens_list", None)
        if toks is None:
            raise RuntimeError(
                f"no tokens on sampled sequence; attrs="
                f"{[a for a in dir(seq) if not a.startswith('__')]}"
            )
        return self.tok.decode(list(toks), skip_special_tokens=True)

    def predictions(self, rows: list[dict], *, workers: int = WORKERS,
                    progress: bool = True, budget=None,
                    probe: int = 25) -> list[str]:
        """Predict all rows, but sample a probe wave FIRST.

        A broken sampler and a terrible model both produce empty strings and a
        zero score. The difference is that a full eval costs ~35 GPU-minutes, so
        the distinction has to be drawn on the first 25 rows, not the last.
        """
        from .budget import check_sampler_alive

        out: list[str | None] = [None] * len(rows)
        done = [0]

        def one(i: int) -> None:
            row = rows[i]
            ids = row.get("_prompt_ids")
            if ids is None:          # not pre-rendered; do it now
                text = render_prompt(self.tok, row["system"], row["turns"],
                                     thinking=bool(row.get("thinking")))
                ids = self.tok(text, add_special_tokens=False)["input_ids"]
            cap = row.get("_max_tokens") or MAX_COMPLETION_BY_TASK.get(
                row.get("task", ""), DEFAULT_MAX_TOKENS)
            try:
                out[i] = self.complete(ids, cap)
            except Exception as e:  # noqa: BLE001
                out[i] = ""
                with _lock:
                    print(f"  row {i} ({row.get('task')}): {e!r}")
            with _lock:
                done[0] += 1
                if progress and done[0] % 50 == 0:
                    tail = f" | {budget.line()}" if budget is not None else ""
                    print(f"  {done[0]}/{len(rows)}{tail}", flush=True)

        head = list(range(min(probe, len(rows))))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, head))
        check_sampler_alive([out[i] or "" for i in head])
        if budget is not None:
            budget.tick("eval-probe", rows=len(head))
        print(f"  probe ok ({len(head)} rows)", flush=True)

        rest = list(range(len(head), len(rows)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, rest))
        if budget is not None:
            budget.tick("eval-done", rows=len(rows))
        return [o or "" for o in out]

    def close(self) -> None:
        if self.sc is None:  # borrowed client; the owner closes it
            return
        try:
            self.sc.close()
        except Exception:  # noqa: BLE001, S110
            pass
