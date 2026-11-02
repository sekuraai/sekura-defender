"""Chat-template rendering and per-token array construction for river SFT.

river takes parallel per-token arrays, not a messages array. Two things here are
deliberate deviations from the shape shown in river's own SFT example, and both
matter:

1. No leading space on the target. The docs write `tok(" " + target)` because
   their example renders a bare "Input: x\\nOutput:" prompt. `apply_chat_template`
   already ends with "<|im_start|>assistant\\n", so a leading space would train a
   spurious space token at the front of every single completion.

2. Weights are normalized by completion length.

WHY NOT river's first-party renderer (river_client.renderers)?

Its docs say "don't hand-roll the template", and for tools or images that is
right -- go through the renderer, which validates placeholder/image pairing.
For plain chat SFT it was measured and deliberately not adopted:

  * The default Qwen renderer is THINKING-ENABLED. Its sample prompt ends at
    "<think>\n" and its training example teaches the model to close the block
    itself ("\n</think>\n\n" + answer). That is a coherent design, just a
    different one from enable_thinking=False, which puts the closed block in the
    prompt so the answer starts immediately.
  * More importantly, the renderer's own sample prompt is NOT an exact token
    prefix of its own training example. They diverge at the think boundary,
    where the prompt ends with a "\n" token but training has "\n\n" merged at
    that position -- an ordinary BPE-boundary effect, but it means the model
    never sees, during training, the exact final prompt token it meets at
    serving time.

Our path keeps that property: the prompt string used for training is byte-
identical to the one used for sampling, and preflight asserts the tokenized
prompt is a strict prefix of prompt+completion. Since river samples from the
raw string we pass, holding both sides to one renderer is what removes skew --
and here that renderer is this module. All river losses are SUMMED over
   tokens, never averaged, so with flat 1.0 weights a 900-token threat model
   contributes ~15x the gradient of a 60-token classification and the carefully
   chosen task mixture silently becomes "whichever task is longest wins."
   Dividing by len(completion_ids) makes each example contribute ~TASK_WEIGHT
   units regardless of length; batching then divides by the batch's example
   count so batch loss stays ~mean(TASK_WEIGHT) and the LR is transferable
   across batches of wildly different shapes.
"""

from __future__ import annotations

from typing import Any

from .config import BASE_MODEL, TASK_WEIGHT, TASK_WEIGHT_DEFAULT

_TOK: dict[str, Any] = {}


def load(base_model: str = BASE_MODEL):
    """Load (and cache) the tokenizer for a base model.

    Always goes through river's alias map: the FP8/NVFP4 serving repos are not
    the tokenizer repos. resolve_tokenizer_name("Qwen/Qwen3.6-35B-A3B-FP8")
    returns "Qwen/Qwen3.6-35B-A3B", and loading the FP8 name directly fails.
    """
    if base_model not in _TOK:
        import river_client as river
        _TOK[base_model] = river.load_tokenizer(river.resolve_tokenizer_name(base_model))
    return _TOK[base_model]


def render_prompt(tok, system: str, turns: list[dict], *, thinking: bool) -> str:
    """Render messages up to (and including) the generation prompt.

    `turns` is an alternating list of {"role": "user"|"assistant", "content": ...}
    ending on a user turn.
    """
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("turns must end with a user message")
    msgs = ([{"role": "system", "content": system}] if system else []) + list(turns)
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def encode(tok, prompt_text: str, target_text: str, *, task: str = "") -> dict:
    """Build the per-token arrays for one cross_entropy datum.

    Alignment: position i predicts ids[i+1]. With P prompt tokens and C
    completion tokens, positions 0..P-2 (weight 0) predict prompt tokens, and
    position P-1 -- the first non-zero weight -- predicts ids[P], the first
    completion token. So the prompt is fully masked and the completion fully
    supervised, and all three arrays are length P+C.
    """
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if not prompt_ids:
        raise ValueError("empty prompt")

    ids = prompt_ids + completion_ids
    target_tokens = ids[1:] + [tok.eos_token_id]
    # The server force-zeros weights[T-1] -- there is no next-token target
    # there -- so the last slot must be zero here too, and must not be counted
    # in the divisor. Weighting it inflated every example's divisor by one
    # position that can never contribute gradient.
    # Divide by the number of SUPERVISED POSITIONS (C+1), not by C. The mask
    # spans len(prompt)-1 .. end, which is C+1 positions, so dividing by C
    # inflates a short example's total mass by (C+1)/C -- 33% for a 3-token
    # completion, 12.5% for 8 tokens, 0.6% for 169. That systematically
    # over-weights exactly the short-completion tasks (classification, CVSS,
    # ATT&CK mapping) against long ones (threat models, remediation), which is
    # the same silent mixture drift the length normalization exists to prevent.
    n_supervised = len(completion_ids)          # positions P-1 .. T-2
    w = TASK_WEIGHT.get(task, TASK_WEIGHT_DEFAULT) / n_supervised
    weights = [0.0] * (len(prompt_ids) - 1) + [w] * n_supervised + [0.0]

    assert len(ids) == len(target_tokens) == len(weights), "array length mismatch"
    return {"input_ids": ids, "target_tokens": target_tokens, "weights": weights,
            "_n_prompt": len(prompt_ids), "_n_completion": len(completion_ids)}


def supervised_tokens(datum: dict) -> int:
    return sum(1 for w in datum["weights"] if w > 0.0)


class PreflightError(RuntimeError):
    pass


def preflight(base_model: str = BASE_MODEL) -> dict:
    """Assertions that must pass before a single training example is written.

    A silent chat-template change corrupts every example identically and
    invisibly -- the loss curve looks normal and the model is simply worse. These
    are cheap; run them in CI and at the top of every dataset build.
    """
    tok = load(base_model)
    out: dict[str, Any] = {"base_model": base_model,
                           "tokenizer": type(tok).__name__,
                           "vocab_size": tok.vocab_size}

    # 1. EOS resolves, and for Qwen chat it is <|im_end|>, not <|endoftext|>.
    if tok.eos_token_id is None:
        raise PreflightError("tokenizer has no eos_token_id")
    out["eos_token"] = tok.eos_token
    out["eos_token_id"] = tok.eos_token_id

    system = "You are Sekura Defender."
    turns = [{"role": "user", "content": "Explain SSRF to IMDS in one sentence."}]

    for thinking in (False, True):
        prompt = render_prompt(tok, system, turns, thinking=thinking)
        key = "thinking_on" if thinking else "thinking_off"
        out[key] = prompt[-80:]

        # 2. Prompt round-trips through the tokenizer unchanged.
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        if tok.decode(pid) != prompt:
            raise PreflightError(f"prompt does not round-trip ({key})")

        # 3. The prompt is a strict token prefix of prompt+completion. This
        #    catches BPE merges across the prompt/completion boundary, which
        #    silently shift every weight by one position.
        d = encode(tok, prompt, "SSRF lets an attacker coerce the server into "
                                "reading cloud instance credentials from IMDS.")
        if d["input_ids"][:len(pid)] != pid:
            raise PreflightError(f"prompt is not a token prefix ({key}) -- BPE merged "
                                 "across the boundary")

        # 4. Masking is exactly right: last prompt position is the first
        #    supervised one, and every earlier position is masked.
        first_sup = next(i for i, w in enumerate(d["weights"]) if w > 0.0)
        if first_sup != d["_n_prompt"] - 1:
            raise PreflightError(
                f"first supervised index {first_sup} != n_prompt-1 "
                f"{d['_n_prompt'] - 1} ({key})")
        if any(w != 0.0 for w in d["weights"][:first_sup]):
            raise PreflightError(f"prompt not fully masked ({key})")
        out[key + "_first_supervised"] = first_sup

    # 5. thinking=False must actually change the rendered prompt, or the
    #    per-task thinking switch is a no-op and we would never notice.
    if out["thinking_off"] == out["thinking_on"]:
        raise PreflightError("enable_thinking has no effect on the rendered prompt; "
                             "the per-task thinking switch would be silently inert")
    return out
