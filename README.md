# Sekura Defender

A rank-32 LoRA on **`qwen3p6-35b-a3b`**, trained with the **Fireworks Training API**, that
turns a 370k-record security knowledge corpus into a model which emits the *machine-checkable
schemas* a security scanner actually parses.

**Result: 0.4690 → 0.6653 (+0.1963)** on 238 held-out examples across 13 tasks.
Headline: the recon planner goes **0.600 → 1.000** with **zero-coverage 20% → 0%** on
coverage rule sets it never trained on — a real production bug driven to zero.

Full write-up, methodology and limitations: **[SUBMISSION.md](SUBMISSION.md)**
Submission video (106 s, built in code with anime.js): **[`video/sekura-defender-submission.mp4`](video/sekura-defender-submission.mp4)** · [source](video/)

---

## The task and the user

Sekura runs a multi-agent security scanner. Its knowledge corpus was used purely as
*retrieval* material, so every scan paid frontier prices and latency for work that is narrow,
schema-constrained and repetitive: authoring attack trees, writing detections, planning
recon coverage, triaging findings, mapping CWE→ATT&CK, computing CVSS.

The intended user is the scanner itself (an in-pipeline model behind `LLM_API_BASE`) and the
security analyst reading its output.

**Why a fine-tune and not a prompt:** these tasks have **deterministic validators** the base
model was never trained against — a JSON-schema attack-tree validator with 8 invariant
classes, a CVSS v3.1 arithmetic scorer, a recon-coverage invariant. That makes the training
signal and the benchmark objective rather than judged.

## Before and after

| | |
|---|---|
| **Baseline model** | `accounts/fireworks/models/qwen3p6-35b-a3b`, untuned |
| **Tuned model** | same base + rank-32 LoRA, 94 steps / 2 epochs / 3,660 examples |
| **Eval dataset** | 238 held-out examples, 20 per task across 13 tasks |
| **Metric** | per-task deterministic scorers (schema validity, exact match, coverage recall, CVSS exactness, citation grounding), weighted mean |
| **Method** | identical prompts, greedy decoding (`temperature=0`), identical stop token, **both models sampled through the same deployment** |

| task | base | tuned | delta |
|---|---|---|---|
| `attacktree_repair` | 0.083 | 0.817 | **+0.733** |
| `triage` | 0.129 | 0.591 | **+0.463** |
| `planner` | 0.600 | 1.000 | **+0.400** |
| `mapping` | 0.208 | 0.551 | +0.343 |
| `attacktree` | 0.255 | 0.594 | +0.339 |
| `remediation` | 0.303 | 0.407 | +0.104 |
| `threat_model` | 0.827 | 0.887 | +0.061 |
| `cvss` | 0.350 | 0.400 | +0.050 |
| `grounded_qa` | 0.809 | 0.841 | +0.032 |
| `detection` | 0.839 | 0.869 | +0.030 |
| `safety` (n=5) | 0.806 | 0.808 | +0.002 |
| `classify_cwe` | 0.600 | 0.600 | 0.000 |
| `classify_category` | 0.450 | 0.350 | **−0.100** |
| **overall** | **0.4690** | **0.6653** | **+0.1963** |

Raw reports: [`results/`](results/) — `base_session.json`, `tuned.json`, `comparison.json`,
plus `base_http.json` (base measured a second time on an independent serving path; the two
agree to 0.0024, so the gain is capability and not train/serve skew).

### What we are not claiming

- **The ship gate FAILS** on three thresholds, and one is a *regression*:
  `triage.dangerous_rate` goes 0.000 → 0.05 (accepting a finding that should be rejected).
  At n=20 that is one example — too thin to size, not a reason to discount.
- **`classify_category` regressed −0.100.** It had 408 examples and was capped at 300, so it
  lost data *and* competed for one rank-32 subspace against twelve other tasks.
- **`cvss` barely moved.** CVSS is arithmetic over a vector string; SFT on 300 examples does
  not teach arithmetic. That task was always earmarked for RL against the scorer.
- **Safety is measured on n=5**, which cannot support a refusal *or* over-refusal claim.
- **Triage measures internal-consistency checking, not exploitability** — `is_reachable` is
  empty across all 6,090 intel rows.

## Setup

Requires **Python ≥3.12** and a Fireworks API key with Training API access.

```bash
git clone <this repo> && cd sekura-defender
python3.12 -m venv .venv && ./.venv/bin/pip install -U pip
./.venv/bin/pip install fireworks-ai river-client transformers pytest
export FIREWORKS_API_KEY=...        # or put it in ~/.zshrc; the code reads an interactive zsh
```

### Run the test suite (no API key, no GPU)

```bash
./.venv/bin/python -m pytest -q          # 226 tests
```

`tests/test_fw_weights.py` is the one worth reading: it pins the per-token weight contract
against the Fireworks SDK's *own* reader, because a misnamed key in `loss_fn_inputs` is
silently dropped (see below).

### Verify the harness offline, before spending anything

```bash
./.venv/bin/python scripts/run_fireworks_campaign.py --mock --epochs 2
```

Runs the entire train→eval→gate pipeline against doubles whose sampler replays each row's own
target, so it **must** land at the oracle ceiling (observed: 0.9946, `GATE: PASS`). If it does
not, the bug is in the harness, not the model. Deployment provisioning costs ~10 GPU-minutes
per launch, so a code bug found on GPU costs more than the training it interrupts.

### Reproduce the evaluation

```bash
./.venv/bin/python scripts/prep_fireworks.py        # CPU only: tokenize, render, subset
./.venv/bin/python scripts/run_fireworks_campaign.py \
      --epochs 2 --token-budget 40955 --budget-usd 40 --absolute-usd 195
```

`--token-budget` matters: omit it and `train()` falls back to 120 steps/epoch, which silently
turned a 94-step run into 252 steps and tripled the bill.

`scripts/eval_or_retrain.py` reattaches to an existing trainer job and evaluates a checkpoint
(retraining only if the checkpoint is gone) — this produced the numbers above.

> **Training data is not included.** The corpus is derived in part from licensed and
> internal-reference material, so this repository ships the pipeline, the evaluation harness,
> the aggregate manifests (`data/manifests/`, statistics only) and the raw result reports —
> everything needed to read and re-run the method — but not the corpus or its curation list.

## Layout

| path | what |
|---|---|
| `sekura_sft/` | corpus curation funnel, grounded bundler, task generators, tokenization |
| `sekura_fireworks/` | Training API loop, sampling, offline doubles, spend guards |
| `sekura_eval/` | deterministic scorers, 13-task suite, regression gate |
| `sekura_rl/` | RL environments and reward functions (planner / triage / attack-tree) |
| `sekura_defender/` | river.ai harness (the surface originally targeted) |
| `scripts/` | pipeline entry points |
| `results/` | the eval reports behind the table above |
| `video/` | the 106-second submission video (committed MP4) and the anime.js source that renders it |

## Three things this build learned the hard way

**1. `loss_fn_inputs` is an untyped dict, so a misnamed weight array is silently dropped.**
Naming it `token_weights` instead of `weights` did not raise — `cross_entropy` scored all
1,510 positions in the batch, prompt included. The loss came back **bit-identical** for
as-built, all-zero and doubled weight arrays, so the run would have trained cleanly,
converged, and taught the model to generate prompts. Nothing local catches this by inspecting
your own arrays; the only real test is written against the consumer.

**2. Per-step loss is not evidence when batches are length-bucketed.** Consecutive steps
differ in task mix, sequence length and batch size, so the series wandered 0.37–1.52 with no
trend *while the model trained well*. What is comparable is the same position in a later
epoch — identical examples, identical bucket: **−47.6% on 5/5 matched batches**. A
trailing-mean-vs-opening-mean tripwire would have aborted a healthy run.

**3. The spend meter lags during a run and settles on teardown.** `monthly-spend-usd` read
$166.75 mid-run and $212.94 moments after the GPUs were released. A cost guard must take
whichever is higher, the meter or an elapsed-time floor. Relatedly,
`cleanup_trainer_on_close` defaults to **False**, so each probe run leaves a trainer job
billing after its process exits.

## Technologies

Fireworks AI Training API (Tinker-compatible, 4×B300 trainer + 1×B300 sampler) ·
`qwen3p6-35b-a3b` 35B MoE / 3B active / 262k ctx · LoRA rank 32 · Python 3.12 ·
`transformers` tokenizer · Claude (via CLI subscription) as teacher and adversarial critic
for ~9,600 synthesized examples at zero API cost.
