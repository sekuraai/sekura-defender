# Sekura Defender — a defensive-security model fine-tuned on Fireworks

**Base:** `qwen3p6-35b-a3b` (35B MoE, 3B active, 262k context) · rank-32 LoRA
**Surface:** Fireworks **Training API** (Tinker-compatible), 4×B300 trainer + 1×B300 sampler
**Repo:** `sekuraai/sekura-kb` (private) — `sekura-sft/`

## One-paragraph version (for the form)

Sekura's security scanner relied on a 370k-record knowledge corpus purely as retrieval
material, paying frontier prices and latency for narrow, schema-constrained tasks. We turned
that corpus into 15,940 training examples across 13 defensive-security tasks and fine-tuned a
rank-32 LoRA on `qwen3p6-35b-a3b` using the Fireworks Training API. The tasks have
**deterministic validators** the base model never trained against — a JSON-schema attack-tree
validator with 8 invariant classes, a CVSS v3.1 scorer, a recon-coverage invariant — so both
the training signal and the eval are objective rather than judged. Base and tuned are measured
on an identical serving path (the base reference is an untrained adapter, which is
mathematically the identity), and the harness is verified offline at the oracle ceiling before
any GPU is provisioned. **Result: 0.4690 → 0.6653 overall (+0.1963) on 238 held-out rows**, with the
headline being the recon planner — 0.600 → 1.000 and zero-coverage 20% → 0% on rule sets it
never trained on, which is a real production bug driven to zero. The ship gate still fails on
three thresholds, including a *regression* in triage dangerous-error rate, and we report that
rather than bury it. Along the way we found that Fireworks' `loss_fn_inputs` silently drops a
misnamed per-token weight array, which trains a model to generate prompts while the loss curve
looks perfect.

## What it is

Sekura runs a multi-agent security scanner. A 370,231-record knowledge corpus sat behind it
purely as *retrieval* material, so every scan paid frontier prices and frontier latency for
tasks that are narrow, schema-constrained, and repetitive. This converts that corpus into a
fine-tuned model that does those tasks directly.

The bet: these tasks have **deterministic validators** the base model was never trained
against — a JSON-schema attack-tree validator with 8 invariant classes, a CVSS v3.1
arithmetic scorer, a recon-coverage invariant. That makes both the training signal and the
eval objective, not vibes.

## The data

370,231 raw records → 160,080 curated (11,403 groups) → 16,503 grounded bundles →
**15,940 training examples** across 13 tasks.

| Task | n | | Task | n |
|---|---|---|---|---|
| grounded_qa | 2,985 | | attacktree_repair | 650 |
| threat_model | 2,900 | | classify_category | 468 |
| detection | 2,666 | | cvss | 440 |
| planner | 2,391 | | safety | 359 |
| triage | 1,067 | | mapping | 205 |
| remediation | 983 | | attacktree | 171 |
| classify_cwe | 655 | | **total** | **15,940** |

Split 14,172 / 724 / 1,044 (train/val/test). **Splits key on entity** — CVE, `source_path`,
`subtree_id`, ruleset version — hashed, so regenerating the dataset cannot move an example
across the boundary. Row-level splitting would have leaked badly: `sekura_rs_intel.json` has
3,612 rows but only **885 unique findings**, the same advisory repeated across up to 60
packages.

Every example carries a `license_tier`, so a tier restriction is a one-flag rebuild rather
than a re-derivation: T3-reference 6,821 · T2-internal 6,469 · T1-sharealike 2,183 ·
T0-permissive 467.

## Three findings that changed the build

**1. `confidence` is a model fingerprint, not a quality signal.** `claude-haiku-4-5` emitted
**0.7 for 100% of its 167,830 records**; DeepSeek spreads 0.8–1.0. Filtering
`confidence >= 0.8` would have deleted 45% of the corpus by extractor identity. It is
excluded from every threshold in the pipeline.

**2. Near-duplicate dedup must compare entities before surface form.** SimHash over claim text
merged *different CVEs on different packages* that shared a sentence template, deleting real
data. Bucketing by an entity fingerprint first fixed it. Separately, Hamming ≤3 — the obvious
threshold — gives **19.4% recall**; calibrated on 9,000 real claim pairs, ≤10 gives 80.6% at a
0.08% false-positive rate.

**3. Triage labels are manufactured, and the honest version says so.** No real ground truth
exists — `verdict == "POTENTIAL"` for all 3,612 rows. Labels come from single-field corruption
(C0–C8), where *the corruption is the label*. One corruption class is **held out of training
entirely**, so transfer to it measures reasoning rather than perturbation-artifact detection.
This teaches internal-consistency checking, which is genuinely the scan-critic's job. It does
**not** teach exploitability: `is_reachable` is 0.0 across all 6,090 intel rows, and nothing
in scope grounds it. The model card says so.

## The bug worth the hackathon

`Datum.loss_fn_inputs` is an untyped `dict[str, TensorData]`, so naming the per-token array
`token_weights` instead of `weights` **did not raise** — the key was silently dropped and
`cross_entropy` scored all 1,510 positions in the batch, prompt included.

The loss came back **bit-identical** for as-built, all-zero, and doubled weight arrays. The
run would have trained cleanly, converged, and taught the model to generate prompts.

After the fix the probe is exact: `response_tokens` = 213 (our nonzero count), all-zero → 0.0,
doubled → precisely 2×. `tests/test_fw_weights.py` asserts against the SDK's *own* reader
(`_count_response_tokens`), not our output, so a future rename fails locally instead of after
a paid run.

Nothing local can catch a dropped key by inspecting your own arrays. That is the general
lesson: when an API accepts a free-form dict, the only real test is one written against the
consumer.

## Engineering notes specific to this surface

- **Billing is GPU-time, not tokens**, and the meter lies while you need it most. The shapes
  are 4×B300 for the trainer and 1×B300 for the sampler; B300 lists at $15/GPU-hour, but
  `use_reservation=True` is the provisioning default, so the effective rate was far lower.
  Worse, `monthly-spend-usd` **lags badly during a run and settles on teardown** — it read
  $166.75 mid-run and $212.94 moments after the GPUs were released. A spend guard must take
  whichever is higher, the meter or an elapsed-time floor; trusting the meter alone reports
  spend as near-zero while GPUs bill.
- **`cleanup_trainer_on_close` defaults to False**, so every probe run leaves a job billing
  after its process exits. Six concurrent leaked jobs hit a spending cap mid-training, which
  412'd the checkpoint save and discarded 31 completed steps.
- **Session checkpoints outlive their session** as long as the trainer job survives. The docs
  warn they become `NOT_FOUND` on teardown, and we designed around that; in practice a
  checkpoint saved before an aborted session sampled fine on reattach, which saved a 15-minute
  retrain. Still provision train and eval as one session — deployment provisioning is ~10
  minutes per launch and is the real cost of a restart.
- **Read sampled tokens off `seq.tokens`, not `seq.tokens_np`.** The latter is `None` unless
  the response came back in array form, so `list(...)` raises `TypeError` on every row. Our
  25-row probe caught it in two minutes, but its message blamed the sampler — an empty
  completion cannot distinguish a broken sampler from a broken decode from a mute model, and
  a guard should not assert a cause it cannot observe.
- **The base reference is an untrained rank-32 adapter.** LoRA zero-initializes B, so ΔW = 0
  and the adapter is the identity — base-model behaviour on a byte-identical serving path.
  Measuring base via serverless and tuned via a session deployment would fold train/serve skew
  into the reported capability delta. (`qwen3p6-35b-a3b` is not on serverless anyway.)
- **FireTitan does not resolve tokenizers server-side.** `tokenizer_model` is passed from
  `river.resolve_tokenizer_name(BASE_MODEL)`, so both sides use one name and cannot drift.
- **Losses are token-summed, not averaged**, so weights are normalized per example. Without
  that, a 900-token threat model outweighs a 60-token classification ~15:1 and the task
  mixture silently becomes "longest task wins."

## Training

94 steps, 2 epochs, 3,660 balanced examples (≤2,048 tokens, ≤300/task, safety kept whole),
15.3 minutes on 4×B300. Peak LR 1e-4 with linear warmup and cosine decay, `train_unembed=False`.

Raw per-step loss is **not** evidence of anything here: batches are emitted bucket by bucket,
so consecutive steps differ in task mix, sequence length and batch size, and the series
wandered 0.37–1.52 with no visible trend. What *is* comparable is the same position in a later
epoch — identical examples, identical bucket, identical batch size, with training the only
variable:

| position | n | epoch 1 | epoch 2 | change |
|---|---|---|---|---|
| 1 | 1,401 | 1.2458 | 0.4705 | **−62.2%** |
| 2 | 263 | 1.0783 | 0.5139 | **−52.3%** |
| 3 | 89 | 0.9431 | 0.3739 | **−60.4%** |
| 4 | 43 | 1.3296 | 0.9307 | −30.0% |
| 5 | 40 | 1.5205 | 1.0142 | −33.3% |

Improved on **5/5 matched batches, mean −47.6%**. Epoch 2 is a second pass over seen data, so
this evidences a working gradient and a correct supervised mask — not held-out capability. The
eval below is what settles that.

## Eval

A 13-task suite with deterministic scorers, gated on absolute thresholds and regression
against a stored baseline. Anchored at both ends: an **oracle ceiling** (targets replayed)
scoring 1.0000 across 12 tasks, and a **degenerate floor** (empty output) at 0.0015. Failure
*rates* are always reported, even at zero — a metric that vanishes exactly when the model is
perfect once made the gate fail the oracle against itself.

Base is the **untrained rank-32 adapter** on the identical serving path, so the delta is
capability rather than train/serve skew.

Base-vs-tuned, 238 rows balanced at 20/task (0/238 empty completions, so the serving path is
sound):

| task | n | base score | base exact | tuned score | tuned exact | delta |
|---|---|---|---|---|---|---|
| `attacktree` | 20 | 0.255 | 0.150 | 0.594 | 0.150 | **+0.339** |
| `attacktree_repair` | 20 | 0.083 | 0.000 | 0.817 | 0.450 | **+0.733** |
| `classify_category` | 20 | 0.450 | 0.450 | 0.350 | 0.350 | **-0.100** |
| `classify_cwe` | 20 | 0.600 | 0.600 | 0.600 | 0.600 | **+0.000** |
| `cvss` | 20 | 0.350 | 0.400 | 0.400 | 0.400 | **+0.050** |
| `detection` | 20 | 0.839 | 0.150 | 0.869 | 0.150 | **+0.030** |
| `grounded_qa` | 20 | 0.809 | 0.000 | 0.841 | 0.000 | **+0.032** |
| `mapping` | 13 | 0.208 | 0.154 | 0.551 | 0.462 | **+0.343** |
| `planner` | 20 | 0.600 | 0.600 | 1.000 | 1.000 | **+0.400** |
| `remediation` | 20 | 0.303 | 0.150 | 0.407 | 0.200 | **+0.104** |
| `safety` | 5 | 0.806 | 0.000 | 0.808 | 0.000 | **+0.002** |
| `threat_model` | 20 | 0.827 | 0.000 | 0.887 | 0.000 | **+0.061** |
| `triage` | 20 | 0.129 | 0.050 | 0.591 | 0.300 | **+0.463** |
| **overall** | **238** | **0.4690** | | **0.6653** | | **+0.1963** |

**Headline: `planner` 0.600 -> 1.000, zero-coverage 20% -> 0%.** That is the real production
bug — the planner called a site static, emitted an empty plan, and the scan returned nothing
— driven to zero on **held-out coverage rule sets it never trained on**, which is the
difference between a planner and a lookup table. `attacktree_repair` 0.083 -> 0.817 and
`triage` 0.129 -> 0.591 are the other large gains, and all three are tasks with deterministic
validators.

**Where it did not help, and one place it hurt.** `classify_category` **regressed -0.100**.
Every task was capped at 300 examples and `classify_category` had 408 available, so it lost
data *and* competed for one rank-32 subspace against twelve other tasks. That is the clearest
evidence in this run that rank 32 is a contested, shared resource — and it is the argument
for the two specialised checkpoints (analyst / pipeline) the plan already calls for rather
than one monolith. `classify_cwe` moved exactly 0.000.

**What the base numbers say.** The base model fails hardest exactly where deterministic
validators exist: `attacktree_repair` exact **0.000** against a 0.60 ship gate, `cvss` exact
0.400 against 0.90, `triage` 0.129, and `planner` zero-coverage **20%** — that last one being
the real production bug (a plan that covers nothing means the scan returns nothing),
reproduced on held-out rule sets. (All figures here are the session-sampler base, matching
the table; the HTTP base put planner zero-coverage at 35% and triage at 0.172.)

Meanwhile `detection` 0.839 and `threat_model` 0.827 at exact **0.000** show the base model
writes plausible security prose but essentially never emits the required structure. So the
honest framing is not "the base model is bad at security" — it is "the base model does not
produce the schemas our pipeline parses, and those schemas are machine-checkable." That is
precisely the gap a rank-32 adapter should close, and it is why the training signal here is
objective rather than judged.

### The ship gate: FAIL — and that is the correct verdict

```
GATE: FAIL
  triage.dangerous_rate     = 0.05  violates <= 0.03
  attacktree_repair.exact   = 0.45  violates >= 0.60
  cvss.exact                = 0.40  violates >= 0.90
```

These are **shippability** thresholds — "may this checkpoint go in front of paying
customers" — not "did training work". Training worked; this checkpoint is not shippable yet.
Both statements are true and the gate exists to keep them apart.

**The failure that matters is `triage.dangerous_rate`, and it is a regression.** Base scored
**0.000** on the same serving path; tuned scores **0.05**. A dangerous error is accepting a
finding that should have been rejected — shipping a bad finding to a customer — and the
asymmetric scoring matrix exists precisely because that error is not symmetric with
over-rejection. So the tuned model is better at triage overall (0.129 → 0.591, reason-exact
0.05 → 0.25) while being *worse* on the one triage metric with real-world cost. On n=20 that
rate is a single example, which is far too thin to size the effect but not a reason to
discount it: the honest reading is "unmeasured, and the measurement we have points the wrong
way." A 250-row held-out triage set is the next thing to build, not an RL run.

`cvss` barely moved (0.350 → 0.400). CVSS base score is pure arithmetic over a vector
string, and supervised fine-tuning on 300 examples does not teach arithmetic. This is exactly
the task earmarked for RL against the deterministic scorer, where the reward is the
arithmetic itself.

**Two honest limitations in the eval, stated rather than smoothed over:**

- **Safety is measured on n=5.** Splits key on entity, and the safety slice shares few
  entities, so 342 of its 359 examples landed in train (val 12, test 5). The design called
  for a 120-item set at 60/60 should-answer / should-decline, which would measure refusal
  rate *and* over-refusal rate — over-refusal being the more likely and more damaging
  failure for a security tool. n=5 cannot support either claim. `mapping` is similarly thin
  at n=13.
- **Triage measures internal-consistency checking, not exploitability.** Labels are
  manufactured by single-field corruption because no real ground truth exists. One
  corruption class is held out entirely so transfer to it separates reasoning from
  artifact-detection, but `is_reachable` is 0.0 across all 6,090 intel rows and nothing in
  scope grounds reachability.

The fix for both is the same and is already specified: log live `scan-critic` decisions and
human overrides from production. That is the missing dataset.

### Serving-path cross-check

Base was measured twice, on two different serving paths: **0.4690** through the session sampler that also served the tuned checkpoint (the comparator used above), and **0.4666** through the independent HTTP inference endpoint. They agree to **0.0024**, so the reported gain is capability rather than train/serve skew -- the single assumption that could have invalidated the whole comparison.

Overall agreement does hide per-task variance, though: `planner` scored 0.600 via the session sampler and 0.450 over HTTP, `triage` 0.129 vs 0.172. Greedy decoding should be deterministic, so that spread is a property of the two serving paths and a reason to quote one path consistently rather than mix them.

## Reproducing

```bash
# 1. Everything that does not need a GPU: tokenize, render eval prompts, pick a
#    balanced length-capped subset. Prints the projected paid cost.
python scripts/prep_fireworks.py          # -> 3,660 examples, 94 steps, 238 eval rows

# 2. Verify the entire harness offline. The mock sampler replays each row's own
#    target, so this MUST land at the oracle ceiling; if it does not, the bug is
#    in the harness, not the model.
python scripts/run_fireworks_campaign.py --mock --epochs 2    # -> 0.9946, GATE: PASS

# 3. Train and evaluate in one session.
python scripts/run_fireworks_campaign.py --epochs 2 --token-budget 40955 \
       --budget-usd 40 --absolute-usd 195
```

`--token-budget` is not optional in spirit: omit it and `train()` falls back to 120 steps per
epoch, which silently turned a 94-step run into 252 steps and tripled the bill.

The numbers in this document came from `scripts/eval_or_retrain.py`, which reattaches to an
existing trainer job and then branches — sample the existing checkpoint if it survived, else
retrain — because deployment provisioning costs ~10 minutes whichever path you need.

A live ledger (`data/fireworks/ledger.jsonl`) records spend per step, and three tripwires
abort the run: a cost ceiling taking the higher of meter and elapsed-time floor; a loss that
shows no epoch-over-epoch improvement on **matched batch positions** (a trailing-vs-opening
mean does not work here — batches are emitted bucket by bucket, so per-example loss wandered
0.37–1.52 with no trend while the model trained well, and that test would have killed a
healthy run); and a 25-row sampler probe that fires before the remaining rows are paid for.

Doing the free work first took the projected cost from **$82 to $36**. Actual month-to-date
spend reached **$212.94**, but the large majority of that was the six leaked trainer jobs and
three restarts, not the training itself: the successful run trained 94 steps in 15.3 minutes
and evaluated 2×238 rows in 11.2 minutes.

226 tests pass, including `tests/test_fw_weights.py`, which pins the weight-key contract
against the SDK's own reader.
