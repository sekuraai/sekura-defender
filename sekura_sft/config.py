"""Central configuration: paths, license tiers, task weights, length budgets, seeds.

Stdlib only. Everything downstream imports its constants from here so a run is
reproducible from one frozen record (see manifest.dataset_lock).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# REPO is sekura-kb; SIBLINGS holds sekura-intel and friends.
PKG_ROOT = Path(__file__).resolve().parent
SFT_ROOT = PKG_ROOT.parent
REPO = SFT_ROOT.parent
SIBLINGS = REPO.parent

KNOWLEDGE_JSONL = REPO / "SocraticAI" / "knowledge.jsonl"
GRAPH_EDGES = REPO / "SocraticAI" / "graph.edges"
KB_ROOT = REPO / "kb" / "sekura-kb"
KB_MANIFEST = KB_ROOT / "manifest.json"
ATTACK_TREES = REPO / "sekura-attack-trees"
ATTACK_CATALOG = ATTACK_TREES / "catalog"
ATTACK_INDEX = ATTACK_CATALOG / "index.json"
ATTACK_SCHEMA = ATTACK_TREES / "schema" / "attack-subtree.schema.json"
ATTACK_TOOLS = ATTACK_TREES / "tools"

INTEL_JSON = REPO / "sekura_rs_intel.json"
CWE_TO_ATTACK = REPO / "cwe_to_attack.json"
RECON_CONTRACT = REPO / "structured-recon-planner-contract.md"

# sekura-intel: breadth + feature grounding. NOT triage ground truth (see plan §2).
INTEL_REPO = SIBLINGS / "sekura-intel"
INTEL_TRAINING = INTEL_REPO / "data" / "training" / "training_data.json"
INTEL_VULN_MAPS = INTEL_REPO / "data" / "vuln_maps"
INTEL_SAST_RESULTS = INTEL_REPO / "data" / "sast_results"

DATA = SFT_ROOT / "data"
DOCSET_TSV = DATA / "docset" / "allow.tsv"
POOL_DIR = DATA / "pool"
BUNDLES_DIR = DATA / "bundles"
EXAMPLES_DIR = DATA / "examples"
DATASET_DIR = DATA / "dataset"
MANIFESTS_DIR = DATA / "manifests"
EVAL_DIR = DATA / "eval"

# --- License tiers ---------------------------------------------------------
# Ordered most-permissive first. `--max-tier` keeps everything at or below the
# named tier, so a share-alike-only rebuild is one flag (plan: Licensing).
TIER_ORDER = ("T0-permissive", "T1-sharealike", "T2-internal", "T3-reference", "DROP")

LICENSE_TO_TIER = {
    "MIT": "T0-permissive",
    "Public Domain (US Govt)": "T0-permissive",
    "MITRE Terms of Use": "T0-permissive",
    "Apache-2.0": "T0-permissive",
    "CC-BY-SA-4.0": "T1-sharealike",
    "internal": "T2-internal",
    "Internal": "T2-internal",
    "Mixed / internal reference": "T3-reference",
    "": "DROP",
}


def tier_for(license_str: str) -> str:
    """Map a record's `license` field to a tier. Unknown licenses are treated as
    the most restrictive real tier rather than silently dropped, so a new source
    shows up in the report instead of vanishing."""
    return LICENSE_TO_TIER.get((license_str or "").strip(), "T3-reference")


def tier_allowed(tier: str, max_tier: str) -> bool:
    if tier == "DROP":
        return False
    return TIER_ORDER.index(tier) <= TIER_ORDER.index(max_tier)


# --- Curation thresholds (plan §2.1) ---------------------------------------
# Measured knees, not guesses. See the funnel table in the plan.
DOC_KEEP_TAG_FRAC = 0.25       # doc-level security purity, tag-based
DOC_KEEP_TEXT_FRAC = 0.30      # doc-level security purity, text-based
GROUP_MIN_SIZE = 4             # (source_path, heading) group floor
GROUP_MIN_SEC_FRAC = 0.40      # measured knee: 11,669 groups / 215,103 records
MIN_CLAIM_CHARS = 60           # below this a claim carries no usable information
SIMHASH_HAMMING = 10           # near-dup threshold; calibrated, see corpus/nearsim.py

# `confidence` is deliberately absent. It is an extractor fingerprint, not a
# quality signal: claude-haiku-4-5 emitted 0.7 for 100% of its 167,830 records.
# Thresholding on it would delete 45% of the corpus by model identity.

# --- Bundling (plan §2.2) --------------------------------------------------
BUNDLE_WINDOW = 10
BUNDLE_OVERLAP = 2
BUNDLE_SHARD_ABOVE = 12

# --- Token budgets (plan §3) -----------------------------------------------
MAX_SEQ = 4096
MAX_PROMPT = 3072
MAX_COMPLETION = 1024

# Per-task completion caps. A plan covering 60 endpoints is legitimately ~3.5k
# tokens -- that IS the output the pipeline consumes, so capping it at the global
# default would silently drop the hardest and most valuable planner examples
# rather than teaching them.
MAX_COMPLETION_BY_TASK = {
    "planner": 4096,
    # Teacher-written analysis is genuinely long-form. A six-section threat model
    # or a Sigma rule plus data sources, false-positive conditions and a tuning
    # step IS ~1.5-2k tokens; squeezing it to fit the default cap produced terse,
    # less useful targets and rejected 13 of 16 samples.
    "threat_model": 2048,
    "detection": 2048,
    "grounded_qa": 1536,
    "remediation": 1536,
    "safety": 1536,
}


def max_completion_for(task: str) -> int:
    return MAX_COMPLETION_BY_TASK.get(task, MAX_COMPLETION)
MAX_SEQ_OVERFLOW = 8192
OVERFLOW_SHARE_CAP = 0.05
LENGTH_BUCKETS = (512, 1024, 2048, 4096, 8192)

# Mixture control. With per-example 1/len(completion) normalization, each example
# contributes ~= TASK_WEIGHT[task] units of loss regardless of its length.
TASK_WEIGHT_DEFAULT = 1.0
TASK_WEIGHT = {
    "triage": 1.5,
    "detection": 1.5,
    "planner": 1.5,
    "attacktree": 1.2,
    "remediation": 1.2,
    "threat_model": 1.2,
    "classify_cwe": 0.5,
    "classify_category": 0.5,
    "relations": 0.5,
}

# --- Model ------------------------------------------------------------------
BASE_MODEL = os.environ.get("SEKURA_BASE_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
BASE_MODEL_FALLBACKS = (
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",   # same $1.00/1M train
    "Qwen/Qwen3.5-9B",                                      # dense: MoE escape hatch
    "Qwen/Qwen3.8-27B-FP8",                                 # 4.1x train cost
)

# river.LoraConfig accepts any rank client-side -- verified against
# river-client 0.9.1, where LoraConfig(rank=33) constructs without complaint.
# The cap is enforced server-side, so an invalid rank would only surface after a
# session and a model have already been created. Assert locally instead.
MAX_LORA_RANK = 32
LORA_RANK = 32
LORA_TRAIN_ATTN = True
LORA_TRAIN_MLP = True
LORA_TRAIN_UNEMBED_SFT = False   # ~150k vocab would dwarf a rank-32 adapter
LORA_TRAIN_UNEMBED_RL = True     # RL reward turns on a few format-critical tokens

# Optimizer. river's guide prose says weight_decay defaults to 0.01 but the
# `optim_step` signature says 0.0 -- confirmed against the installed package.
# Always pass it explicitly and log what was passed.
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
ADAM_BETA2_RL = 0.95
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
LR_PEAK, LR_FLOOR_FRAC, LR_WARMUP_FRAC = 1e-4, 0.1, 0.03
LR_PEAK_RL = 4e-5
# Optimizer steps per epoch, NOT a fixed token budget.
#
# river's docs advise large batches ("millions of tokens per step"), which is
# advice about THROUGHPUT. It is the wrong objective for a small dataset. The
# plan's 262,144 supervised tokens/step assumed ~1k supervised tokens per
# example; measured, this corpus averages 198 (targets are mostly short
# structured output), so that budget swallows a quarter of the training set in
# one step and yields SIX steps per epoch -- twelve optimizer updates for a
# two-epoch run, which would learn essentially nothing.
#
# Expressing the target as steps per epoch keeps the update count sane as the
# corpus grows (the teacher slices will roughly triple it). Per-batch weight
# normalization keeps batch loss ~mean(TASK_WEIGHT) regardless of batch size, so
# one LR schedule stays valid across budgets.
TARGET_STEPS_PER_EPOCH = 120
MIN_SUPERVISED_TOKENS_PER_STEP = 4_096
MAX_SUPERVISED_TOKENS_PER_STEP = 262_144


def supervised_budget(total_supervised_tokens: int,
                      target_steps: int = TARGET_STEPS_PER_EPOCH) -> int:
    """Per-step supervised-token budget that yields ~target_steps per epoch."""
    if total_supervised_tokens <= 0:
        return MIN_SUPERVISED_TOKENS_PER_STEP
    raw = total_supervised_tokens // max(1, target_steps)
    return max(MIN_SUPERVISED_TOKENS_PER_STEP,
               min(MAX_SUPERVISED_TOKENS_PER_STEP, raw))


def check_lora_rank(rank: int) -> int:
    if not 1 <= rank <= MAX_LORA_RANK:
        raise ValueError(
            f"LoRA rank {rank} outside river's supported range 1..{MAX_LORA_RANK}. "
            "The client accepts it but the server rejects it, after you have "
            "already paid for session and model setup.")
    return rank

# --- Splits -----------------------------------------------------------------
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.90, 0.05, 0.05
HELDOUT_MOD = 100
HELDOUT_CUT = 5                # sha256(datum_id) % 100 < 5 -> held out, frozen
SEED = 20260909
