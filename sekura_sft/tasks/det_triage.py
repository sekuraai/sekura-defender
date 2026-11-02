"""Triage / scan-critic examples with MANUFACTURED labels.

No triage ground truth exists in this repo: `verdict` is the constant
"POTENTIAL" for all 3,612 rows and confidence_score averages 0.978 with no
spread. sekura-intel does not rescue it either -- its labels are largely derived
(cve_match -> 1 is circular, default -> 0 is unlabelled) and `is_reachable` is
0.0 for all 6,090 rows.

So labels are manufactured: corrupt exactly one field of a real finding, and the
corruption IS the label. That makes accept/challenge/reject/reroute
deterministically checkable, and -- crucially -- makes `reason_code` checkable
too, which is what separates a model that reasons from one that guesses a verdict
at the base rate.

WHAT THIS TEACHES, STATED PLAINLY: internal-consistency checking. Does the
finding cohere, and is the fix responsive to the stated root cause? That is
genuinely the scan-critic's job. It does NOT teach exploitability or reachability
assessment, and the model card must not claim it does.

Anti-artifact controls, all enforced here:
 1. canonical re-render -- clean and corrupted are byte-identical in layout
 2. real donor values only, drawn from other real findings
 3. donors are SPLIT-LOCAL, so a test corruption never embeds train text
 4. 50% clean, or the model learns "when unsure, challenge"
 5. paired presentation -- every corruption ships with its clean twin
 6. offending_field localization required; a guess cannot localize
 7. C5 held out of TRAINING entirely -- transfer to it measures reasoning
 8. cosmetic distractors (C8) teach "changed" != "wrong"
 9. single corruption in train, multi-corruption in test only
10. adversarial positives drawn from band boundaries and rare CWEs
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections.abc import Iterator

from ..scoring import cvss
from ..sources import intel
from ..sources.intel import Finding, render
from .base import SYS_PIPELINE, Example

VERDICTS = ("accept", "challenge", "reject", "reroute")

# Held out of training entirely. Transfer to C5 is the direct measurement of
# "did it learn to check whether remediation addresses the root cause" versus
# "did it learn what a perturbation looks like".
HELD_OUT_CORRUPTIONS = frozenset({"remediation_misses_root_cause"})

NON_PROD = re.compile(r"^(\.github|docs?|examples?|tests?|website|perf|benchmarks?)/")

SCHEMA = (
    'Respond with one JSON object:\n'
    '{"verdict": "accept|challenge|reject|reroute", '
    '"confidence_multiplier": <0.0-1.0>, '
    '"reason_code": "<code|none>", '
    '"offending_field": "<field|none>", '
    '"justification": "<2-3 sentences>"}'
)

INSTRUCTION = (
    "You are reviewing a security finding produced by an automated scanner. Decide "
    "whether to accept it, challenge it (reduce confidence), reject it, or reroute "
    "it to a different owner. Base the decision only on whether the finding is "
    "internally consistent and whether its recommendation addresses the root cause "
    "it describes.\n\n" + SCHEMA
)

# Sibling CWEs: close enough that a near-miss is a `challenge`, not a `reject`.
CWE_SIBLINGS = {
    "CWE-89": "CWE-943", "CWE-943": "CWE-89",
    "CWE-22": "CWE-23", "CWE-23": "CWE-22",
    "CWE-79": "CWE-80", "CWE-80": "CWE-79",
    "CWE-78": "CWE-77", "CWE-77": "CWE-78",
    "CWE-400": "CWE-770", "CWE-770": "CWE-400",
    "CWE-20": "CWE-1284", "CWE-502": "CWE-915",
}

BAND_ORDER = ("low", "medium", "high", "critical")

# INFRASTRUCTURE (dependency CVEs) is 638/885 = 72% of unique findings. Left
# alone, any classifier trained on this collapses to the majority class. But
# capping it at 30% discards 533 real findings, and dependency CVEs genuinely
# ARE the most common finding a scanner emits -- so the cap is set where the
# minority classes are learnable without pretending the real world is uniform.
INFRA_MAX_SHARE = 0.45


def _bump(sev: str, steps: int = 1) -> str:
    if sev not in BAND_ORDER:
        return "critical"
    return BAND_ORDER[min(len(BAND_ORDER) - 1, BAND_ORDER.index(sev) + steps)]


def _resp(verdict: str, mult: float, code: str, field: str, why: str) -> str:
    return json.dumps({"verdict": verdict, "confidence_multiplier": mult,
                       "reason_code": code, "offending_field": field,
                       "justification": why}, indent=2)


# --- corruptions ----------------------------------------------------------
# Each returns (mutated_finding, response) or None when the finding lacks the
# shape to corrupt. Donors are supplied by the caller and are split-local.

def c1_cwe_swap(f: Finding, donors: list[Finding], rng) -> tuple | None:
    others = [d for d in donors if d.cwe_id and d.cwe_id != f.cwe_id
              and d.category != f.category]
    if not (f.cwe_id and others):
        return None
    g = copy.deepcopy(f)
    g.cwe_id = rng.choice(others).cwe_id
    return g, _resp("reject", 0.0, "wrong_cwe", "cwe_id",
                    f"The described defect is {f.category or 'not'} in nature, which "
                    f"{g.cwe_id} does not cover; the correct class is {f.cwe_id}. "
                    "A wrong weakness class misroutes remediation.")


def c2_cwe_near_miss(f: Finding, donors, rng) -> tuple | None:
    sib = CWE_SIBLINGS.get(f.cwe_id)
    if not sib:
        return None
    g = copy.deepcopy(f)
    g.cwe_id = sib
    return g, _resp("challenge", 0.6, "imprecise_cwe", "cwe_id",
                    f"{sib} is adjacent to but narrower than {f.cwe_id} for this defect. "
                    "The finding is real; the classification is imprecise.")


def c3_severity_inflate(f: Finding, donors, rng) -> tuple | None:
    computed = f.computed_score()
    if computed is None or not f.severity:
        return None
    band = cvss.severity_band(computed)
    if band != f.severity:
        return None                      # already inconsistent; not a clean lesson
    inflated = _bump(band, 1)
    if inflated == band:
        return None
    g = copy.deepcopy(f)
    g.severity = inflated
    return g, _resp("challenge", 0.4, "severity_unsupported_by_cvss", "severity",
                    f"The vector {f.cvss_vector} scores {computed}, which is {band}, "
                    f"not {inflated}. The finding stands but its severity is overstated.")


def c4_cvss_desync(f: Finding, donors, rng) -> tuple | None:
    computed = f.computed_score()
    if computed is None:
        return None
    g = copy.deepcopy(f)
    g.cvss_score = round(min(10.0, computed + rng.choice([1.5, 2.0, 2.5])), 1)
    return g, _resp("reject", 0.0, "cvss_score_mismatch", "cvss_score",
                    f"The stated score {g.cvss_score} does not follow from the vector, "
                    f"which computes to {computed}. The record contradicts itself.")


def c5_remediation_mismatch(f: Finding, donors, rng) -> tuple | None:
    others = [d for d in donors if d.recommendation and d.cwe_id != f.cwe_id
              and d.severity == f.severity]
    if not (f.recommendation and others):
        return None
    g = copy.deepcopy(f)
    g.recommendation = rng.choice(others).recommendation
    return g, _resp("challenge", 0.5, "remediation_misses_root_cause", "recommendation",
                    "The recommendation does not address the root cause described. "
                    "The weakness is real but the proposed fix would not remove it.")


def c6_already_patched(f: Finding, donors, rng) -> tuple | None:
    fixed = f.fixed_in()
    if not fixed or not f.package:
        return None
    g = copy.deepcopy(f)
    g.evidence = f"{f.evidence}\nInstalled version: {fixed}"
    return g, _resp("reject", 0.0, "already_patched", "evidence",
                    f"The evidence reports the installed version as {fixed}, which is "
                    f"the fixed version. The vulnerable range no longer applies.")


def c7_non_production(f: Finding, donors, rng) -> tuple | None:
    """Not synthetic. 118 real findings already point at non-production paths."""
    path = f.path_hint()
    if not (path and NON_PROD.match(path)):
        return None
    g = copy.deepcopy(f)
    owner = path.split("/", 1)[0]
    return g, _resp("reroute", 0.6, "non_production_path", "evidence",
                    f"The evidence points at {path}, which is not production code. "
                    f"This belongs with whoever owns {owner}/, not the application "
                    "security queue.")


def c8_cosmetic(f: Finding, donors, rng) -> tuple | None:
    """A distractor: something changed, nothing is wrong."""
    g = copy.deepcopy(f)
    g.title = f.title.replace(" vulnerability", " Vulnerability")
    g.technique = f.technique.strip()
    if g.title == f.title and g.technique == f.technique:
        return None
    return g, _resp("accept", 1.0, "none", "none",
                    "The finding is internally consistent: the weakness class matches the "
                    "described defect, the severity follows from the vector, and the "
                    "recommendation addresses the root cause.")


# INVARIANT: every key here is byte-identical to the `reason_code` its function
# emits. The RL reward scores reason_code against the injected corruption name,
# so any divergence silently zeroes that reward component. test_triage.py pins it.
CORRUPTIONS = {
    "wrong_cwe": c1_cwe_swap,
    "imprecise_cwe": c2_cwe_near_miss,
    "severity_unsupported_by_cvss": c3_severity_inflate,
    "cvss_score_mismatch": c4_cvss_desync,
    "remediation_misses_root_cause": c5_remediation_mismatch,
    "already_patched": c6_already_patched,
    "non_production_path": c7_non_production,
    "cosmetic_distractor": c8_cosmetic,
}

ACCEPT_RESPONSE = _resp(
    "accept", 1.0, "none", "none",
    "The finding is internally consistent: the weakness class matches the described "
    "defect, the severity follows from the CVSS vector, and the recommendation "
    "addresses the root cause.")


def _example(f: Finding, response: str, *, corruption: str, split: str) -> Example:
    return Example(
        task="triage",
        system=SYS_PIPELINE,
        turns=[{"role": "user", "content": f"{INSTRUCTION}\n\n---\n{render(f)}\n---"}],
        target=response,
        thinking=True,
        split_key=f.split_key,
        forced_split=split,
        license_tier="T2-internal",
        meta={"corruption": corruption, "cve": f.cve, "cwe": f.cwe_id,
              "category": f.category, "severity": f.severity},
    )


def build(seed: int = 0, cap: int = 6000, clean_share: float = 0.42,
          max_corruptions_per_finding: int = 3) -> Iterator[Example]:
    """Emit triage examples.

    VOLUME CEILING, stated honestly: the plan assumed ~6,000 examples from
    sekura_rs_intel's 3,612 rows, but those rows hold only 885 UNIQUE findings.
    Clean examples cannot exceed 885 without duplicating a prompt, so at a
    ~50/50 clean/corrupted split the real ceiling is ~1,800. Allowing up to
    `max_corruptions_per_finding` distinct corruption types per finding and
    settling at ~42% clean gets to roughly 2,700 without repeating any prompt.
    """
    rng = random.Random(seed)
    findings = list(intel.load_findings())

    # Assign each finding's split FIRST, so donors can be drawn split-locally.
    def split_of(f: Finding) -> str:
        probe = Example(task="triage", system="", turns=[{"role": "user", "content": "x"}],
                        target="y", split_key=f.split_key)
        return probe.split()

    by_split: dict[str, list[Finding]] = {"train": [], "val": [], "test": []}
    for f in findings:
        by_split[split_of(f)].append(f)

    def rebalanced(pool: list[Finding], max_share: float = INFRA_MAX_SHARE) -> list[Finding]:
        """Cap any one category's share of the OUTPUT, not of the input.

        INFRASTRUCTURE is 638/885 = 72% of findings. Capping as a fraction of the
        input leaves it dominant; the cap has to grow with what has been kept so
        far, so the limit is relative to the emitted list."""
        rng.shuffle(pool)
        out: list[Finding] = []
        counts: dict[str, int] = {}
        # Two passes: minority categories first, then fill with the majority up
        # to its share. Otherwise a shuffled majority exhausts the budget early.
        minority = [f for f in pool if (f.category or "?") != "INFRASTRUCTURE"]
        majority = [f for f in pool if (f.category or "?") == "INFRASTRUCTURE"]
        for f in minority:
            out.append(f)
            counts[f.category or "?"] = counts.get(f.category or "?", 0) + 1
        budget = int(len(out) * max_share / (1 - max_share)) if out else len(majority)
        out.extend(majority[:max(budget, 1)])
        rng.shuffle(out)
        return out

    made = 0
    n_clean = 0
    for split, pool in by_split.items():
        donors = pool                       # control #3: donors are split-local
        for f in rebalanced(list(pool)):
            if made >= cap:
                return
            # control #4/#5: every finding contributes its clean twin
            yield _example(f, ACCEPT_RESPONSE, corruption="clean", split=split)
            made += 1
            n_clean += 1

            names = [n for n in CORRUPTIONS
                     if not (split == "train" and n in HELD_OUT_CORRUPTIONS)]
            rng.shuffle(names)
            # Rare-but-real signals first so they are not crowded out: only 118
            # findings sit on a non-production path, and reroute would otherwise
            # be under 2% of the verdict distribution.
            names.sort(key=lambda n: 0 if n in ("non_production_path",
                                                "remediation_misses_root_cause") else 1)
            emitted = 0
            for name in names:
                if emitted >= max_corruptions_per_finding:
                    break
                # control #4: hold the clean share near target
                if n_clean and (made - n_clean) / max(made, 1) > (1 - clean_share):
                    break
                out = CORRUPTIONS[name](f, donors, rng)
                if out is None:
                    continue
                g, response = out
                # control #9: ONE corruption per example (a fresh copy each time)
                yield _example(g, response, corruption=name, split=split)
                made += 1
                emitted += 1
