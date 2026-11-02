"""ENV-B adapter: wires the planner reward to the RL loop.

Prompts come from the same synthetic generator the SFT slice uses, so the RL
distribution matches what the model was trained on -- including the adversarial
`likely_static` trap at ~1/3 of inventories and the ~10% of targets whose correct
plan is empty.

The held-out rule set is used ONLY at eval. Training on it would destroy the
anti-memorization property: the whole point is that the model applies rules it is
handed rather than recalling a fixed surface->task mapping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sekura_sft.sources import recon as R
from sekura_sft.tasks.base import SYS_PIPELINE
from sekura_sft.tasks.det_recon import PLAN_INSTRUCTION, _inventory, ARCHETYPES
from sekura_sft.tokenization import load as load_tok
from sekura_sft.tokenization import render_prompt

from . import planner as reward


@dataclass
class PlannerEnv:
    seed: int = 0
    adversarial_rate: float = 0.35
    empty_rate: float = 0.08
    base_model: str | None = None
    _tok: object = field(default=None, init=False)
    _rng: random.Random = field(default=None, init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    @property
    def tok(self):
        if self._tok is None:
            self._tok = load_tok(self.base_model) if self.base_model else load_tok()
        return self._tok

    def prompts(self, step: int, n: int) -> tuple[list[str], list[dict]]:
        rng = random.Random(self.seed * 1_000_003 + step)
        texts, contexts = [], []
        for i in range(n):
            ruleset = rng.choice(R.TRAIN_RULESETS)      # never the held-out set
            adversarial = rng.random() < self.adversarial_rate
            if rng.random() < self.empty_rate:
                inv = R.TargetInventory(
                    base_url=f"https://static-{step}-{i}.example",
                    pages=[f"/p{j}" for j in range(rng.randint(2, 8))],
                    architecture_hints={"likely_static": True})
                adversarial = False
            else:
                inv = _inventory(rng, ARCHETYPES[i % len(ARCHETYPES)], adversarial)
            user = (f"{PLAN_INSTRUCTION}\n\n{R.render_rules(ruleset)}\n\n"
                    f"Inventory:\n{inv.render()}")
            texts.append(render_prompt(self.tok, SYS_PIPELINE,
                                       [{"role": "user", "content": user}],
                                       thinking=True))
            contexts.append({"inventory": inv, "ruleset": ruleset,
                             "adversarial": adversarial,
                             "required": len(R.required_pairs(inv, ruleset))})
        return texts, contexts

    def prompt_tokens(self, prompts: list[str]) -> list[list[int]]:
        return [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

    def reward(self, text: str, ctx: dict, stop_reason: str = "stop") -> float:
        s = reward.score(text, ctx["inventory"], ctx["ruleset"])
        # Truncation is scored down rather than ignored: a cut-off plan is
        # unusable by the pipeline even when the visible part looks right.
        return s.reward * (1.0 if stop_reason == "stop" else 0.8)
