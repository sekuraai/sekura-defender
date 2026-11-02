"""Recon -> coverage planner examples. Oracle-generated, no teacher.

Four task variants, all scored against `deterministic_fallback_plan`, which is
the rule cross-product and therefore correct by construction.

The generator deliberately over-represents the failure the contract was written
to kill: `architecture_hints.likely_static = true` WHILE forms and endpoints
exist. In production the planner believed the hint, emitted an empty plan, and
the scan returned nothing. Roughly a third of generated inventories carry that
trap.

Anti-shortcut: the coverage rule set is passed IN THE PROMPT and varies between
examples, and the test split uses a rule set that never appears in training. A
model that memorized a fixed surface->task mapping scores zero there; only one
that actually applies the rules it is given transfers.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator

from ..sources import recon as R
from ..sources.recon import TargetInventory
from .base import SYS_PIPELINE, Example

PLAN_INSTRUCTION = (
    "You are planning a penetration test. Below is a typed inventory of surfaces "
    "observed during reconnaissance, and the coverage rules in force.\n\n"
    "Produce a plan covering EVERY surface as the rules require. Architecture "
    "hints are advisory: they may inform priority, never coverage. An empty plan "
    "against a non-empty inventory is always an error.\n\n"
    'Answer as a JSON array of {"surface_ref": "<kind>:<id>", "kind": "<TaskKind>"}.'
)

ARCHETYPES = (
    "static-brochure", "spa-json-api", "graphql-bff", "server-rendered-forms",
    "file-sharing", "oauth-saas", "websocket-app", "openapi-microservice",
)


def _inventory(rng: random.Random, archetype: str, adversarial: bool) -> TargetInventory:
    def ids(prefix, n):
        return [{"id": f"{prefix}{i}", "url": f"/{prefix}/{i}"} for i in range(n)]

    inv = TargetInventory(pages=[f"/p{i}" for i in range(rng.randint(0, 14))])
    if archetype == "static-brochure":
        inv.forms = ids("form", rng.randint(0, 1))
        inv.endpoints = ids("ep", rng.randint(0, 2))
    elif archetype == "spa-json-api":
        inv.endpoints = ids("ep", rng.randint(8, 40))
        inv.js_bundles = [f"/static/{i}.js" for i in range(rng.randint(1, 4))]
        inv.auth_surfaces = ids("auth", rng.randint(0, 2))
    elif archetype == "graphql-bff":
        inv.graphql_endpoints = ids("gql", rng.randint(1, 2))
        inv.endpoints = ids("ep", rng.randint(2, 10))
    elif archetype == "server-rendered-forms":
        inv.forms = ids("form", rng.randint(2, 8))
        inv.endpoints = ids("ep", rng.randint(1, 8))
        inv.auth_surfaces = ids("auth", 1)
    elif archetype == "file-sharing":
        inv.upload_surfaces = ids("up", rng.randint(1, 3))
        inv.forms = ids("form", rng.randint(1, 3))
        inv.endpoints = ids("ep", rng.randint(2, 12))
    elif archetype == "oauth-saas":
        inv.auth_surfaces = ids("auth", rng.randint(1, 3))
        inv.endpoints = ids("ep", rng.randint(5, 20))
        inv.cookies = [{"id": f"c{i}", "name": f"sess{i}"} for i in range(rng.randint(1, 3))]
    elif archetype == "websocket-app":
        inv.websocket_endpoints = ids("ws", rng.randint(1, 3))
        inv.endpoints = ids("ep", rng.randint(1, 8))
    else:  # openapi-microservice
        inv.openapi_specs = ids("spec", 1)
        inv.endpoints = ids("ep", rng.randint(10, 60))

    inv.architecture_hints = {
        "likely_static": False, "likely_spa": archetype == "spa-json-api",
        "framework_fingerprints": rng.sample(
            ["django", "rails", "express", "spring", "laravel", "next.js"], k=rng.randint(0, 2)),
    }
    if adversarial:
        # THE production failure, reproduced as training data: the hint says
        # static while forms and endpoints demonstrably exist.
        inv.architecture_hints["likely_static"] = True
        if not inv.forms:
            inv.forms = ids("form", rng.randint(1, 3))
        if not inv.endpoints:
            inv.endpoints = ids("ep", rng.randint(1, 5))
    return inv


def _canonical(plan: list[dict]) -> str:
    return json.dumps(sorted(plan, key=lambda t: (t["surface_ref"], t["kind"])), indent=2)


def build(seed: int = 0, n: int = 2400) -> Iterator[Example]:
    rng = random.Random(seed)
    seen: set[str] = set()
    for ex in _build(rng, n):
        # Empty inventories render identically, so they collapse to one prompt.
        did = ex.datum_id()
        if did in seen:
            continue
        seen.add(did)
        yield ex


def _build(rng: random.Random, n: int) -> Iterator[Example]:
    for i in range(n):
        # Test uses ONLY the held-out rule set; training never sees it.
        split = "test" if i % 20 == 0 else ("val" if i % 20 == 1 else "train")
        ruleset = (R.HELD_OUT_RULESET if split == "test"
                   else rng.choice(R.TRAIN_RULESETS))
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        adversarial = rng.random() < 0.35
        empty_case = rng.random() < 0.08

        if empty_case:
            # Genuinely varied "nothing to test" targets, not one repeated stub.
            # A real static brochure site HAS pages and bundles; it simply has no
            # surface the coverage rules apply to. Emitting a single identical
            # empty inventory would collapse to one example after dedup, and the
            # model would never learn that an empty plan is sometimes correct --
            # the necessary counterweight to "never emit an empty plan".
            inv = TargetInventory(
                base_url=f"https://static-{i}.example",
                pages=[f"/{w}" for w in rng.sample(
                    ["", "about", "team", "pricing", "blog", "careers", "legal",
                     "privacy", "terms", "contact-info", "press", "faq"],
                    k=rng.randint(2, 8))],
                js_bundles=[f"/assets/{rng.randrange(1000)}.js"
                            for _ in range(rng.randint(0, 3))],
                headers={"server": rng.choice(["nginx", "cloudfront", "netlify"])},
                architecture_hints={"likely_static": True, "likely_spa": False,
                                    "framework_fingerprints": rng.sample(
                                        ["hugo", "jekyll", "gatsby", "eleventy"],
                                        k=rng.randint(0, 2))})
            # An empty inventory is the LEGITIMATE empty-plan case, not the trap.
            # The trap is specifically "hint says static WHILE surfaces exist", so
            # the two flags are mutually exclusive by definition.
            adversarial = False
        else:
            inv = _inventory(rng, archetype, adversarial)
        oracle = R.deterministic_fallback_plan(inv, ruleset)

        prompt = (f"{PLAN_INSTRUCTION}\n\n{R.render_rules(ruleset)}\n\n"
                  f"Inventory:\n{inv.render()}")
        # The property that matters is whether the RULES require anything, not
        # whether the inventory is literally bare. A static brochure site has
        # pages and bundles yet no rule-covered surface, so its correct plan is
        # empty -- and that is the counterweight the model must learn alongside
        # "never emit an empty plan when coverage IS required".
        meta = {"archetype": archetype, "ruleset": ruleset, "adversarial": adversarial,
                "no_required_coverage": len(oracle) == 0,
                "required_tasks": len(oracle)}

        variant = i % 4
        if variant == 0 or empty_case:
            yield Example(task="planner", system=SYS_PIPELINE,
                          turns=[{"role": "user", "content": prompt}],
                          target=_canonical(oracle), thinking=True,
                          split_key=f"recon:{ruleset}:{i}", forced_split=split,
                          license_tier="T2-internal", meta={**meta, "variant": "plan"})
            continue

        if variant == 1:
            # Judge a candidate plan: half are deliberately incomplete.
            broken = bool(oracle) and rng.random() < 0.5
            candidate = list(oracle)
            dropped = None
            if broken:
                dropped = candidate.pop(rng.randrange(len(candidate)))
            try:
                R.validate_plan(inv, candidate, ruleset)
                target = json.dumps({"valid": True, "missing": []}, indent=2)
            except R.PlanValidationError as e:
                target = json.dumps(
                    {"valid": False,
                     "missing": [f"{s} needs {k}" for s, k in e.missing]}, indent=2)
            yield Example(task="planner", system=SYS_PIPELINE,
                          turns=[{"role": "user", "content":
                                  f"{R.render_rules(ruleset)}\n\nInventory:\n{inv.render()}\n\n"
                                  f"Candidate plan:\n{_canonical(candidate)}\n\n"
                                  "Does this plan satisfy the coverage rules? Answer as JSON: "
                                  '{"valid": <bool>, "missing": ["<surface> needs <kind>", ...]}'}],
                          target=target, thinking=True,
                          split_key=f"recon:{ruleset}:{i}", forced_split=split,
                          license_tier="T2-internal",
                          meta={**meta, "variant": "validate", "was_broken": broken,
                                "dropped": dropped})
            continue

        if variant == 2 and oracle:
            # Repair from validator feedback -- mirrors plan_with_retry.
            candidate = list(oracle)
            for _ in range(min(3, len(candidate))):
                candidate.pop(rng.randrange(len(candidate)))
            try:
                R.validate_plan(inv, candidate, ruleset)
                continue
            except R.PlanValidationError as e:
                feedback = e.feedback()
            yield Example(task="planner", system=SYS_PIPELINE,
                          turns=[{"role": "user", "content":
                                  f"{R.render_rules(ruleset)}\n\nInventory:\n{inv.render()}\n\n"
                                  f"This plan was rejected:\n{_canonical(candidate)}\n\n"
                                  f"Validator feedback: {feedback}\n\n"
                                  "Emit the corrected plan as the same JSON array."}],
                          target=_canonical(oracle), thinking=True,
                          split_key=f"recon:{ruleset}:{i}", forced_split=split,
                          license_tier="T2-internal", meta={**meta, "variant": "repair"})
            continue

        if oracle:
            # Prioritization -- the one judgment the contract actually delegates.
            order = {"auth_surface": 0, "upload_surface": 1, "graphql_endpoint": 2,
                     "websocket_endpoint": 3, "form": 4, "openapi_spec": 5,
                     "endpoint": 6, "cookie": 7}
            ranked = sorted(oracle, key=lambda t: (order.get(t["surface_ref"].split(":")[0], 9),
                                                   t["surface_ref"], t["kind"]))
            yield Example(task="planner", system=SYS_PIPELINE,
                          turns=[{"role": "user", "content":
                                  f"{R.render_rules(ruleset)}\n\nInventory:\n{inv.render()}\n\n"
                                  "Produce the plan ordered by risk priority, highest first. "
                                  "Coverage is still mandatory: every required task must appear. "
                                  'Answer as a JSON array of {"surface_ref", "kind"}.'}],
                          target=json.dumps(ranked, indent=2), thinking=True,
                          split_key=f"recon:{ruleset}:{i}", forced_split=split,
                          license_tier="T2-internal", meta={**meta, "variant": "prioritize"})
