"""Typed TargetInventory, versioned CoverageRules, and the coverage oracle.

A faithful Python port of structured-recon-planner-contract.md. The contract
exists because of a real production failure: the planner judged a site "static",
emitted an empty plan, no agent got a task, and the scan returned nothing. The
architectural principle is "LLM proposes, deterministic system disposes" --
prioritization may be model-driven, coverage may not.

The invariant, verbatim from the contract:
  "For every surface in the inventory, the plan must contain at least one task of
   every required kind for that surface type."
  "An empty plan against a non-empty inventory is always a validation error."

`deterministic_fallback_plan` is the ground-truth oracle: it is the rule
cross-product, so it needs no teacher and cannot be wrong by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# The 16 vuln classes sekura-rs dispatches, plus the recon-side task kinds. Using
# these exact strings means a generated plan is pipeline-parseable as-is.
TASK_KINDS = (
    "FormFuzz", "InjectionProbe", "CsrfCheck", "AuthProbe", "SessionProbe",
    "AuthzProbe", "IdorProbe", "UploadProbe", "GraphQlIntrospect", "GraphQlAbuse",
    "WsProbe", "SchemaFuzz", "EndpointFuzz", "HeaderAudit", "CookieAudit",
    "SsrfProbe", "CorsAudit", "SecretScan",
)

SURFACE_TYPES = ("form", "endpoint", "auth_surface", "upload_surface",
                 "graphql_endpoint", "websocket_endpoint", "openapi_spec",
                 "cookie", "header_set", "js_bundle", "page")


@dataclass
class SurfaceRef:
    kind: str
    ident: str

    def __hash__(self):
        return hash((self.kind, self.ident))

    def __str__(self):
        return f"{self.kind}:{self.ident}"


@dataclass
class TargetInventory:
    """Observed facts only. No interpretations -- see the contract's rule that
    the LLM never sees raw HTML in lieu of a typed field."""
    base_url: str = "https://target.example"
    pages: list[str] = field(default_factory=list)
    forms: list[dict] = field(default_factory=list)
    endpoints: list[dict] = field(default_factory=list)
    js_bundles: list[str] = field(default_factory=list)
    cookies: list[dict] = field(default_factory=list)
    headers: dict = field(default_factory=dict)
    auth_surfaces: list[dict] = field(default_factory=list)
    upload_surfaces: list[dict] = field(default_factory=list)
    graphql_endpoints: list[dict] = field(default_factory=list)
    websocket_endpoints: list[dict] = field(default_factory=list)
    openapi_specs: list[dict] = field(default_factory=list)
    architecture_hints: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Advisory hints deliberately excluded: a target with hints but no
        observed surfaces genuinely has nothing to test."""
        return not any([self.forms, self.endpoints, self.auth_surfaces,
                        self.upload_surfaces, self.graphql_endpoints,
                        self.websocket_endpoints, self.openapi_specs, self.pages])

    def render(self) -> str:
        d = asdict(self)
        hints = d.pop("architecture_hints")
        body = json.dumps(d, indent=2)
        return (f"{body}\n\nArchitecture hints (ADVISORY ONLY -- they may inform "
                f"priority but never coverage):\n{json.dumps(hints, indent=2)}")


# --- Coverage rules --------------------------------------------------------

@dataclass(frozen=True)
class CoverageRule:
    surface: str                 # inventory attribute name
    kind: str                    # SurfaceRef kind
    required: tuple[str, ...]    # TaskKinds every such surface must receive

    def applies_to(self, inv: TargetInventory) -> list[SurfaceRef]:
        items = getattr(inv, self.surface, []) or []
        out = []
        for i, item in enumerate(items):
            ident = (item.get("id") or item.get("url") or item.get("path") or str(i)
                     if isinstance(item, dict) else str(item))
            out.append(SurfaceRef(self.kind, ident))
        return out


RULESETS: dict[str, tuple[CoverageRule, ...]] = {
    # The shipped baseline.
    "v1": (
        CoverageRule("forms", "form", ("FormFuzz", "InjectionProbe", "CsrfCheck")),
        CoverageRule("auth_surfaces", "auth_surface", ("AuthProbe", "SessionProbe")),
        CoverageRule("upload_surfaces", "upload_surface", ("UploadProbe",)),
        CoverageRule("graphql_endpoints", "graphql_endpoint",
                     ("GraphQlIntrospect", "GraphQlAbuse")),
        CoverageRule("endpoints", "endpoint", ("EndpointFuzz",)),
    ),
    # Adds websockets and authz; drops CsrfCheck.
    "v2": (
        CoverageRule("forms", "form", ("FormFuzz", "InjectionProbe")),
        CoverageRule("auth_surfaces", "auth_surface", ("AuthProbe", "AuthzProbe")),
        CoverageRule("upload_surfaces", "upload_surface", ("UploadProbe", "SecretScan")),
        CoverageRule("graphql_endpoints", "graphql_endpoint", ("GraphQlIntrospect",)),
        CoverageRule("websocket_endpoints", "websocket_endpoint", ("WsProbe",)),
        CoverageRule("endpoints", "endpoint", ("EndpointFuzz", "IdorProbe")),
    ),
    # API-centric: OpenAPI-driven, header/cookie hygiene.
    "v3": (
        CoverageRule("endpoints", "endpoint", ("EndpointFuzz", "SsrfProbe")),
        CoverageRule("openapi_specs", "openapi_spec", ("SchemaFuzz",)),
        CoverageRule("cookies", "cookie", ("CookieAudit",)),
        CoverageRule("auth_surfaces", "auth_surface", ("AuthProbe",)),
        CoverageRule("forms", "form", ("FormFuzz",)),
    ),
    # HELD OUT of training. The test split uses only this, so the model is
    # measured on applying rules it has never seen rather than on memorizing a
    # fixed surface->task mapping.
    "v4-heldout": (
        CoverageRule("forms", "form", ("FormFuzz", "CsrfCheck", "AuthzProbe")),
        CoverageRule("graphql_endpoints", "graphql_endpoint", ("GraphQlAbuse",)),
        CoverageRule("websocket_endpoints", "websocket_endpoint", ("WsProbe", "AuthProbe")),
        CoverageRule("upload_surfaces", "upload_surface", ("UploadProbe", "SecretScan")),
        CoverageRule("endpoints", "endpoint", ("EndpointFuzz", "CorsAudit")),
    ),
}

HELD_OUT_RULESET = "v4-heldout"
TRAIN_RULESETS = tuple(k for k in RULESETS if k != HELD_OUT_RULESET)


def render_rules(name: str) -> str:
    lines = [f"Coverage rule set: {name}"]
    for r in RULESETS[name]:
        lines.append(f"- every {r.kind} requires: {', '.join(r.required)}")
    return "\n".join(lines)


# --- Oracle ----------------------------------------------------------------

def required_pairs(inv: TargetInventory, ruleset: str) -> list[tuple[SurfaceRef, str]]:
    out = []
    for rule in RULESETS[ruleset]:
        for ref in rule.applies_to(inv):
            for kind in rule.required:
                out.append((ref, kind))
    return out


def deterministic_fallback_plan(inv: TargetInventory, ruleset: str) -> list[dict]:
    """The rule cross-product: minimum viable, complete by construction."""
    return [{"surface_ref": str(ref), "kind": kind}
            for ref, kind in required_pairs(inv, ruleset)]


class PlanValidationError(Exception):
    def __init__(self, missing, hallucinated=(), empty=False):
        self.missing = list(missing)
        self.hallucinated = list(hallucinated)
        self.empty = empty
        super().__init__(self.feedback())

    def feedback(self) -> str:
        parts = []
        if self.empty:
            parts.append("plan is empty but the inventory is not")
        if self.missing:
            shown = ", ".join(f"{s} needs {k}" for s, k in self.missing[:6])
            parts.append(f"missing coverage: {shown}"
                         + (f" (+{len(self.missing) - 6} more)" if len(self.missing) > 6 else ""))
        if self.hallucinated:
            parts.append("plan references surfaces not in the inventory: "
                         + ", ".join(self.hallucinated[:5]))
        return "; ".join(parts)


def validate_plan(inv: TargetInventory, plan: list[dict], ruleset: str) -> None:
    """Raises PlanValidationError, or returns None when the plan covers."""
    real = {str(ref) for rule in RULESETS[ruleset] for ref in rule.applies_to(inv)}
    have = {(t.get("surface_ref"), t.get("kind")) for t in plan}
    hallucinated = sorted({t.get("surface_ref") for t in plan
                           if t.get("surface_ref") not in real})
    missing = [(ref, kind) for ref, kind in required_pairs(inv, ruleset)
               if (str(ref), kind) not in have]
    empty = not plan and not inv.is_empty()
    if missing or hallucinated or empty:
        raise PlanValidationError(missing, hallucinated, empty)


def plan_is_valid(inv: TargetInventory, plan: list[dict], ruleset: str) -> bool:
    try:
        validate_plan(inv, plan, ruleset)
        return True
    except PlanValidationError:
        return False
