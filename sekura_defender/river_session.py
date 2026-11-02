"""Client factory and the capabilities gate."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sekura_sft.config import BASE_MODEL, BASE_MODEL_FALLBACKS, check_lora_rank

API_KEY_ENV = "RIVER_API_KEY"

# Train-call timeout. river defaults every call to 86400s (24h); at that setting
# a wedged run is indistinguishable from a slow one for a whole day.
TRAIN_TIMEOUT = 1800.0


class MissingApiKey(RuntimeError):
    pass


class NoUsableModel(RuntimeError):
    pass


@dataclass
class Capabilities:
    healthy: bool
    models: tuple[str, ...]
    selected: str


def client(api_key: str | None = None):
    """river.Client does NOT read RIVER_API_KEY itself -- the key is passed
    explicitly, so a missing one must fail loudly here."""
    import river_client as river
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise MissingApiKey(
            f"{API_KEY_ENV} is not set. Create a key at https://console.river.ai/")
    return river.Client(api_key=key)


def select_model(cl, preferred: str = BASE_MODEL,
                 fallbacks: tuple[str, ...] = BASE_MODEL_FALLBACKS) -> Capabilities:
    """Hard gate at process start.

    get_capabilities() is scoped to the account, so the published catalog is not
    what this key can use. Resolving here -- and recording the answer in the run
    manifest -- also prevents a silent base swap on resume, which would produce
    garbage from a checkpoint bound to a different base.
    """
    healthy = bool(cl.health_check())
    models = tuple(cl.get_capabilities())
    for candidate in (preferred, *fallbacks):
        if candidate in models:
            return Capabilities(healthy, models, candidate)
    raise NoUsableModel(
        f"none of {(preferred, *fallbacks)} is enabled for this key; available: {models}")


def lora(rank: int, *, train_unembed: bool):
    import river_client as river
    from sekura_sft.config import LORA_TRAIN_ATTN, LORA_TRAIN_MLP
    return river.LoraConfig(rank=check_lora_rank(rank), train_attn=LORA_TRAIN_ATTN,
                            train_mlp=LORA_TRAIN_MLP, train_unembed=train_unembed)
