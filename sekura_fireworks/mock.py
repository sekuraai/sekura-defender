"""Offline doubles for the Fireworks training surface.

Provisioning is a fixed ~$12.50 tax per job, so a code-level bug found on GPU
costs more than the training it interrupts. Every failure in this build so far
was code-level -- a silently dropped weight key, a report iterated as a list
when it is a dict, a duplicated keyword argument -- and each would have been
caught here for nothing.

The sampler replays each row's own target, so a mock campaign should score at
the oracle ceiling. That makes the mock a test of the HARNESS: if the mock run
does not come out at ~1.0, the scoring, gating or comparison code is wrong, not
the model.
"""
from __future__ import annotations

import hashlib
import math


class _Future:
    def __init__(self, value):
        self._v = value

    def result(self, *a, **k):
        return self._v


class _Out:
    def __init__(self, metrics):
        self.metrics = metrics
        self.loss_fn_outputs = []
        self.loss_fn_output_type = "logprobs"


class _Save:
    def __init__(self, path):
        self.path = path
        self.type = None


class MockSampler:
    """Replays the target for each prompt, keyed by prompt token ids."""

    def __init__(self, by_prompt: dict[tuple, str]):
        self.by_prompt = by_prompt

    def sample(self, model_input, num_samples, sampling_params, **kw):
        ids = tuple(model_input.to_ints() if hasattr(model_input, "to_ints")
                    else model_input)
        return _Future(_Seq(self.by_prompt.get(ids, "")))


class _Seq:
    def __init__(self, text):
        self.sequences = [self]
        self.text = text
        self.stop_reason = "stop"
        self.tokens_np = []


class MockTrainingClient:
    """Produces a plausible, monotonically improving loss curve."""

    def __init__(self, by_prompt: dict[tuple, str] | None = None):
        self.run_id = "mock-run"
        self.steps = 0
        self.saved: list[str] = []
        self.by_prompt = by_prompt or {}

    def forward_backward(self, data, loss_fn, loss_fn_config=None):
        assert loss_fn == "cross_entropy", loss_fn
        n = len(data)
        # Weighted loss falls like a real curve so LossTripwire sees progress.
        per_ex = 1.4 * math.exp(-self.steps / 60) + 0.55
        resp = 0
        for d in data:
            w = d.loss_fn_inputs.get("weights")
            if w is not None:
                resp += int(sum(1 for v in w.to_numpy() if v != 0))
        return _Future(_Out({
            "loss:sum": per_ex * n, "response_tokens": float(resp),
            "total_tokens:sum": float(resp * 3), "local_input_sequences:sum": float(n),
        }))

    def optim_step(self, adam_params, *a, **k):
        self.steps += 1
        return _Future(None)

    def save_weights_for_sampler(self, name, *a, **k):
        h = hashlib.sha1(f"{name}:{self.steps}".encode()).hexdigest()[:8]
        p = f"{name}-{h}"
        self.saved.append(p)
        return _Future(_Save(p))

    def create_sampling_client(self, path, *a, **k):
        return MockSampler(self.by_prompt)


class MockServiceClient:
    def __init__(self):
        self.training_session_id = "mock-session"

    def close(self):
        pass
