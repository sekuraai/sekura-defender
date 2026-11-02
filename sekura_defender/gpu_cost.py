"""Self-hosted inference cost for Qwen3.6-35B-A3B-FP8 vs river's per-token price.

CONFIDENCE. Two inputs here are measured and two are estimated, and they should
not be read with equal weight:

  MEASURED   model size (37.5 GB FP8, ungated, Apache-2.0 on HF), and the traffic
             shape taken from this repo's own corpus -- ~4k prompt / 600
             completion per agent call, 200 calls per scan.
  ESTIMATED  GPU hourly rates and serving throughput. Rates move monthly and vary
             several-fold between hyperscalers and neoclouds; throughput depends
             on the serving stack, batch size and MoE kernel quality. Treat the
             break-even POINT as the output, not the absolute dollar figures, and
             re-run with measured throughput before committing to hardware.

Why an A3B MoE is awkward to size: only ~3B parameters activate per token, but
ALL 37.5 GB must be resident because any expert may be routed to. So memory is
sized by total parameters while compute is sized by active ones -- it needs a big
card but does not use most of its FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL_GB_FP8 = 37.5          # measured from the HF repo
KV_HEADROOM_GB = 8.0         # working set beyond weights before KV cache


@dataclass(frozen=True)
class Gpu:
    name: str
    vram_gb: int
    usd_hr_ondemand: float
    usd_hr_spot: float
    hbm_tbs: float
    note: str = ""

    def fits(self, model_gb: float = MODEL_GB_FP8) -> bool:
        return self.vram_gb >= model_gb + KV_HEADROOM_GB


# Indicative mid-2026 neocloud rates. Verify before committing -- these move.
GPUS = [
    Gpu("H100 80GB SXM", 80, 2.50, 1.80, 3.35, "the default choice"),
    Gpu("H200 141GB", 141, 3.50, 2.60, 4.80, "more KV cache, longer contexts"),
    Gpu("A100 80GB", 80, 1.60, 1.10, 2.04, "cheaper, ~40% less bandwidth"),
    Gpu("L40S 48GB", 48, 1.00, 0.75, 0.86, "fits, but bandwidth-starved for MoE"),
]


@dataclass
class Traffic:
    """Measured from this corpus."""
    prompt_tokens: int = 4000
    completion_tokens: int = 600
    calls_per_scan: int = 200
    cache_hit_rate: float = 0.80        # byte-identical system prefix

    def fresh_prefill_per_scan(self) -> int:
        return int(self.calls_per_scan * self.prompt_tokens * (1 - self.cache_hit_rate))

    def decode_per_scan(self) -> int:
        return self.calls_per_scan * self.completion_tokens


@dataclass
class Throughput:
    """Aggregate tokens/sec at serving batch sizes. ESTIMATED.

    Decode on an MoE is bandwidth-bound: at batch 1 you read ~3B active params
    per token, but at serving batch sizes expert diversity means you read most of
    the 37.5 GB each step, so throughput tracks (bandwidth / model size) x batch.
    The efficiency factor covers attention, routing overhead and real-world MFU.
    """
    decode_efficiency: float = 0.45
    prefill_tok_s: int = 20_000
    batch: int = 48

    def decode_tok_s(self, gpu: Gpu) -> float:
        steps_s = gpu.hbm_tbs * 1000 / MODEL_GB_FP8      # full-weight reads per second
        return steps_s * self.batch * self.decode_efficiency


@dataclass
class Result:
    gpu: Gpu
    spot: bool
    gpu_seconds_per_scan: float
    usd_per_scan: float
    scans_per_hour: float
    breakeven_scans_month: float
    breakeven_utilisation: float
    monthly_247: float


def analyse(gpu: Gpu, traffic: Traffic, thr: Throughput,
            river_usd_per_scan: float, spot: bool = False) -> Result:
    rate = gpu.usd_hr_spot if spot else gpu.usd_hr_ondemand
    prefill_s = traffic.fresh_prefill_per_scan() / thr.prefill_tok_s
    decode_s = traffic.decode_per_scan() / thr.decode_tok_s(gpu)
    gpu_s = prefill_s + decode_s
    usd = gpu_s * rate / 3600
    scans_hr = 3600 / gpu_s
    # A rented GPU bills whether or not it is busy, so the comparison is not
    # per-scan cost but "how much traffic justifies keeping one warm".
    breakeven_scans_hr = rate / river_usd_per_scan
    return Result(gpu=gpu, spot=spot, gpu_seconds_per_scan=gpu_s, usd_per_scan=usd,
                  scans_per_hour=scans_hr,
                  breakeven_scans_month=breakeven_scans_hr * 730,
                  breakeven_utilisation=breakeven_scans_hr / scans_hr,
                  monthly_247=rate * 730)
