"""Self-hosting cost model.

The absolute dollars rest on estimated GPU rates and throughput; what these
tests pin is the SHAPE of the answer, which is what the decision turns on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_defender.gpu_cost import (GPUS, MODEL_GB_FP8, Gpu, Throughput, Traffic,
                                      analyse)

RIVER_SCAN = 0.193
H100 = GPUS[0]


def test_model_needs_a_big_card_despite_activating_few_parameters():
    """An A3B MoE activates ~3B params but must hold all 37.5 GB resident,
    because any expert may be routed to."""
    assert MODEL_GB_FP8 == 37.5
    assert H100.fits()
    assert not Gpu("A10 24GB", 24, 0.75, 0.5, 0.6).fits()


def test_traffic_is_the_measured_shape():
    t = Traffic()
    assert t.decode_per_scan() == 200 * 600
    # prefix caching removes most prefill, which is why it is worth protecting
    assert t.fresh_prefill_per_scan() < 0.25 * t.calls_per_scan * t.prompt_tokens


def test_self_hosting_is_cheaper_per_scan_at_full_utilisation():
    r = analyse(H100, Traffic(), Throughput(), RIVER_SCAN)
    assert r.usd_per_scan < RIVER_SCAN / 3


def test_break_even_volume_is_set_by_hourly_rate_not_throughput():
    """A rented GPU bills whether or not it is busy, so the question is never
    per-scan cost -- it is how much traffic justifies keeping one warm."""
    slow = analyse(H100, Traffic(), Throughput(decode_efficiency=0.25), RIVER_SCAN)
    fast = analyse(H100, Traffic(), Throughput(decode_efficiency=0.70), RIVER_SCAN)
    assert slow.breakeven_scans_month == pytest.approx(fast.breakeven_scans_month)
    # throughput decides only whether one card can absorb that volume
    assert fast.breakeven_utilisation < slow.breakeven_utilisation


def test_one_card_can_absorb_its_own_break_even_volume():
    r = analyse(H100, Traffic(), Throughput(), RIVER_SCAN)
    assert r.scans_per_hour * 730 > r.breakeven_scans_month


def test_spot_pricing_lowers_the_break_even_point():
    on = analyse(H100, Traffic(), Throughput(), RIVER_SCAN, spot=False)
    sp = analyse(H100, Traffic(), Throughput(), RIVER_SCAN, spot=True)
    assert sp.breakeven_scans_month < on.breakeven_scans_month
    assert sp.usd_per_scan < on.usd_per_scan


def test_bandwidth_drives_decode_throughput():
    """Decode on an MoE is bandwidth-bound, so the ranking follows HBM."""
    thr = Throughput()
    ranked = sorted(GPUS, key=lambda g: -thr.decode_tok_s(g))
    assert ranked[0].hbm_tbs == max(g.hbm_tbs for g in GPUS)


def test_low_volume_favours_the_api():
    r = analyse(H100, Traffic(), Throughput(), RIVER_SCAN)
    low = 1_000
    assert low * RIVER_SCAN < r.monthly_247
    high = int(r.breakeven_scans_month * 3)
    assert high * RIVER_SCAN > r.monthly_247
