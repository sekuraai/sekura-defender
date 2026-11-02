"""Self-hosted GPU inference vs river per-token pricing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_defender.cost_model import inference_cost
from sekura_defender.gpu_cost import GPUS, MODEL_GB_FP8, Throughput, Traffic, analyse
from sekura_sft.config import BASE_MODEL

def main() -> int:
    t, thr = Traffic(), Throughput()
    river_call = inference_cost(BASE_MODEL, prompt_tokens=t.prompt_tokens,
                                completion_tokens=t.completion_tokens, calls=1)
    river_scan = river_call * t.calls_per_scan

    print(f"model: {BASE_MODEL}  ({MODEL_GB_FP8} GB FP8, Apache-2.0, ungated)")
    print(f"traffic (measured): {t.calls_per_scan} calls/scan, "
          f"{t.prompt_tokens}p/{t.completion_tokens}c, {t.cache_hit_rate:.0%} prefix cache")
    print(f"  -> {t.fresh_prefill_per_scan():,} fresh prefill + "
          f"{t.decode_per_scan():,} decode tokens per scan")
    print(f"\nriver: ${river_call:.5f}/call = ${river_scan:.3f}/scan\n")

    print(f"{'GPU':18s} {'$/hr':>6s} {'tok/s':>7s} {'s/scan':>7s} {'$/scan':>8s} "
          f"{'scans/hr':>9s} {'break-even':>11s} {'util':>6s}")
    print("-" * 82)
    for gpu in GPUS:
        if not gpu.fits():
            print(f"{gpu.name:18s}  does not fit {MODEL_GB_FP8} GB + KV headroom")
            continue
        for spot in (False, True):
            r = analyse(gpu, t, thr, river_scan, spot=spot)
            tag = f"{gpu.name}{' spot' if spot else ''}"
            rate = gpu.usd_hr_spot if spot else gpu.usd_hr_ondemand
            print(f"{tag:18s} {rate:6.2f} {thr.decode_tok_s(gpu):7.0f} "
                  f"{r.gpu_seconds_per_scan:7.1f} {r.usd_per_scan:8.4f} "
                  f"{r.scans_per_hour:9.1f} {r.breakeven_scans_month:11,.0f} "
                  f"{r.breakeven_utilisation:5.0%}")

    print("\n'break-even' = scans/month above which one always-on GPU beats river.")
    print("'util' = the GPU utilisation that break-even implies.\n")

    h100 = GPUS[0]
    print("monthly comparison at a few volumes (1x H100 on-demand @ $2.50/hr):")
    print(f"  {'scans/mo':>10s} {'river':>10s} {'GPU 24/7':>10s}   cheaper")
    for scans in (500, 2_000, 5_000, 10_000, 25_000, 100_000):
        r = analyse(h100, t, thr, river_scan)
        n_gpu = max(1, -(-scans // int(r.scans_per_hour * 730)))
        gpu_cost = n_gpu * r.monthly_247
        rv = scans * river_scan
        print(f"  {scans:10,d} ${rv:9,.0f} ${gpu_cost:9,.0f}   "
              f"{'river' if rv < gpu_cost else f'GPU ({n_gpu}x)'}")

    base = analyse(h100, t, thr, river_scan)
    print(f"\ncapacity: 1x H100 serves ~{base.scans_per_hour * 730:,.0f} scans/month flat out, "
          f"so it can comfortably cover the {base.breakeven_scans_month:,.0f} break-even volume.")

    print("\nsensitivity -- decode throughput is the least certain input:")
    print(f"  {'efficiency':>10s} {'tok/s':>7s} {'$/scan':>8s} {'capacity/mo':>12s} {'util @ b/e':>11s}")
    for eff in (0.25, 0.45, 0.70):
        th = Throughput(decode_efficiency=eff)
        r = analyse(h100, t, th, river_scan)
        print(f"  {eff:>9.0%} {th.decode_tok_s(h100):7.0f} {r.usd_per_scan:8.4f} "
              f"{r.scans_per_hour * 730:12,.0f} {r.breakeven_utilisation:11.0%}")
    print("  Note the break-even VOLUME does not move with throughput -- it is set by")
    print("  the hourly rate against river's per-scan price. Throughput decides only")
    print("  whether one card can absorb that volume, and how much headroom is left.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
