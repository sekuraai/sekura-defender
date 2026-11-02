"""Whole-project river cost estimate from the measured corpus."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_defender.cost_model import (PRICES, Estimate, inference_cost, rl_cost,
                                        sft_cost)
from sekura_sft.config import BASE_MODEL, MANIFESTS_DIR


def main() -> int:
    tok_path = MANIFESTS_DIR / "tokenization.json"
    if not tok_path.exists():
        print("run scripts/tokenize_dataset.py first", file=sys.stderr)
        return 2
    tk = json.loads(tok_path.read_text())
    model = tk.get("base_model", BASE_MODEL)
    train = tk["splits"]["train"]
    val = tk["splits"].get("val", {})

    teacher_extra = 0
    tpath = MANIFESTS_DIR / "teacher.json"
    if tpath.exists():
        t = json.loads(tpath.read_text())
        # measured mean tokens/example from the deterministic corpus, scaled by
        # the teacher slices' longer targets
        teacher_extra = int(t.get("kept", 0) * 2100)

    est = Estimate(model=model)

    epoch_tokens = train["billed_tokens"] + teacher_extra
    usd, detail = sft_cost(model, epoch_tokens, epochs=2,
                           eval_tokens=val.get("billed_tokens", 0), eval_passes=11)
    est.add("Phase 1  SFT (2 epochs)", usd, detail)

    usd, detail = sft_cost(model, epoch_tokens, epochs=1, eval_tokens=0, eval_passes=0)
    est.add("         LR probe (3 x 20 steps)", usd * 0.10, "3-point probe at 10% of an epoch")

    # RL: prompt/completion sizes measured from the corresponding SFT slices.
    usd, detail = rl_cost(model, steps=300, prompts_per_step=64, group_size=6,
                          prompt_tokens=800, completion_tokens=450)
    est.add("Phase 2  RL ENV-B planner (300)", usd, detail)

    usd, detail = rl_cost(model, steps=500, prompts_per_step=96, group_size=6,
                          prompt_tokens=900, completion_tokens=1400)
    est.add("Phase 3  RL ENV-A attack trees (500)", usd, detail)

    usd, detail = rl_cost(model, steps=400, prompts_per_step=128, group_size=8,
                          prompt_tokens=550, completion_tokens=220)
    est.add("Phase 4  RL ENV-C triage (400)", usd, detail)

    usd, detail = rl_cost(model, steps=400, prompts_per_step=96, group_size=6,
                          prompt_tokens=750, completion_tokens=700)
    est.add("Phase 5  mixed RL (400)", usd, detail)

    usd, _ = sft_cost(model, epoch_tokens, epochs=1, eval_tokens=0, eval_passes=0)
    est.add("Phase 6  SFT refresh (0.5 epoch)", usd * 0.5, "anti-forgetting pass")

    n_eval = tk["splits"].get("test", {}).get("examples", 0)
    eval_run = inference_cost(model, prompt_tokens=700, completion_tokens=400,
                              calls=n_eval)
    est.add("         eval suite (15 runs)", eval_run * 15,
            f"{n_eval} examples x 15 gated runs")

    est.add("         checkpoint storage", 0.10 * 3,
            "rank-32 LoRA adapters, ~30 checkpoints, 3 months")

    print(est.render())
    print()
    print("  Phase 0 smoke test, run FIRST, before any of the above:")
    smoke = sft_cost(model, 64 * 700, epochs=1, eval_tokens=8 * 700, eval_passes=1)[0]
    smoke += rl_cost(model, steps=3, prompts_per_step=8, group_size=4,
                     prompt_tokens=800, completion_tokens=450)[0]
    print(f"    ~${smoke:.2f} measured; budget $15 with retries")
    print()
    print("  Serving economics (what replaces the incumbent):")
    per_call = inference_cost(model, prompt_tokens=4000, completion_tokens=600, calls=1)
    print(f"    one agent call (4k prompt, mostly cached / 600 completion): ${per_call:.5f}")
    print(f"    a 200-call scan:                                           ${per_call * 200:.2f}")
    print(f"    1,000 scans/month:                                         ${per_call * 200 * 1000:,.0f}")
    print()
    print("  Cheapest viable path if the full programme is not funded:")
    only_sft, d = sft_cost(model, epoch_tokens, 2, val.get("billed_tokens", 0), 11)
    envb, _ = rl_cost(model, steps=300, prompts_per_step=64, group_size=6,
                      prompt_tokens=800, completion_tokens=450)
    print(f"    Phase 0 + Phase 1 SFT only          ~${smoke + only_sft:,.2f}")
    print(f"    ...plus ENV-B (the production bug)  ~${smoke + only_sft + envb:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
