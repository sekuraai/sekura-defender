"""Assemble every generator into the sharded dataset + lock manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft.assemble import generate, write
from sekura_sft.config import MANIFESTS_DIR, TIER_ORDER


def main() -> int:
    max_tier = "T3-reference"
    for a in sys.argv[1:]:
        if a.startswith("--max-tier="):
            max_tier = a.split("=", 1)[1]
    if max_tier not in TIER_ORDER:
        print(f"unknown tier {max_tier!r}", file=sys.stderr)
        return 2
    examples, rep = generate(max_tier=max_tier)
    lock = write(examples, rep, max_tier=max_tier)
    print(rep.render())
    print(f"\nshards: {len(lock['shards'])}  corpus sha256: {lock['corpus_sha256'][:16]}...")
    print(f"wrote {MANIFESTS_DIR / 'dataset.lock.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
