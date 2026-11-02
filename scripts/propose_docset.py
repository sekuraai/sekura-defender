"""Generate data/docset/allow.tsv for human review."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sekura_sft.corpus.docset import propose

if __name__ == "__main__":
    print(f"wrote {propose()}")
