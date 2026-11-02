"""SimHash-64 over character 3-gram shingles, for near-duplicate claim collapse.

Used only WITHIN a (dedup_family, heading) bucket. A global pass at 370k scale
would collapse legitimately distinct facts about the same CWE -- cross-document
duplication is already handled at the document level (docset.py), which is where
it belongs given that duplicate book editions share only ~2% of exact claims.
"""

from __future__ import annotations

import hashlib
import re

SHINGLE = 3
BITS = 64
_MASK = (1 << BITS) - 1

# Calibrated on 9,000 real within-bucket claim pairs, labelled by token Jaccard:
#   hamming <=  3   recall 19.4%   false-pos 0.000%   <- the value the plan assumed
#   hamming <=  8   recall 72.2%   false-pos 0.034%
#   hamming <= 10   recall 80.6%   false-pos 0.080%   <- chosen
#   hamming <= 14   recall 91.7%   false-pos 1.315%
# Distinct pairs bottom out at hamming 8 (p10 = 20), so 10 sits in the gap.
# We are deleting training data here, so a false positive (dropping a genuinely
# distinct fact) costs more than a false negative (keeping one duplicate);
# 10 buys 4x the recall of the assumed value at a 0.08% false-positive rate.
SIMHASH_THRESHOLD = 10


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def simhash(text: str) -> int:
    """64-bit SimHash. Empty/short text hashes to 0, which callers must treat as
    'no signal' rather than 'identical to every other empty string'."""
    norm = normalize(text)
    if len(norm) < SHINGLE:
        return 0
    v = [0] * BITS
    for i in range(len(norm) - SHINGLE + 1):
        h = int.from_bytes(
            hashlib.blake2b(norm[i:i + SHINGLE].encode(), digest_size=8).digest(), "big")
        for b in range(BITS):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(BITS):
        if v[b] > 0:
            out |= 1 << b
    return out & _MASK


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & _MASK).bit_count()


class DupIndex:
    """Exact near-duplicate lookup within one (dedup_family, heading) bucket.

    Deliberately a linear scan, not an LSH banded index. Buckets are small
    (median 17 claims, p99 41), so an exact O(n) scan costs ~3M comparisons over
    the whole corpus -- negligible -- and it has no recall loss. Banding would
    only pay off at a much larger bucket size, and at threshold 10 the
    pigeonhole guarantee that makes banding lossless no longer holds anyway
    (it requires threshold < number of bands).
    """

    def __init__(self, threshold: int = SIMHASH_THRESHOLD):
        self.threshold = threshold
        self._seen: list[int] = []

    def find(self, h: int) -> int | None:
        if h == 0:                      # no signal, never a duplicate
            return None
        for cand in self._seen:
            if hamming(h, cand) <= self.threshold:
                return cand
        return None

    def add(self, h: int) -> None:
        if h != 0:
            self._seen.append(h)

    def add_if_new(self, h: int) -> bool:
        """True if `h` is new (and was added); False if it duplicates a member."""
        if self.find(h) is not None:
            return False
        self.add(h)
        return True
