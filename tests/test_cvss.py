"""CVSS scorer tests. Values are from the FIRST CVSS v3.1 specification and from
hand-computation against it, not from the implementation's own output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

from sekura_sft.scoring.cvss import (CvssError, base_score, has_temporal,
                                     parse_vector, score, severity_band,
                                     temporal_score)

BASE_CASES = [
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "critical"),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0, "none"),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N", 6.1, "medium"),
    ("CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H", 7.8, "high"),
    ("CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N", 5.9, "medium"),
    ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0, "critical"),
    ("CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "critical"),
    # Hand-computed: ISS .525448, Impact 3.373376, Exploitability .121090.
    ("CVSS:3.1/AV:P/AC:H/PR:H/UI:R/S:U/C:L/I:L/A:L", 3.5, "low"),
    # Scope-changed, hand-computed: raw 1.08 * (6.047730 + 2.515145) = 9.247906.
    ("CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 9.3, "critical"),
]


@pytest.mark.parametrize("vector,expected,band", BASE_CASES)
def test_base_score(vector, expected, band):
    assert base_score(vector) == pytest.approx(expected)
    assert severity_band(base_score(vector)) == band


def test_temporal_multiplies_the_rounded_base():
    # 9.8 * E:P(.94) * RL:O(.95) = 8.7514 -> 8.8
    v = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H/E:P/RL:O/RC:C"
    assert base_score(v) == pytest.approx(9.8)
    assert temporal_score(v) == pytest.approx(8.8)
    assert has_temporal(v)
    # 8.1 * E:U(.91) * RL:O(.95) = 7.00245, and roundup takes it to 7.1 not 7.0.
    v2 = "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H/E:U/RL:O/RC:C"
    assert temporal_score(v2) == pytest.approx(7.1)


def test_score_picks_temporal_only_when_supplied():
    plain = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert not has_temporal(plain)
    assert score(plain) == base_score(plain)
    assert not has_temporal(plain + "/E:X/RL:X/RC:X")   # X means "not defined"


def test_scope_changes_privileges_required_weighting():
    """PR is the one metric whose weight depends on scope."""
    u = base_score("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H")
    c = base_score("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:H/A:H")
    assert c > u


@pytest.mark.parametrize("bad", [
    "", "not-a-vector", "CVSS:3.1/AV:N", "CVSS:3.1/AV:Z/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
    "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H",          # missing A
])
def test_malformed_vectors_raise(bad):
    with pytest.raises(CvssError):
        base_score(bad)


def test_vector_without_prefix_is_accepted():
    assert base_score("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") == pytest.approx(9.8)


def test_bands_are_first_scale():
    assert severity_band(0.0) == "none"
    assert severity_band(3.9) == "low"
    assert severity_band(4.0) == "medium"
    assert severity_band(6.9) == "medium"
    assert severity_band(7.0) == "high"
    assert severity_band(8.9) == "high"
    assert severity_band(9.0) == "critical"


def test_parse_ignores_duplicate_components():
    m = parse_vector("CVSS:3.1/AV:N/AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    assert m["AV"] == "N"          # first wins
