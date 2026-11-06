"""Range notation parsing/formatting and blocker-aware combo counting."""

from __future__ import annotations

import numpy as np
import pytest

from gto_trainer.poker.cards import (
    CLASS_INDEX,
    CLASS_LABELS,
    CLASS_NCOMBOS,
    combo_from_str,
    parse_cards,
)
from gto_trainer.poker.ranges import (
    RangeParseError,
    class_weights,
    compare_ranges,
    count_combos,
    format_range,
    parse_range,
)


def classes(text: str) -> set[str]:
    cw = class_weights(parse_range(text))
    return {CLASS_LABELS[i] for i in np.flatnonzero(cw > 0)}


def test_grid_layout() -> None:
    assert CLASS_LABELS[0] == "AA" and CLASS_LABELS[1] == "AKs" and CLASS_LABELS[13] == "AKo"
    assert CLASS_LABELS[168] == "22"
    assert CLASS_NCOMBOS.sum() == 1326
    assert CLASS_NCOMBOS[CLASS_INDEX["AKs"]] == 4
    assert CLASS_NCOMBOS[CLASS_INDEX["AKo"]] == 12
    assert CLASS_NCOMBOS[CLASS_INDEX["77"]] == 6


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("77+", {"77", "88", "99", "TT", "JJ", "QQ", "KK", "AA"}),
        ("ATs+", {"ATs", "AJs", "AQs", "AKs"}),
        ("KQo", {"KQo"}),
        ("KQ", {"KQs", "KQo"}),
        ("A5s-A2s", {"A5s", "A4s", "A3s", "A2s"}),
        ("A2s-A5s", {"A5s", "A4s", "A3s", "A2s"}),
        ("99-66", {"99", "88", "77", "66"}),
        ("T9s-76s", {"T9s", "98s", "87s", "76s"}),
        ("K9o+", {"K9o", "KTo", "KJo", "KQo"}),
        ("22+", {f"{r}{r}" for r in "23456789TJQKA"}),
    ],
)
def test_parse_shorthand(text: str, expected: set[str]) -> None:
    assert classes(text) == expected


def test_combined_notation_counts() -> None:
    w = parse_range("77+, ATs+, KQo")
    assert w.sum() == 8 * 6 + 4 * 4 + 12  # 76 combos
    assert parse_range("random").sum() == 1326


def test_weights_and_override_order() -> None:
    w = parse_range("AQs+,AKs:0.5")
    assert w[combo_from_str("AhKh")] == 0.5
    assert w[combo_from_str("AhQh")] == 1.0
    assert parse_range("QQ:50%")[combo_from_str("QsQh")] == 0.5
    assert parse_range("QQ:25")[combo_from_str("QsQh")] == 0.25


def test_specific_combos() -> None:
    w = parse_range("AhKh, AsKs:0.5")
    assert w.sum() == 1.5
    assert class_weights(w)[CLASS_INDEX["AKs"]] == pytest.approx(1.5 / 4)


@pytest.mark.parametrize("bad", ["AKx", "A", "77s", "A5s-K2s", "AKs-QJo", "AKs:2.5x", "ZZ"])
def test_parse_errors(bad: str) -> None:
    with pytest.raises(RangeParseError):
        parse_range(bad)


@pytest.mark.parametrize(
    "text",
    ["77+,ATs+,KQo", "22-55,A5s-A2s:0.5,KJo+,T9s", "AhKh,QQ:0.25,JTs-54s,random:0.1", ""],
)
def test_format_roundtrip(text: str) -> None:
    w = parse_range(text)
    assert np.allclose(parse_range(format_range(w)), w)


def test_format_is_compact() -> None:
    assert format_range(parse_range("77+,ATs+,KQo")) == "77+,ATs+,KQo"
    assert format_range(parse_range("A5s-A2s")) == "A5s-A2s"


def test_combo_counting_with_blockers() -> None:
    aa = parse_range("AA")
    assert count_combos(aa) == 6
    assert count_combos(aa, parse_cards("As")) == 3  # one ace out: C(3,2)
    assert count_combos(aa, parse_cards("AsAh")) == 1
    ak = parse_range("AK")
    assert count_combos(ak, parse_cards("Kd")) == 12  # 16 - 4
    assert count_combos(ak, parse_cards("AsKs")) == 9  # 3 aces x 3 kings
    # Board blockers on a flush texture
    assert count_combos(parse_range("AKs"), parse_cards("Ks9s2s")) == 3


def test_compare_ranges() -> None:
    user = parse_range("88+,AQs+")
    ref = parse_range("TT+,AJs+")
    d = compare_ranges(user, ref)
    assert d.overlap_combos == 5 * 6 + 2 * 4  # TT+ and AQs+
    assert {c for c, _ in d.over_included} == {"88", "99"}
    assert {c for c, _ in d.under_included} == {"AJs"}
    assert 0 < d.jaccard < 1
