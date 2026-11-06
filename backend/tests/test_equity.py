"""Equity against known values, exact vs Monte Carlo agreement, and blocker handling."""

from __future__ import annotations

import pytest

from gto_trainer.poker import odds
from gto_trainer.poker.cards import combo_from_str, parse_cards
from gto_trainer.poker.equity import hand_vs_hand, hand_vs_range, range_vs_range
from gto_trainer.poker.evaluator import evaluate_hand
from gto_trainer.poker.ranges import parse_range


def c(s: str) -> int:
    return combo_from_str(s)


@pytest.mark.parametrize(
    ("a", "b", "expected", "tol"),
    [
        ("AhKh", "QsQc", 0.462, 0.004),  # AKs vs QQ ~ 46/54
        ("AsAh", "KdKc", 0.8125, 0.004),  # AA vs KK, no shared suits ~ 81/19
        ("AcKd", "2h2s", 0.470, 0.004),  # the classic "coin flip": 22 slight favourite
    ],
)
def test_preflop_known_values_exact(a: str, b: str, expected: float, tol: float) -> None:
    r = hand_vs_hand(c(a), c(b))
    assert r.method == "exact"
    assert r.samples == 1_712_304
    assert r.equity == pytest.approx(expected, abs=tol)
    assert hand_vs_hand(c(b), c(a)).equity == pytest.approx(1 - r.equity, abs=1e-12)


def test_monte_carlo_agrees_with_exact() -> None:
    exact = hand_vs_hand(c("AhKh"), c("QsQc"))
    mc = hand_vs_hand(c("AhKh"), c("QsQc"), method="monte_carlo", samples=200_000, seed=3)
    assert abs(mc.equity - exact.equity) < 4 * mc.std_error + 1e-3


def test_turn_exact_matches_brute_force() -> None:
    board = parse_cards("Ks7s2h9d")
    hero, vill = parse_cards("AsQs"), parse_cards("Kd7c")
    used = {*board, *hero, *vill}
    wins = ties = n = 0
    for river in range(52):
        if river in used:
            continue
        a = evaluate_hand([*hero, *board, river])
        b = evaluate_hand([*vill, *board, river])
        wins += a > b
        ties += a == b
        n += 1
    r = hand_vs_hand(c("AsQs"), c("Kd7c"), board)
    assert r.method == "exact"
    assert r.equity == pytest.approx((wins + ties / 2) / n, abs=1e-12)


def test_card_removal_against_range() -> None:
    # AsAh vs "AA": only AdAc survives the blockers -> pure chop most of the time.
    r = hand_vs_range(c("AsAh"), parse_range("AA"))
    assert r.matchups == 1
    assert r.equity == pytest.approx(hand_vs_hand(c("AsAh"), c("AdAc")).equity, abs=1e-12)
    # On a board of three spades, AsKs blocks villain's AxKx spade combos entirely.
    board = parse_cards("Qs7s2s")
    r2 = hand_vs_range(c("AsKs"), parse_range("AKs"), board)
    assert r2.matchups == 3  # AcKc, AdKd, AhKh remain


def test_range_vs_itself_is_half() -> None:
    rng = parse_range("TT+,AKs")
    r = range_vs_range(
        rng, rng, parse_cards("8c5d2h"), method="monte_carlo", samples=100_000, seed=1
    )
    assert r.equity == pytest.approx(0.5, abs=4 * r.std_error + 1e-3)


def test_per_combo_equity_on_flop() -> None:
    board = parse_cards("Kc7d2s")
    r = range_vs_range(parse_range("KK,77,AK"), parse_range("QQ"), board, per_combo=True)
    assert r.method == "exact"
    sets = [v for k, v in r.hero_combo_equity.items() if k in {c("KhKd"), c("7h7s")}]
    assert len(sets) == 2 and all(v > 0.9 for v in sets)
    # AK (top pair) is a big favourite but QQ has outs.
    ak = r.hero_combo_equity[c("AhKd")]
    assert 0.6 < ak < 0.95


def test_odds_formulas() -> None:
    assert odds.pot_odds(100, 50) == pytest.approx(0.25)
    assert odds.minimum_defence_frequency(100, 50) == pytest.approx(2 / 3)
    assert odds.bluff_breakeven(100, 100) == pytest.approx(0.5)
    assert odds.optimal_bluff_fraction(100, 100) == pytest.approx(1 / 3)
    assert odds.value_to_bluff_ratio(100, 100) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        odds.pot_odds(0, 10)
