"""Evaluator correctness: exhaustive 5-card checks, 7-card consistency, treys cross-check."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from gto_trainer.poker.cards import parse_cards
from gto_trainer.poker.equity import enumerate_runouts
from gto_trainer.poker.evaluator import (
    CAT_SHIFT,
    FLUSH,
    FULL_HOUSE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TWO_PAIR,
    evaluate,
    evaluate_hand,
)


def ev(text: str) -> int:
    return evaluate_hand(parse_cards(text))


@pytest.mark.slow
def test_exhaustive_five_card_hands() -> None:
    """All C(52,5) hands: textbook category counts and exactly 7,462 equivalence classes."""
    hands = enumerate_runouts(list(range(52)), 5)
    assert len(hands) == 2_598_960
    scores = evaluate(hands)
    counts = np.bincount(scores >> CAT_SHIFT, minlength=9)
    expected = [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40]
    assert counts.tolist() == expected
    assert len(np.unique(scores)) == 7_462


def test_seven_card_equals_best_five_card_subset() -> None:
    rng = np.random.default_rng(7)
    hands = np.argpartition(rng.random((3000, 52)), 7, axis=1)[:, :7]
    seven = evaluate(hands)
    subsets = np.array(list(itertools.combinations(range(7), 5)))
    five = evaluate(hands[:, subsets].reshape(-1, 5)).reshape(3000, 21)
    np.testing.assert_array_equal(seven, five.max(axis=1))


def test_matches_treys_ordering() -> None:
    treys = pytest.importorskip("treys")
    evaluator = treys.Evaluator()
    rng = np.random.default_rng(11)
    from gto_trainer.poker.cards import card_to_str

    def t(cards: np.ndarray) -> int:
        return int(
            evaluator.evaluate(
                [treys.Card.new(card_to_str(int(c))) for c in cards[:2]],
                [treys.Card.new(card_to_str(int(c))) for c in cards[2:]],
            )
        )

    for _ in range(1500):
        deck = rng.permutation(52)
        board, a, b = deck[:5], deck[5:7], deck[7:9]
        mine = np.sign(evaluate_hand([*a, *board]) - evaluate_hand([*b, *board]))
        ref = np.sign(t(np.array([*b, *board])) - t(np.array([*a, *board])))  # treys: lower=better
        assert mine == ref


@pytest.mark.parametrize(
    ("better", "worse"),
    [
        ("6c5d4h3s2c", "5c4d3h2sAc"),  # six-high straight beats the wheel
        ("5s4s3s2sAs", "KhKdKcKs2d"),  # steel wheel (straight flush) beats quads
        ("AhAdAcKsKd", "AsQsTs8s6s"),  # full house beats flush
        ("AsQsTs8s6s", "KhQhJhTh8h"),  # ace-high flush beats king-high flush
        ("AhKdQcJsTd", "KhQdJcTs9d"),  # broadway beats king-high straight
        ("AhAdKc7s2d", "AsAcQd7h3h"),  # same pair, better kicker
        ("KhKdQcQs4d", "KsKcQdQh3h"),  # two pair decided by kicker
        ("7h7d7c2s3d", "6h6d6cAsKd"),  # higher trips win regardless of kickers
    ],
)
def test_pairwise_ordering(better: str, worse: str) -> None:
    assert ev(better) > ev(worse)


def test_seven_card_selection_rules() -> None:
    # Three pairs: best two pairs plus the best remaining card as kicker.
    assert ev("AsAdKsKdQsQd2c") == ev("AsAdKsKdQc")
    # Two sets make a full house of the higher set over the lower.
    s = ev("AsAdAcKsKdKc2h")
    assert s >> CAT_SHIFT == FULL_HOUSE and s == ev("AsAdAcKsKd")
    # Quads with a pair on board: kicker is the highest remaining rank, not the pair.
    assert ev("AsAdAcAhKsKd2c") == ev("AsAdAcAhQc") + (1 << 12)
    # Six suited cards: flush uses the top five.
    assert ev("AsKs9s7s5s3s2d") == ev("AsKs9s7s5s")
    assert ev("AsKs9s7s5s3s2d") >> CAT_SHIFT == FLUSH
    # Straight + flush in seven cards that is not a straight flush.
    assert ev("9s8s7d6s5c2s3s") >> CAT_SHIFT == FLUSH
    assert ev("9s8s7s6s5s4d") >> CAT_SHIFT == STRAIGHT_FLUSH
    assert ev("Ac2d3h4s5c9d9h") >> CAT_SHIFT == STRAIGHT


def test_ties_and_split_pots() -> None:
    board = "AsKsQsJsTs"  # royal flush on board: every hand ties
    assert ev("2c3d" + board) == ev("AhAd" + board)
    board2 = "KhKdQcQs3d"  # 44 is counterfeited: KKQQ with a 4 kicker loses to KKQQ-A
    assert ev("Ac2c" + board2) > ev("4c4d" + board2)
    board3 = "KhKdQcQsAd"  # both players play the board -> split pot
    assert ev("2c3d" + board3) == ev("4h5h" + board3)
    # Same straight from different hole cards splits.
    assert ev("Ah2c" + "KdQcJsTh3d") == ev("Ad4c" + "KdQcJsTh3d")
    assert ev("9c9d" + "KdKcQsQh3d") >> CAT_SHIFT == TWO_PAIR


def test_batch_shape_and_scalar_agree() -> None:
    hands = np.array([parse_cards("AsKsQsJsTs2c3d"), parse_cards("2c3d4h5s7c8dTh")])
    batch = evaluate(hands)
    assert batch.shape == (2,)
    assert batch[0] == evaluate_hand(hands[0].tolist())
