"""Vectorised 5/6/7-card hand evaluator.

Why not treys / eval7 / phevaluator?
------------------------------------
Every hot path in this project evaluates *batches*: Monte Carlo equity draws 10^5-10^6
boards, and the CFR solver needs the rank of ~1000 combos on every sampled river. Libraries
that evaluate one hand per Python call (treys, phevaluator's Python API) spend most of their
time in interpreter overhead; eval7 is fast but unmaintained with patchy 3.12 wheels. A
batched NumPy evaluator costs ~15 array operations *per batch*, independent of batch size,
needs no compiled extension, and is easy to verify exhaustively (see tests: all 2,598,960
five-card hands reproduce the textbook category counts and exactly 7,462 distinct ranks).

Score encoding
--------------
A score is a single int, higher = better, so comparing hands is integer comparison::

    score = category << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5

``category`` is 0 (high card) .. 8 (straight flush); ``k1..k5`` are the tie-break ranks
(0..12) in significance order, e.g. two pair = (high pair, low pair, kicker).

Algorithm (per batch of N hands with k cards each)
--------------------------------------------------
1. ``rank_counts`` (N, 13) and per-suit 13-bit rank masks (N, 4) via ``np.bincount`` on
   flattened (row, rank) / (row, suit) indices. Each card is unique, so summing ``1<<rank``
   within a suit equals OR-ing them.
2. Flush: a suit with >= 5 cards; its rank mask feeds two 8192-entry lookup tables:
   ``STRAIGHT_HIGH[mask]`` (highest straight in the mask, wheel aware, else -1) and
   ``TOP5[mask]`` (the five highest ranks pre-packed as nibbles).
3. Paired hands: sort ``count * 16 + rank`` per row. The largest key is the "main" group
   (quads > trips > pair, ties broken by rank) and the second largest is the secondary group,
   which is exactly what full house / two pair need. Kickers come from ``TOP5``/``HIGHEST``
   applied to the overall rank mask with the grouped ranks removed -- that handles corner
   cases like ``AAAA KK Q`` (kicker K) and three pairs in 7 cards (best two + best kicker).
4. ``np.select`` picks the category in strength order.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

from .cards import RANKS, IntArray, card_to_str

CATEGORY_NAMES: Final = (
    "High card",
    "Pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
)
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CAT_SHIFT: Final = 20


def _build_tables() -> tuple[IntArray, IntArray, IntArray]:
    straight_high = np.full(8192, -1, dtype=np.int64)
    top5 = np.zeros(8192, dtype=np.int64)
    highest = np.full(8192, -1, dtype=np.int64)
    wheel = (1 << 12) | 0b1111  # A-2-3-4-5
    for mask in range(8192):
        for high in range(12, 3, -1):  # A-high (12) down to 6-high (4)
            window = 0b11111 << (high - 4)
            if mask & window == window:
                straight_high[mask] = high
                break
        else:
            if mask & wheel == wheel:
                straight_high[mask] = 3  # five-high
        packed, taken = 0, 0
        for r in range(12, -1, -1):
            if mask >> r & 1 and taken < 5:
                packed |= r << (4 * (4 - taken))
                taken += 1
        top5[mask] = packed
        if mask:
            highest[mask] = mask.bit_length() - 1
    return straight_high, top5, highest


STRAIGHT_HIGH, TOP5, HIGHEST = _build_tables()
_RANK_ORDER: Final = np.arange(13, dtype=np.int64)


def _top_k(mask: IntArray, k: int) -> IntArray:
    """Top ``k`` ranks of each 13-bit mask packed as ``4k`` bits (most significant first)."""
    return TOP5[mask] >> (4 * (5 - k))


def evaluate(cards: npt.ArrayLike) -> IntArray:
    """Score a batch of hands. ``cards`` has shape (N, k) or (k,), 5 <= k <= 7, card ids.

    Returns an int64 array of shape (N,) (or a 0-d array for a single hand).
    """
    arr = np.asarray(cards, dtype=np.int64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    n, k = arr.shape
    if not 5 <= k <= 7:
        raise ValueError(f"evaluate needs 5-7 cards per hand, got {k}")
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    ranks = arr >> 2
    suits = arr & 3
    rows = np.repeat(np.arange(n, dtype=np.int64), k)

    # Step 1: per-rank counts and per-suit rank masks.
    counts = np.bincount(rows * 13 + ranks.ravel(), minlength=n * 13).reshape(n, 13)
    suit_idx = rows * 4 + suits.ravel()
    suit_mask = (
        np.bincount(suit_idx, weights=(1 << ranks.ravel()).astype(np.float64), minlength=n * 4)
        .reshape(n, 4)
        .astype(np.int64)
    ) & 0x1FFF  # duplicates (only in rows callers mask out) must not index past the tables
    suit_count = np.bincount(suit_idx, minlength=n * 4).reshape(n, 4)
    rank_mask = np.bitwise_or.reduce(suit_mask, axis=1)

    # Step 2: flushes and straights through the lookup tables.
    flush_suit = np.argmax(suit_count, axis=1)
    is_flush = suit_count[np.arange(n), flush_suit] >= 5
    flush_mask = np.where(is_flush, suit_mask[np.arange(n), flush_suit], 0)
    sf_high = np.where(is_flush, STRAIGHT_HIGH[flush_mask], -1)
    st_high = STRAIGHT_HIGH[rank_mask]

    # Step 3: grouped ranks. key = count*16 + rank, absent ranks pushed to -1.
    keys = np.where(counts > 0, counts * 16 + _RANK_ORDER, -1)
    keys.sort(axis=1)
    top, second = keys[:, -1], keys[:, -2]
    top_cnt, top_rank = top >> 4, top & 15
    sec_cnt, sec_rank = second >> 4, second & 15
    without_top = rank_mask & ~(1 << top_rank)
    without_two = without_top & ~(1 << np.maximum(sec_rank, 0))

    c = CAT_SHIFT
    conditions = [
        sf_high >= 0,
        top_cnt == 4,
        (top_cnt == 3) & (sec_cnt >= 2),
        is_flush,
        st_high >= 0,
        top_cnt == 3,
        (top_cnt == 2) & (sec_cnt == 2),
        top_cnt == 2,
    ]
    choices = [
        (STRAIGHT_FLUSH << c) | (sf_high << 16),
        (QUADS << c) | (top_rank << 16) | (HIGHEST[without_top] << 12),
        (FULL_HOUSE << c) | (top_rank << 16) | (sec_rank << 12),
        (FLUSH << c) | TOP5[flush_mask],
        (STRAIGHT << c) | (st_high << 16),
        (TRIPS << c) | (top_rank << 16) | (_top_k(without_top, 2) << 8),
        (TWO_PAIR << c) | (top_rank << 16) | (sec_rank << 12) | (HIGHEST[without_two] << 8),
        (PAIR << c) | (top_rank << 16) | (_top_k(without_top, 3) << 4),
    ]
    scores: IntArray = np.select(conditions, choices, default=(HIGH_CARD << c) | TOP5[rank_mask])
    return scores[0] if single else scores


def evaluate_hand(cards: Sequence[int]) -> int:
    """Scalar convenience wrapper (use :func:`evaluate` for anything in a loop)."""
    return int(evaluate(np.asarray(cards, dtype=np.int64)))


def category(score: int | IntArray) -> int | IntArray:
    return score >> CAT_SHIFT


_PLURAL: Final = dict(
    zip(
        RANKS,
        [
            "Twos",
            "Threes",
            "Fours",
            "Fives",
            "Sixes",
            "Sevens",
            "Eights",
            "Nines",
            "Tens",
            "Jacks",
            "Queens",
            "Kings",
            "Aces",
        ],
        strict=True,
    )
)


def describe(score: int) -> str:
    """Human-readable description, e.g. ``"Two pair, Kings and Sevens (A kicker)"``."""
    cat = score >> CAT_SHIFT
    nib = [(score >> (4 * (4 - i))) & 15 for i in range(5)]
    name = CATEGORY_NAMES[cat]
    plural = _PLURAL

    def rk(i: int) -> str:
        return str(RANKS[nib[i]])

    if cat in (STRAIGHT, STRAIGHT_FLUSH):
        return f"{name}, {rk(0)}-high"
    if cat == QUADS:
        return f"{name}, {plural[rk(0)]} ({rk(1)} kicker)"
    if cat == FULL_HOUSE:
        return f"{name}, {plural[rk(0)]} full of {plural[rk(1)]}"
    if cat == TRIPS:
        return f"{name}, {plural[rk(0)]}"
    if cat == TWO_PAIR:
        return f"{name}, {plural[rk(0)]} and {plural[rk(1)]} ({rk(2)} kicker)"
    if cat == PAIR:
        return f"{name} of {plural[rk(0)]}"
    if cat == FLUSH:
        return f"{name}, {rk(0)}-high"
    return f"{name}, {rk(0)}-high"


def describe_cards(cards: Sequence[int]) -> str:
    return f"{' '.join(card_to_str(c) for c in cards)}: {describe(evaluate_hand(cards))}"
