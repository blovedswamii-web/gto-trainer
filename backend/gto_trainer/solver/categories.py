"""Postflop hand categories (made hand x draw), for strategy display and leak aggregation.

These are a *view* of the solution, not a solving abstraction: the solver works on exact
combos, and categories only group them so "bet/check by hand category" is readable.
Categories are ordered strong -> weak; a weak made hand with a strong draw is promoted to
"pair + draw" because players think about (and misplay) those differently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..poker.cards import COMBOS, IntArray
from ..poker.evaluator import (
    FLUSH,
    FULL_HOUSE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    STRAIGHT_HIGH,
    TRIPS,
    evaluate,
)

CATEGORIES = (
    "Straight flush / quads",
    "Full house",
    "Flush",
    "Straight",
    "Set",
    "Trips",
    "Two pair",
    "Overpair",
    "Top pair, good kicker",
    "Top pair, weak kicker",
    "Middle pair",
    "Weak pair / underpair",
    "Pair + draw",
    "Flush draw",
    "Open-ender / combo draw",
    "Gutshot",
    "Overcards",
    "Ace high",
    "Air",
)


def _straight_outs(hand_mask: int, board_mask: int) -> int:
    """Number of ranks that complete a straight using at least one hole card."""
    outs = 0
    for r in range(13):
        bit = 1 << r
        if hand_mask & bit:
            continue
        if STRAIGHT_HIGH[hand_mask | bit] >= 0 and STRAIGHT_HIGH[board_mask | bit] < 0:
            outs += 1
    return outs


def classify(hole: Sequence[int], board: Sequence[int]) -> str:
    """Category for one hand on a 3- or 4-card board (river boards get no draw categories)."""
    cards = [*hole, *board]
    score = int(evaluate(np.array(cards)))
    cat = score >> 20
    hr = sorted((c >> 2 for c in hole), reverse=True)
    br = [c >> 2 for c in board]
    distinct = sorted(set(br), reverse=True)
    top = distinct[0]
    if cat >= FULL_HOUSE and not (cat == FULL_HOUSE and _board_plays_boat(hole, board)):
        return "Straight flush / quads" if cat > FULL_HOUSE else "Full house"
    if cat == STRAIGHT_FLUSH:
        return "Straight flush / quads"
    if cat == FLUSH:
        return "Flush"
    if cat == STRAIGHT:
        return "Straight"

    pocket = hr[0] == hr[1]
    matches = [r for r in hr if r in br]
    if cat == TRIPS or (pocket and hr[0] in br):
        return "Set" if pocket else "Trips"
    if len(set(matches)) == 2:
        return "Two pair"

    # Draws (flop/turn only)
    draw = ""
    if len(board) < 5:
        suits = [c & 3 for c in cards]
        hole_suits = {c & 3 for c in hole}
        if any(suits.count(s) == 4 for s in hole_suits):
            draw = "flush"
        hand_mask = sum(1 << r for r in set(hr + br))
        board_mask = sum(1 << r for r in set(br))
        outs = _straight_outs(hand_mask, board_mask)
        if outs >= 2 or (draw and outs >= 1):
            draw = "strong" if draw else "oesd"
        elif outs == 1 and not draw:
            draw = "gutshot"

    pair_cat = ""
    if pocket:
        pair_cat = "Overpair" if hr[0] > top else "Weak pair / underpair"
    elif matches:
        m = matches[0]
        kicker = hr[1] if hr[0] == m else hr[0]
        if m == top:
            pair_cat = "Top pair, good kicker" if kicker >= 9 else "Top pair, weak kicker"
        elif len(distinct) > 1 and m == distinct[1]:
            pair_cat = "Middle pair"
        else:
            pair_cat = "Weak pair / underpair"
    if pair_cat:
        if draw and pair_cat in ("Middle pair", "Weak pair / underpair"):
            return "Pair + draw"
        return pair_cat
    if draw == "flush":
        return "Flush draw"
    if draw in ("oesd", "strong"):
        return "Open-ender / combo draw"
    if draw == "gutshot":
        return "Gutshot"
    if hr[1] > top:
        return "Overcards"
    if hr[0] == 12:
        return "Ace high"
    return "Air"


def _board_plays_boat(hole: Sequence[int], board: Sequence[int]) -> bool:
    """True if the full house is entirely on the board (only possible with 5 board cards)."""
    if len(board) < 5:
        return False
    return int(evaluate(np.array(board))) == int(evaluate(np.array([*hole, *board])))


def classify_combos(combos: IntArray, board: Sequence[int]) -> list[str]:
    return [classify([int(a), int(b)], board) for a, b in COMBOS[combos]]
