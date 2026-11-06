"""Card, combo and hand-class primitives.

Everything downstream (evaluator, equity, ranges, CFR) works on small integers so that
NumPy can vectorise over them. The encodings are fixed and global:

* **card id** ``c = rank * 4 + suit`` with rank 0..12 = ``2..A`` and suit 0..3 = ``c d h s``.
  So ``2c = 0``, ``As = 51``. ``c >> 2`` is the rank, ``c & 3`` the suit.
* **combo id** 0..1325 indexes :data:`COMBOS`, the 1326 unordered two-card holdings, each
  stored as ``(high_card_id, low_card_id)``.
* **hand class** 0..168 is a cell of the 13x13 grid, ``row * 13 + col`` with row/col 0 = Ace.
  Pairs sit on the diagonal, suited hands above it (row < col), offsuit below (row > col).
  That is the standard chart layout, so the frontend can index the grid directly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Final

import numpy as np
import numpy.typing as npt

RANKS: Final = "23456789TJQKA"
SUITS: Final = "cdhs"
SUIT_SYMBOLS: Final = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def card_from_str(text: str) -> int:
    """Parse ``"As"``/``"td"`` into a card id. Rank is case-insensitive, suit lower-cased."""
    if len(text) != 2:
        raise ValueError(f"card must be 2 characters like 'As', got {text!r}")
    rank = RANKS.find(text[0].upper())
    suit = SUITS.find(text[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"invalid card {text!r}")
    return rank * 4 + suit


def card_to_str(card: int) -> str:
    return RANKS[card >> 2] + SUITS[card & 3]


_CARD_RE = re.compile(r"([2-9TJQKAtjqka])([cdhsCDHS])")


def parse_cards(text: str) -> list[int]:
    """Parse a run of cards such as ``"AsKd 7h"`` or ``"As,Kd,7h"``. Rejects duplicates."""
    cleaned = re.sub(r"[\s,]+", "", text)
    if not cleaned:
        return []
    matches = _CARD_RE.findall(cleaned)
    if "".join(r + s for r, s in matches) != cleaned:
        raise ValueError(f"could not parse cards from {text!r}")
    cards = [card_from_str(r + s) for r, s in matches]
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate card in {text!r}")
    return cards


def cards_to_str(cards: Iterable[int]) -> str:
    return "".join(card_to_str(c) for c in cards)


def cards_mask(cards: Iterable[int]) -> int:
    """Python-int bitmask of a card set (bit ``c`` set for card ``c``)."""
    mask = 0
    for c in cards:
        mask |= 1 << int(c)
    return mask


# --------------------------------------------------------------------------------------
# Combos
# --------------------------------------------------------------------------------------


def _build_combos() -> tuple[IntArray, IntArray]:
    combos = [(hi, lo) for hi in range(52) for lo in range(hi)]
    arr = np.array(combos, dtype=np.int64)
    index = np.full((52, 52), -1, dtype=np.int64)
    for i, (hi, lo) in enumerate(combos):
        index[hi, lo] = i
        index[lo, hi] = i
    return arr, index


COMBOS, COMBO_INDEX = _build_combos()
N_COMBOS: Final = 1326
#: uint64 bitmask of each combo's two cards -- blocker checks become one ``&``.
COMBO_MASK: Final[npt.NDArray[np.uint64]] = (np.uint64(1) << COMBOS[:, 0].astype(np.uint64)) | (
    np.uint64(1) << COMBOS[:, 1].astype(np.uint64)
)


def combo_index(c1: int, c2: int) -> int:
    idx = int(COMBO_INDEX[c1, c2])
    if idx < 0:
        raise ValueError("a combo needs two distinct cards")
    return idx


def combo_from_str(text: str) -> int:
    cards = parse_cards(text)
    if len(cards) != 2:
        raise ValueError(f"expected exactly two cards, got {text!r}")
    return combo_index(cards[0], cards[1])


def combo_to_str(idx: int) -> str:
    hi, lo = COMBOS[idx]
    return card_to_str(int(hi)) + card_to_str(int(lo))


def blocked_combos(dead: Iterable[int]) -> BoolArray:
    """Boolean (1326,) mask: True where the combo uses any dead card."""
    mask = np.uint64(cards_mask(dead))
    out: BoolArray = (COMBO_MASK & mask) != 0
    return out


def combos_conflict(a: IntArray, b: IntArray) -> BoolArray:
    """Pairwise card-sharing matrix between two lists of combo ids (shape ``len(a) x len(b)``)."""
    out: BoolArray = (COMBO_MASK[a][:, None] & COMBO_MASK[b][None, :]) != 0
    return out


# --------------------------------------------------------------------------------------
# Hand classes (the 169 grid cells)
# --------------------------------------------------------------------------------------


def class_label(row: int, col: int) -> str:
    hi_rank = RANKS[12 - min(row, col)]
    lo_rank = RANKS[12 - max(row, col)]
    if row == col:
        return hi_rank + lo_rank
    return hi_rank + lo_rank + ("s" if row < col else "o")


CLASS_LABELS: Final[list[str]] = [class_label(r, c) for r in range(13) for c in range(13)]
CLASS_INDEX: Final[dict[str, int]] = {lab: i for i, lab in enumerate(CLASS_LABELS)}
N_CLASSES: Final = 169


def _combo_class(hi: int, lo: int) -> int:
    rh, rl = hi >> 2, lo >> 2
    if rh == rl:
        row = col = 12 - rh
    elif (hi & 3) == (lo & 3):  # suited -> above diagonal (row = high rank)
        row, col = 12 - max(rh, rl), 12 - min(rh, rl)
    else:  # offsuit -> below diagonal
        row, col = 12 - min(rh, rl), 12 - max(rh, rl)
    return row * 13 + col


COMBO_CLASS: Final = np.array([_combo_class(int(h), int(lo)) for h, lo in COMBOS], dtype=np.int64)
CLASS_COMBOS: Final[list[IntArray]] = [np.flatnonzero(k == COMBO_CLASS) for k in range(169)]
CLASS_NCOMBOS: Final = np.array([len(c) for c in CLASS_COMBOS], dtype=np.int64)


def class_of_label(label: str) -> int:
    """Normalise ``"aks"``/``"KAs"``/``"TT"`` to a class index."""
    t = label.strip()
    if len(t) not in (2, 3):
        raise ValueError(f"bad hand class {label!r}")
    r1, r2 = RANKS.find(t[0].upper()), RANKS.find(t[1].upper())
    if r1 < 0 or r2 < 0:
        raise ValueError(f"bad hand class {label!r}")
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        if len(t) == 3:
            raise ValueError(f"pairs take no suit qualifier: {label!r}")
        return (12 - hi) * 14
    if len(t) == 2:
        raise ValueError(f"non-pair class needs 's' or 'o': {label!r}")
    q = t[2].lower()
    if q == "s":
        return (12 - hi) * 13 + (12 - lo)
    if q == "o":
        return (12 - lo) * 13 + (12 - hi)
    raise ValueError(f"bad suitedness in {label!r}")


def class_ranks(cls: int) -> tuple[int, int, str]:
    """(high rank, low rank, kind) with kind in {'pair','suited','offsuit'}."""
    row, col = divmod(cls, 13)
    hi, lo = 12 - min(row, col), 12 - max(row, col)
    kind = "pair" if row == col else ("suited" if row < col else "offsuit")
    return hi, lo, kind


def combos_per_class(weights: FloatArray) -> FloatArray:
    """Sum combo weights (1326,) into the 169 grid cells."""
    return np.bincount(COMBO_CLASS, weights=weights, minlength=169).astype(np.float64)


def class_to_combo_weights(class_weights: Sequence[float] | FloatArray) -> FloatArray:
    """Broadcast per-class weights (169,) to per-combo weights (1326,)."""
    cw = np.asarray(class_weights, dtype=np.float64)
    return cw[COMBO_CLASS]
