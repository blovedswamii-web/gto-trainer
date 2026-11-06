"""Chance-node abstraction: suit isomorphism + stratified runout sampling.

Suit isomorphism (lossless)
---------------------------
Relabelling suits does not change poker. For a board ``B``, any suit permutation ``phi`` with
``phi(B) = B`` (the board's *stabiliser*) maps the game after dealing card ``c`` onto the game
after dealing ``phi(c)``, with every hand ``h`` mapped to ``phi(h)``. Cards related this way
form an *orbit*; we solve one representative per orbit and recover the others by permuting
hand indices (the engine's "gathers"). Examples:

* ``Ks7d2c`` (rainbow): only the identity fixes the board -> no reduction (49 turn cards).
* ``As9s3d`` (two-tone): clubs <-> hearts swap is a symmetry -> 36 turn orbits.
* ``9h8h7h`` (monotone): any permutation of the 3 other suits -> 23 turn orbits.

Runout sampling (lossy, the actual abstraction)
-----------------------------------------------
Solving every turn *and* river card in pure Python is too slow for an interactive tool, so
the solver can keep only ``k`` representative orbits per chance node. Orbits are sorted by
(suit relation to the board, rank) and chosen at evenly spaced *card-count quantiles*, so the
sample spans flush-completing cards, overcards, bricks and pairing cards in proportion.
Each kept orbit inherits the probability mass of its stratum. The resulting game is a
slightly different game; exploitability is reported *within* it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

from ..poker.cards import COMBO_INDEX, COMBOS, IntArray

Perm = tuple[int, ...]
SUIT_PERMS: list[Perm] = [tuple(p) for p in permutations(range(4))]


def permute_card(card: int, perm: Perm) -> int:
    return (card >> 2) * 4 + perm[card & 3]


def inverse(perm: Perm) -> Perm:
    inv = [0] * 4
    for s, t in enumerate(perm):
        inv[t] = s
    return tuple(inv)


def stabilizer(board: tuple[int, ...]) -> list[Perm]:
    target = set(board)
    return [p for p in SUIT_PERMS if {permute_card(c, p) for c in board} == target]


@dataclass(frozen=True)
class Orbit:
    rep: int
    members: tuple[tuple[int, Perm], ...]  # (card, perm with perm(rep) == card)

    @property
    def size(self) -> int:
        return len(self.members)


def card_orbits(board: tuple[int, ...], isomorphism: bool = True) -> list[Orbit]:
    group = stabilizer(board) if isomorphism else [(0, 1, 2, 3)]
    used = set(board)
    seen: set[int] = set()
    out: list[Orbit] = []
    for c in range(52):
        if c in used or c in seen:
            continue
        members: dict[int, Perm] = {}
        for p in group:
            members.setdefault(permute_card(c, p), p)
        seen.update(members)
        out.append(Orbit(c, tuple(sorted(members.items()))))
    return out


def _suit_group(card: int, board: tuple[int, ...]) -> int:
    """Sort key: suits with more board cards first (flush-relevant cards cluster together)."""
    counts = [sum(1 for b in board if b & 3 == s) for s in range(4)]
    return -counts[card & 3]


@dataclass(frozen=True)
class SampledOrbit:
    orbit: Orbit
    mass: float  # number of real cards this representative stands for


def sample_orbits(
    board: tuple[int, ...], k: int | None, isomorphism: bool = True
) -> list[SampledOrbit]:
    """All orbits (k=None or k >= #orbits) or k stratified representatives."""
    orbits = card_orbits(board, isomorphism)
    if k is None or k >= len(orbits):
        return [SampledOrbit(o, float(o.size)) for o in orbits]
    orbits.sort(key=lambda o: (_suit_group(o.rep, board), -(o.rep >> 2), o.rep))
    sizes = np.array([o.size for o in orbits], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(sizes)])
    total = edges[-1]
    # Split the card mass into k equal strata and keep the orbit at each stratum's centre.
    picks: dict[int, float] = {}
    for j in range(k):
        lo, hi = total * j / k, total * (j + 1) / k
        centre = (lo + hi) / 2
        idx = int(np.searchsorted(edges, centre, side="right") - 1)
        picks[idx] = picks.get(idx, 0.0) + (hi - lo)
    return [SampledOrbit(orbits[i], m) for i, m in sorted(picks.items())]


@lru_cache(maxsize=4096)
def _perm_combo_map(perm: Perm) -> IntArray:
    """combo id -> combo id after applying ``perm`` to both cards."""
    hi = np.array([permute_card(int(c), perm) for c in COMBOS[:, 0]])
    lo = np.array([permute_card(int(c), perm) for c in COMBOS[:, 1]])
    out: IntArray = COMBO_INDEX[hi, lo]
    return out


def gather_for(perm: Perm, combos: IntArray, index_of: dict[int, int]) -> IntArray:
    """g[h] = index of perm^-1(hand h), so that V_{perm(c)}(h) = V_c[g][h].

    A permutation that fixes the board *set* may swap two board cards (e.g. 3d <-> 3h on
    As9s3d + 3h). A hand holding the turn card can then map onto a hand holding a flop card,
    which is not in the hand list. Such hands are already dead (zeroed by the live mask of
    the chance node that dealt the turn), so mapping them to themselves is safe.
    """
    mapped = _perm_combo_map(inverse(perm))[combos]
    return np.array([index_of.get(int(m), i) for i, m in enumerate(mapped)], dtype=np.int64)
