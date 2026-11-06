"""Dealing preflop drill spots.

Which hand to deal is a training-design decision, not a poker one. Dealing uniformly
reproduces real frequencies (most hands are trivial folds) and wastes most reps. The
default ``focus="boundary"`` reweights hands by how *instructive* they are::

    weight(combo) = reach * (base + mixedness + 0.6 * on_boundary)

* ``reach``      -- the hand must be able to arrive at the node (vs 3-bet: in your open range)
* ``mixedness``  -- ``1 - max action frequency``: 0 for pure hands, up to 2/3 for 3-way mixes
* ``on_boundary``-- the cell's dominant action differs from a grid neighbour's, i.e. the
  hand sits at the edge of a range where mistakes actually happen
* ``base``       -- keeps trivial hands in rotation so you also confirm the easy ones

``focus="uniform"`` deals with the true reach distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..poker.cards import (
    CLASS_LABELS,
    COMBO_CLASS,
    COMBOS,
    FloatArray,
    card_to_str,
    combo_to_str,
)
from ..strategies.loader import CompiledSpot, Registry

Focus = Literal["boundary", "uniform"]


@dataclass(frozen=True)
class DealtSpot:
    spot: CompiledSpot
    combo: int
    reason: str  # "new" | "review" | "leak"

    @property
    def cards(self) -> list[str]:
        hi, lo = COMBOS[self.combo]
        return [card_to_str(int(hi)), card_to_str(int(lo))]

    @property
    def hand_class(self) -> str:
        return CLASS_LABELS[int(COMBO_CLASS[self.combo])]

    @property
    def combo_str(self) -> str:
        return combo_to_str(self.combo)


def _boundary_mask(spot: CompiledSpot) -> FloatArray:
    """1.0 for grid cells whose dominant action differs from any 4-neighbour's."""
    combos, freqs, _ = spot.class_view()
    dom = np.where(combos > 0, freqs.argmax(axis=1), -1).reshape(13, 13)
    edge = np.zeros((13, 13))
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        shifted = np.full((13, 13), -2)
        rs = slice(max(dr, 0), 13 + min(dr, 0))
        rd = slice(max(-dr, 0), 13 + min(-dr, 0))
        cs = slice(max(dc, 0), 13 + min(dc, 0))
        cd = slice(max(-dc, 0), 13 + min(-dc, 0))
        shifted[rd, cd] = dom[rs, cs]
        edge = np.maximum(edge, (shifted != -2) & (shifted != dom) & (dom >= 0))
    return edge.ravel()


def hand_weights(spot: CompiledSpot, focus: Focus = "boundary", base: float = 0.03) -> FloatArray:
    if focus == "uniform":
        return spot.reach.copy()
    mixed = 1.0 - spot.freqs.max(axis=1)
    edge = _boundary_mask(spot)[COMBO_CLASS]
    return spot.reach * (base + mixed + 0.6 * edge)


def deal_preflop(
    registry: Registry,
    rng: np.random.Generator,
    format_id: str = "6max_100bb",
    scenarios: list[str] | None = None,
    positions: list[str] | None = None,
    focus: Focus = "boundary",
    forced: tuple[str, str] | None = None,
) -> DealtSpot:
    """Deal a random spot + hand. ``forced=(spot_id, hand_class)`` replays an SRS item."""
    if forced is not None:
        spot = registry.spots[forced[0]]
        cls = CLASS_LABELS.index(forced[1])
        combos = np.flatnonzero((cls == COMBO_CLASS) & (spot.reach > 0))
        if len(combos):
            return DealtSpot(spot, int(rng.choice(combos)), "review")
    pool = [
        s
        for s in registry.list(format_id)
        if (not scenarios or s.file.scenario in scenarios)
        and (not positions or s.file.hero in positions)
        and s.reach.sum() > 0
    ]
    if not pool:
        raise LookupError("no spots match the selected filters")
    spot = pool[int(rng.integers(len(pool)))]
    w = hand_weights(spot, focus)
    combo = int(rng.choice(len(w), p=w / w.sum()))
    return DealtSpot(spot, combo, "new")
