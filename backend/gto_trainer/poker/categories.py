"""Hand categories used to aggregate decisions into human-sized leaks.

Preflop: the 169 classes grouped by how players actually think about them ("suited gappers",
"offsuit broadways"). Postflop categories live in :mod:`gto_trainer.solver.categories`.
"""

from __future__ import annotations

from .cards import CLASS_INDEX, class_ranks

PREFLOP_CATEGORIES = (
    "big pairs",
    "medium pairs",
    "small pairs",
    "suited aces",
    "suited broadways",
    "suited kings/queens",
    "suited connectors",
    "suited one-gappers",
    "suited gappers",
    "offsuit aces",
    "offsuit broadways",
    "offsuit connectors",
    "weak offsuit",
)

T = 8  # rank index of the Ten (2 = 0 ... A = 12)


def preflop_category(label: str) -> str:
    hi, lo, kind = class_ranks(CLASS_INDEX[label])
    if kind == "pair":
        return "big pairs" if hi >= 10 else ("medium pairs" if hi >= 5 else "small pairs")
    gap = hi - lo - 1
    broadway = lo >= T
    if kind == "suited":
        if broadway:
            return "suited broadways"
        if hi == 12:
            return "suited aces"
        if gap == 0:
            return "suited connectors"
        if gap == 1:
            return "suited one-gappers"
        if hi >= 10:
            return "suited kings/queens"
        return "suited gappers"
    if broadway:
        return "offsuit broadways"
    if hi == 12:
        return "offsuit aces"
    if gap == 0:
        return "offsuit connectors"
    return "weak offsuit"
