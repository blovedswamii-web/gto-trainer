"""Range notation: parsing, formatting and combo arithmetic.

A *range* is a ``(1326,)`` float vector of combo weights in [0, 1]. Working at combo level
(not 169 classes) keeps blockers exact: ``AsKs`` can be in a range while ``AhKh`` is not,
and card removal is a mask multiply.

Supported notation (case-insensitive ranks, comma and/or whitespace separated)::

    77          pair                    77+       77 through AA
    99-66       pair run                AKs/AKo   suited/offsuit class
    AK          both suitednesses       ATs+      ATs AJs AQs AKs  ("+" raises the kicker)
    A5s-A2s     kicker run              T9s-54s   connector run (same gap on both ends)
    AhKh        one specific combo      random    every combo (also "any", "100%")
    token:0.5   weight 0.5 (Pio/GTO+ style); ":50" or ":50%" are read as percent

Later tokens override earlier ones, so ``"AQs+, AKs:0.5"`` keeps AKs at 50%.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .cards import (
    CLASS_COMBOS,
    CLASS_LABELS,
    CLASS_NCOMBOS,
    COMBO_CLASS,
    N_COMBOS,
    RANKS,
    FloatArray,
    blocked_combos,
    class_ranks,
    combo_from_str,
    combo_to_str,
)

_COMBO_RE = re.compile(r"^[2-9TJQKA][cdhs][2-9TJQKA][cdhs]$", re.IGNORECASE)
_CLASS_RE = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)$", re.IGNORECASE)


class RangeParseError(ValueError):
    pass


def _rank(ch: str) -> int:
    return int(RANKS.index(ch.upper()))


def _class_idx(hi: int, lo: int, kind: str) -> int:
    if hi == lo:
        return (12 - hi) * 14
    if kind == "s":
        return (12 - hi) * 13 + (12 - lo)
    return (12 - lo) * 13 + (12 - hi)


def _expand_kinds(hi: int, lo: int, kind: str) -> list[int]:
    if hi == lo:
        return [_class_idx(hi, lo, "")]
    kinds = [kind] if kind else ["s", "o"]
    return [_class_idx(hi, lo, k) for k in kinds]


def _parse_class(text: str) -> tuple[int, int, str]:
    m = _CLASS_RE.match(text)
    if not m:
        raise RangeParseError(f"not a hand class: {text!r}")
    r1, r2, kind = _rank(m.group(1)), _rank(m.group(2)), m.group(3).lower()
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo and kind:
        raise RangeParseError(f"pairs take no s/o suffix: {text!r}")
    return hi, lo, kind


def _token_classes(body: str) -> list[int]:
    """Class indices covered by one token body (no weight, not a specific combo)."""
    if "-" in body:
        left, right = (p.strip() for p in body.split("-", 1))
        h1, l1, k1 = _parse_class(left)
        h2, l2, k2 = _parse_class(right)
        if k1 != k2:
            raise RangeParseError(f"mismatched suitedness in {body!r}")
        out: list[int] = []
        if h1 == l1 and h2 == l2:  # pair run 99-66
            for r in range(min(h1, h2), max(h1, h2) + 1):
                out += _expand_kinds(r, r, "")
        elif h1 == h2:  # kicker run A5s-A2s
            for lo in range(min(l1, l2), max(l1, l2) + 1):
                out += _expand_kinds(h1, lo, k1)
        elif h1 - l1 == h2 - l2:  # connector run T9s-54s
            gap = h1 - l1
            for hi in range(min(h1, h2), max(h1, h2) + 1):
                out += _expand_kinds(hi, hi - gap, k1)
        else:
            raise RangeParseError(f"cannot interpret run {body!r}")
        return out
    plus = body.endswith("+")
    hi, lo, kind = _parse_class(body.rstrip("+"))
    if not plus:
        return _expand_kinds(hi, lo, kind)
    if hi == lo:
        return [c for r in range(hi, 13) for c in _expand_kinds(r, r, "")]
    return [c for k in range(lo, hi) for c in _expand_kinds(hi, k, kind)]


def _parse_weight(text: str) -> float:
    t = text.strip().rstrip("%")
    try:
        w = float(t)
    except ValueError as e:
        raise RangeParseError(f"bad weight {text!r}") from e
    if text.strip().endswith("%") or w > 1.0:
        w /= 100.0
    if not 0.0 <= w <= 1.0:
        raise RangeParseError(f"weight out of range: {text!r}")
    return w


def parse_range(text: str) -> FloatArray:
    """Parse range notation into a (1326,) weight vector. Raises :class:`RangeParseError`."""
    weights = np.zeros(N_COMBOS)
    for raw in re.split(r"[,\s]+", text.strip()):
        if not raw:
            continue
        body, _, wtxt = raw.partition(":")
        w = _parse_weight(wtxt) if wtxt else 1.0
        low = body.lower()
        if low in {"random", "any", "all", "100%"}:
            weights[:] = w
        elif _COMBO_RE.match(body):
            try:
                weights[combo_from_str(body)] = w
            except ValueError as e:
                raise RangeParseError(str(e)) from e
        else:
            for cls in _token_classes(body):
                weights[CLASS_COMBOS[cls]] = w
    return weights


def class_weights(weights: FloatArray) -> FloatArray:
    """Average weight per grid cell (169,), i.e. the fraction of that cell's combos included."""
    sums = np.bincount(COMBO_CLASS, weights=weights, minlength=169)
    return sums / CLASS_NCOMBOS


def from_class_weights(cw: Sequence[float] | FloatArray) -> FloatArray:
    return np.asarray(cw, dtype=np.float64)[COMBO_CLASS]


def from_class_dict(d: dict[str, float]) -> FloatArray:
    weights = np.zeros(N_COMBOS)
    for label, w in d.items():
        for cls in _token_classes(label):
            weights[CLASS_COMBOS[cls]] = w
    return weights


def count_combos(weights: FloatArray, dead: Iterable[int] = ()) -> float:
    """Weighted combo count after removing combos that use dead/board cards."""
    return float(np.where(blocked_combos(dead), 0.0, weights).sum())


def range_percent(weights: FloatArray) -> float:
    """Share of all 1326 starting combos (the usual "VPIP-style" % for a chart)."""
    return float(weights.sum() / N_COMBOS)


# --------------------------------------------------------------------------------------
# Formatting (weights -> compact notation)
# --------------------------------------------------------------------------------------


def _runs(values: list[int]) -> list[tuple[int, int]]:
    """Group sorted ints into maximal consecutive runs (lo, hi)."""
    out: list[tuple[int, int]] = []
    for v in sorted(values):
        if out and v == out[-1][1] + 1:
            out[-1] = (out[-1][0], v)
        else:
            out.append((v, v))
    return out


def _fmt_weight(w: float) -> str:
    return "" if w >= 0.9995 else f":{round(w, 3):g}"


def format_range(weights: FloatArray, precision: int = 3) -> str:
    """Compress a range into notation. Exact inverse of :func:`parse_range` up to rounding.

    Cells whose combos carry different weights (e.g. after suit-specific edits) are written
    as individual combos so nothing is lost.
    """
    w = np.round(np.asarray(weights, dtype=np.float64), precision)
    groups: dict[float, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"pair": [], "suited": [], "offsuit": []}
    )
    combo_tokens: list[str] = []
    for cls in range(169):
        cw = w[CLASS_COMBOS[cls]]
        if np.all(cw == 0):
            continue
        if np.all(cw == cw[0]):
            hi, lo, kind = class_ranks(cls)
            groups[float(cw[0])][kind].append((hi, lo))
        else:
            for combo, cwi in zip(CLASS_COMBOS[cls], cw, strict=True):
                if cwi > 0:
                    combo_tokens.append(combo_to_str(int(combo)) + _fmt_weight(float(cwi)))
    tokens: list[str] = []
    for wt in sorted(groups, reverse=True):
        g, suffix = groups[wt], _fmt_weight(wt)
        for lo, hi in reversed(_runs([h for h, _ in g["pair"]])):
            a, b = RANKS[hi] * 2, RANKS[lo] * 2
            tokens.append((f"{b}+" if hi == 12 else (a if lo == hi else f"{a}-{b}")) + suffix)
        for kind, q in (("suited", "s"), ("offsuit", "o")):
            by_high: dict[int, list[int]] = defaultdict(list)
            for h, lo_r in g[kind]:
                by_high[h].append(lo_r)
            for h in sorted(by_high, reverse=True):
                for k_lo, k_hi in reversed(_runs(by_high[h])):
                    H = RANKS[h]
                    if k_hi == h - 1 and k_lo != k_hi:
                        tok = f"{H}{RANKS[k_lo]}{q}+"
                    elif k_lo == k_hi:
                        tok = f"{H}{RANKS[k_lo]}{q}"
                    else:
                        tok = f"{H}{RANKS[k_hi]}{q}-{H}{RANKS[k_lo]}{q}"
                    tokens.append(tok + suffix)
    return ",".join(tokens + combo_tokens)


@dataclass(frozen=True)
class RangeDiff:
    """Comparison of a user range against a reference, at grid-cell level."""

    overlap_combos: float
    user_combos: float
    reference_combos: float
    jaccard: float
    over_included: list[tuple[str, float]]  # (class, user - ref) where user > ref
    under_included: list[tuple[str, float]]  # (class, ref - user) where ref > user
    cell_diff: list[float]  # (169,) user - ref, in fraction of the cell


def compare_ranges(user: FloatArray, reference: FloatArray, tol: float = 0.05) -> RangeDiff:
    """Overlap = sum(min(u, r)) combos; Jaccard = overlap / sum(max(u, r))."""
    overlap = float(np.minimum(user, reference).sum())
    union = float(np.maximum(user, reference).sum())
    cu, cr = class_weights(user), class_weights(reference)
    diff = cu - cr
    order = np.argsort(-np.abs(diff) * CLASS_NCOMBOS)
    over = [(CLASS_LABELS[i], float(diff[i])) for i in order if diff[i] > tol]
    under = [(CLASS_LABELS[i], float(-diff[i])) for i in order if diff[i] < -tol]
    return RangeDiff(
        overlap_combos=overlap,
        user_combos=float(user.sum()),
        reference_combos=float(reference.sum()),
        jaccard=overlap / union if union > 0 else 1.0,
        over_included=over,
        under_included=under,
        cell_diff=[float(x) for x in diff],
    )
