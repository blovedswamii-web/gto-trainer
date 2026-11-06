"""Hand/range equity: exact enumeration when cheap, vectorised Monte Carlo otherwise.

Ranges are (1326,) weight vectors over combos (see :mod:`gto_trainer.poker.ranges`); a single
hand is just a one-hot range, so one code path serves hand-vs-hand, hand-vs-range and
range-vs-range.

What "equity" means here
------------------------
The joint distribution is ``P(h, v, runout) ∝ w_hero[h] * w_villain[v] * [h, v, board, dead and
runout share no card]``. Hero equity is ``E[win + tie/2]`` under that distribution. Card
removal is therefore exact: villain combos that collide with hero's cards get zero weight in
that matchup, which is what "hand vs range" must mean.

Exact enumeration
-----------------
Enumerate every completion of the board (``R`` runouts), rank every live combo on every
runout once (``(n_hero + n_villain) * R`` evaluations) and then compare all compatible
(hero, villain) pairs runout by runout. Comparison work is ``n_hero * n_villain * R``; we
enumerate when both budgets are small. Examples:

* AhKh vs QsQc preflop: R = C(48,5) = 1,712,304, 3.4M evaluations -> exact (~1 s).
* hand vs range on the flop: R ~ 1000, ~1000 villain combos -> exact (fast).
* range vs range on the flop: ~600*600*990 comparisons -> Monte Carlo.

Monte Carlo
-----------
Draw hero and villain combos by weight, discard draws that share a card (rejection sampling
*is* conditioning on compatibility, so this is unbiased), then complete the board by giving
every remaining card a random key and taking the ``need`` smallest (``argpartition``) --
a vectorised sample without replacement. Standard error is reported from the sample
variance of the {0, 0.5, 1} outcomes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from .cards import (
    COMBO_MASK,
    COMBOS,
    N_COMBOS,
    BoolArray,
    FloatArray,
    IntArray,
    blocked_combos,
    cards_mask,
    combos_conflict,
)
from .evaluator import evaluate

Method = Literal["exact", "monte_carlo"]

#: Upper bounds for choosing exact enumeration (pairwise comparisons, hand evaluations).
EXACT_COMPARISON_BUDGET = 4e7
EXACT_EVAL_BUDGET = 4e6


@dataclass(frozen=True)
class EquityResult:
    equity: float
    win: float
    tie: float
    method: Method
    samples: int  # runouts per matchup (exact) or accepted samples (Monte Carlo)
    std_error: float = 0.0
    matchups: int = 0  # compatible combo pairs considered (exact only)
    hero_combo_equity: dict[int, float] = field(default_factory=dict)

    @property
    def lose(self) -> float:
        return 1.0 - self.win - self.tie


def one_hot(combo: int) -> FloatArray:
    w = np.zeros(N_COMBOS)
    w[combo] = 1.0
    return w


def live_combos(weights: FloatArray, dead: Sequence[int]) -> IntArray:
    """Indices of combos with positive weight that avoid all dead cards."""
    ok = (np.asarray(weights) > 0) & ~blocked_combos(dead)
    return np.flatnonzero(ok)


def enumerate_runouts(deck: Sequence[int], need: int) -> IntArray:
    """All ``need``-card completions from ``deck`` as an (R, need) array."""
    if need == 0:
        return np.zeros((1, 0), dtype=np.int64)
    count = math.comb(len(deck), need)
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(deck, need)),
        dtype=np.int64,
        count=count * need,
    )
    return flat.reshape(count, need)


def _runout_masks(runouts: IntArray) -> npt.NDArray[np.uint64]:
    masks = np.zeros(len(runouts), dtype=np.uint64)
    for j in range(runouts.shape[1]):
        masks |= np.uint64(1) << runouts[:, j].astype(np.uint64)
    return masks


def rank_table(
    combos: IntArray, board: Sequence[int], runouts: IntArray
) -> tuple[IntArray, BoolArray]:
    """Scores of each combo on each runout, plus an ``alive`` mask (combo avoids runout).

    Returns ``(ranks, alive)`` both shaped (n_combos, R). Dead entries hold garbage ranks and
    must be masked by ``alive``.
    """
    n, r = len(combos), len(runouts)
    board_arr = np.asarray(board, dtype=np.int64)
    alive = (COMBO_MASK[combos][:, None] & _runout_masks(runouts)[None, :]) == 0
    ranks = np.empty((n, r), dtype=np.int64)
    rows_per_chunk = max(1, 400_000 // max(r, 1))
    for s in range(0, n, rows_per_chunk):
        block = combos[s : s + rows_per_chunk]
        m = len(block)
        hole = np.repeat(COMBOS[block], r, axis=0)  # (m*R, 2)
        brd = np.broadcast_to(board_arr, (m * r, len(board_arr)))
        run = np.tile(runouts, (m, 1))
        ranks[s : s + m] = evaluate(np.concatenate([hole, brd, run], axis=1)).reshape(m, r)
    return ranks, alive


def showdown_counts(
    hero: IntArray, villain: IntArray, board: Sequence[int], dead: Sequence[int] = ()
) -> tuple[IntArray, IntArray, IntArray]:
    """Exact (win, tie, total) runout counts for every hero x villain combo pair.

    This is also what the CFR solver uses to build all-in equity matrices.
    """
    # Cards held by *every* hero (or every villain) combo can never appear on the board in a
    # compatible matchup, so drop them from the deck too (makes hand-vs-hand ~2x cheaper).
    always = int(np.bitwise_and.reduce(COMBO_MASK[hero])) | int(
        np.bitwise_and.reduce(COMBO_MASK[villain])
    )
    excluded = cards_mask([*board, *dead]) | always
    deck = [c for c in range(52) if not excluded >> c & 1]
    runouts = enumerate_runouts(deck, 5 - len(board))
    rh, ah = rank_table(hero, board, runouts)
    rv, av = rank_table(villain, board, runouts)
    nh, nv = len(hero), len(villain)
    win = np.zeros((nh, nv), dtype=np.int64)
    tie = np.zeros((nh, nv), dtype=np.int64)
    tot = np.zeros((nh, nv), dtype=np.int64)
    step = max(1, int(2e7 // max(1, nh * nv)))
    for s in range(0, len(runouts), step):
        sl = slice(s, s + step)
        both = ah[:, None, sl] & av[None, :, sl]
        a, b = rh[:, None, sl], rv[None, :, sl]
        win += ((a > b) & both).sum(axis=-1)
        tie += ((a == b) & both).sum(axis=-1)
        tot += both.sum(axis=-1)
    # Pairs that share a hole card can never occur.
    conflict = combos_conflict(hero, villain)
    win[conflict] = tie[conflict] = tot[conflict] = 0
    return win, tie, tot


def _choose_method(n_hero: int, n_villain: int, board_len: int, n_deck: int) -> Method:
    runouts = math.comb(n_deck, 5 - board_len)
    if (
        n_hero * n_villain * runouts <= EXACT_COMPARISON_BUDGET
        and (n_hero + n_villain) * runouts <= EXACT_EVAL_BUDGET
    ):
        return "exact"
    return "monte_carlo"


def range_vs_range(
    hero: FloatArray,
    villain: FloatArray,
    board: Sequence[int] = (),
    dead: Sequence[int] = (),
    *,
    samples: int = 200_000,
    method: Method | None = None,
    seed: int | None = None,
    per_combo: bool = False,
) -> EquityResult:
    """Hero's equity (win + tie/2) against villain. See module docstring for semantics."""
    if len(board) > 5:
        raise ValueError("board has more than 5 cards")
    dead_all = [*board, *dead]
    h_idx, v_idx = live_combos(hero, dead_all), live_combos(villain, dead_all)
    if len(h_idx) * len(v_idx) <= 2_000_000:
        # Drop combos that conflict with *every* opposing combo: they never occur, and
        # removing them lets e.g. AsAh vs "AA" collapse to a cheap exact AdAc matchup.
        conflict = combos_conflict(h_idx, v_idx)
        h_idx = h_idx[~conflict.all(axis=1)]
        v_idx = v_idx[~conflict.all(axis=0)]
    if len(h_idx) == 0 or len(v_idx) == 0:
        raise ValueError("no compatible matchups after removing board/dead cards and blockers")
    always = int(np.bitwise_and.reduce(COMBO_MASK[h_idx])) | int(
        np.bitwise_and.reduce(COMBO_MASK[v_idx])
    )
    n_deck = 52 - (cards_mask(dead_all) | always).bit_count()
    chosen = method or _choose_method(len(h_idx), len(v_idx), len(board), n_deck)
    if chosen == "exact":
        return _exact(hero, villain, h_idx, v_idx, board, dead, per_combo)
    return _monte_carlo(hero, villain, h_idx, v_idx, board, dead, samples, seed)


def hand_vs_range(
    hero_combo: int,
    villain: FloatArray,
    board: Sequence[int] = (),
    dead: Sequence[int] = (),
    **kw: object,
) -> EquityResult:
    return range_vs_range(one_hot(hero_combo), villain, board, dead, **kw)  # type: ignore[arg-type]


def hand_vs_hand(
    a: int, b: int, board: Sequence[int] = (), dead: Sequence[int] = (), **kw: object
) -> EquityResult:
    return range_vs_range(one_hot(a), one_hot(b), board, dead, **kw)  # type: ignore[arg-type]


def _exact(
    hero: FloatArray,
    villain: FloatArray,
    h_idx: IntArray,
    v_idx: IntArray,
    board: Sequence[int],
    dead: Sequence[int],
    per_combo: bool,
) -> EquityResult:
    win, tie, tot = showdown_counts(h_idx, v_idx, board, dead)
    valid = tot > 0
    safe = np.where(valid, tot, 1)
    p_win = np.where(valid, win / safe, 0.0)
    p_tie = np.where(valid, tie / safe, 0.0)
    # Joint matchup weight: w_h * w_v for compatible pairs (each pair has the same number of
    # runouts, so weighting matchups equally per runout is the same as weighting by w_h*w_v).
    joint = np.asarray(hero)[h_idx][:, None] * np.asarray(villain)[v_idx][None, :] * valid
    z = joint.sum()
    if z <= 0:
        raise ValueError("no compatible matchups between the two ranges")
    w = float((joint * p_win).sum() / z)
    t = float((joint * p_tie).sum() / z)
    per: dict[int, float] = {}
    if per_combo:
        rows = joint.sum(axis=1)
        eq_rows = (joint * (p_win + 0.5 * p_tie)).sum(axis=1)
        per = {int(h): float(e / r) for h, e, r in zip(h_idx, eq_rows, rows, strict=True) if r > 0}
    return EquityResult(
        equity=w + 0.5 * t,
        win=w,
        tie=t,
        method="exact",
        samples=int(tot[valid].max()) if valid.any() else 0,
        matchups=int(valid.sum()),
        hero_combo_equity=per,
    )


def _monte_carlo(
    hero: FloatArray,
    villain: FloatArray,
    h_idx: IntArray,
    v_idx: IntArray,
    board: Sequence[int],
    dead: Sequence[int],
    samples: int,
    seed: int | None,
) -> EquityResult:
    rng = np.random.default_rng(seed)
    ph = np.asarray(hero)[h_idx] / np.asarray(hero)[h_idx].sum()
    pv = np.asarray(villain)[v_idx] / np.asarray(villain)[v_idx].sum()
    board_arr = np.asarray(board, dtype=np.int64)
    need = 5 - len(board)
    static = np.array(sorted({*board, *dead}), dtype=np.int64)
    outcomes: list[FloatArray] = []
    accepted, attempts = 0, 0
    chunk = 100_000
    while accepted < samples and attempts < 50:
        attempts += 1
        hs = h_idx[rng.choice(len(h_idx), size=chunk, p=ph)]
        vs = v_idx[rng.choice(len(v_idx), size=chunk, p=pv)]
        ok = (COMBO_MASK[hs] & COMBO_MASK[vs]) == 0
        hs, vs = hs[ok], vs[ok]
        m = len(hs)
        if m == 0:
            continue
        hc, vc = COMBOS[hs], COMBOS[vs]
        rows = np.arange(m)[:, None]
        if need:
            keys = rng.random((m, 52))
            keys[:, static] = 2.0
            keys[rows, hc] = 2.0
            keys[rows, vc] = 2.0
            run = np.argpartition(keys, need - 1, axis=1)[:, :need]
        else:
            run = np.zeros((m, 0), dtype=np.int64)
        brd = np.broadcast_to(board_arr, (m, len(board_arr)))
        sh = evaluate(np.concatenate([hc, brd, run], axis=1))
        sv = evaluate(np.concatenate([vc, brd, run], axis=1))
        outcomes.append(np.where(sh > sv, 1.0, np.where(sh == sv, 0.5, 0.0)))
        accepted += m
    if accepted == 0:
        raise ValueError("ranges never produce a compatible matchup")
    out = np.concatenate(outcomes)[:samples]
    n = len(out)
    return EquityResult(
        equity=float(out.mean()),
        win=float((out == 1.0).mean()),
        tie=float((out == 0.5).mean()),
        method="monte_carlo",
        samples=n,
        std_error=float(out.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
    )
