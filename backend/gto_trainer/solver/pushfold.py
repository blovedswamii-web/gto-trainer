"""Short-stack push/fold equilibrium with ICM (or chip-EV) payoffs.

The game
--------
Everyone posts antes and blinds. Action folds to the first player who either **shoves all
in** or folds; after a shove, each later player in turn either **calls** or folds, and the
first call ends the action (heads-up all-in; *no overcalls* -- the standard simplification
for push/fold charts, and a small effect at these depths). If everyone folds to the BB, the
BB takes the blinds.

Each strategy is a frequency per 169 hand class: ``push[i]`` for the first-in shove of
player ``i`` and ``call[i][j]`` for player ``j`` facing ``i``'s shove.

Payoffs
-------
An outcome is a vector of end-of-hand stacks; the utility of an outcome is either the stacks
themselves (chip EV) or each player's ICM equity of those stacks (tournament $EV, in % of
the prize pool). ICM makes the game non-zero-sum in chips: losing a coin flip costs more
equity than winning it gains -- the *bubble factor* -- which is why ICM ranges are tighter.

Card removal and equities
-------------------------
Hand-vs-range quantities use the 169x169 class table: ``pairs[a, b]`` (compatible combo
pairs) gives the probability an opponent holds class ``b`` when you hold ``a``; ``eq[a, b]``
is the all-in equity. Opponents are treated as independent given your own hand (folded
players' cards are not removed) -- again standard for push/fold tools.

Solving: fictitious play
------------------------
Every iteration computes, for every decision and hand, ``EV(shove or call) - EV(fold)``
against the *current average* strategies (expected utilities of folding include everything
that can still happen behind you), takes the best response, and moves each average
strategy a step ``1/t`` toward it. The averages converge to an equilibrium; we report
convergence as the largest gain any player could still get by deviating, in bb (chip EV)
or % of the prize pool (ICM).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ..poker.cards import CLASS_NCOMBOS
from ..poker.icm import bubble_factor, icm_equity
from ..poker.preflop_equity import load_table

Arr = npt.NDArray[np.float64]

POSITIONS: dict[int, list[str]] = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    9: ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
}


@dataclass(frozen=True)
class PushFoldSetup:
    stacks: tuple[float, ...]  # in bb, preflop action order (first to act ... BB)
    payouts: tuple[float, ...] = ()  # empty = chip EV
    ante: float = 0.0  # per player, in bb (dead money)
    sb: float = 0.5
    bb: float = 1.0

    def __post_init__(self) -> None:
        n = len(self.stacks)
        if not 2 <= n <= 9:
            raise ValueError("2 to 9 players")
        if any(s <= self.bb + self.ante for s in self.stacks):
            raise ValueError("every stack must cover the big blind plus the ante")
        if len(self.payouts) > n or any(p < 0 for p in self.payouts):
            raise ValueError("payouts: at most one per player, non-negative")

    @property
    def icm(self) -> bool:
        return len(self.payouts) > 0

    @property
    def positions(self) -> list[str]:
        return POSITIONS[len(self.stacks)]

    def blinds(self) -> list[float]:
        n = len(self.stacks)
        out = [0.0] * n
        if n == 2:
            out[0], out[1] = self.sb, self.bb  # heads-up: the button posts the small blind
        else:
            out[n - 2], out[n - 1] = self.sb, self.bb
        return out

    def code(self) -> str:
        """Compact, reversible text form (fits the decision log's ``node`` column)."""
        return (
            f"s={','.join(f'{x:g}' for x in self.stacks)};"
            f"p={','.join(f'{x:g}' for x in self.payouts)};a={self.ante:g}"
        )

    @classmethod
    def from_code(cls, code: str) -> PushFoldSetup:
        parts = dict(kv.split("=", 1) for kv in code.split(";"))

        def nums(t: str) -> tuple[float, ...]:
            return tuple(float(x) for x in t.split(",") if x)

        return cls(nums(parts["s"]), nums(parts.get("p", "")), float(parts.get("a", "0")))

    def key(self) -> str:
        blob = json.dumps([self.stacks, self.payouts, self.ante, self.sb, self.bb])
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


@dataclass
class PushFoldResult:
    setup: PushFoldSetup
    push: list[Arr]  # per first-in player (BB excluded): (169,) shove frequency
    call: dict[tuple[int, int], Arr]  # (shover, caller) -> (169,) call frequency
    push_margin: list[Arr] = field(default_factory=list)  # EV(shove) - EV(fold)
    call_margin: dict[tuple[int, int], Arr] = field(default_factory=dict)
    iterations: int = 0
    exploitability: float = 0.0  # largest per-decision gain from deviating (units below)
    warnings: list[str] = field(default_factory=list)
    units: str = "bb"  # "bb" or "% prize pool"
    start_equity: list[float] = field(default_factory=list)


class _Game:
    """Outcome stacks/utilities for one setup (computed once)."""

    def __init__(self, st: PushFoldSetup) -> None:
        self.st = st
        n = len(st.stacks)
        self.n = n
        s = np.asarray(st.stacks, dtype=np.float64)
        blinds = np.asarray(st.blinds())
        self.after_posts = s - st.ante - blinds  # stacks of players who fold
        live = s - st.ante  # live stack incl. blind
        total = float(s.sum())
        self._util_cache: dict[tuple[float, ...], Arr] = {}

        def util(final: Arr) -> Arr:
            k = tuple(np.round(final, 9))
            if k not in self._util_cache:
                self._util_cache[k] = icm_equity(final, st.payouts) if st.icm else final.copy()
            return self._util_cache[k]

        # BB walk
        walk = self.after_posts.copy()
        walk[n - 1] = total - (walk.sum() - walk[n - 1])
        self.u_walk = util(walk)
        # steals and all-ins
        self.u_steal: list[Arr] = []
        self.u_win: dict[tuple[int, int], Arr] = {}  # shover i beats caller j
        self.u_lose: dict[tuple[int, int], Arr] = {}
        for i in range(n - 1):
            st_ = self.after_posts.copy()
            st_[i] = total - (st_.sum() - st_[i])
            self.u_steal.append(util(st_))
            for j in range(i + 1, n):
                eff = min(live[i], live[j])
                base = self.after_posts.copy()
                base[i] = live[i] - eff
                base[j] = live[j] - eff
                pot = total - base.sum()
                w, lo = base.copy(), base.copy()
                w[i] += pot
                lo[j] += pot
                self.u_win[(i, j)] = util(w)
                self.u_lose[(i, j)] = util(lo)


def solve(st: PushFoldSetup, iterations: int = 600) -> PushFoldResult:
    table = load_table()
    if table is None:
        raise FileNotFoundError("preflop equity table missing: run `make precompute`")
    eq, pairs = table
    prior = (CLASS_NCOMBOS / CLASS_NCOMBOS.sum()).astype(np.float64)
    row_tot = pairs.sum(axis=1)  # compatible combos of an opponent given my class
    peq = pairs * eq
    g = _Game(st)
    n = g.n

    push = [np.full(169, 0.5) for _ in range(n - 1)]
    call = {(i, j): np.full(169, 0.5) for i in range(n - 1) for j in range(i + 1, n)}

    def p_act(r: Arr) -> Arr:
        """P(opponent's hand is in range r | my class), per my class."""
        out: Arr = (pairs @ r) / row_tot
        return out

    def eq_vs(r: Arr) -> Arr:
        """My all-in equity vs range r, per my class (0.5 where r is empty for me)."""
        w = pairs @ r
        out: Arr = np.where(w > 1e-12, peq @ r / np.maximum(w, 1e-12), 0.5)
        return out

    def range_vs_range(a: Arr, b: Arr) -> tuple[float, float]:
        """(P(b's holder plays | a's holder shoved), equity of a's range vs b's range)."""
        wa = a * prior
        num = wa @ pairs @ b
        den = wa @ pairs @ np.ones(169)
        p = float(num / den) if den > 1e-12 else 0.0
        e = float((wa @ peq @ b) / num) if num > 1e-12 else 0.5
        return p, e

    def values() -> tuple[list[Arr], dict[tuple[int, int], Arr]]:
        """F[i]: utilities when folded to i; G[(i, j)]: i shoved, folded to j."""
        G: dict[tuple[int, int], Arr] = {}
        for i in range(n - 1):
            G[(i, n)] = g.u_steal[i]
            for j in range(n - 1, i, -1):
                c, e = range_vs_range(push[i], call[(i, j)])
                showdown = e * g.u_win[(i, j)] + (1 - e) * g.u_lose[(i, j)]
                G[(i, j)] = c * showdown + (1 - c) * G[(i, j + 1)]
        F: list[Arr] = [np.zeros(n)] * n
        F[n - 1] = g.u_walk
        for i in range(n - 2, -1, -1):
            pp = float(prior @ push[i])
            F[i] = pp * G[(i, i + 1)] + (1 - pp) * F[i + 1]
        return F, G

    def margins() -> tuple[list[Arr], dict[tuple[int, int], Arr]]:
        F, G = values()
        pm: list[Arr] = []
        for i in range(n - 1):
            reach = np.ones(169)
            ev = np.zeros(169)
            for j in range(i + 1, n):
                cp = p_act(call[(i, j)])
                e = eq_vs(call[(i, j)])
                ev += reach * cp * (e * g.u_win[(i, j)][i] + (1 - e) * g.u_lose[(i, j)][i])
                reach = reach * (1 - cp)
            ev += reach * g.u_steal[i][i]
            pm.append(ev - F[i + 1][i])
        cm: dict[tuple[int, int], Arr] = {}
        for (i, j), _ in call.items():
            e = eq_vs(push[i])  # my equity as the caller vs i's shoving range
            ev_call = e * g.u_lose[(i, j)][j] + (1 - e) * g.u_win[(i, j)][j]
            cm[(i, j)] = ev_call - G[(i, j + 1)][j]
        return pm, cm

    t = 0
    for t in range(1, iterations + 1):
        pm, cm = margins()
        step = 1.0 / t  # t=1 replaces the 0.5 starting guess with the first best response
        for i in range(n - 1):
            push[i] += step * ((pm[i] > 0).astype(float) - push[i])
        for k in call:
            call[k] += step * ((cm[k] > 0).astype(float) - call[k])

    pm, cm = margins()
    # Exploitability: largest expected gain (over hands) any single decision could get by
    # switching to a best response against the current averages.
    gains = [float(prior @ (np.maximum(m, 0) - m * push[i])) for i, m in enumerate(pm)]
    gains += [float(prior @ (np.maximum(m, 0) - m * call[k])) for k, m in cm.items()]
    start = icm_equity(np.asarray(st.stacks), st.payouts) if st.icm else np.asarray(st.stacks)
    warnings = []
    deep = [p for p, x in zip(st.positions, st.stacks, strict=True) if x > 20]
    if deep:
        warnings.append(
            f"{', '.join(deep)} {'has' if len(deep) == 1 else 'have'} more than 20bb: "
            "real strategies there use smaller raises, so shove-or-fold is only "
            "an approximation at that depth"
        )
    return PushFoldResult(
        setup=st,
        push=push,
        call=call,
        push_margin=pm,
        call_margin=cm,
        iterations=t,
        exploitability=max(gains, default=0.0),
        warnings=warnings,
        units="% prize pool" if st.icm else "bb",
        start_equity=[float(x) for x in start],
    )


@lru_cache(maxsize=32)
def solve_cached(st: PushFoldSetup) -> PushFoldResult:
    return solve(st)


def bubble_matrix(st: PushFoldSetup) -> list[list[float | None]]:
    n = len(st.stacks)
    if not st.icm:
        return [[None if i == j else 1.0 for j in range(n)] for i in range(n)]
    return [
        [None if i == j else bubble_factor(st.stacks, st.payouts, i, j) for j in range(n)]
        for i in range(n)
    ]


PRESETS: list[dict[str, object]] = [
    {
        "id": "hu_chip_10",
        "label": "Heads-up, 10bb, chip EV",
        "stacks": [10, 10],
        "payouts": [],
        "ante": 0.0,
    },
    {
        "id": "6max_chip_10",
        "label": "6-handed, 10bb each, chip EV (classic Nash charts)",
        "stacks": [10] * 6,
        "payouts": [],
        "ante": 0.0,
    },
    {
        "id": "sng_bubble",
        "label": "9-player SNG bubble: 4 left, 50/30/20",
        "stacks": [12, 8, 15, 10],
        "payouts": [50, 30, 20],
        "ante": 0.0,
    },
    {
        "id": "sng_6max_itm",
        "label": "6-max SNG, 3 left, 65/35",
        "stacks": [15, 10, 20],
        "payouts": [65, 35],
        "ante": 0.0,
    },
    {
        "id": "mtt_ft",
        "label": "MTT final table, 9 left, with antes",
        "stacks": [14, 9, 18, 11, 16, 7, 20, 12, 10],
        "payouts": [25, 17, 12.5, 10, 8.5, 7.5, 7, 6.5, 6],
        "ante": 0.125,
    },
]
