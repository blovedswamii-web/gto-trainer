"""Batched vector-form CFR for postflop subgames (the production engine).

Why a second engine
-------------------
The reference engine (:mod:`cfr`) builds one Python node per (betting state, board). With
8 turn x 5 river samples that is ~4,800 decision nodes and ~15k node visits per iteration,
and every showdown is an ``n x n`` matrix-vector product. Profiling showed time split
between Python overhead, regret matching on many small arrays and those O(n^2) showdowns.

Key observation: every river subtree on the same betting line has the *same shape*; only
the board differs. So here a node is a **betting state only**, and every array carries a
leading **board axis** ``B``: the flop has ``B = 1``, the turn ``B = #turn cards``, the river
``B = #river boards``. The number of Python-level nodes no longer depends on how many
runouts you keep, and each operation runs on one large array.

Terminals in O(n) per board instead of O(n^2)
---------------------------------------------
* **Fold**: ``sum_j compat(i, j) r_j = total - r(card1 of i) - r(card2 of i) + r(same combo)``,
  where ``r(card c)`` is the opponent reach on combos containing ``c`` (one ``(B, n) @ (n, 52)``
  product). Exact card removal, no compatibility matrix.
* **River showdown**: ``sum_j r_j sign(rank_i - rank_j)`` over compatible ``j``. Opponent combos
  are pre-sorted by rank on every board, so for hand ``i`` the reach of weaker and stronger
  hands are two lookups into a cumulative sum. Card removal is corrected exactly with the
  same trick per card: opponent combos are also grouped by card and sorted by rank inside each
  group, giving the weaker/stronger reach *among the combos that share card c with i*, which
  is subtracted. (The identical combo is never stronger or weaker -- it ties -- so it needs no
  correction.) All index arrays are precomputed once per board.
* **All-in before the river** keeps exact equity matrices (one per flop/turn board).

Chance nodes move between batches: reach goes down by indexing the parent board of every
child board and masking dead hands; values come back through the suit-isomorphism gathers
(see :mod:`abstraction`) and a small weighted scatter matrix ``(B_parent, M)``.

Algorithms, payoffs and update order are identical to the reference engine, which the tests
use as ground truth (same game -> same values and best responses to float precision).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ..poker.cards import COMBOS, card_to_str
from ..poker.equity import showdown_counts
from ..poker.evaluator import evaluate
from . import kernels
from .abstraction import gather_for, sample_orbits
from .cfr import DT, Arr, Idx
from .tree import STREETS, HandSet, TreeConfig

I32 = npt.NDArray[np.int32]
_BIG = np.int64(1 << 25)  # > any hand score, so (card, rank) pack into one sortable key


# ---------------------------------------------------------------------------- batches


@dataclass(eq=False)
class SDIndex:
    """Precomputed lookups for river showdowns of hero hands vs opponent hands (all boards)."""

    order: Idx  # (B, n_o) opponent combos sorted by rank
    lo: Idx  # (B, n_h) #opponent combos strictly weaker than hero hand
    hi: Idx  # (B, n_h) #opponent combos weaker or tied
    e_combo: Idx  # (B, 2 n_o) opponent combo of each (card, rank)-sorted entry
    c_lo: Idx  # (2, B, n_h) per hero card: entries of that card group weaker than hero
    c_hi: Idx
    c_start: Idx  # group bounds of that card inside the entry array
    c_end: Idx


@dataclass(eq=False)
class Batch:
    street: int
    boards: list[tuple[int, ...]]
    live: tuple[Arr, Arr]  # (B, n_p) 1.0 if the hand avoids every board card
    sd: tuple[SDIndex, SDIndex] | None = None  # river: (hero 0 vs 1, hero 1 vs 0)
    eq: Arr | None = None  # (B, n0, n1) all-in equity, P(win) - P(lose), flop/turn

    @property
    def size(self) -> int:
        return len(self.boards)


@dataclass(eq=False)
class Transition:
    parent: Batch
    child: Batch
    pidx: Idx  # (B_c,) parent board of each child board
    k_idx: Idx  # (M,) child board of each aggregation row (one row per orbit member)
    gathers: tuple[Idx, Idx]  # (M, n_p) hand-index permutation per row
    scatter: Arr  # (B_p, M) row weight placed at the parent board


# ---------------------------------------------------------------------------- nodes


@dataclass(eq=False)
class BFold:
    folder: int
    amount: float
    batch: Batch


@dataclass(eq=False)
class BShowdown:
    amount: float
    batch: Batch


@dataclass(eq=False)
class BChance:
    trans: Transition
    child: BNode
    tokens: tuple[str, ...]  # betting tokens of the street just closed (for paths)


@dataclass(eq=False)
class BPlayer:
    player: int
    actions: list[str]
    children: list[BNode]
    batch: Batch
    info: dict[str, Any] = field(default_factory=dict)
    record: bool = False
    regret: Arr = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=DT))
    strat_sum: Arr = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=DT))
    ev: Arr | None = None  # (B, n_actor, A)
    reach: tuple[Arr, Arr] | None = None  # (B, n0), (B, n1)
    br_q: Arr | None = None
    br_ro: Arr | None = None


BNode = BFold | BShowdown | BChance | BPlayer


def iter_bnodes(node: BNode) -> list[BPlayer]:
    """Decision nodes in a deterministic depth-first order (used to store/restore)."""
    out: list[BPlayer] = []
    stack: list[BNode] = [node]
    while stack:
        nd = stack.pop()
        if isinstance(nd, BPlayer):
            out.append(nd)
            stack.extend(reversed(nd.children))
        elif isinstance(nd, BChance):
            stack.append(nd.child)
    return out


@dataclass
class BGame:
    hands: tuple[HandSet, HandSet]
    weights: tuple[Arr, Arr]
    root: BNode
    batches: list[Batch]
    starting_pot: float
    cards: tuple[Idx, Idx]  # (n_p, 2) card ids per hand
    card_mat: tuple[Arr, Arr]  # (n_p, 52) one-hot of each hand's two cards
    same: tuple[Idx, Idx]  # per hand: index of the identical combo in the other list, or n_o

    @property
    def n(self) -> tuple[int, int]:
        return (len(self.hands[0].combos), len(self.hands[1].combos))


# ---------------------------------------------------------------------------- building


def _sd_index(h_cards: Idx, o_cards: Idx, boards: list[tuple[int, ...]]) -> SDIndex:
    nb, nh, no = len(boards), len(h_cards), len(o_cards)
    order = np.zeros((nb, no), np.int64)
    lo = np.zeros((nb, nh), np.int64)
    hi = np.zeros((nb, nh), np.int64)
    e_combo = np.zeros((nb, 2 * no), np.int64)
    c_lo = np.zeros((2, nb, nh), np.int64)
    c_hi = np.zeros((2, nb, nh), np.int64)
    c_start = np.zeros((2, nb, nh), np.int64)
    c_end = np.zeros((2, nb, nh), np.int64)
    for b, board in enumerate(boards):
        brd = np.asarray(board, dtype=np.int64)
        rh = evaluate(np.concatenate([h_cards, np.broadcast_to(brd, (nh, 5))], axis=1))
        ro = evaluate(np.concatenate([o_cards, np.broadcast_to(brd, (no, 5))], axis=1))
        ordr = np.argsort(ro, kind="stable")
        rs = ro[ordr]
        order[b], lo[b] = ordr, np.searchsorted(rs, rh, "left")
        hi[b] = np.searchsorted(rs, rh, "right")
        keys = np.concatenate([o_cards[:, 0] * _BIG + ro, o_cards[:, 1] * _BIG + ro])
        combo = np.concatenate([np.arange(no), np.arange(no)])
        srt = np.argsort(keys, kind="stable")
        ks = keys[srt]
        e_combo[b] = combo[srt]
        for k in range(2):
            c = h_cards[:, k]
            c_lo[k, b] = np.searchsorted(ks, c * _BIG + rh, "left")
            c_hi[k, b] = np.searchsorted(ks, c * _BIG + rh, "right")
            c_start[k, b] = np.searchsorted(ks, c * _BIG, "left")
            c_end[k, b] = np.searchsorted(ks, (c + 1) * _BIG, "left")
    return SDIndex(order, lo, hi, e_combo, c_lo, c_hi, c_start, c_end)


def _live(hands: HandSet, board: tuple[int, ...]) -> Arr:
    cards = COMBOS[hands.combos]
    ok = np.ones(len(cards), dtype=bool)
    for c in board:
        ok &= (cards != c).all(axis=1)
    out: Arr = ok.astype(DT)
    return out


def _equity_matrix(h: tuple[HandSet, HandSet], board: tuple[int, ...]) -> Arr:
    l0, l1 = _live(h[0], board), _live(h[1], board)
    i0, i1 = np.flatnonzero(l0), np.flatnonzero(l1)
    m = np.zeros((len(l0), len(l1)), dtype=DT)
    win, tie, tot = showdown_counts(h[0].combos[i0], h[1].combos[i1], board)
    m[np.ix_(i0, i1)] = np.where(tot > 0, (2 * win + tie - tot) / np.maximum(tot, 1), 0.0)
    return m


def _make_batch(
    street: int,
    boards: list[tuple[int, ...]],
    h: tuple[HandSet, HandSet],
    c: tuple[Idx, Idx],
    need_eq: bool,
) -> Batch:
    live = (np.stack([_live(h[0], b) for b in boards]), np.stack([_live(h[1], b) for b in boards]))
    bt = Batch(street, boards, live)
    if street == 2:
        bt.sd = (_sd_index(c[0], c[1], boards), _sd_index(c[1], c[0], boards))
    elif need_eq:
        bt.eq = np.stack([_equity_matrix(h, b) for b in boards]).astype(DT)
    return bt


def _transition(
    parent: Batch,
    k_cards: int | None,
    h: tuple[HandSet, HandSet],
    c: tuple[Idx, Idx],
    iso: bool,
    need_eq: bool,
) -> Transition:
    child_boards: list[tuple[int, ...]] = []
    pidx: list[int] = []
    rows_k: list[int] = []
    rows_g: tuple[list[Idx], list[Idx]] = ([], [])
    rows_p: list[int] = []
    rows_w: list[float] = []
    for p, board in enumerate(parent.boards):
        n_rem = 52 - len(board)
        sampled = sample_orbits(board, k_cards, iso)
        kept = sum(so.mass for so in sampled)
        for so in sampled:
            k = len(child_boards)
            child_boards.append((*board, so.orbit.rep))
            pidx.append(p)
            # same per-card probability as the reference engine (see tree._street_end)
            w = (so.mass / so.orbit.size) * n_rem / ((n_rem - 4) * kept)
            for _, perm in so.orbit.members:
                rows_k.append(k)
                rows_p.append(p)
                rows_w.append(w)
                for q in (0, 1):
                    rows_g[q].append(gather_for(perm, h[q].combos, h[q].index_of))
    child = _make_batch(parent.street + 1, child_boards, h, c, need_eq)
    scatter = np.zeros((parent.size, len(rows_k)), dtype=DT)
    scatter[rows_p, np.arange(len(rows_k))] = rows_w
    return Transition(
        parent,
        child,
        np.asarray(pidx),
        np.asarray(rows_k),
        (np.stack(rows_g[0]), np.stack(rows_g[1])),
        scatter,
    )


class BTreeBuilder:
    """Same betting tree as :class:`tree.TreeBuilder` (identical semantics and tokens)."""

    def __init__(self, cfg: TreeConfig, board: tuple[int, ...], h: tuple[HandSet, HandSet]):
        self.cfg, self.h = cfg, h
        self.cards = (COMBOS[h[0].combos].astype(np.int64), COMBOS[h[1].combos].astype(np.int64))
        has_allin = any(cfg.sizes(s).allin for s in range(3))
        flop = _make_batch(
            cfg.start_street, [board], h, self.cards, has_allin and cfg.start_street < 2
        )
        self.batches = [flop]
        self.trans: list[Transition] = []
        cur = flop
        for st in range(cfg.start_street, 2):
            k = cfg.turn_cards if st == 0 else cfg.river_cards
            t = _transition(cur, k, h, self.cards, cfg.isomorphism, has_allin and st + 1 < 2)
            self.trans.append(t)
            self.batches.append(t.child)
            cur = t.child
        self.p0half = cfg.starting_pot / 2
        self.n_player_nodes = 0

    def batch(self, street: int) -> Batch:
        return self.batches[street - self.cfg.start_street]

    def build(self) -> BNode:
        return self._decision(self.cfg.start_street, (0.0, 0.0), (0.0, 0.0), 0, 0, False, (), ())

    def _put(
        self, inv: tuple[float, float], sinv: tuple[float, float], p: int, amt: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        i, s = list(inv), list(sinv)
        i[p] += amt
        s[p] += amt
        return (i[0], i[1]), (s[0], s[1])

    def _decision(
        self,
        street: int,
        inv: tuple[float, float],
        sinv: tuple[float, float],
        p: int,
        raises: int,
        checked: bool,
        toks: tuple[str, ...],
        done: tuple[tuple[str, ...], ...],
    ) -> BNode:
        cfg, o = self.cfg.sizes(street), 1 - p
        pot = self.cfg.starting_pot + inv[0] + inv[1]
        to_call = sinv[o] - sinv[p]
        left = self.cfg.stack - inv[p]
        thr = self.cfg.allin_threshold
        bt = self.batch(street)
        acts: list[str] = []
        kids: list[BNode] = []

        def add(tok: str, node: BNode) -> None:
            acts.append(tok)
            kids.append(node)

        if to_call <= 1e-9:
            if checked:
                add("check", self._street_end(street, inv, (*toks, "check"), done))
            else:
                add(
                    "check",
                    self._decision(street, inv, sinv, o, raises, True, (*toks, "check"), done),
                )
            sizes = cfg.bets if (p == 1 or cfg.oop_bets is None) else cfg.oop_bets
            seen: set[float] = set()
            for f in sizes:
                amt = round(f * pot, 4)
                if amt >= thr * left or amt in seen:
                    continue
                seen.add(amt)
                i2, s2 = self._put(inv, sinv, p, amt)
                add(
                    f"bet:{f:g}",
                    self._decision(street, i2, s2, o, raises, checked, (*toks, f"bet:{f:g}"), done),
                )
            if cfg.allin and left > 0:
                i2, s2 = self._put(inv, sinv, p, left)
                add(
                    "allin",
                    self._decision(street, i2, s2, o, raises, checked, (*toks, "allin"), done),
                )
        else:
            add("fold", BFold(p, self.p0half + inv[p], bt))
            i2, s2 = self._put(inv, sinv, p, min(to_call, left))
            if self.cfg.stack - i2[0] <= 1e-9 or self.cfg.stack - i2[1] <= 1e-9:
                add("call", self._showdown(street, i2))
            else:
                add("call", self._street_end(street, i2, (*toks, "call"), done))
            if left > to_call + 1e-9:
                if raises < cfg.max_raises:
                    for m in cfg.raises:
                        extra = m * sinv[o] - sinv[p]
                        if sinv[p] + extra >= thr * (sinv[p] + left):
                            continue
                        i3, s3 = self._put(inv, sinv, p, extra)
                        add(
                            f"raise:{m:g}",
                            self._decision(
                                street,
                                i3,
                                s3,
                                o,
                                raises + 1,
                                checked,
                                (*toks, f"raise:{m:g}"),
                                done,
                            ),
                        )
                if cfg.allin:
                    i3, s3 = self._put(inv, sinv, p, left)
                    add(
                        "allin",
                        self._decision(
                            street, i3, s3, o, raises + 1, checked, (*toks, "allin"), done
                        ),
                    )
        self.n_player_nodes += bt.size
        # lines = betting tokens of each street so far; node_paths() interleaves board cards
        info = {
            "street": STREETS[street],
            "pot": pot,
            "to_call": to_call,
            "stack": left,
            "inv": list(inv),
            "lines": (*done, toks),
            "card_offset": 3 + self.cfg.start_street,
        }
        return BPlayer(p, acts, kids, bt, info=info, record=street < 2)

    def _showdown(self, street: int, inv: tuple[float, float]) -> BShowdown:
        return BShowdown(self.p0half + inv[0], self.batch(street))

    def _street_end(
        self,
        street: int,
        inv: tuple[float, float],
        toks: tuple[str, ...],
        done: tuple[tuple[str, ...], ...],
    ) -> BNode:
        if street == 2:
            return self._showdown(street, inv)
        t = self.trans[street - self.cfg.start_street]
        child = self._decision(street + 1, inv, (0.0, 0.0), 0, 0, False, (), (*done, toks))
        return BChance(t, child, (*[x for line in done for x in line], *toks))


def build_bgame(
    cfg: TreeConfig, board: tuple[int, ...], hands: tuple[HandSet, HandSet]
) -> tuple[BGame, BTreeBuilder]:
    tb = BTreeBuilder(cfg, board, hands)
    root = tb.build()
    n = (len(hands[0].combos), len(hands[1].combos))
    card_mat = []
    for q in (0, 1):
        m = np.zeros((n[q], 52), dtype=DT)
        m[np.arange(n[q]), tb.cards[q][:, 0]] = 1
        m[np.arange(n[q]), tb.cards[q][:, 1]] = 1
        card_mat.append(m)
    same = tuple(
        np.array(
            [hands[1 - q].index_of.get(int(c), n[1 - q]) for c in hands[q].combos], dtype=np.int64
        )
        for q in (0, 1)
    )
    g = BGame(
        hands,
        (hands[0].weights, hands[1].weights),
        root,
        tb.batches,
        cfg.starting_pot,
        tb.cards,
        (card_mat[0], card_mat[1]),
        (same[0], same[1]),
    )
    return g, tb


# ---------------------------------------------------------------------------- solving


def _take(a: Arr, idx: Idx) -> Arr:
    out: Arr = np.take_along_axis(a, idx, axis=1)
    return out


class BSolver:
    """Batched DCFR / CFR+ with the reference engine's update rules."""

    def __init__(
        self,
        game: BGame,
        algo: str = "dcfr",
        alpha: float = 1.5,
        beta: float = 0.0,
        gamma: float = 2.0,
        train: bool = True,
        use_numba: bool = True,
    ) -> None:
        self.g, self.algo = game, algo
        self.numba = use_numba and kernels.HAVE_NUMBA
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.t = 0
        self.player_nodes = iter_bnodes(game.root)
        n = game.n
        for nd in self.player_nodes:
            shape = (nd.batch.size, n[nd.player], len(nd.actions))
            nd.regret = np.zeros(shape if train else (0, 0, 0), dtype=DT)
            nd.strat_sum = np.zeros(shape, dtype=DT)
        self._disc = (1.0, 1.0, 1.0)

    # ------------------------------------------------------------------ terminals
    def compat_sum(self, p: int, ro: Arr) -> Arr:
        """sum_j compat(i, j) * ro[b, j] for every hero hand i of player p, per board."""
        o = 1 - p
        g = self.g
        if self.numba:
            res: Arr = kernels.compat_sum(
                np.ascontiguousarray(ro, dtype=DT), g.cards[p], g.cards[o], g.same[p]
            )
            return res
        cs = ro @ g.card_mat[o]  # (B, 52) opponent reach per card
        ro_ext = np.concatenate([ro, np.zeros((ro.shape[0], 1), DT)], axis=1)
        c = g.cards[p]
        blocked = cs[:, c[:, 0]] + cs[:, c[:, 1]] - ro_ext[:, g.same[p]]
        out: Arr = ro.sum(axis=1, keepdims=True) - blocked
        return out

    def _river_sd(self, ix: SDIndex, ro: Arr) -> Arr:
        nb = ro.shape[0]
        zero = np.zeros((nb, 1), DT)
        cum = np.concatenate([zero, np.cumsum(_take(ro, ix.order), axis=1)], axis=1)
        total = cum[:, -1:]
        out: Arr = (_take(cum, ix.lo) - (total - _take(cum, ix.hi))).astype(DT)
        ce = np.concatenate([zero, np.cumsum(_take(ro, ix.e_combo), axis=1)], axis=1)
        for k in range(2):
            weaker = _take(ce, ix.c_lo[k]) - _take(ce, ix.c_start[k])
            stronger = _take(ce, ix.c_end[k]) - _take(ce, ix.c_hi[k])
            out -= weaker - stronger
        return out

    def _terminal(self, node: BFold | BShowdown, p: int, ro: Arr) -> Arr:
        if isinstance(node, BFold):
            sign = -1.0 if node.folder == p else 1.0
            out: Arr = self.compat_sum(p, ro) * DT(sign * node.amount)
            return out
        bt = node.batch
        if bt.sd is not None:
            ix = bt.sd[p]
            if self.numba:
                k: Arr = kernels.river_showdown(
                    np.ascontiguousarray(ro, dtype=DT),
                    ix.order,
                    ix.lo,
                    ix.hi,
                    ix.e_combo,
                    ix.c_lo,
                    ix.c_hi,
                    ix.c_start,
                    ix.c_end,
                    float(node.amount),
                )
                return k
            res: Arr = self._river_sd(ix, ro) * DT(node.amount)
            return res
        assert bt.eq is not None
        if p == 0:
            v: Arr = np.matmul(bt.eq, ro[:, :, None])[:, :, 0]
        else:
            v = -np.matmul(ro[:, None, :], bt.eq)[:, 0, :]
        out2: Arr = v * DT(node.amount)
        return out2

    def _chance_up(self, tr: Transition, p: int, vc: Arr) -> Arr:
        vl = vc * tr.child.live[p]
        rows = vl[tr.k_idx[:, None], tr.gathers[p]]  # (M, n)
        out: Arr = tr.scatter @ rows
        return out

    def _down(self, tr: Transition, q: int, r: Arr) -> Arr:
        out: Arr = r[tr.pidx] * tr.child.live[q]
        return out

    # ------------------------------------------------------------------ regret matching
    @staticmethod
    def rm(regret: Arr) -> Arr:
        pos = np.maximum(regret, 0.0)
        tot = pos.sum(axis=2, keepdims=True)
        out: Arr = np.where(
            tot > 0, pos / np.where(tot > 0, tot, 1.0), DT(1.0 / regret.shape[2])
        ).astype(DT)
        return out

    @staticmethod
    def avg(nd: BPlayer) -> Arr:
        s = nd.strat_sum
        tot = s.sum(axis=2, keepdims=True)
        out: Arr = np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), DT(1.0 / s.shape[2])).astype(
            DT
        )
        return out

    def _avg(self, nd: BPlayer) -> Arr:
        if self.numba:
            out: Arr = kernels.normalize(nd.strat_sum)
            return out
        return self.avg(nd)

    # ------------------------------------------------------------------ passes
    def _cfr(self, node: BNode, p: int, rp: Arr, ro: Arr) -> Arr:
        if isinstance(node, (BFold, BShowdown)):
            return self._terminal(node, p, ro)
        if isinstance(node, BChance):
            tr, o = node.trans, 1 - p
            vc = self._cfr(node.child, p, self._down(tr, p, rp), self._down(tr, o, ro))
            return self._chance_up(tr, p, vc)
        sigma = kernels.regret_match(node.regret) if self.numba else self.rm(node.regret)
        if node.player != p:
            v = np.zeros_like(rp)
            for a, ch in enumerate(node.children):
                v += self._cfr(ch, p, rp, ro * sigma[:, :, a])
            return v
        q = np.stack(
            [self._cfr(ch, p, rp * sigma[:, :, a], ro) for a, ch in enumerate(node.children)],
            axis=2,
        )
        pos_d, neg_d, strat_d = self._disc
        if self.numba:
            vk: Arr = kernels.dcfr_update(
                node.regret,
                node.strat_sum,
                sigma,
                q,
                np.ascontiguousarray(rp, dtype=DT),
                pos_d,
                neg_d,
                strat_d,
                self.algo == "cfr+",
                float(self.t),
            )
            return vk
        v = (sigma * q).sum(axis=2)
        inst = q - v[:, :, None]
        if self.algo == "cfr+":
            node.regret = np.maximum(node.regret + inst, 0.0).astype(DT)
            node.strat_sum += (rp[:, :, None] * sigma) * DT(self.t)
        else:
            r = node.regret
            node.regret = (np.where(r > 0, r * DT(pos_d), r * DT(neg_d)) + inst).astype(DT)
            node.strat_sum = (node.strat_sum * DT(strat_d) + rp[:, :, None] * sigma).astype(DT)
        out: Arr = v.astype(DT)
        return out

    def iterate(self, n: int = 1) -> None:
        w0, w1 = (w.astype(DT)[None, :] for w in self.g.weights)
        for _ in range(n):
            self.t += 1
            t = float(self.t)
            self._disc = (
                t**self.alpha / (t**self.alpha + 1),
                t**self.beta / (t**self.beta + 1),
                (t / (t + 1)) ** self.gamma,
            )
            for p in (0, 1):
                rp, ro = (w0, w1) if p == 0 else (w1, w0)
                self._cfr(self.g.root, p, rp, ro)

    def _br(self, node: BNode, p: int, ro: Arr, record: bool) -> Arr:
        if isinstance(node, (BFold, BShowdown)):
            return self._terminal(node, p, ro)
        if isinstance(node, BChance):
            tr = node.trans
            vc = self._br(node.child, p, self._down(tr, 1 - p, ro), record)
            return self._chance_up(tr, p, vc)
        if node.player == p:
            vals = np.stack([self._br(ch, p, ro, record) for ch in node.children], axis=2)
            if record:
                node.br_q, node.br_ro = vals, ro.copy()
            out: Arr = vals.max(axis=2)
            return out
        sigma = self._avg(node)
        v: Arr | None = None
        for a, ch in enumerate(node.children):
            x = self._br(ch, p, ro * sigma[:, :, a], record)
            v = x if v is None else v + x
        assert v is not None
        return v

    def normaliser(self) -> float:
        """Total mass of compatible deals: sum_ij w0_i w1_j compat_ij."""
        w0, w1 = self.g.weights
        cs = self.compat_sum(0, w1.astype(DT)[None, :])[0]
        return float(w0.astype(np.float64) @ cs.astype(np.float64))

    def best_response_value(self, p: int, record: bool = False) -> float:
        if record:
            for nd in self.player_nodes:
                if nd.player == p:
                    nd.br_q = nd.br_ro = None
        w = self.g.weights
        v = self._br(self.g.root, p, w[1 - p].astype(DT)[None, :], record)[0]
        return float(w[p].astype(np.float64) @ v.astype(np.float64)) / self.normaliser()

    def exploitability(self) -> tuple[float, float, float]:
        b0, b1 = self.best_response_value(0), self.best_response_value(1)
        return b0, b1, (b0 + b1) / 2

    def _ev(self, node: BNode, r0: Arr, r1: Arr) -> tuple[Arr, Arr]:
        if isinstance(node, (BFold, BShowdown)):
            return self._terminal(node, 0, r1), self._terminal(node, 1, r0)
        if isinstance(node, BChance):
            tr = node.trans
            v0, v1 = self._ev(node.child, self._down(tr, 0, r0), self._down(tr, 1, r1))
            return self._chance_up(tr, 0, v0), self._chance_up(tr, 1, v1)
        sigma = self._avg(node)
        ap = node.player
        vs = [
            self._ev(ch, r0 * sigma[:, :, a], r1)
            if ap == 0
            else self._ev(ch, r0, r1 * sigma[:, :, a])
            for a, ch in enumerate(node.children)
        ]
        q = np.stack([v[ap] for v in vs], axis=2)
        if node.record:
            node.ev = q.astype(DT)
            node.reach = (r0.copy(), r1.copy())
        va = (sigma * q).sum(axis=2).astype(DT)
        vo = np.sum([v[1 - ap] for v in vs], axis=0).astype(DT)
        return (va, vo) if ap == 0 else (vo, va)

    def evaluate(self) -> tuple[float, float]:
        w0, w1 = (w.astype(DT)[None, :] for w in self.g.weights)
        v0, v1 = self._ev(self.g.root, w0, w1)
        z = self.normaliser()
        return (
            float(self.g.weights[0].astype(np.float64) @ v0[0]) / z,
            float(self.g.weights[1].astype(np.float64) @ v1[0]) / z,
        )


# ---------------------------------------------------------------------------- paths


def node_paths(nd: BPlayer) -> list[str]:
    """Concrete path of every board in a node's batch, e.g. 'check/bet:0.33/call/9d/check'."""
    lines: tuple[tuple[str, ...], ...] = nd.info["lines"]
    off: int = nd.info["card_offset"]
    out = []
    for board in nd.batch.boards:
        parts: list[str] = []
        for k, line in enumerate(lines):
            parts.extend(line)
            if k < len(lines) - 1:
                parts.append(card_to_str(board[off + k]))
        out.append("/".join(parts))
    return out


def chance_paths(root: BNode) -> dict[str, list[str]]:
    """Flop line -> turn cards. Only flop->turn chance nodes: their children are stored."""
    out: dict[str, list[str]] = {}
    stack: list[BNode] = [root]
    while stack:
        nd = stack.pop()
        if isinstance(nd, BPlayer):
            stack.extend(nd.children)
        elif isinstance(nd, BChance) and nd.trans.parent.street == 0:
            out["/".join(nd.tokens)] = [card_to_str(b[3]) for b in nd.trans.child.boards]
    return out


ProgressFn = Callable[[str], None]
