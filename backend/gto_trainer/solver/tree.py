"""Heads-up postflop game construction: betting tree, chance nodes and showdown matrices.

Player 0 is out of position (acts first on every street), player 1 is in position.

Bet-size tree (per street, see :class:`StreetSizes`)
-----------------------------------------------------
* no bet faced: check, bet ``f * pot`` for each size (OOP may use a different list --
  ``()`` disables donk-betting), all-in
* facing a bet: fold, call, raise to ``m x`` the bet (up to ``max_raises`` per street), all-in
* any bet or raise >= ``allin_threshold`` of the remaining stack becomes an all-in
* a call that leaves someone all-in before the river ends the hand at an *all-in equity*
  terminal (exact equity over every remaining runout, independent of runout sampling)

Showdown matrices ``E[i, j] = P(win) - P(lose)`` are built per board after the tree is
known, only for the boards that actually occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..poker.cards import COMBOS, IntArray, card_to_str, cards_to_str
from ..poker.equity import showdown_counts
from ..poker.evaluator import evaluate
from .abstraction import gather_for, sample_orbits
from .cfr import DT, Arr, ChanceBranch, ChanceNode, FoldNode, Game, Node, PlayerNode, ShowdownNode

STREETS = ("flop", "turn", "river")


@dataclass(frozen=True)
class StreetSizes:
    bets: tuple[float, ...] = (0.75,)
    oop_bets: tuple[float, ...] | None = None  # None: same as `bets`; (): no leads/donks
    raises: tuple[float, ...] = (3.0,)  # raise-to multiples of the bet faced
    max_raises: int = 1  # non-all-in raises allowed per street
    allin: bool = True


@dataclass(frozen=True)
class TreeConfig:
    starting_pot: float
    stack: float  # effective stack behind at the start of the subgame
    flop: StreetSizes = StreetSizes()
    turn: StreetSizes = StreetSizes()
    river: StreetSizes = StreetSizes()
    allin_threshold: float = 0.67
    turn_cards: int | None = 8  # representative turn orbits (None = all)
    river_cards: int | None = 5  # representative river orbits per turn (None = all)
    start_street: int = 0  # 0 = flop; tests use 1/2 for small turn/river games
    isomorphism: bool = True  # merge suit-isomorphic cards (lossless); off only for testing

    def sizes(self, street: int) -> StreetSizes:
        return (self.flop, self.turn, self.river)[street]


ACTION_LABELS = {"check": "Check", "call": "Call", "fold": "Fold", "allin": "All-in"}


def action_label(a: str) -> str:
    kind, _, size = a.partition(":")
    if kind == "bet":
        return f"Bet {float(size) * 100:.0f}%"
    if kind == "raise":
        return f"Raise {float(size):g}x"
    return ACTION_LABELS.get(a, a)


@dataclass
class _State:
    street: int
    board: tuple[int, ...]
    inv: tuple[float, float]  # chips invested in the subgame
    street_inv: tuple[float, float]
    to_act: int
    raises: int
    checked: bool
    path: tuple[str, ...]


@dataclass
class HandSet:
    combos: IntArray  # combo ids
    weights: Arr
    index_of: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index_of = {int(c): i for i, c in enumerate(self.combos)}

    def live(self, card: int) -> Arr:
        out: Arr = ((COMBOS[self.combos] != card).all(axis=1)).astype(DT)
        return out


class TreeBuilder:
    def __init__(self, cfg: TreeConfig, board: tuple[int, ...], hands: tuple[HandSet, HandSet]):
        self.cfg = cfg
        self.board = board
        self.hands = hands
        self.matrix_boards: dict[str, tuple[str, tuple[int, ...]]] = {}  # key -> (kind, board)
        self.p0half = cfg.starting_pot / 2.0
        self.n_player_nodes = 0

    # ------------------------------------------------------------------ helpers
    def pot(self, s: _State) -> float:
        return self.cfg.starting_pot + s.inv[0] + s.inv[1]

    def stack_left(self, s: _State, p: int) -> float:
        return self.cfg.stack - s.inv[p]

    def _put(self, s: _State, p: int, amount: float, token: str, *, raised: bool) -> _State:
        inv = list(s.inv)
        sinv = list(s.street_inv)
        inv[p] += amount
        sinv[p] += amount
        return _State(
            s.street,
            s.board,
            (inv[0], inv[1]),
            (sinv[0], sinv[1]),
            1 - p,
            s.raises + (1 if raised else 0),
            s.checked,
            (*s.path, token),
        )

    def _showdown(self, s: _State) -> ShowdownNode:
        kind = "sd" if len(s.board) == 5 else "eq"
        key = f"{kind}:{cards_to_str(s.board)}"
        self.matrix_boards[key] = (kind, s.board)
        return ShowdownNode(key, self.p0half + s.inv[0])

    # ------------------------------------------------------------------ recursion
    def build(self) -> Node:
        b0 = self.board
        s = _State(self.cfg.start_street, b0, (0.0, 0.0), (0.0, 0.0), 0, 0, False, ())
        return self._decision(s)

    def _decision(self, s: _State) -> Node:
        p, o = s.to_act, 1 - s.to_act
        cfg = self.cfg.sizes(s.street)
        to_call = s.street_inv[o] - s.street_inv[p]
        left = self.stack_left(s, p)
        thr = self.cfg.allin_threshold
        actions: list[str] = []
        children: list[Node] = []

        def add(token: str, child: Node) -> None:
            actions.append(token)
            children.append(child)

        if to_call <= 1e-9:
            # Check: closes the street if the other player already checked.
            if s.checked:
                add("check", self._street_end(self._put(s, p, 0.0, "check", raised=False)))
            else:
                nxt = self._put(s, p, 0.0, "check", raised=False)
                nxt.checked = True
                add("check", self._decision(nxt))
            sizes = cfg.bets if (p == 1 or cfg.oop_bets is None) else cfg.oop_bets
            seen: set[float] = set()
            for f in sizes:
                amt = round(f * self.pot(s), 4)
                if amt >= thr * left or amt in seen:
                    continue
                seen.add(amt)
                add(f"bet:{f:g}", self._decision(self._put(s, p, amt, f"bet:{f:g}", raised=False)))
            if cfg.allin and left > 0:
                add("allin", self._decision(self._put(s, p, left, "allin", raised=False)))
        else:
            add("fold", FoldNode(p, self.p0half + s.inv[p]))
            call_amt = min(to_call, left)
            called = self._put(s, p, call_amt, "call", raised=False)
            if self.stack_left(called, 0) <= 1e-9 or self.stack_left(called, 1) <= 1e-9:
                add("call", self._showdown(called))
            else:
                add("call", self._street_end(called))
            if left > to_call + 1e-9:
                if s.raises < cfg.max_raises:
                    for m in cfg.raises:
                        raise_to = m * s.street_inv[o]
                        extra = raise_to - s.street_inv[p]
                        if s.street_inv[p] + extra >= thr * (s.street_inv[p] + left):
                            continue
                        add(
                            f"raise:{m:g}",
                            self._decision(self._put(s, p, extra, f"raise:{m:g}", raised=True)),
                        )
                if cfg.allin:
                    add("allin", self._decision(self._put(s, p, left, "allin", raised=True)))

        self.n_player_nodes += 1
        info: dict[str, Any] = {
            "street": STREETS[s.street],
            "board": list(s.board),
            "pot": self.pot(s),
            "to_call": to_call,
            "path": "/".join(s.path),
            "stack": left,
            "inv": list(s.inv),
        }
        return PlayerNode(p, actions, children, info=info, record=s.street < 2)

    def _street_end(self, s: _State) -> Node:
        if s.street == 2:
            return self._showdown(s)
        k = self.cfg.turn_cards if s.street == 0 else self.cfg.river_cards
        n_remaining = 52 - len(s.board)
        sampled = sample_orbits(s.board, k, self.cfg.isomorphism)
        n_kept_mass = sum(so.mass for so in sampled)
        branches: list[ChanceBranch] = []
        for so in sampled:
            card = so.orbit.rep
            nxt = _State(
                s.street + 1,
                (*s.board, card),
                s.inv,
                (0.0, 0.0),
                0,
                0,
                False,
                (*s.path, card_to_str(card)),
            )
            # Each real card has probability 1/(n_remaining - 4) given both hands; a kept
            # orbit carries the mass of its stratum, rescaled so the average deal sums to 1.
            weight = (so.mass / so.orbit.size) * n_remaining / ((n_remaining - 4) * n_kept_mass)
            gathers = tuple(
                [gather_for(perm, h.combos, h.index_of) for _, perm in so.orbit.members]
                for h in self.hands
            )
            branches.append(
                ChanceBranch(
                    card=card,
                    node=self._decision(nxt),
                    live=(self.hands[0].live(card), self.hands[1].live(card)),
                    gathers=(gathers[0], gathers[1]),
                    weight=weight,
                )
            )
        return ChanceNode(
            branches,
            unit_weight=1.0,
            info={"street": STREETS[s.street + 1], "path": "/".join(s.path)},
        )


def build_matrices(builder: TreeBuilder) -> dict[str, Arr]:
    """E[i, j] = P(win) - P(lose) for every board used by a showdown terminal."""
    h0, h1 = builder.hands
    c0, c1 = COMBOS[h0.combos], COMBOS[h1.combos]
    compat = ((c0[:, None, :, None] != c1[None, :, None, :]).all(axis=(2, 3))).astype(DT)
    out: dict[str, Arr] = {}
    for key, (kind, board) in builder.matrix_boards.items():
        live0 = np.ones(len(c0), dtype=DT)
        live1 = np.ones(len(c1), dtype=DT)
        for card in board:
            live0 *= h0.live(card)
            live1 *= h1.live(card)
        if kind == "sd":
            b = np.broadcast_to(np.array(board), (len(c0), 5))
            r0 = evaluate(np.concatenate([c0, b], axis=1))
            r1 = evaluate(
                np.concatenate([c1, np.broadcast_to(np.array(board), (len(c1), 5))], axis=1)
            )
            m = np.sign(r0[:, None] - r1[None, :]).astype(DT)
        else:
            idx0, idx1 = np.flatnonzero(live0), np.flatnonzero(live1)
            win, tie, tot = showdown_counts(h0.combos[idx0], h1.combos[idx1], board)
            sub = np.where(tot > 0, (2 * win + tie - tot) / np.maximum(tot, 1), 0.0)
            m = np.zeros((len(c0), len(c1)), dtype=DT)
            m[np.ix_(idx0, idx1)] = sub
        out[key] = (m * compat * live0[:, None] * live1[None, :]).astype(DT)
    out["__compat__"] = compat
    return out


def build_game(
    cfg: TreeConfig, board: tuple[int, ...], hands: tuple[HandSet, HandSet]
) -> tuple[Game, TreeBuilder]:
    tb = TreeBuilder(cfg, board, hands)
    root = tb.build()
    mats = build_matrices(tb)
    compat = mats.pop("__compat__")
    game = Game(
        n=(len(hands[0].combos), len(hands[1].combos)),
        weights=(hands[0].weights, hands[1].weights),
        compat=compat,
        matrices=mats,
        root=root,
        starting_pot=cfg.starting_pot,
    )
    return game, tb
