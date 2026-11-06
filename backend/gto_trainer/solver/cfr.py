"""Vector-form CFR (DCFR or CFR+) for two-player zero-sum games with private hands.

Representation
--------------
The tree contains only *public* states: betting histories and dealt board cards. At every
node each player holds one of ``n_p`` private hands, so instead of one information set per
(node, hand) object we keep **arrays**: a player node stores ``regret`` and ``strat_sum`` of
shape ``(n_actor, n_actions)``, and a traversal passes *reach vectors* down and *value
vectors* up. One recursive pass therefore updates every hand at once, and the expensive work
at terminals is matrix-vector algebra (NumPy/BLAS) rather than Python loops.

Payoffs (zero-sum, chips relative to the subgame start)
-------------------------------------------------------
The starting pot ``P0`` is treated as owned half by each player, so for player ``p``:

* fold terminal:     ``+/- (P0/2 + invested_by_folder)`` -- sign + for the non-folder
* showdown terminal: ``(P0/2 + invested) * E[i, j]``, where ``E[i, j] in [-1, 1]`` is
  ``P(win) - P(lose)`` of hand i vs hand j on that board (exact river comparison, or exact
  all-in equity over remaining runouts), already zeroed for pairs that share a card.

Counterfactual value of hand ``i`` at a terminal is then a matrix-vector product with the
opponent's reach: ``v_p = amount * M_p @ reach_opp``. Card removal is exact because
``compat``/``E`` are zero for incompatible pairs.

Chance nodes deal public cards. Each branch carries a ``live`` mask per player (hands that
do not contain the dealt card) and, for suit isomorphism, a list of index *gathers*: a branch
for a representative card ``c`` stands for every card ``c' = phi(c)`` in its orbit under the
board's suit symmetries, with ``V_{c'}(h) = V_c(phi^-1(h))``. Summing ``V_c[gather]`` over the
orbit gives the exact contribution of all those cards while solving the subtree only once.
Chance probabilities are applied on the way *up* (``unit_weight``); scaling a whole subtree
by a positive constant does not change regret matching, and root values stay exact.

Algorithms (alternating updates; ``t`` = iteration, 1-based)
------------------------------------------------------------
* **DCFR** (Brown & Sandholm 2019, alpha=1.5, beta=0, gamma=2): before adding iteration
  ``t``'s instantaneous regret, positive accumulated regrets are scaled by
  ``t^a/(t^a+1)`` and negative ones by ``t^b/(t^b+1)``; the average-strategy sum is scaled by
  ``(t/(t+1))^g``. Discounting the early, bad iterations makes it converge much faster.
* **CFR+**: regrets floored at zero after every update; linear averaging (weight ``t``).

Exploitability is measured with a best-response pass against the *average* strategy:
``expl = (BR_0 + BR_1) / 2`` in chips (game values cancel because the game is zero-sum).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float32]
Idx = npt.NDArray[np.int64]
Algo = Literal["dcfr", "cfr+"]
DT = np.float32


@dataclass(eq=False)
class FoldNode:
    folder: int
    amount: float  # P0/2 + chips the folder invested in the subgame


@dataclass(eq=False)
class ShowdownNode:
    matrix: str  # key into Game.matrices
    amount: float  # P0/2 + chips each player invested (equal at showdown)


@dataclass(eq=False)
class ChanceBranch:
    card: int
    node: Node
    live: tuple[Arr, Arr]  # per player: 1.0 where the hand avoids `card`
    gathers: tuple[list[Idx], list[Idx]]  # per player: one gather per orbit member
    weight: float = 1.0  # probability of each card in this branch's orbit (given both hands)


@dataclass(eq=False)
class ChanceNode:
    branches: list[ChanceBranch]
    unit_weight: float  # probability of each single dealt card (orbit members included)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(eq=False)
class PlayerNode:
    player: int
    actions: list[str]
    children: list[Node]
    info: dict[str, Any] = field(default_factory=dict)
    record: bool = False  # keep EVs / reach for this node when evaluating
    regret: Arr = field(default_factory=lambda: np.zeros((0, 0), dtype=DT))
    strat_sum: Arr = field(default_factory=lambda: np.zeros((0, 0), dtype=DT))
    ev: Arr | None = None  # (n_actor, A) action values, filled by evaluate()
    reach: tuple[Arr, Arr] | None = None  # own-strategy reach of each player at this node
    br_q: Arr | None = None  # (n_actor, A) best-response action values (best_response_value)
    br_ro: Arr | None = None  # opponent reach seen by that best-response pass


Node = FoldNode | ShowdownNode | ChanceNode | PlayerNode


@dataclass
class Game:
    n: tuple[int, int]
    weights: tuple[Arr, Arr]  # initial range weights of each player's hands
    compat: Arr  # (n0, n1) 1.0 if the two hands share no card
    matrices: dict[str, Arr]  # showdown/equity matrices from player 0's perspective
    root: Node
    starting_pot: float = 1.0


def iter_nodes(node: Node) -> Iterator[Node]:
    stack: list[Node] = [node]
    while stack:
        nd = stack.pop()
        yield nd
        if isinstance(nd, PlayerNode):
            stack.extend(nd.children)
        elif isinstance(nd, ChanceNode):
            stack.extend(b.node for b in nd.branches)


def regret_matching(regret: Arr) -> Arr:
    """Strategy proportional to positive regret; uniform where no action has positive regret."""
    pos = np.maximum(regret, 0.0)
    tot = pos.sum(axis=1, keepdims=True)
    n_act = regret.shape[1]
    out: Arr = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), DT(1.0 / n_act)).astype(DT)
    return out


def average_strategy(node: PlayerNode) -> Arr:
    tot = node.strat_sum.sum(axis=1, keepdims=True)
    n_act = node.strat_sum.shape[1]
    out: Arr = np.where(
        tot > 0, node.strat_sum / np.where(tot > 0, tot, 1.0), DT(1.0 / n_act)
    ).astype(DT)
    return out


class Solver:
    def __init__(
        self,
        game: Game,
        algo: Algo = "dcfr",
        alpha: float = 1.5,
        beta: float = 0.0,
        gamma: float = 2.0,
        train: bool = True,
    ) -> None:
        """``train=False`` skips regret arrays: for evaluating stored strategies only."""
        self.g = game
        self.algo = algo
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.t = 0
        self.player_nodes: list[PlayerNode] = [
            nd for nd in iter_nodes(game.root) if isinstance(nd, PlayerNode)
        ]
        for nd in self.player_nodes:
            shape = (game.n[nd.player], len(nd.actions))
            nd.regret = np.zeros(shape if train else (0, len(nd.actions)), dtype=DT)
            nd.strat_sum = np.zeros(shape, dtype=DT)
        # Player-specific views so every terminal is a single contiguous matvec.
        self._compat = (
            np.ascontiguousarray(game.compat, dtype=DT),
            np.ascontiguousarray(game.compat.T, dtype=DT),
        )
        self._mats: dict[str, tuple[Arr, Arr]] = {
            k: (np.ascontiguousarray(m, dtype=DT), np.ascontiguousarray(-m.T, dtype=DT))
            for k, m in game.matrices.items()
        }
        self._disc = (1.0, 1.0, 1.0)

    # ------------------------------------------------------------------ terminals
    def _terminal(self, node: FoldNode | ShowdownNode, p: int, ro: Arr) -> Arr:
        if isinstance(node, FoldNode):
            sign = -1.0 if node.folder == p else 1.0
            out: Arr = (self._compat[p] @ ro) * DT(sign * node.amount)
            return out
        res: Arr = (self._mats[node.matrix][p] @ ro) * DT(node.amount)
        return res

    def _chance(self, node: ChanceNode, p: int, child: Callable[[ChanceBranch], Arr]) -> Arr:
        out = np.zeros(self.g.n[p], dtype=DT)
        for br in node.branches:
            v = child(br) * (br.live[p] * DT(br.weight))
            for gi in br.gathers[p]:
                out += v[gi]
        res: Arr = out * DT(node.unit_weight)
        return res

    # ------------------------------------------------------------------ CFR pass
    def _cfr(self, node: Node, p: int, rp: Arr, ro: Arr) -> Arr:
        if isinstance(node, (FoldNode, ShowdownNode)):
            return self._terminal(node, p, ro)
        if isinstance(node, ChanceNode):
            o = 1 - p
            return self._chance(
                node, p, lambda br: self._cfr(br.node, p, rp * br.live[p], ro * br.live[o])
            )
        if not ro.any():
            return np.zeros(self.g.n[p], dtype=DT)
        sigma = regret_matching(node.regret)
        if node.player != p:
            v = np.zeros(self.g.n[p], dtype=DT)
            for a, ch in enumerate(node.children):
                v += self._cfr(ch, p, rp, ro * sigma[:, a])
            return v
        q = np.stack(
            [self._cfr(ch, p, rp * sigma[:, a], ro) for a, ch in enumerate(node.children)], axis=1
        )
        v = (sigma * q).sum(axis=1)
        inst = q - v[:, None]
        pos_d, neg_d, strat_d = self._disc
        if self.algo == "cfr+":
            node.regret = np.maximum(node.regret + inst, 0.0).astype(DT)
            node.strat_sum += (rp[:, None] * sigma) * DT(self.t)
        else:
            r = node.regret
            node.regret = (np.where(r > 0, r * DT(pos_d), r * DT(neg_d)) + inst).astype(DT)
            node.strat_sum = (node.strat_sum * DT(strat_d) + rp[:, None] * sigma).astype(DT)
        out: Arr = v.astype(DT)
        return out

    def iterate(self, n: int = 1) -> None:
        w0, w1 = self.g.weights
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
                self._cfr(self.g.root, p, rp.astype(DT), ro.astype(DT))

    # ------------------------------------------------------------------ best response
    def _br(self, node: Node, p: int, ro: Arr, record: bool = False) -> Arr:
        if isinstance(node, (FoldNode, ShowdownNode)):
            return self._terminal(node, p, ro)
        if isinstance(node, ChanceNode):
            o = 1 - p
            return self._chance(node, p, lambda br: self._br(br.node, p, ro * br.live[o], record))
        if not ro.any():
            return np.zeros(self.g.n[p], dtype=DT)
        if node.player == p:
            vals = np.stack([self._br(ch, p, ro, record) for ch in node.children], axis=1)
            if record:
                node.br_q, node.br_ro = vals, ro.copy()
            out: Arr = vals.max(axis=1)
            return out
        sigma = average_strategy(node)
        v = np.zeros(self.g.n[p], dtype=DT)
        for a, ch in enumerate(node.children):
            v += self._br(ch, p, ro * sigma[:, a], record)
        return v

    def best_response_value(self, p: int, record: bool = False) -> float:
        """Expected value (chips per deal) of a best response by ``p`` to the other player's
        average strategy. With ``record``, every node of ``p`` keeps ``br_q``/``br_ro`` so the
        best-response *strategy* can be read off afterwards (argmax of ``br_q``)."""
        if record:
            for nd in self.player_nodes:
                if nd.player == p:
                    nd.br_q = nd.br_ro = None
        w = self.g.weights
        v = self._br(self.g.root, p, w[1 - p].astype(DT), record)
        return float(w[p].astype(np.float64) @ v) / self.normaliser()

    def normaliser(self) -> float:
        """Total probability mass of compatible deals: sum_ij w0_i w1_j compat_ij."""
        w0, w1 = self.g.weights
        return float(w0.astype(np.float64) @ (self.g.compat.astype(np.float64) @ w1))

    def exploitability(self) -> tuple[float, float, float]:
        """(BR value of player 0, BR value of player 1, exploitability), in chips per deal."""
        w0, w1 = self.g.weights
        z = self.normaliser()
        br0 = float(w0.astype(np.float64) @ self._br(self.g.root, 0, w1.astype(DT))) / z
        br1 = float(w1.astype(np.float64) @ self._br(self.g.root, 1, w0.astype(DT))) / z
        return br0, br1, (br0 + br1) / 2.0

    # ------------------------------------------------------------------ evaluation
    def _ev(self, node: Node, r0: Arr, r1: Arr) -> tuple[Arr, Arr]:
        """Values of both players when both play the average strategy; records EVs."""
        if isinstance(node, (FoldNode, ShowdownNode)):
            return self._terminal(node, 0, r1), self._terminal(node, 1, r0)
        if isinstance(node, ChanceNode):
            out0 = np.zeros(self.g.n[0], dtype=DT)
            out1 = np.zeros(self.g.n[1], dtype=DT)
            for br in node.branches:
                v0, v1 = self._ev(br.node, r0 * br.live[0], r1 * br.live[1])
                w = DT(br.weight)
                v0, v1 = v0 * br.live[0] * w, v1 * br.live[1] * w
                for gi in br.gathers[0]:
                    out0 += v0[gi]
                for gi in br.gathers[1]:
                    out1 += v1[gi]
            w = DT(node.unit_weight)
            return out0 * w, out1 * w
        sigma = average_strategy(node)
        a_p = node.player
        vs = []
        for a, ch in enumerate(node.children):
            if a_p == 0:
                vs.append(self._ev(ch, r0 * sigma[:, a], r1))
            else:
                vs.append(self._ev(ch, r0, r1 * sigma[:, a]))
        q_actor = np.stack([v[a_p] for v in vs], axis=1)
        if node.record:
            node.ev = q_actor.astype(DT)
            node.reach = (r0.copy(), r1.copy())
        other = 1 - a_p
        v_actor = (sigma * q_actor).sum(axis=1).astype(DT)
        v_other = np.sum([v[other] for v in vs], axis=0).astype(DT)
        return (v_actor, v_other) if a_p == 0 else (v_other, v_actor)

    def evaluate(self) -> tuple[float, float]:
        """Fill ``ev``/``reach`` on recorded nodes. Returns each player's expected value."""
        w0, w1 = self.g.weights
        v0, v1 = self._ev(self.g.root, w0.astype(DT), w1.astype(DT))
        z = self.normaliser()
        return float(w0 @ v0) / z, float(w1 @ v1) / z
