"""Kuhn poker expressed in the vector-form engine (the convergence sanity check).

Three cards J < Q < K, each player antes 1 and is dealt one card. Player 0 checks or bets 1;
the classic equilibrium gives player 0 a game value of exactly -1/18. Because the engine is
generic over "private hands + public tree", Kuhn needs no special code: hands are the three
cards, ``compat = 1 - I`` (both players cannot hold the same card) and the showdown matrix is
``sign(rank_i - rank_j)``.
"""

from __future__ import annotations

import numpy as np

from .cfr import DT, FoldNode, Game, PlayerNode, ShowdownNode

KUHN_VALUE = -1.0 / 18.0


def kuhn_game() -> Game:
    ranks = np.arange(3)
    compat = (1.0 - np.eye(3)).astype(DT)
    sd = (np.sign(ranks[:, None] - ranks[None, :]) * compat).astype(DT)
    p0 = 1.0  # starting pot 2 (antes) -> each player "owns" 1
    root = PlayerNode(
        0,
        ["check", "bet"],
        [
            PlayerNode(
                1,
                ["check", "bet"],
                [
                    ShowdownNode("sd", p0 + 0),
                    PlayerNode(
                        0,
                        ["fold", "call"],
                        [FoldNode(0, p0 + 0), ShowdownNode("sd", p0 + 1)],
                        info={"path": "check/bet"},
                    ),
                ],
                info={"path": "check"},
            ),
            PlayerNode(
                1,
                ["fold", "call"],
                [FoldNode(1, p0 + 0), ShowdownNode("sd", p0 + 1)],
                info={"path": "bet"},
            ),
        ],
        info={"path": ""},
    )
    w = np.ones(3, dtype=DT)
    return Game(
        n=(3, 3),
        weights=(w, w.copy()),
        compat=compat,
        matrices={"sd": sd},
        root=root,
        starting_pot=2.0,
    )
