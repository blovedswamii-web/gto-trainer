"""CFR engine: Kuhn convergence, poker subgame convergence, isomorphism exactness."""

from __future__ import annotations

import numpy as np
import pytest

from gto_trainer.poker.cards import blocked_combos, parse_cards
from gto_trainer.poker.ranges import parse_range
from gto_trainer.solver.abstraction import card_orbits, sample_orbits, stabilizer
from gto_trainer.solver.cfr import PlayerNode, Solver, average_strategy
from gto_trainer.solver.kuhn import KUHN_VALUE, kuhn_game
from gto_trainer.solver.tree import HandSet, StreetSizes, TreeConfig, build_game


@pytest.mark.parametrize("algo", ["dcfr", "cfr+"])
def test_kuhn_exploitability_goes_to_zero(algo: str) -> None:
    s = Solver(kuhn_game(), algo=algo)  # type: ignore[arg-type]
    s.iterate(20)
    early = s.exploitability()[2]
    s.iterate(1980)
    late = s.exploitability()[2]
    assert late < early / 20
    assert late < 3e-4
    v0, v1 = s.evaluate()
    assert v0 == pytest.approx(KUHN_VALUE, abs=2e-3)
    assert v1 == pytest.approx(-KUHN_VALUE, abs=2e-3)
    # Equilibrium structure: Q never bets first; K bets 3x as often as J bluffs.
    root = s.g.root
    assert isinstance(root, PlayerNode)
    bet = average_strategy(root)[:, 1]
    assert bet[1] < 0.02
    assert bet[2] == pytest.approx(3 * bet[0], abs=0.05)


def _hands(board: tuple[int, ...], text: str) -> HandSet:
    w = parse_range(text) * ~blocked_combos(board)
    idx = np.flatnonzero(w > 0)
    return HandSet(idx, w[idx].astype(np.float32))


def test_river_subgame_converges() -> None:
    board = tuple(parse_cards("Ks7d2c5h9s"))
    h = (_hands(board, "22+,A2s+,KTo+,QJs,JTs,T9s,98s"), _hands(board, "22+,AJs+,KQs,T8s,64s"))
    cfg = TreeConfig(starting_pot=10, stack=50, river=StreetSizes(bets=(0.5, 1.0)), start_street=2)
    game, _ = build_game(cfg, board, h)
    s = Solver(game)
    s.iterate(30)
    e30 = s.exploitability()[2]
    s.iterate(370)
    e400 = s.exploitability()[2]
    assert e400 < e30
    assert e400 / 10 * 100 < 0.1  # < 0.1% of the pot


def test_stabilizer_and_orbits() -> None:
    assert len(stabilizer(tuple(parse_cards("Ks7d2c")))) == 1
    assert len(stabilizer(tuple(parse_cards("As9s3d")))) == 2
    assert len(stabilizer(tuple(parse_cards("9h8h7h")))) == 6
    assert len(card_orbits(tuple(parse_cards("Ks7d2c")))) == 49
    assert len(card_orbits(tuple(parse_cards("As9s3d")))) == 36
    assert len(card_orbits(tuple(parse_cards("9h8h7h")))) == 23
    sampled = sample_orbits(tuple(parse_cards("Ks7d2c")), 8)
    assert len(sampled) == 8
    assert sum(so.mass for so in sampled) == pytest.approx(49)


def test_isomorphism_is_lossless() -> None:
    """Merging suit-isomorphic river cards must give the same game as dealing them all.

    With a fixed symmetric strategy (uniform: nothing solved yet) the two representations
    must agree to float precision on both the game value and both best-response values.
    After solving, the runs take slightly different paths (regret matching amplifies float32
    rounding) but must agree on the game value.
    """
    board = tuple(parse_cards("As9s3s7d"))  # hearts <-> clubs symmetric
    h = (_hands(board, "TT+,AQs+,KQs,QJs,T9s,65s"), _hands(board, "99+,AJs+,KJs+,JTs,87s"))
    uniform, solved = [], []
    for iso in (True, False):
        cfg = TreeConfig(
            starting_pot=10,
            stack=40,
            turn=StreetSizes(bets=(0.75,)),
            river=StreetSizes(bets=(0.75,)),
            start_street=1,
            river_cards=None,
            isomorphism=iso,
        )
        game, _ = build_game(cfg, board, h)
        s = Solver(game)
        uniform.append((s.evaluate()[0], *s.exploitability()[:2]))
        s.iterate(150)
        solved.append(s.evaluate()[0])
    np.testing.assert_allclose(uniform[0], uniform[1], atol=1e-4)
    assert solved[0] == pytest.approx(solved[1], abs=2e-3)


def test_flop_game_with_board_swapping_symmetries() -> None:
    """Turn cards that pair a flop card in a swappable suit (As9s3d + 3c) once broke gathers."""
    board = tuple(parse_cards("As9s3d"))
    h = (_hands(board, "33,99,A5s,KQs,T9s,QJo"), _hands(board, "33,AQs,KJs,87s,65s"))
    cfg = TreeConfig(
        starting_pot=6,
        stack=30,
        flop=StreetSizes(bets=(0.5,), oop_bets=()),
        turn=StreetSizes(bets=(0.75,)),
        river=StreetSizes(bets=(0.75,)),
        turn_cards=None,
        river_cards=3,
    )
    game, _ = build_game(cfg, board, h)
    s = Solver(game)
    s.iterate(30)
    br0, br1, expl = s.exploitability()
    assert np.isfinite(expl) and expl < 6 * 0.25  # well under a quarter pot after 30 its
    assert br0 + br1 >= -1e-6  # exploitability is never negative
