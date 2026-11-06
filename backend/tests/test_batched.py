"""The batched production engine must compute the same game as the reference engine."""

from __future__ import annotations

import numpy as np
import pytest

from gto_trainer.poker.cards import blocked_combos, parse_cards
from gto_trainer.poker.ranges import parse_range
from gto_trainer.solver import kernels
from gto_trainer.solver.batched import BSolver, build_bgame, node_paths
from gto_trainer.solver.cfr import Solver
from gto_trainer.solver.postflop import walk_player_nodes
from gto_trainer.solver.tree import HandSet, StreetSizes, TreeConfig, build_game

BOARD = tuple(parse_cards("As9s3d"))  # two-tone: exercises suit-isomorphic merging


def _hands(text: str) -> HandSet:
    w = parse_range(text) * ~blocked_combos(BOARD)
    idx = np.flatnonzero(w > 0)
    return HandSet(idx, w[idx].astype(np.float32))


HANDS = (_hands("33,99,A5s,KQs,T9s,QJo,76s,AJo"), _hands("33,AQs,KJs,87s,65s,TT+,A9o"))
CFG = TreeConfig(
    starting_pot=6,
    stack=30,
    flop=StreetSizes(bets=(0.5,), oop_bets=(0.33,), raises=(3.0,)),
    turn=StreetSizes(bets=(0.75,)),
    river=StreetSizes(bets=(0.75,)),
    turn_cards=5,
    river_cards=3,
)


def _pair(use_numba: bool) -> tuple[Solver, BSolver]:
    g_old, _ = build_game(CFG, BOARD, HANDS)
    g_new, _ = build_bgame(CFG, BOARD, HANDS)
    return Solver(g_old), BSolver(g_new, use_numba=use_numba)


@pytest.mark.parametrize("use_numba", [False, True])
def test_same_values_and_best_responses_for_any_strategy(use_numba: bool) -> None:
    if use_numba and not kernels.HAVE_NUMBA:
        pytest.skip("numba not installed")
    old, new = _pair(use_numba)
    concrete = {p: (nd, b) for nd in new.player_nodes for b, p in enumerate(node_paths(nd))}
    old_nodes = walk_player_nodes(old.g.root)
    assert {p for p, _ in old_nodes} == set(concrete)  # identical trees, path for path
    rng = np.random.default_rng(0)
    for path, nd in old_nodes:  # same random (non-equilibrium) strategy in both engines
        x = rng.random(nd.strat_sum.shape).astype(np.float32)
        nd.strat_sum = x
        bn, b = concrete[path]
        bn.strat_sum[b] = x
    assert new.normaliser() == pytest.approx(old.normaliser())
    np.testing.assert_allclose(new.evaluate(), old.evaluate(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(new.exploitability(), old.exploitability(), rtol=1e-5)


@pytest.mark.skipif(not kernels.HAVE_NUMBA, reason="numba not installed")
def test_kernels_match_numpy() -> None:
    g, _ = build_bgame(CFG, BOARD, HANDS)
    fast, slow = BSolver(g, use_numba=True), BSolver(g, use_numba=False)
    river = g.batches[-1]
    assert river.sd is not None
    rng = np.random.default_rng(1)
    for p in (0, 1):
        ro = rng.random((river.size, g.n[1 - p])).astype(np.float32) * river.live[1 - p]
        np.testing.assert_allclose(fast.compat_sum(p, ro), slow.compat_sum(p, ro), rtol=1e-5)
        sd = river.sd[p]
        k = kernels.river_showdown(
            ro, sd.order, sd.lo, sd.hi, sd.e_combo, sd.c_lo, sd.c_hi, sd.c_start, sd.c_end, 1.0
        )
        np.testing.assert_allclose(k, slow._river_sd(sd, ro), rtol=1e-4, atol=1e-4)
    reg = rng.normal(size=(4, 7, 3)).astype(np.float32)
    np.testing.assert_allclose(kernels.regret_match(reg), BSolver.rm(reg), atol=1e-7)


def test_cfr_iterations_track_the_reference() -> None:
    old, new = _pair(use_numba=False)
    old.iterate(2)
    new.iterate(2)
    np.testing.assert_allclose(new.exploitability()[2], old.exploitability()[2], rtol=1e-5)
    np.testing.assert_allclose(new.evaluate(), old.evaluate(), rtol=1e-5, atol=1e-5)
    # Later iterates can split where two actions tie exactly (float summation order decides
    # 0 vs +-1e-7 regret, and regret matching is discontinuous there); both runs must still
    # converge to the same game value.
    old.iterate(148)
    new.iterate(148)
    for s in (old, new):
        assert s.exploitability()[2] / 6 * 100 < 1.0  # < 1% of the pot
    assert new.evaluate()[0] == pytest.approx(old.evaluate()[0], abs=0.02 * 6)
