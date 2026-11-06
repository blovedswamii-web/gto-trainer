"""Postflop pipeline end-to-end on a tiny spot: solve -> save -> load -> view -> drill."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gto_trainer.poker.cards import parse_cards
from gto_trainer.solver.categories import classify
from gto_trainer.solver.postflop import SPOTS, load_solution, save_solution, solve_spot
from gto_trainer.solver.tree import StreetSizes
from gto_trainer.training import postflop_view as pv


@pytest.fixture(scope="module")
def solution(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    base = SPOTS["3bp_sb_btn_Q84r"]
    tree = replace(
        base.tree,
        turn_cards=2,
        river_cards=2,
        flop=StreetSizes(bets=(0.5,), oop_bets=(0.5,), raises=(), max_raises=0),
        turn=StreetSizes(bets=(0.75,), raises=(), max_raises=0),
        river=StreetSizes(bets=(0.75,), raises=(), max_raises=0),
    )
    d = replace(base, id="tiny_test_spot", tree=tree)
    s, rep, hands = solve_spot(d, iterations=40, check_every=20, tree=tree)
    assert rep.history[-1][1] < rep.history[0][1]  # exploitability went down
    out = tmp_path_factory.mktemp("solutions")
    save_solution(d, s, rep, hands, out)
    return out, d.id


def test_load_and_view(solution: tuple[Path, str]) -> None:
    out, sid = solution
    sol = load_solution(sid, out)
    assert sol is not None
    root = sol.nodes[""]
    assert root.street == "flop" and root.actor == 0
    np.testing.assert_allclose(root.strategy[root.reach > 0].sum(axis=1), 1.0, atol=1e-2)
    view = pv.node_view(sol, "")
    assert view["actor_position"] == "SB" and view["source"]["type"] == "solver"
    assert view["source"]["exploitability_pct"] is not None
    assert "APPROXIMATE" in view["source"]["notes"]
    assert sum(c["combos"] for c in view["categories"]) == pytest.approx(
        view["range_combos"], rel=1e-3
    )
    assert len(view["grid"]) == 169
    # check -> IP decision node, and check/check -> turn chance with 2 cards
    assert view["children"]["check"]["kind"] == "node"
    ip = pv.node_view(sol, "check")
    assert ip["actor_position"] == "BTN"
    assert ip["children"]["check"]["kind"] == "chance"
    turn_cards = ip["children"]["check"]["cards"]
    assert len(turn_cards) == 2
    turn = pv.node_view(sol, f"check/check/{turn_cards[0]}")
    assert turn["street"] == "turn" and len(turn["board"]) == 4


def test_drill_deal_and_grade(solution: tuple[Path, str]) -> None:
    out, sid = solution
    sol = load_solution(sid, out)
    assert sol is not None
    d = pv.deal([sol], random.Random(3))
    nd = sol.nodes[d.path]
    freqs = pv.combo_detail(sol, d.path, d.combo)["freqs"]
    best = max(freqs, key=freqs.get)
    assert best in nd.actions
    g, extra = pv.grade_and_record(sol, d.path, d.combo, best, 1500)
    assert g.grade == "best"
    assert g.ev_loss is not None and g.ev_loss < 0.5  # top action is ~EV-maximising
    assert extra["category"]


@pytest.mark.parametrize(
    ("hole", "board", "cat"),
    [
        ("KhKd", "Ks7d2c", "Set"),
        ("AsKd", "Ks7d2c", "Top pair, good kicker"),
        ("AhAd", "Ks7d2c", "Overpair"),
        ("7c6c", "Ks7d2c", "Middle pair"),
        ("Ah5h", "9h8h7c", "Open-ender / combo draw"),
        ("JhTh", "9h8h7c", "Straight"),
        ("Jc4s", "9h8h7c", "Gutshot"),
        ("AdKd", "9h8h7c", "Overcards"),
        ("Ah4h", "Ks7h2h", "Flush draw"),
        ("6h5d", "Ks7d2c", "Air"),
    ],
)
def test_categories(hole: str, board: str, cat: str) -> None:
    assert classify(parse_cards(hole), parse_cards(board)) == cat
