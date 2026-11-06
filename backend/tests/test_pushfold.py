"""ICM equity and the push/fold equilibrium solver."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gto_trainer.api.app import app
from gto_trainer.poker.cards import CLASS_INDEX, CLASS_NCOMBOS
from gto_trainer.poker.icm import bubble_factor, icm_equity
from gto_trainer.solver.pushfold import PushFoldSetup, solve

PRIOR = CLASS_NCOMBOS / CLASS_NCOMBOS.sum()


def brute_icm(stacks: list[float], pay: list[float]) -> np.ndarray:
    n = len(stacks)
    eq = np.zeros(n)
    for perm in itertools.permutations(range(n)):
        p, rem = 1.0, sum(stacks)
        for i in perm:
            p *= stacks[i] / rem
            rem -= stacks[i]
        for k, i in enumerate(perm[: len(pay)]):
            eq[i] += p * pay[k]
    return eq


@pytest.mark.parametrize(
    ("stacks", "pay"),
    [
        ([50, 30, 20], [50, 30, 20]),
        ([10, 20, 30, 40], [50, 30, 20]),
        ([12, 8, 15, 10, 3, 2], [40, 25, 15, 10, 6, 4]),
        ([5, 5, 5, 5, 5], [65, 35]),
    ],
)
def test_icm_matches_brute_force(stacks: list[float], pay: list[float]) -> None:
    np.testing.assert_allclose(icm_equity(stacks, pay), brute_icm(stacks, pay), atol=1e-9)
    assert icm_equity(stacks, pay).sum() == pytest.approx(sum(pay[: len(stacks)]))


def test_icm_properties() -> None:
    assert icm_equity([10, 30], [100]).tolist() == pytest.approx([25, 75])  # winner-take-all
    np.testing.assert_allclose(icm_equity([0, 30, 20], [50, 30, 20])[0], 20)  # bust = last place
    assert bubble_factor([10, 10, 10, 10], [50, 30, 20], 0, 1) > 1.5  # bubble: risk premium
    assert bubble_factor([10, 10], [100], 0, 1) == pytest.approx(1.0)  # WTA heads-up = chips


@pytest.fixture(scope="module")
def hu10() -> object:
    return solve(PushFoldSetup((10.0, 10.0)), iterations=800)


def test_heads_up_nash_10bb(hu10: object) -> None:
    r = hu10
    push, call = r.push[0], r.call[(0, 1)]  # type: ignore[attr-defined]
    # Published HU Nash push/fold at 10bb (no ante): SB shoves ~58%, BB calls ~37%.
    assert 0.55 <= float(PRIOR @ push) <= 0.62
    assert 0.33 <= float(PRIOR @ call) <= 0.42
    assert push[CLASS_INDEX["AA"]] == pytest.approx(1.0) and push[CLASS_INDEX["72o"]] < 0.05
    assert call[CLASS_INDEX["A2o"]] > 0.95 and call[CLASS_INDEX["J4o"]] < 0.05
    assert r.exploitability < 0.01  # type: ignore[attr-defined]  # bb per hand


def test_icm_tightens_calls_and_antes_widen_shoves() -> None:
    stacks = (12.0, 8.0, 15.0, 10.0)
    chip = solve(PushFoldSetup(stacks), iterations=300)
    icm = solve(PushFoldSetup(stacks, (50.0, 30.0, 20.0)), iterations=300)
    bb_vs_co_chip = float(PRIOR @ chip.call[(0, 3)])
    bb_vs_co_icm = float(PRIOR @ icm.call[(0, 3)])
    assert bb_vs_co_icm < bb_vs_co_chip - 0.05  # bubble pressure: call much tighter
    no_ante = solve(PushFoldSetup((10.0,) * 6), iterations=300)
    ante = solve(PushFoldSetup((10.0,) * 6, ante=0.125), iterations=300)
    assert float(PRIOR @ ante.push[0]) > float(PRIOR @ no_ante.push[0])  # more dead money


def test_setup_validation() -> None:
    with pytest.raises(ValueError):
        PushFoldSetup((10.0,))
    with pytest.raises(ValueError):
        PushFoldSetup((0.8, 10.0))
    with pytest.raises(ValueError):
        PushFoldSetup((10.0, 10.0), (50.0, 30.0, 20.0))


def test_pushfold_api_and_drill() -> None:
    c = TestClient(app)
    presets = c.get("/api/pushfold/presets").json()
    p = next(x for x in presets if x["id"] == "sng_bubble")
    body = {"stacks": p["stacks"], "payouts": p["payouts"], "ante": p["ante"]}
    r = c.post("/api/pushfold/solve", json=body).json()
    assert r["icm"] and r["units"] == "% prize pool" and len(r["push"]) == 3
    assert sum(r["equity"]) == pytest.approx(100)
    assert r["bubble_factors"][0][1] > 1
    d = c.post("/api/pushfold/drill/next", json={**body, "seed": 1}).json()
    best = "fold"
    k = r["labels"].index(d["hand_class"])
    rng = (
        r["push"][d["shover"]]
        if d["kind"] == "push"
        else next(
            x
            for x in r["call"]
            if x["shover"] == r["positions"][d["shover"]]
            and x["caller"] == r["positions"][d["caller"]]
        )
    )
    if rng["freqs"][k] > 0.5:
        best = d["kind"]
    a = c.post(
        "/api/pushfold/drill/answer",
        json={
            **body,
            "kind": d["kind"],
            "shover": d["shover"],
            "caller": d["caller"],
            "hand": d["hand"],
            "hand_class": d["hand_class"],
            "action": best,
        },
    ).json()
    assert a["grade"]["grade"] in ("best", "good")
    assert c.get("/api/stats/summary", params={"mode": "pushfold"}).json()["total"]["n"] == 1
    assert c.post("/api/pushfold/solve", json={"stacks": [1, 10]}).status_code == 422
