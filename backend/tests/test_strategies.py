"""Strategy loading/compilation and the spot API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gto_trainer.api.app import app
from gto_trainer.config import FORMATS
from gto_trainer.poker.cards import CLASS_INDEX, combo_from_str
from gto_trainer.strategies.context import build_context
from gto_trainer.strategies.loader import StrategyError, compile_spot, get_registry
from gto_trainer.strategies.schemas import SourceTag, SpotFile

SRC = SourceTag(type="approximate", label="test")


def test_seed_pack_loads_cleanly() -> None:
    reg = get_registry()
    assert reg.errors == []
    assert len(reg.spots) >= 20
    assert all(s.source.type == "approximate" for s in reg.spots.values())
    for s in reg.spots.values():
        live = s.reach > 0
        np.testing.assert_allclose(s.freqs[live].sum(axis=1), 1.0)


def test_reach_chains_through_previous_spots() -> None:
    reg = get_registry()
    rfi = reg.spots["6max_100bb.rfi.CO"]
    vs3 = reg.spots["6max_100bb.vs_3bet.CO_vs_BTN"]
    np.testing.assert_allclose(vs3.reach, rfi.action_range("raise"))
    vs4 = reg.spots["6max_100bb.vs_4bet.BTN_vs_CO"]
    np.testing.assert_allclose(
        vs4.reach, reg.spots["6max_100bb.vs_open.BTN_vs_CO"].action_range("raise")
    )


def test_context_sizes_and_pot() -> None:
    fmt = FORMATS["6max_100bb"]
    c = build_context(fmt, "vs_open", "BTN", "CO")
    assert c.pot_bb == 4.0 and c.to_call_bb == 2.5
    assert [a.size_bb for a in c.actions] == [None, 2.5, 7.5]  # IP 3-bet = 3x
    c = build_context(fmt, "bb_defense", "BB", "BTN")
    assert c.actions[2].size_bb == 10.0  # OOP 3-bet = 4x
    c = build_context(fmt, "vs_4bet", "SB", "BTN")
    assert [h.position for h in c.history if h.action != "fold"] == ["BTN", "SB", "BTN"]
    assert c.to_call_bb == 22.0 - 10.0


def test_overlapping_weights_rejected() -> None:
    sf = SpotFile(scenario="rfi", hero="BTN", source=SRC, ranges={"raise": "AA"})
    compile_spot(sf, Path("x"))
    bad = SpotFile(
        scenario="vs_open",
        hero="BTN",
        villain="CO",
        source=SRC,
        ranges={"raise": "AA,KK:0.6", "call": "KK:0.6"},
    )
    with pytest.raises(StrategyError, match="KK"):
        compile_spot(bad, Path("x"))


def test_explicit_strategy_with_combo_override() -> None:
    sf = SpotFile(
        scenario="vs_open",
        hero="BTN",
        villain="CO",
        source=SRC,
        strategy={"AKs": {"raise": 0.6, "call": 0.4}, "AsKs": {"raise": 1.0}},
        ev={"AKs": {"raise": 3.0, "call": 2.5, "fold": 0.0}},
    )
    cs = compile_spot(sf, Path("x"))
    assert cs.combo_freqs(combo_from_str("AhKh"))["raise"] == pytest.approx(0.6)
    assert cs.combo_freqs(combo_from_str("AsKs"))["raise"] == pytest.approx(1.0)
    assert cs.combo_freqs(combo_from_str("7c2d"))["fold"] == 1.0
    _, freqs, ev = cs.class_view()
    k = CLASS_INDEX["AKs"]
    assert freqs[k, cs.action_index("raise")] == pytest.approx(0.7)  # (1 + 3*0.6) / 4
    assert ev is not None and ev[k, cs.action_index("raise")] == pytest.approx(3.0)


def test_api_spots_and_detail() -> None:
    client = TestClient(app)
    spots = client.get("/api/spots", params={"format_id": "6max_100bb"}).json()
    assert len(spots) == len(get_registry().spots)
    d = client.get("/api/spots/6max_100bb.bb_defense.BB_vs_BTN").json()
    assert len(d["grid"]) == 169
    assert d["summary"]["source"]["type"] == "approximate"
    assert {a["id"] for a in d["actions"]} == {"fold", "call", "raise"}
    assert d["stats"]["value_bluff"][0]["action"] == "raise"
    cell = client.get("/api/spots/6max_100bb.rfi.UTG/cell/AKs").json()
    assert len(cell) == 4
    assert client.get("/api/spots/nope").status_code == 404
    r = client.post("/api/ranges/parse", json={"text": "77+, ATs+, KQo"}).json()
    assert r["combos"] == 76 and r["normalized"] == "77+,ATs+,KQo"
    assert client.post("/api/ranges/parse", json={"text": "XYZ"}).status_code == 422
