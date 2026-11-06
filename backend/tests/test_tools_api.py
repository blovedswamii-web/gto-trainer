"""Equity/odds/range-editor endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gto_trainer.api.app import app
from gto_trainer.training.math_drills import make_question

client = TestClient(app)


def test_equity_endpoint_hand_vs_hand_exact() -> None:
    r = client.post("/api/equity", json={"hero": "AhKh", "villain": "QsQc"}).json()
    assert r["method"] == "exact"
    assert r["equity"] == pytest.approx(0.462, abs=0.004)


def test_equity_endpoint_range_vs_range_mc() -> None:
    r = client.post(
        "/api/equity", json={"hero": "TT+,AK", "villain": "22+,AJs+", "board": "Kd7c2s"}
    ).json()
    assert r["method"] == "exact"  # small ranges on a flop: enumeration is cheap
    r = client.post(
        "/api/equity", json={"hero": "TT+,AK", "villain": "22+,AJs+", "samples": 20000}
    ).json()
    assert r["method"] == "monte_carlo" and 0 < r["std_error"] < 0.01
    assert client.post("/api/equity", json={"hero": "AhKh", "villain": "AhKh"}).status_code == 422
    assert (
        client.post(
            "/api/equity", json={"hero": "AA", "villain": "KK", "board": "Kd7c"}
        ).status_code
        == 422
    )


def test_odds_and_math_drill() -> None:
    o = client.post("/api/odds", json={"pot": 100, "bet": 50}).json()
    assert o["pot_odds"] == pytest.approx(0.25) and o["mdf"] == pytest.approx(2 / 3)
    q = client.get("/api/math/question", params={"kind": "mdf"}).json()
    good = client.post(
        "/api/math/answer", json={**q, "guess_pct": 100 * q["pot"] / (q["pot"] + q["bet"])}
    ).json()
    assert good["grade"] == "best"
    bad = client.post("/api/math/answer", json={**q, "guess_pct": 1.0}).json()
    assert bad["grade"] == "mistake"
    s = client.get("/api/stats/summary", params={"mode": "math"}).json()
    assert s["total"]["n"] == 2
    assert make_question("bluff").answer < 1


def test_compare_and_custom_ranges() -> None:
    weights = [0.0] * 169
    weights[0] = 1.0  # AA only
    r = client.post(
        "/api/ranges/compare",
        json={"class_weights": weights, "spot_id": "6max_100bb.rfi.UTG", "actions": ["raise"]},
    ).json()
    assert r["overlap_combos"] == 6 and r["user_combos"] == 6
    assert r["reference_combos"] > 200
    assert any(lab == "KK" for lab, _ in r["under_included"])
    saved = client.post("/api/ranges/custom", json={"name": "my UTG", "text": "QQ+,AKs"}).json()
    assert saved["notation"] == "QQ+,AKs"
    listed = client.get("/api/ranges/custom").json()
    assert listed[0]["name"] == "my UTG"
    assert client.delete(f"/api/ranges/custom/{saved['id']}").json()["deleted"]
    assert client.get("/api/ranges/custom").json() == []
