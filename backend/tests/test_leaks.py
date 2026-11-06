"""Leak mining and progress stats over a synthetic decision log."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from gto_trainer.api.app import app
from gto_trainer.training import leaks

client = TestClient(app)
SPOT = "6max_100bb.bb_defense.BB_vs_BTN"


def answer(combo: str, action: str, spot: str = SPOT) -> dict[str, Any]:
    r = client.post(
        "/api/drill/preflop/answer", json={"spot_id": spot, "combo": combo, "action": action}
    )
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def test_overfold_suited_gappers_is_top_leak() -> None:
    # 96s/85s/74s are calls in the seed BB-vs-BTN chart; folding them is a mistake.
    for combo in ["9h6h", "8d5d", "7c4c", "9s6s"]:
        assert answer(combo, "fold")["grade"]["grade"] == "mistake"
    # some correct answers elsewhere, and a single one-off error that should rank lower
    answer("AsAh", "raise")
    answer("KdKc", "raise")
    answer("7d2c", "fold")
    answer("7c2d", "raise", "6max_100bb.rfi.BTN")
    top = leaks.top_leaks()
    assert top[0]["category"] == "suited gappers"
    assert top[0]["chosen"] == "fold" and top[0]["best"] == "call"
    assert "over-fold BB vs BTN open with suited gappers" in str(top[0]["text"])
    assert top[0]["errors"] == 4
    # the single RFI misclick is below the min_errors threshold
    assert all(int(str(lk["errors"])) >= 2 for lk in top)


def test_summary_and_rolling() -> None:
    answer("AsAh", "raise")
    answer("9h6h", "fold")
    s = client.get("/api/stats/summary").json()
    assert s["total"]["n"] == 2
    assert s["total"]["accuracy"] == 0.5
    assert {r["key"] for r in s["by_position"]} == {"BB"}
    assert s["srs"]["items"] == 1
    r = client.get("/api/stats/rolling", params={"window": 2}).json()
    assert [p["accuracy"] for p in r] == [1.0, 0.5]
    assert client.get("/api/stats/recent").json()[0]["hand"] == "9h6h"
