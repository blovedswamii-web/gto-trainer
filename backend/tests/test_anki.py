"""Anki export: deck contents, .apkg structure, stable note GUIDs, API."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import genanki
import pytest
from fastapi.testclient import TestClient

from gto_trainer.api.app import app
from gto_trainer.export import anki
from gto_trainer.solver.postflop import available_solutions
from gto_trainer.solver.pushfold import PushFoldSetup

client = TestClient(app)


def test_preflop_deck_close_vs_all_and_svg() -> None:
    close = anki.build_preflop(scenarios=["rfi"])
    full = anki.build_preflop(scenarios=["rfi"], hands="all")
    assert 0 < len(close.cards) < len(full.cards)
    assert len(close.media) == 5  # one shared range image per RFI spot
    for svg in close.media.values():
        ElementTree.fromstring(svg)  # well-formed
    c = close.cards[0]
    assert "approximate seed chart" in c.back and "<img src=" in c.back
    assert "gto_trainer" in c.tags and "preflop" in c.tags


def test_apkg_structure_and_stable_guids(tmp_path: Path) -> None:
    b1 = anki.build_preflop(scenarios=["vs_4bet"])
    b2 = anki.build_preflop(scenarios=["vs_4bet"])
    assert [c.key for c in b1.cards] == [c.key for c in b2.cards]  # deterministic keys
    out = anki.write_apkg(b1, tmp_path / "d.apkg")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert "collection.anki2" in names and "media" in names
        media = json.loads(z.read("media"))
        assert sorted(media.values()) == sorted(b1.media)
        z.extract("collection.anki2", tmp_path)
    con = sqlite3.connect(tmp_path / "collection.anki2")
    guids = {g for (g,) in con.execute("select guid from notes")}
    assert guids == {genanki.guid_for(c.key) for c in b1.cards}


def test_pushfold_deck() -> None:
    b = anki.build_pushfold(PushFoldSetup((12.0, 8.0, 15.0, 10.0), (50.0, 30.0, 20.0)))
    assert b.cards and all("Push/fold" in c.front for c in b.cards)
    assert "% of the prize pool" in b.cards[0].back


@pytest.mark.skipif(not available_solutions(), reason="no cached postflop solutions")
def test_postflop_deck() -> None:
    sid = available_solutions()[0]
    b = anki.build_postflop([sid])
    assert b.cards and all(c.key.startswith(f"postflop|{sid}|") for c in b.cards)
    assert "Solver-converged" in b.cards[0].back or "solver" in b.cards[0].back


def test_leaks_deck_groups_mistakes_across_sources() -> None:
    for combo in ("9h6h", "9s6s"):  # same spot + hand class twice -> one card
        client.post(
            "/api/drill/preflop/answer",
            json={"spot_id": "6max_100bb.bb_defense.BB_vs_BTN", "combo": combo, "action": "fold"},
        )
    client.post(
        "/api/drill/preflop/answer",
        json={  # minor deviation (25% play)
            "spot_id": "6max_100bb.rfi.HJ",
            "combo": "QcTd",
            "action": "raise",
        },
    )
    setup = {"stacks": [12, 8, 15, 10], "payouts": [50, 30, 20], "ante": 0}
    r = client.post("/api/pushfold/solve", json=setup).json()
    k = r["labels"].index("72o")
    assert r["push"][0]["freqs"][k] < 0.05
    client.post(
        "/api/pushfold/drill/answer",
        json={
            **setup,
            "kind": "push",
            "shover": 0,
            "caller": None,
            "hand": "7c2d",
            "hand_class": "72o",
            "action": "push",
        },
    )
    b = anki.build_leaks()
    keys = [c.key for c in b.cards]
    assert len(keys) == 3
    leak96 = next(c for c in b.cards if "96s" in c.key)
    assert "Flagged 2×" in leak96.back and "fold ×2" in leak96.back
    assert any(key.startswith("leak|pushfold|") for key in keys)
    assert len(anki.build_leaks(include_minor=False).cards) == 2


def test_export_api() -> None:
    p = client.post("/api/export/anki/preview", json={"kind": "preflop", "scenarios": ["vs_4bet"]})
    body = p.json()
    assert body["cards"] > 0 and body["samples"]
    assert "data:image/svg+xml" in body["samples"][0]["back"]  # media inlined for preview
    d = client.post("/api/export/anki", json={"kind": "preflop", "scenarios": ["vs_4bet"]})
    assert d.status_code == 200 and d.content[:2] == b"PK"
    assert "gto-trainer-preflop.apkg" in d.headers["content-disposition"]
    assert client.post("/api/export/anki", json={"kind": "leaks"}).status_code == 422  # empty
    bad = client.post("/api/export/anki/preview", json={"kind": "pushfold", "stacks": [1]})
    assert bad.status_code == 422
