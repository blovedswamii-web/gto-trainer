"""End-to-end drill flow: deal -> answer -> persisted decision -> spaced repetition."""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from gto_trainer.api.app import app
from gto_trainer.db.models import Decision, SrsItem
from gto_trainer.db.session import session
from gto_trainer.training import srs

client = TestClient(app)


def test_deal_and_answer_records_decision() -> None:
    deal = client.post("/api/drill/preflop/next", json={"seed": 1, "use_srs": False}).json()
    assert len(deal["hero_cards"]) == 2
    action = deal["actions"][0]["id"]
    fb = client.post(
        "/api/drill/preflop/answer",
        json={
            "spot_id": deal["spot"]["id"],
            "combo": deal["combo"],
            "action": action,
            "response_ms": 1234,
        },
    ).json()
    assert fb["grade"]["grade"] in {"best", "good", "minor", "mistake"}
    assert set(fb["combo_freqs"]) == {a["id"] for a in deal["actions"]}
    assert fb["detail"]["summary"]["id"] == deal["spot"]["id"]
    with session() as s:
        assert s.scalar(select(func.count()).select_from(Decision)) == 1


def test_table_state_invariants_every_scenario() -> None:
    seen: set[str] = set()
    for seed in range(200):
        deal = client.post("/api/drill/preflop/next", json={"seed": seed, "use_srs": False}).json()
        sc = deal["spot"]["scenario"]
        t = deal["table"]
        assert sum(x["committed_bb"] for x in t["seats"]) == pytest.approx(t["pot_bb"])
        assert t["pot_bb"] == deal["pot_bb"] and t["to_call_bb"] == deal["to_call_bb"]
        assert sum(x["is_hero"] for x in t["seats"]) == 1
        assert sum(x["is_actor"] for x in t["seats"]) == 1
        assert t["board"] == []
        seen.add(sc)
    assert seen == {"rfi", "vs_open", "bb_defense", "vs_3bet", "vs_4bet"}


def test_btn_vs_co_open_table() -> None:
    d = client.get("/api/spots/6max_100bb.vs_open.BTN_vs_CO").json()
    seats = {x["position"]: x for x in d["table"]["seats"]}
    assert seats["CO"]["committed_bb"] == 2.5 and seats["CO"]["is_villain"]
    assert seats["SB"]["committed_bb"] == 0.5 and seats["BB"]["committed_bb"] == 1.0
    assert seats["UTG"]["folded"] and seats["HJ"]["folded"]
    assert seats["BTN"]["is_hero"] and seats["BTN"]["is_actor"]
    assert d["table"]["pot_bb"] == d["pot_bb"]


def test_mistake_creates_due_review_item() -> None:
    # 72o UTG is a pure fold in the seed chart: raising is a mistake.
    fb = client.post(
        "/api/drill/preflop/answer",
        json={"spot_id": "6max_100bb.rfi.UTG", "combo": "7c2d", "action": "raise"},
    ).json()
    assert fb["grade"]["grade"] == "mistake"
    with session() as s:
        item = s.scalars(select(SrsItem)).one()
        assert item.lapses == 1 and item.interval_min == 2.0
    # Not due yet (2 minutes), so no forced review
    assert srs.due_item("6max_100bb", rng=random.Random(0)) is None
    # Make it due and check it gets served
    with session() as s:
        it = s.scalars(select(SrsItem)).one()
        it.due_at = it.due_at.replace(year=2000)
    rng = random.Random(1)
    served = [srs.due_item("6max_100bb", rng=rng) for _ in range(20)]
    assert ("6max_100bb.rfi.UTG", "72o") in served
    # correct answer schedules it a day out
    client.post(
        "/api/drill/preflop/answer",
        json={"spot_id": "6max_100bb.rfi.UTG", "combo": "7h2s", "action": "fold"},
    )
    with session() as s:
        item = s.scalars(select(SrsItem)).one()
        assert item.reps == 1 and item.interval_min == 24 * 60


def test_bad_inputs() -> None:
    assert (
        client.post(
            "/api/drill/preflop/answer",
            json={"spot_id": "6max_100bb.rfi.UTG", "combo": "7c2d", "action": "call"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/drill/preflop/answer", json={"spot_id": "nope", "combo": "7c2d", "action": "fold"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/drill/preflop/next", json={"scenarios": ["vs_4bet"], "positions": ["UTG"]}
        ).status_code
        == 404
    )
