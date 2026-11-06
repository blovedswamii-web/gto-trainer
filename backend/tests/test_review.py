"""Hand-history parsing, decision mapping/grading, and the review store/API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from gto_trainer.api.app import app
from gto_trainer.db.models import Decision, SrsItem
from gto_trainer.db.session import session
from gto_trainer.strategies.loader import get_registry
from gto_trainer.training import review_store
from gto_trainer.training.hand_history import parse_hand, parse_many, split_hands
from gto_trainer.training.review import assign_positions, review_hand

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
PS = (SAMPLES / "pokerstars_sample.txt").read_text()
GG = (SAMPLES / "ggpoker_sample.txt").read_text()


def ps_hand(n: int) -> str:
    return split_hands(PS)[n - 1]


def test_parse_pokerstars() -> None:
    hands, errors = parse_many(PS)
    assert errors == [] and len(hands) == 9
    h = hands[0]
    assert (h.site, h.hand_id, h.sb, h.bb, h.button_seat) == (
        "pokerstars",
        "250000000001",
        0.25,
        0.5,
        3,
    )
    assert h.hero == "Hero" and h.hero_cards == ["Kc", "9d"]
    assert h.net("Hero") == pytest.approx(0.75)  # won the blinds
    h8 = hands[7]
    assert h8.board == ["Qd", "7h", "3s", "2d", "9c"]
    assert h8.actions[-1].allin and h8.net("Hero") == pytest.approx(-50.0)
    assert h8.invested_on("Hero", "preflop") == pytest.approx(50.0)


def test_parse_ggpoker() -> None:
    hands, errors = parse_many(GG)
    assert errors == [] and len(hands) == 3
    h = hands[0]
    assert h.site == "ggpoker" and h.hand_id == "HD100000001"
    assert h.hero == "Hero" and h.hero_cards == ["Ah", "5h"]  # "Dealt to x " lines ignored
    assert h.net("Hero") == pytest.approx(1.5)


def test_positions() -> None:
    pos = assign_positions(parse_hand(ps_hand(1)))
    assert pos == {
        "Villain4": "SB",
        "Villain5": "BB",
        "Villain6": "UTG",
        "Villain1": "HJ",
        "Villain2": "CO",
        "Hero": "BTN",
    }
    # 5-handed (one player sitting out): first to act is named HJ, i.e. 4 players behind.
    txt = ps_hand(1).replace(
        "Seat 6: Villain6 ($50.00 in chips)", "Seat 6: Villain6 ($50.00 in chips) is sitting out"
    )
    txt = txt.replace("Villain6: folds\n", "")
    pos5 = assign_positions(parse_hand(txt))
    assert pos5["Villain1"] == "HJ" and pos5["Hero"] == "BTN" and "Villain6" not in pos5


def _summary(n: int) -> list[tuple[str, str, str | None]]:
    rv = review_hand(parse_hand(ps_hand(n)), get_registry())
    return [(d.street, d.scenario, d.grade.grade if d.grade else None) for d in rv.decisions]


def test_preflop_classification_and_grades() -> None:
    assert _summary(1) == [("preflop", "rfi", "best")]
    assert _summary(2) == [("preflop", "bb_defense", "mistake")]  # folded 96s vs BTN
    assert _summary(5) == [("preflop", "rfi", "best"), ("preflop", "vs_3bet", "mistake")]
    assert _summary(6) == [("preflop", "rfi", "mistake")]  # UTG limp is never in the chart
    assert _summary(7) == [("preflop", "", None)]  # multiway: not scored
    assert _summary(8) == [("preflop", "vs_open", "best"), ("preflop", "vs_4bet", "best")]


def test_postflop_graded_in_isomorphic_solved_spot() -> None:
    rv = review_hand(parse_hand(ps_hand(4)), get_registry())
    flop, turn = rv.decisions[1], rv.decisions[2]
    assert flop.street == "flop" and flop.action == "bet:0.33" and flop.scored
    assert flop.spot_id == "srp_btn_bb_K72r" and flop.node == "check"
    assert any("Ks 7d 2c" in n for n in flop.notes)
    assert turn.street == "turn" and turn.scored and turn.action is not None
    # 75% pot maps to the closest modelled turn size (log distance)
    sizes = [float(a.split(":")[1]) for a in (turn.freqs or {}) if a.startswith("bet:")]
    assert turn.action == f"bet:{min(sizes, key=lambda f: abs(np.log(0.75 / f))):g}"
    assert turn.node == "check/bet:0.33/call/9d/check"  # 9c remapped to the solved 9d
    assert flop.grade is not None and flop.grade.ev_loss is not None
    # BTN vs CO 3-bet pot: no solved spot -> explained, not graded
    rv3 = review_hand(parse_hand(ps_hand(3)), get_registry())
    assert rv3.postflop_note and "no solved spot" in rv3.postflop_note


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (
            lambda t: t.replace(
                "Hold'em No Limit ($0.25/$0.50 USD)",
                "Tournament #1, $5+$0.5 USD Hold'em No Limit - Level I (10/20)",
            ),
            "tournament",
        ),
        (
            lambda t: "\n".join(
                x
                for x in t.split("\n")
                if not any(v in x for v in ("Villain6", "Villain1", "Villain2", "Villain4"))
            ),
            "heads-up",
        ),
    ],
)
def test_unreviewable_hands_are_skipped(edit: object, reason: str) -> None:
    rv = review_hand(parse_hand(edit(ps_hand(1))), get_registry())  # type: ignore[operator]
    assert rv.skip_reason is not None and reason in rv.skip_reason
    assert rv.decisions == []


def test_short_stack_not_scored() -> None:
    txt = ps_hand(1).replace("Seat 3: Hero ($50.00 in chips)", "Seat 3: Hero ($10.00 in chips)")
    rv = review_hand(parse_hand(txt), get_registry())
    assert not rv.decisions[0].scored and "push/fold" in rv.decisions[0].reason


def test_store_dedupes_and_feeds_stats_and_srs() -> None:
    r = review_store.import_text(PS + "\n\n" + GG)
    assert r["parsed"] == 12 and r["imported"] == 12 and r["flagged_decisions"] >= 6
    again = review_store.import_text(PS)
    assert again["imported"] == 0 and again["duplicates"] == 9
    with session() as s:
        rows = s.scalars(select(Decision).where(Decision.mode == "review")).all()
        assert len(rows) == r["scored_decisions"]
        assert all(d.hand_ref is not None for d in rows)
        keys = {it.item_key for it in s.scalars(select(SrsItem)).all()}
    assert any("bb_defense.BB_vs_BTN" in k and "96s" in k for k in keys)  # real mistake -> drill
    devs = review_store.deviations()
    assert devs[0]["grade"] == "mistake"
    assert devs[0]["ev_loss_bb"] is not None  # the costly postflop mistake ranks first
    assert review_store.clear_all() == 12
    assert review_store.list_hands() == []


def test_review_api() -> None:
    c = TestClient(app)
    text = c.get("/api/review/sample").json()["text"]
    assert c.post("/api/review/import", json={"text": text}).json()["imported"] == 12
    assert c.post("/api/review/import", json={"text": "nonsense"}).status_code == 422
    s = c.get("/api/review/summary").json()
    assert s["hands"] == 12 and s["decisions"]["n"] > 0
    flagged = c.get("/api/review/hands", params={"filter": "flagged"}).json()
    d = c.get(f"/api/review/hands/{flagged[0]['id']}").json()
    assert d["timeline"][0]["kind"] == "action" and d["decisions"]
    assert c.get("/api/stats/summary", params={"mode": "review"}).json()["total"]["n"] > 0
    assert c.delete("/api/review/hands").json()["deleted"] == 12
