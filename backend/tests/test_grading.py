"""Mixed-strategy grading and the drill dealer."""

from __future__ import annotations

import numpy as np
import pytest

from gto_trainer.poker.cards import CLASS_INDEX, COMBO_CLASS
from gto_trainer.poker.categories import preflop_category
from gto_trainer.strategies.loader import get_registry
from gto_trainer.training.grading import grade_decision
from gto_trainer.training.spots import deal_preflop, hand_weights


@pytest.mark.parametrize(
    ("freqs", "chosen", "grade"),
    [
        ({"fold": 0.0, "call": 0.35, "raise": 0.65}, "raise", "best"),
        ({"fold": 0.0, "call": 0.35, "raise": 0.65}, "call", "good"),
        ({"fold": 0.05, "call": 0.25, "raise": 0.70}, "fold", "minor"),
        ({"fold": 0.05, "call": 0.25, "raise": 0.70}, "call", "minor"),
        ({"fold": 0.0, "call": 1.0}, "fold", "mistake"),
        ({"fold": 0.02, "call": 0.98}, "fold", "mistake"),
        ({"fold": 0.5, "raise": 0.5}, "fold", "best"),  # a true 50/50 tie is never penalised
        ({"fold": 0.30, "raise": 0.70}, "fold", "good"),  # boundary: >= 30% is fine
    ],
)
def test_grade_thresholds(freqs: dict[str, float], chosen: str, grade: str) -> None:
    assert grade_decision(freqs, chosen).grade == grade


def test_freq_error_and_points() -> None:
    g = grade_decision({"fold": 0.0, "call": 1.0}, "fold")
    assert g.freq_error == pytest.approx(1.0) and g.points == 0.0
    g = grade_decision({"fold": 0.1, "call": 0.6, "raise": 0.3}, "raise")
    assert g.freq_error == pytest.approx(0.3) and g.points == 1.0
    assert grade_decision({"fold": 0.1, "call": 0.9}, "fold").points == 0.5


def test_ev_loss_uses_best_ev_not_most_frequent() -> None:
    g = grade_decision({"call": 0.6, "raise": 0.4}, "call", ev={"call": 1.0, "raise": 1.2})
    assert g.ev_loss == pytest.approx(0.2)
    assert grade_decision({"fold": 1.0, "call": 0.0}, "fold", ev={"fold": 0.0}).ev_loss == 0.0


def test_unknown_action_rejected() -> None:
    with pytest.raises(ValueError):
        grade_decision({"fold": 1.0}, "raise")


def test_dealer_respects_reach_and_filters() -> None:
    reg = get_registry()
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = deal_preflop(reg, rng, scenarios=["vs_3bet"], positions=["CO"])
        assert d.spot.file.scenario == "vs_3bet" and d.spot.file.hero == "CO"
        assert d.spot.reach[d.combo] > 0  # only hands that opened can face a 3-bet
    spot = reg.spots["6max_100bb.rfi.BTN"]
    w = hand_weights(spot, "boundary")
    trivial = w[np.flatnonzero(CLASS_INDEX["AA"] == COMBO_CLASS)].mean()
    edge = w[np.flatnonzero(CLASS_INDEX["K7o"] == COMBO_CLASS)].mean()
    assert edge > trivial  # boundary/mixed hands are dealt more often than AA
    with pytest.raises(LookupError):
        deal_preflop(reg, rng, scenarios=["vs_4bet"], positions=["UTG"])


@pytest.mark.parametrize(
    ("label", "cat"),
    [
        ("AA", "big pairs"),
        ("88", "medium pairs"),
        ("33", "small pairs"),
        ("A5s", "suited aces"),
        ("KQs", "suited broadways"),
        ("76s", "suited connectors"),
        ("T8s", "suited one-gappers"),
        ("96s", "suited gappers"),
        ("K5s", "suited kings/queens"),
        ("AJo", "offsuit broadways"),
        ("A7o", "offsuit aces"),
        ("87o", "offsuit connectors"),
        ("Q4o", "weak offsuit"),
    ],
)
def test_preflop_categories(label: str, cat: str) -> None:
    assert preflop_category(label) == cat
