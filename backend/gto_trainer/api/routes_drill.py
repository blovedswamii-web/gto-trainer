"""Preflop drill: deal a spot, grade the answer, return full feedback."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..poker.cards import COMBO_CLASS, combo_from_str
from ..poker.preflop_equity import class_equity_vs_range
from ..poker.ranges import class_weights
from ..strategies.loader import get_registry
from ..strategies.schemas import ActionDef, HistoryStep, SpotDetail, SpotSummary
from ..training.grading import GradeResult, grade_decision
from ..training.progress import record_preflop_decision
from ..training.spots import Focus, deal_preflop
from ..training.srs import due_item

router = APIRouter(prefix="/api/drill", tags=["drill"])
_rng = np.random.default_rng()


class PreflopNextIn(BaseModel):
    format_id: str = "6max_100bb"
    scenarios: list[str] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)
    focus: Focus = "boundary"
    use_srs: bool = True
    seed: int | None = None


class PreflopDeal(BaseModel):
    spot: SpotSummary
    history: list[HistoryStep]
    pot_bb: float
    to_call_bb: float
    actions: list[ActionDef]
    hero_cards: list[str]
    hand_class: str
    combo: str
    reason: str


class GradeOut(BaseModel):
    grade: str
    chosen: str
    chosen_freq: float
    best_action: str
    best_freq: float
    freq_error: float
    ev_loss: float | None
    explanation: str
    points: float

    @classmethod
    def of(cls, g: GradeResult) -> GradeOut:
        return cls(
            grade=g.grade,
            chosen=g.chosen,
            chosen_freq=g.chosen_freq,
            best_action=g.best_action,
            best_freq=g.best_freq,
            freq_error=g.freq_error,
            ev_loss=g.ev_loss,
            explanation=g.explanation,
            points=g.points,
        )


class PreflopAnswerIn(BaseModel):
    spot_id: str
    combo: str
    action: str
    response_ms: int | None = None
    reason: str = "new"


class PreflopFeedback(BaseModel):
    grade: GradeOut
    combo_freqs: dict[str, float]
    equity_vs_villain: float | None
    detail: SpotDetail
    decision_id: int | None = None


@router.post("/preflop/next")
def preflop_next(body: PreflopNextIn) -> PreflopDeal:
    reg = get_registry()
    rng = np.random.default_rng(body.seed) if body.seed is not None else _rng
    forced = None
    if body.use_srs:
        item = due_item(body.format_id)
        if item and item[0] in reg.spots:
            sp = reg.spots[item[0]]
            ok = (not body.scenarios or sp.file.scenario in body.scenarios) and (
                not body.positions or sp.file.hero in body.positions
            )
            forced = item if ok else None
    try:
        dealt = deal_preflop(
            reg, rng, body.format_id, body.scenarios, body.positions, body.focus, forced
        )
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    s = dealt.spot
    return PreflopDeal(
        spot=s.summary(),
        history=s.ctx.history,
        pot_bb=s.ctx.pot_bb,
        to_call_bb=s.ctx.to_call_bb,
        actions=s.ctx.actions,
        hero_cards=dealt.cards,
        hand_class=dealt.hand_class,
        combo=dealt.combo_str,
        reason=dealt.reason,
    )


@router.post("/preflop/answer")
def preflop_answer(body: PreflopAnswerIn) -> PreflopFeedback:
    spot = get_registry().spots.get(body.spot_id)
    if spot is None:
        raise HTTPException(404, f"unknown spot {body.spot_id}")
    try:
        combo = combo_from_str(body.combo)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    freqs = spot.combo_freqs(combo)
    ev = None
    if spot.ev is not None:
        ev = {a: float(spot.ev[combo, i]) for i, a in enumerate(spot.action_ids)}
    labels = {a.id: a.label for a in spot.ctx.actions}
    try:
        g = grade_decision(freqs, body.action, ev, labels)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    eq = None
    if spot.villain_range is not None and spot.villain_range.sum() > 0:
        table = class_equity_vs_range(class_weights(spot.villain_range))
        if table is not None:
            eq = float(table[COMBO_CLASS[combo]])

    decision_id = record_preflop_decision(spot, combo, g, body.response_ms, body.reason)
    return PreflopFeedback(
        grade=GradeOut.of(g),
        combo_freqs=freqs,
        equity_vs_villain=eq,
        detail=spot.detail(),
        decision_id=decision_id,
    )
