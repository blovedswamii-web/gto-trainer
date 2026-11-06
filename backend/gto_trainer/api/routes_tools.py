"""Equity calculator, odds drills, range comparison and saved custom ranges."""

from __future__ import annotations

import random
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..db.models import CustomRange
from ..db.session import local_user_id, session
from ..poker import odds
from ..poker.cards import N_COMBOS, combo_to_str, parse_cards
from ..poker.equity import range_vs_range
from ..poker.ranges import (
    RangeParseError,
    class_weights,
    compare_ranges,
    format_range,
    from_class_weights,
    parse_range,
)
from ..strategies.loader import get_registry
from ..training import math_drills

router = APIRouter(prefix="/api", tags=["tools"])


class EquityIn(BaseModel):
    hero: str  # range notation or a single combo like "AhKh"
    villain: str
    board: str = ""
    dead: str = ""
    samples: int = Field(default=200_000, ge=1000, le=2_000_000)
    method: Literal["auto", "exact", "monte_carlo"] = "auto"


@router.post("/equity")
def equity(body: EquityIn) -> dict[str, Any]:
    try:
        hero, villain = parse_range(body.hero), parse_range(body.villain)
        board, dead = parse_cards(body.board), parse_cards(body.dead)
    except (RangeParseError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    if len(board) not in (0, 3, 4, 5):
        raise HTTPException(422, "board must have 0, 3, 4 or 5 cards")
    method = None if body.method == "auto" else body.method
    try:
        r = range_vs_range(
            hero,
            villain,
            board,
            dead,
            samples=body.samples,
            method=method,
            per_combo=bool((hero > 0).sum() <= 400),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    per = sorted(r.hero_combo_equity.items(), key=lambda kv: -kv[1])
    return {
        "equity": r.equity,
        "win": r.win,
        "tie": r.tie,
        "lose": r.lose,
        "method": r.method,
        "samples": r.samples,
        "std_error": r.std_error,
        "matchups": r.matchups,
        "hero_combos": float(hero.sum()),
        "villain_combos": float(villain.sum()),
        "per_combo": [{"combo": combo_to_str(c), "equity": e} for c, e in per[:60]],
    }


class OddsIn(BaseModel):
    pot: float = Field(gt=0)
    bet: float = Field(gt=0)


@router.post("/odds")
def odds_calc(body: OddsIn) -> dict[str, float]:
    return {
        "pot_odds": odds.pot_odds(body.pot, body.bet),
        "mdf": odds.minimum_defence_frequency(body.pot, body.bet),
        "bluff_breakeven": odds.bluff_breakeven(body.pot, body.bet),
        "optimal_bluff_fraction": odds.optimal_bluff_fraction(body.pot, body.bet),
        "value_to_bluff": odds.value_to_bluff_ratio(body.pot, body.bet),
    }


@router.get("/math/question")
def math_question(kind: str | None = None) -> dict[str, Any]:
    if kind not in (None, "pot_odds", "mdf", "bluff"):
        raise HTTPException(422, "kind must be pot_odds, mdf or bluff")
    q = math_drills.make_question(kind, random.Random())  # type: ignore[arg-type]
    return {"kind": q.kind, "pot": q.pot, "bet": q.bet, "prompt": q.prompt}


class MathAnswerIn(BaseModel):
    kind: Literal["pot_odds", "mdf", "bluff"]
    pot: float
    bet: float
    guess_pct: float


@router.post("/math/answer")
def math_answer(body: MathAnswerIn) -> dict[str, object]:
    fn = {
        "pot_odds": odds.pot_odds,
        "mdf": odds.minimum_defence_frequency,
        "bluff": odds.bluff_breakeven,
    }[body.kind]
    q = math_drills.MathQuestion(
        body.kind,
        body.pot,
        body.bet,
        math_drills.PROMPTS[body.kind].format(pot=body.pot, bet=body.bet),
        fn(body.pot, body.bet),
    )
    return math_drills.grade_answer(q, body.guess_pct)


# ----------------------------------------------------------------------------- range editor


class CompareIn(BaseModel):
    class_weights: list[float] | None = None  # 169 values in [0, 1]
    text: str | None = None
    spot_id: str
    actions: list[str]  # which actions make up the reference range, e.g. ["raise"]
    tolerance: float = 0.05


def _user_range(class_w: list[float] | None, text: str | None) -> np.ndarray:
    if class_w is not None:
        if len(class_w) != 169:
            raise HTTPException(422, "class_weights must have 169 entries")
        return from_class_weights(np.clip(np.asarray(class_w, dtype=float), 0.0, 1.0))
    if text is not None:
        try:
            return parse_range(text)
        except RangeParseError as e:
            raise HTTPException(422, str(e)) from e
    raise HTTPException(422, "send class_weights or text")


@router.post("/ranges/compare")
def compare(body: CompareIn) -> dict[str, Any]:
    spot = get_registry().spots.get(body.spot_id)
    if spot is None:
        raise HTTPException(404, f"unknown spot {body.spot_id}")
    bad = [a for a in body.actions if a not in spot.action_ids]
    if bad or not body.actions:
        raise HTTPException(422, f"actions must be a subset of {spot.action_ids}")
    ref = sum((spot.action_range(a) for a in body.actions), start=np.zeros(N_COMBOS))
    user = _user_range(body.class_weights, body.text)
    d = compare_ranges(user, ref, body.tolerance)
    return {
        "overlap_combos": d.overlap_combos,
        "user_combos": d.user_combos,
        "reference_combos": d.reference_combos,
        "jaccard": d.jaccard,
        "over_included": d.over_included,
        "under_included": d.under_included,
        "cell_diff": d.cell_diff,
        "reference_class_weights": [float(x) for x in class_weights(ref)],
        "reference_text": format_range(ref),
        "user_text": format_range(user),
        "source": spot.source.model_dump(),
        "title": spot.ctx.title,
    }


class SaveRangeIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    class_weights: list[float] | None = None
    text: str | None = None
    spot_id: str | None = None
    action: str | None = None


@router.get("/ranges/custom")
def list_custom() -> list[dict[str, Any]]:
    with session() as s:
        uid = local_user_id(s)
        rows = s.scalars(
            select(CustomRange)
            .where(CustomRange.user_id == uid)
            .order_by(CustomRange.created_at.desc())
        ).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "notation": r.notation,
                "spot_id": r.spot_id,
                "action": r.action,
                "class_weights": [float(x) for x in class_weights(parse_range(r.notation))],
            }
            for r in rows
        ]


@router.post("/ranges/custom")
def save_custom(body: SaveRangeIn) -> dict[str, Any]:
    notation = format_range(_user_range(body.class_weights, body.text))
    with session() as s:
        r = CustomRange(
            user_id=local_user_id(s),
            name=body.name,
            notation=notation,
            spot_id=body.spot_id,
            action=body.action,
        )
        s.add(r)
        s.flush()
        return {"id": r.id, "notation": notation}


@router.delete("/ranges/custom/{range_id}")
def delete_custom(range_id: int) -> dict[str, bool]:
    with session() as s:
        r = s.get(CustomRange, range_id)
        if r is None or r.user_id != local_user_id(s):
            raise HTTPException(404, "not found")
        s.delete(r)
    return {"deleted": True}
