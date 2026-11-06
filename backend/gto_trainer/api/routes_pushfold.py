"""Push/fold (ICM) charts and drill."""

from __future__ import annotations

import json
import random
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db.models import Decision
from ..db.session import local_user_id, session
from ..poker.cards import CLASS_COMBOS, CLASS_INDEX, CLASS_LABELS, CLASS_NCOMBOS, combo_to_str
from ..poker.categories import preflop_category
from ..solver.pushfold import PRESETS, PushFoldResult, PushFoldSetup, bubble_matrix, solve_cached
from ..training.grading import grade_decision
from .routes_drill import GradeOut

router = APIRouter(prefix="/api/pushfold", tags=["pushfold"])
_rng = random.Random()
PRIOR = CLASS_NCOMBOS / CLASS_NCOMBOS.sum()


class SetupIn(BaseModel):
    stacks: list[float] = Field(min_length=2, max_length=9)
    payouts: list[float] = Field(default_factory=list)
    ante: float = Field(default=0.0, ge=0.0, le=1.0)


def _setup(body: SetupIn) -> PushFoldSetup:
    try:
        return PushFoldSetup(tuple(body.stacks), tuple(body.payouts), body.ante)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


def _solve(body: SetupIn) -> PushFoldResult:
    try:
        return solve_cached(_setup(body))
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/presets")
def presets() -> list[dict[str, object]]:
    return PRESETS


def _rng_arr(a: np.ndarray) -> list[float]:
    return [round(float(x), 5) for x in a]


@router.post("/solve")
def solve(body: SetupIn) -> dict[str, Any]:
    r = _solve(body)
    st = r.setup
    pos = st.positions
    total = sum(st.stacks)
    return {
        "key": st.key(),
        "labels": list(CLASS_LABELS),
        "positions": pos,
        "stacks": list(st.stacks),
        "chip_pct": [x / total * 100 for x in st.stacks],
        "equity": r.start_equity,
        "icm": st.icm,
        "units": r.units,
        "iterations": r.iterations,
        "exploitability": r.exploitability,
        "warnings": r.warnings,
        "push": [
            {
                "position": pos[i],
                "freqs": _rng_arr(f),
                "margin": _rng_arr(r.push_margin[i]),
                "pct": float(PRIOR @ f),
            }
            for i, f in enumerate(r.push)
        ],
        "call": [
            {
                "shover": pos[i],
                "caller": pos[j],
                "freqs": _rng_arr(f),
                "margin": _rng_arr(r.call_margin[(i, j)]),
                "pct": float(PRIOR @ f),
            }
            for (i, j), f in r.call.items()
        ],
        "bubble_factors": bubble_matrix(st),
    }


class DrillNextIn(SetupIn):
    seed: int | None = None


@router.post("/drill/next")
def drill_next(body: DrillNextIn) -> dict[str, Any]:
    r = _solve(body)
    st, n = r.setup, len(r.setup.stacks)
    rng = random.Random(body.seed) if body.seed is not None else _rng
    decisions: list[tuple[str, int, int | None]] = [("push", i, None) for i in range(n - 1)]
    decisions += [("call", i, j) for (i, j) in r.call]
    kind, i, j = rng.choice(decisions)
    freqs = r.push[i] if kind == "push" else r.call[(i, j)]  # type: ignore[index]
    margin = r.push_margin[i] if kind == "push" else r.call_margin[(i, j)]  # type: ignore[index]
    # Close decisions are the instructive ones: weight by how small the EV margin is.
    scale = float(np.abs(margin).max()) or 1.0
    w = PRIOR * (0.1 + np.exp(-np.abs(margin) / (0.05 * scale)) + freqs * (1 - freqs) * 2)
    cls = int(rng.choices(range(169), weights=w.tolist(), k=1)[0])
    combo = int(rng.choice(list(CLASS_COMBOS[cls])))
    hero = j if kind == "call" else i
    assert hero is not None
    return {
        "kind": kind,
        "shover": i,
        "caller": j,
        "hero": hero,
        "positions": st.positions,
        "stacks": list(st.stacks),
        "ante": st.ante,
        "icm": st.icm,
        "payouts": list(st.payouts),
        "hand": combo_to_str(combo),
        "hand_class": CLASS_LABELS[cls],
        "actions": ["push", "fold"] if kind == "push" else ["call", "fold"],
    }


class DrillAnswerIn(SetupIn):
    kind: Literal["push", "call"]
    shover: int
    caller: int | None = None
    hand: str
    hand_class: str
    action: Literal["push", "call", "fold"]
    response_ms: int | None = None


@router.post("/drill/answer")
def drill_answer(body: DrillAnswerIn) -> dict[str, Any]:
    r = _solve(body)
    st = r.setup
    if body.hand_class not in CLASS_INDEX:
        raise HTTPException(422, "bad hand class")
    k = CLASS_INDEX[body.hand_class]
    if body.kind == "push":
        if not 0 <= body.shover < len(st.stacks) - 1:
            raise HTTPException(422, "bad shover")
        f, m, act = float(r.push[body.shover][k]), float(r.push_margin[body.shover][k]), "push"
        hero, villain = body.shover, None
    else:
        key = (body.shover, body.caller or -1)
        if key not in r.call:
            raise HTTPException(422, "bad caller")
        f, m, act = float(r.call[key][k]), float(r.call_margin[key][k]), "call"
        hero, villain = body.caller or 0, body.shover
    if body.action not in (act, "fold"):
        raise HTTPException(422, f"action must be {act} or fold")
    freqs = {act: f, "fold": 1 - f}
    labels = {"push": "Shove", "call": "Call", "fold": "Fold"}
    ev = {act: m, "fold": 0.0}
    # Grades carry EV loss in bb; ICM losses are in % of the prize pool, reported separately.
    g = grade_decision(freqs, body.action, None if st.icm else ev, labels)
    loss = max(ev.values()) - ev[body.action]
    pos = st.positions
    with session() as s:
        s.add(
            Decision(
                user_id=local_user_id(s),
                mode="pushfold",
                format_id=f"pushfold.{st.key()}",
                spot_id=f"pushfold.{st.key()}.{act}.{pos[hero]}"
                + (f"_vs_{pos[villain]}" if villain is not None else ""),
                scenario=f"{act} ({'ICM' if st.icm else 'chip EV'})",
                hero_position=pos[hero],
                villain_position=pos[villain] if villain is not None else None,
                street="preflop",
                node=st.code(),
                hand=body.hand,
                hand_class=body.hand_class,
                hand_category=preflop_category(body.hand_class),
                chosen_action=body.action,
                best_action=g.best_action,
                chosen_freq=g.chosen_freq,
                best_freq=g.best_freq,
                freqs_json=json.dumps(freqs),
                grade=g.grade,
                points=g.points,
                freq_error=g.freq_error,
                ev_loss_bb=None if st.icm else g.ev_loss,
                source_type="solver",
                response_ms=body.response_ms,
            )
        )
    return {
        "grade": GradeOut.of(g).model_dump(),
        "freqs": freqs,
        "margin": m,
        "units": r.units,
        "ev_loss": loss,
    }
