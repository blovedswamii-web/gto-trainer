"""Postflop trainer: solved-spot explorer and drill."""

from __future__ import annotations

import random
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..db.models import SrsItem, utcnow
from ..db.session import local_user_id, session
from ..poker.cards import COMBOS, card_to_str
from ..solver.postflop import SPOTS, Solution, available_solutions, load_solution
from ..training import postflop_view as pv
from ..training.srs import REVIEW_SHARE
from .routes_drill import GradeOut

router = APIRouter(prefix="/api/postflop", tags=["postflop"])
_rng = random.Random()


def _solution(spot_id: str) -> Solution:
    sol = load_solution(spot_id)
    if sol is None:
        raise HTTPException(404, f"{spot_id} is not solved yet - run `make precompute`")
    return sol


@router.get("/spots")
def list_spots() -> list[dict[str, Any]]:
    out = []
    solved = set(available_solutions())
    for sid, d in SPOTS.items():
        item: dict[str, Any] = {
            "id": sid,
            "title": d.title,
            "board": d.board,
            "pot_type": d.pot_type,
            "oop": d.oop,
            "ip": d.ip,
            "solved": sid in solved,
        }
        if sid in solved:
            sol = _solution(sid)
            item["source"] = pv.source_tag(sol).model_dump()
            item["pot"] = sol.spot["tree"]["starting_pot"]
            item["stack"] = sol.spot["tree"]["stack"]
        out.append(item)
    return out


@router.get("/spots/{spot_id}/node")
def node(spot_id: str, path: str = "") -> dict[str, Any]:
    sol = _solution(spot_id)
    if path not in sol.nodes:
        raise HTTPException(404, f"no decision node at {path!r}")
    return pv.node_view(sol, path)


@router.get("/spots/{spot_id}/root")
def first_decision(spot_id: str) -> dict[str, Any]:
    """The root, or the first node with a real choice (e.g. skip BB's forced check)."""
    sol = _solution(spot_id)
    path = ""
    while path in sol.nodes and len(sol.nodes[path].actions) == 1:
        a = sol.nodes[path].actions[0]
        path = f"{path}/{a}" if path else a
    if path not in sol.nodes:
        path = ""
    return pv.node_view(sol, path)


@router.get("/spots/{spot_id}/combo")
def combo(spot_id: str, path: str, combo: str) -> dict[str, Any]:
    sol = _solution(spot_id)
    try:
        return pv.combo_detail(sol, path, pv.parse_combo(combo))
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e)) from e


class NextIn(BaseModel):
    spot_ids: list[str] = Field(default_factory=list)
    streets: list[str] = Field(default_factory=list)
    role: str | None = None  # "OOP" | "IP"
    use_srs: bool = True
    seed: int | None = None


def _due_postflop() -> tuple[str, str, str] | None:
    with session() as s:
        uid = local_user_id(s)
        it = s.scalars(
            select(SrsItem)
            .where(
                SrsItem.user_id == uid,
                SrsItem.mode == "postflop",
                SrsItem.due_at <= utcnow(),
            )
            .order_by(SrsItem.lapses.desc(), SrsItem.due_at)
            .limit(1)
        ).first()
        return (it.spot_id, it.node, it.hand_class) if it else None


@router.post("/drill/next")
def drill_next(body: NextIn) -> dict[str, Any]:
    ids = body.spot_ids or available_solutions()
    sols = [s for s in (load_solution(i) for i in ids) if s is not None]
    if not sols:
        raise HTTPException(404, "no solved postflop spots - run `make precompute`")
    rng = random.Random(body.seed) if body.seed is not None else _rng
    forced = None
    if body.use_srs and rng.random() < REVIEW_SHARE:
        forced = _due_postflop()
    try:
        d = pv.deal(sols, rng, body.streets or None, body.role, forced)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    sol = _solution(d.spot_id)
    view = pv.node_view(sol, d.path)
    hi, lo = COMBOS[d.combo]
    # Strip the answer: the drill shows the spot, not the strategy.
    for k in ("overall", "categories", "grid"):
        view.pop(k)
    view["hero_cards"] = [card_to_str(int(hi)), card_to_str(int(lo))]
    view["combo"] = view["hero_cards"][0] + view["hero_cards"][1]
    view["reason"] = "review" if forced and (d.spot_id, d.path) == forced[:2] else "new"
    return view


class AnswerIn(BaseModel):
    spot_id: str
    path: str
    combo: str
    action: str
    response_ms: int | None = None
    reason: str = "new"


@router.post("/drill/answer")
def drill_answer(body: AnswerIn) -> dict[str, Any]:
    sol = _solution(body.spot_id)
    if body.path not in sol.nodes:
        raise HTTPException(404, "unknown node")
    try:
        g, extra = pv.grade_and_record(
            sol, body.path, pv.parse_combo(body.combo), body.action, body.response_ms, body.reason
        )
    except KeyError as e:
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    nd = sol.nodes[body.path]
    return {
        "grade": GradeOut.of(g).model_dump(),
        "ev_loss_pct_pot": (g.ev_loss / nd.pot * 100) if g.ev_loss is not None else None,
        **extra,
        "node": pv.node_view(sol, body.path),
    }
