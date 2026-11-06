"""Formats, preflop spots and range-notation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import FORMATS, GameFormat
from ..poker.cards import CLASS_COMBOS, CLASS_INDEX, combo_to_str
from ..poker.ranges import RangeParseError, class_weights, format_range, parse_range
from ..strategies.loader import get_registry, reload_registry
from ..strategies.schemas import SpotDetail, SpotSummary

router = APIRouter(prefix="/api", tags=["spots"])


class FormatOut(BaseModel):
    format: GameFormat
    positions: list[str]
    spot_count: int


@router.get("/formats")
def list_formats() -> list[FormatOut]:
    reg = get_registry()
    return [
        FormatOut(format=f, positions=f.positions, spot_count=len(reg.list(f.id)))
        for f in FORMATS.values()
    ]


@router.get("/spots")
def list_spots(format_id: str | None = None, scenario: str | None = None) -> list[SpotSummary]:
    return [s.summary() for s in get_registry().list(format_id, scenario)]


@router.get("/spots/errors")
def load_errors() -> list[str]:
    return get_registry().errors


@router.post("/spots/reload")
def reload_spots() -> dict[str, int]:
    reg = reload_registry()
    return {"spots": len(reg.spots), "errors": len(reg.errors)}


@router.get("/spots/{spot_id}")
def spot_detail(spot_id: str) -> SpotDetail:
    spot = get_registry().spots.get(spot_id)
    if spot is None:
        raise HTTPException(404, f"unknown spot {spot_id}")
    return spot.detail()


class ComboFreq(BaseModel):
    combo: str
    reach: float
    freqs: dict[str, float]


@router.get("/spots/{spot_id}/cell/{label}")
def spot_cell(spot_id: str, label: str) -> list[ComboFreq]:
    """Per-combo breakdown of one grid cell (suit-specific imports can differ by combo)."""
    spot = get_registry().spots.get(spot_id)
    if spot is None or label not in CLASS_INDEX:
        raise HTTPException(404, "unknown spot or hand class")
    return [
        ComboFreq(
            combo=combo_to_str(int(c)), reach=float(spot.reach[c]), freqs=spot.combo_freqs(int(c))
        )
        for c in CLASS_COMBOS[CLASS_INDEX[label]]
    ]


class RangeText(BaseModel):
    text: str


class ParsedRange(BaseModel):
    class_weights: list[float]
    combos: float
    percent: float
    normalized: str


@router.post("/ranges/parse")
def parse(body: RangeText) -> ParsedRange:
    try:
        w = parse_range(body.text)
    except RangeParseError as e:
        raise HTTPException(422, str(e)) from e
    return ParsedRange(
        class_weights=[float(x) for x in class_weights(w)],
        combos=float(w.sum()),
        percent=float(w.sum() / 1326),
        normalized=format_range(w),
    )
