"""Hand-history import and review."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import DATA_DIR
from ..training import leaks, review_store

router = APIRouter(prefix="/api/review", tags=["review"])
MAX_IMPORT_CHARS = 20_000_000  # ~40k hands; it's a local app, but keep requests sane


class ImportIn(BaseModel):
    text: str = Field(max_length=MAX_IMPORT_CHARS)


@router.post("/import")
def import_hands(body: ImportIn) -> dict[str, Any]:
    if not body.text.strip():
        raise HTTPException(422, "no hand history text")
    report = review_store.import_text(body.text)
    if report["parsed"] == 0:
        raise HTTPException(422, "no PokerStars/GGPoker hands found in the text")
    return report


@router.get("/sample")
def sample() -> dict[str, str]:
    """Synthetic sample histories (for trying the feature without real files)."""
    files = sorted((DATA_DIR / "samples").glob("*.txt"))
    return {"text": "\n\n\n".join(f.read_text() for f in files)}


@router.get("/summary")
def summary() -> dict[str, Any]:
    hands = review_store.list_hands("all", 100_000)
    s = leaks.summary("review")
    return {
        "hands": len(hands),
        "skipped": sum(1 for h in hands if h["skip_reason"]),
        "flagged_hands": sum(1 for h in hands if h["worst_grade"] in ("minor", "mistake")),
        "net_bb": sum(float(h["net_bb"]) for h in hands),
        "decisions": s["total"],
        "grades": s["grades"],
        "by_scenario": s["by_scenario"],
        "by_position": s["by_position"],
    }


@router.get("/deviations")
def deviations(limit: int = 20) -> list[dict[str, Any]]:
    return review_store.deviations(limit)


@router.get("/hands")
def hands(
    filter: Literal["all", "flagged", "skipped"] = "all", limit: int = 500
) -> list[dict[str, Any]]:
    return review_store.list_hands(filter, limit)


@router.get("/hands/{hand_ref}")
def hand(hand_ref: int) -> dict[str, Any]:
    d = review_store.hand_detail(hand_ref)
    if d is None:
        raise HTTPException(404, "hand not found")
    return d


@router.delete("/hands")
def clear() -> dict[str, int]:
    return {"deleted": review_store.clear_all()}
