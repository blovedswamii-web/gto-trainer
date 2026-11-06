"""Progress dashboard endpoints (all computed from the decision log)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..training import leaks

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/summary")
def summary(mode: str | None = None, days: int | None = None) -> dict[str, Any]:
    return leaks.summary(mode, days)


@router.get("/rolling")
def rolling(mode: str | None = None, window: int = 25) -> list[dict[str, Any]]:
    return leaks.rolling(mode, window)


@router.get("/leaks")
def top_leaks(mode: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    return leaks.top_leaks(mode, limit)


@router.get("/recent")
def recent(limit: int = 30) -> list[dict[str, Any]]:
    return leaks.recent(limit)
