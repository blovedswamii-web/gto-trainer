"""Anki export: preview a deck and download it as .apkg."""

from __future__ import annotations

import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..export import anki
from ..solver.pushfold import PushFoldSetup

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportIn(BaseModel):
    kind: Literal["leaks", "preflop", "postflop", "pushfold"]
    # leaks
    days: int | None = Field(default=None, ge=1)
    include_minor: bool = True
    # preflop / postflop
    scenarios: list[str] = Field(default_factory=list)
    spot_ids: list[str] = Field(default_factory=list)
    hands: Literal["close", "all"] = "close"
    # push/fold
    stacks: list[float] = Field(default_factory=list)
    payouts: list[float] = Field(default_factory=list)
    ante: float = 0.0


def _build(body: ExportIn) -> anki.DeckBuild:
    try:
        if body.kind == "leaks":
            return anki.build_leaks(body.days, body.include_minor)
        if body.kind == "preflop":
            return anki.build_preflop(body.scenarios, body.spot_ids, body.hands)
        if body.kind == "postflop":
            return anki.build_postflop(body.spot_ids)
        setup = PushFoldSetup(tuple(body.stacks), tuple(body.payouts), body.ante)
        return anki.build_pushfold(setup, body.hands)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e)) from e


def _inline_media(text: str, media: dict[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        svg = media.get(m.group(1), "")
        return f'src="data:image/svg+xml;utf8,{urllib.parse.quote(svg)}"'

    return re.sub(r'src="(gto_[0-9a-f]+\.svg)"', sub, text)


@router.post("/anki/preview")
def preview(body: ExportIn) -> dict[str, Any]:
    b = _build(body)
    return {
        "deck": b.name,
        "cards": len(b.cards),
        "media": len(b.media),
        "skipped": dict(b.skipped),
        "samples": [
            {"front": c.front, "back": _inline_media(c.back, b.media), "tags": c.tags}
            for c in b.cards[:3]
        ],
        "css": anki.CSS,
    }


@router.post("/anki")
def download(body: ExportIn) -> Response:
    b = _build(body)
    if not b.cards:
        raise HTTPException(422, "this selection produces no cards")
    with tempfile.TemporaryDirectory() as tmp:
        path = anki.write_apkg(b, Path(tmp) / "deck.apkg")
        data = path.read_bytes()
    name = f"gto-trainer-{body.kind}.apkg"
    return Response(
        data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
