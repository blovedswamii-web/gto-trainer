"""FastAPI application. Run with ``uvicorn gto_trainer.api.app:app --reload``."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routes_spots

app = FastAPI(
    title="GTO Trainer", version="0.1.0", description="Local-first Texas Hold'em GTO training API"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes_spots.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
