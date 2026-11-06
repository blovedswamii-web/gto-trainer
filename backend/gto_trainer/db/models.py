"""SQLAlchemy 2.0 models. One row per decision is the source of truth for every statistic;
aggregates are computed on read (a few thousand rows a month is nothing for SQLite)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)  # SQLite stores naive; we always mean UTC


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (
        Index("ix_decisions_user_time", "user_id", "created_at"),
        Index("ix_decisions_spot_hand", "spot_id", "hand_class"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    mode: Mapped[str] = mapped_column(String(16))  # preflop | postflop | math
    format_id: Mapped[str] = mapped_column(String(32))
    spot_id: Mapped[str] = mapped_column(String(128))
    scenario: Mapped[str] = mapped_column(String(32))  # rfi, vs_open, ... / srp, 3bp / math kind
    hero_position: Mapped[str] = mapped_column(String(8))
    villain_position: Mapped[str | None] = mapped_column(String(8), nullable=True)
    street: Mapped[str] = mapped_column(String(8))  # preflop | flop | turn | river
    node: Mapped[str] = mapped_column(String(128), default="")  # postflop action path
    board: Mapped[str] = mapped_column(String(16), default="")
    hand: Mapped[str] = mapped_column(String(8))  # exact combo, e.g. "AsKd"
    hand_class: Mapped[str] = mapped_column(String(4))  # 169-class label
    hand_category: Mapped[str] = mapped_column(String(48))
    chosen_action: Mapped[str] = mapped_column(String(32))
    best_action: Mapped[str] = mapped_column(String(32))
    chosen_freq: Mapped[float] = mapped_column(Float)
    best_freq: Mapped[float] = mapped_column(Float)
    freqs_json: Mapped[str] = mapped_column(Text)
    grade: Mapped[str] = mapped_column(String(8))
    points: Mapped[float] = mapped_column(Float)
    freq_error: Mapped[float] = mapped_column(Float)
    ev_loss_bb: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_loss_pct_pot: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_type: Mapped[str] = mapped_column(String(16))
    response_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason: Mapped[str] = mapped_column(String(8), default="new")  # new | review


class SrsItem(Base):
    """Spaced-repetition state for one (mode, spot, hand class) item."""

    __tablename__ = "srs_items"
    __table_args__ = (UniqueConstraint("user_id", "item_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    item_key: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(16))
    format_id: Mapped[str] = mapped_column(String(32))
    spot_id: Mapped[str] = mapped_column(String(128))
    hand_class: Mapped[str] = mapped_column(String(4))
    node: Mapped[str] = mapped_column(String(128), default="")
    ease: Mapped[float] = mapped_column(Float, default=2.5)
    interval_min: Mapped[float] = mapped_column(Float, default=0.0)
    reps: Mapped[int] = mapped_column(Integer, default=0)
    lapses: Mapped[int] = mapped_column(Integer, default=0)
    due_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_grade: Mapped[str] = mapped_column(String(8), default="")
    updated_at: Mapped[datetime] = mapped_column(default=utcnow)


class CustomRange(Base):
    __tablename__ = "custom_ranges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(80))
    notation: Mapped[str] = mapped_column(Text)
    spot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
