"""Engine/session management. The engine is created lazily so tests can point it elsewhere."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from .. import config
from .models import Base, User

_engine: Engine | None = None
_factory: sessionmaker[Session] | None = None
_lock = threading.Lock()  # FastAPI runs sync endpoints in a threadpool: init exactly once
LOCAL_USER = "local"


def configure(path: Path | str | None = None) -> Engine:
    """(Re)create the engine. ``path=":memory:"`` gives a throwaway in-memory DB."""
    with _lock:
        return _configure_locked(path)


def _configure_locked(path: Path | str | None) -> Engine:
    global _engine, _factory
    target = path if path is not None else config.DB_PATH
    if target == ":memory:":
        url = "sqlite+pysqlite:///:memory:"
    else:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+pysqlite:///{target}"
    from sqlalchemy.pool import StaticPool

    _engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool if target == ":memory:" else None,
    )
    Base.metadata.create_all(_engine)
    _migrate(_engine)
    _factory = sessionmaker(_engine, expire_on_commit=False)
    return _engine


# Columns added after the first release: (table, column, SQL type). create_all() only creates
# missing *tables*, so existing databases get new columns through ALTER TABLE.
_ADDED_COLUMNS = [("decisions", "hand_ref", "INTEGER REFERENCES hands(id)")]


def _migrate(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, column, sql_type in _ADDED_COLUMNS:
            cols = {c["name"] for c in insp.get_columns(table)}
            if column not in cols:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))


@contextmanager
def session() -> Iterator[Session]:
    if _factory is None:
        with _lock:
            if _factory is None:
                _configure_locked(None)
    assert _factory is not None
    s = _factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def local_user_id(s: Session) -> int:
    """Single-user app: one implicit local user, created on first use."""
    uid = s.scalar(select(User.id).where(User.name == LOCAL_USER))
    if uid is None:
        u = User(name=LOCAL_USER)
        s.add(u)
        s.flush()
        uid = u.id
    return uid
