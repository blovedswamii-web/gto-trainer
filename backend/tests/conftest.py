"""Test-wide fixtures: every test run uses a fresh in-memory database."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from gto_trainer.db import session as db_session


@pytest.fixture(autouse=True)
def memory_db() -> Iterator[None]:
    db_session.configure(":memory:")
    yield
