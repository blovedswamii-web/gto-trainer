"""Spaced repetition for drill items (an SM-2 variant with minute-level short intervals).

Item = (mode, spot, hand class[, postflop node]). Items are created the first time you get
one *wrong-ish* (minor or mistake) -- a correctly answered spot is not worth scheduling.

Update rule (quality q: best=5, good=4, minor=3, mistake=1)::

    mistake: lapses += 1, reps = 0, interval = 2 min, ease -= 0.20   -> comes back this session
    minor:   interval = max(10 min, 1.2 * interval), ease -= 0.14
    good+:   reps += 1, interval = 1 day, 3 days, then interval * ease  (SM-2 growth)
             ease += 0.1 - (5-q)(0.08 + 0.02(5-q))

``ease`` is clamped to >= 1.3 as in SM-2. When drilling, a due item is served with
probability ``REVIEW_SHARE`` so sessions mix review with fresh hands; more lapses and
longer overdue -> served first.
"""

from __future__ import annotations

import random
from datetime import timedelta

from sqlalchemy import select

from ..db.models import SrsItem, utcnow
from ..db.session import local_user_id, session

QUALITY = {"best": 5, "good": 4, "minor": 3, "mistake": 1}
REVIEW_SHARE = 0.6


def item_key(mode: str, spot_id: str, hand_class: str, node: str = "") -> str:
    return f"{mode}|{spot_id}|{node}|{hand_class}"


def schedule(item: SrsItem, grade: str) -> None:
    q = QUALITY[grade]
    if q < 3:
        item.lapses += 1
        item.reps = 0
        item.interval_min = 2.0
        item.ease = max(1.3, item.ease - 0.2)
    elif q == 3:
        item.interval_min = max(10.0, item.interval_min * 1.2)
        item.ease = max(1.3, item.ease - 0.14)
    else:
        item.reps += 1
        if item.reps == 1:
            item.interval_min = 24 * 60.0
        elif item.reps == 2:
            item.interval_min = 3 * 24 * 60.0
        else:
            item.interval_min *= item.ease
        item.ease = max(1.3, item.ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    item.last_grade = grade
    now = utcnow()
    item.updated_at = now
    item.due_at = now + timedelta(minutes=item.interval_min)


def update_item(
    mode: str, format_id: str, spot_id: str, hand_class: str, grade: str, node: str = ""
) -> None:
    key = item_key(mode, spot_id, hand_class, node)
    with session() as s:
        uid = local_user_id(s)
        item = s.scalar(select(SrsItem).where(SrsItem.user_id == uid, SrsItem.item_key == key))
        if item is None:
            if grade in ("best", "good"):
                return  # nothing to review
            item = SrsItem(
                user_id=uid,
                item_key=key,
                mode=mode,
                format_id=format_id,
                spot_id=spot_id,
                hand_class=hand_class,
                node=node,
                ease=2.5,
                interval_min=0.0,
                reps=0,
                lapses=0,
            )
            s.add(item)
        schedule(item, grade)


def due_items(mode: str, format_id: str, limit: int = 20) -> list[SrsItem]:
    with session() as s:
        uid = local_user_id(s)
        rows = s.scalars(
            select(SrsItem)
            .where(
                SrsItem.user_id == uid,
                SrsItem.mode == mode,
                SrsItem.format_id == format_id,
                SrsItem.due_at <= utcnow(),
            )
            .order_by(SrsItem.lapses.desc(), SrsItem.due_at)
            .limit(limit)
        ).all()
        return list(rows)


def due_item(
    format_id: str, mode: str = "preflop", rng: random.Random | None = None
) -> tuple[str, str] | None:
    """(spot_id, hand_class) to review now, or None to deal a fresh hand."""
    r = rng or random
    items = due_items(mode, format_id, limit=5)
    if not items or r.random() > REVIEW_SHARE:
        return None
    it = items[0] if r.random() < 0.7 else r.choice(items)
    return it.spot_id, it.hand_class


def due_count(format_id: str, mode: str = "preflop") -> int:
    return len(due_items(mode, format_id, limit=1000))
