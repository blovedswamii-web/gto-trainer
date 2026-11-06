"""Progress statistics and leak mining over the decision log.

Leak model
----------
A *leak* is a systematic deviation, not a single bad click. We group graded decisions by

    (mode, spot, hand category, chosen action, solution's preferred action)

e.g. ("preflop", "BB vs BTN open", "suited gappers", fold, call) reads as *"you over-fold BB
vs BTN with suited gappers; the solution mostly calls"*. Each group is scored by

    severity = sum(freq_error) * n_errors / (n_errors + 2)

i.e. total frequency error with a small-sample shrinkage (one stray misclick ranks below
three consistent errors of the same size). When EV loss is available (solver/import data),
groups also report total EV loss, and EV loss replaces frequency error as the primary sort.
We also report the error *rate* within the (spot, category) cell so "3 of 4 times" and
"3 of 40 times" are distinguishable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select

from ..db.models import Decision, SrsItem, utcnow
from ..db.session import local_user_id, session
from ..strategies.loader import get_registry

VERB = {
    "fold": "over-fold",
    "call": "over-call",
    "check": "over-check",
    "raise": "raise too often",
    "allin": "jam too often",
}


def _verb(action: str) -> str:
    kind = action.split(":")[0]
    if kind == "bet":
        return "bet too often"
    return VERB.get(kind, f"choose {action} too often")


def _plain(action: str) -> str:
    kind, _, size = action.partition(":")
    if kind == "bet" and size:
        return f"bet {float(size):.0%}"
    if kind == "raise" and size:
        return "raise"
    return {"allin": "jam", "raise": "raise"}.get(kind, kind)


def spot_title(d: Decision) -> str:
    if d.mode == "preflop":
        sp = get_registry().spots.get(d.spot_id)
        if sp is not None:
            return sp.ctx.title
    if d.mode == "postflop":
        return f"{d.spot_id.split('.')[-1]} {d.node}".strip()
    return d.spot_id


def load_decisions(mode: str | None = None, days: int | None = None) -> list[Decision]:
    with session() as s:
        uid = local_user_id(s)
        q = select(Decision).where(Decision.user_id == uid)
        if mode:
            q = q.where(Decision.mode == mode)
        if days:
            q = q.where(Decision.created_at >= utcnow() - timedelta(days=days))
        return list(s.scalars(q.order_by(Decision.created_at, Decision.id)).all())


@dataclass
class Bucket:
    key: str
    n: int = 0
    points: float = 0.0
    freq_error: float = 0.0
    ev_loss: float = 0.0
    ev_n: int = 0

    def add(self, d: Decision) -> None:
        self.n += 1
        self.points += d.points
        self.freq_error += d.freq_error
        if d.ev_loss_bb is not None:
            self.ev_loss += d.ev_loss_bb
            self.ev_n += 1

    def out(self) -> dict[str, float | int | str | None]:
        return {
            "key": self.key,
            "n": self.n,
            "accuracy": self.points / self.n if self.n else 0.0,
            "avg_freq_error": self.freq_error / self.n if self.n else 0.0,
            "avg_ev_loss_bb": self.ev_loss / self.ev_n if self.ev_n else None,
        }


def breakdown(decisions: list[Decision], attr: str) -> list[dict[str, float | int | str | None]]:
    buckets: dict[str, Bucket] = {}
    for d in decisions:
        k = str(getattr(d, attr) or "—")
        buckets.setdefault(k, Bucket(k)).add(d)
    return sorted((b.out() for b in buckets.values()), key=lambda r: -int(r["n"] or 0))


def summary(mode: str | None = None, days: int | None = None) -> dict[str, object]:
    ds = load_decisions(mode, days)
    total = Bucket("all")
    for d in ds:
        total.add(d)
    with session() as s:
        uid = local_user_id(s)
        srs_items = list(s.scalars(select(SrsItem).where(SrsItem.user_id == uid)).all())
    now = utcnow()
    return {
        "total": total.out(),
        "grades": {
            g: sum(1 for d in ds if d.grade == g) for g in ("best", "good", "minor", "mistake")
        },
        "by_position": breakdown(ds, "hero_position"),
        "by_street": breakdown(ds, "street"),
        "by_scenario": breakdown(ds, "scenario"),
        "by_category": breakdown(ds, "hand_category"),
        "by_mode": breakdown(ds, "mode"),
        "srs": {
            "items": len(srs_items),
            "due": sum(1 for it in srs_items if it.due_at <= now),
            "learning": sum(1 for it in srs_items if it.reps < 2),
        },
    }


def rolling(mode: str | None = None, window: int = 25) -> list[dict[str, float | int | None]]:
    """Rolling means over the last ``window`` decisions, one point per decision."""
    ds = load_decisions(mode)
    out: list[dict[str, float | int | None]] = []
    fe: list[float] = []
    pts: list[float] = []
    ev: list[float] = []
    for i, d in enumerate(ds):
        fe.append(d.freq_error)
        pts.append(d.points)
        if d.ev_loss_pct_pot is not None:
            ev.append(d.ev_loss_pct_pot)
        w_fe, w_pts, w_ev = fe[-window:], pts[-window:], ev[-window:]
        out.append(
            {
                "i": i + 1,
                "t": int(d.created_at.timestamp()),
                "freq_error": sum(w_fe) / len(w_fe),
                "accuracy": sum(w_pts) / len(w_pts),
                "ev_loss_pct_pot": (sum(w_ev) / len(w_ev)) if w_ev else None,
            }
        )
    return out


def top_leaks(
    mode: str | None = None, limit: int = 10, min_errors: int = 2
) -> list[dict[str, object]]:
    ds = load_decisions(mode)
    cell_total: dict[tuple[str, str, str], int] = defaultdict(int)
    groups: dict[tuple[str, str, str, str, str], list[Decision]] = defaultdict(list)
    for d in ds:
        cell = (d.mode, d.spot_id + "|" + d.node, d.hand_category)
        cell_total[cell] += 1
        if d.grade in ("minor", "mistake"):
            groups[(*cell, d.chosen_action, d.best_action)].append(d)

    leaks = []
    for (m, spot_node, cat, chosen, best), items in groups.items():
        if len(items) < min_errors:
            continue
        n = len(items)
        fe = sum(d.freq_error for d in items)
        evs = [d.ev_loss_bb for d in items if d.ev_loss_bb is not None]
        severity = fe * n / (n + 2)
        ev_total = sum(evs) if evs else None
        title = spot_title(items[-1])
        hands = sorted({d.hand_class for d in items})
        avg_best = sum(json.loads(d.freqs_json).get(best, 0.0) for d in items) / n
        leaks.append(
            {
                "mode": m,
                "spot_id": items[-1].spot_id,
                "node": items[-1].node,
                "scenario": items[-1].scenario,
                "hero": items[-1].hero_position,
                "category": cat,
                "chosen": chosen,
                "best": best,
                "errors": n,
                "cell_decisions": cell_total[(m, spot_node, cat)],
                "avg_freq_error": fe / n,
                "ev_loss_bb_total": ev_total,
                "severity": severity if ev_total is None else ev_total * n / (n + 2),
                "hands": hands[:8],
                "text": (
                    f"You {_verb(chosen)} {title} with {cat} — {n} of "
                    f"{cell_total[(m, spot_node, cat)]} decisions off; the solution mostly "
                    f"plays {_plain(best)} ({avg_best:.0%})."
                ),
            }
        )
    leaks.sort(key=lambda r: -float(r["severity"]))  # type: ignore[arg-type]
    return leaks[:limit]


def recent(limit: int = 30) -> list[dict[str, object]]:
    ds = load_decisions()[-limit:]
    return [
        {
            "id": d.id,
            "t": int(d.created_at.timestamp()),
            "mode": d.mode,
            "spot": spot_title(d),
            "spot_id": d.spot_id,
            "board": d.board,
            "hand": d.hand,
            "hand_class": d.hand_class,
            "chosen": d.chosen_action,
            "best": d.best_action,
            "chosen_freq": d.chosen_freq,
            "grade": d.grade,
            "freq_error": d.freq_error,
            "ev_loss_bb": d.ev_loss_bb,
            "source": d.source_type,
        }
        for d in reversed(ds)
    ]
