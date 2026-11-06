"""Persisting hand-history reviews.

Each imported hand becomes a :class:`HandRecord` (raw text + the full review, scored and
unscored decisions alike) and every *scored* decision is also logged as a ``Decision`` with
``mode="review"``, so real-play results flow into the progress dashboard and leak mining.
Real-play mistakes are also pushed into the spaced-repetition queue of the matching drill.
Hands are deduplicated by site + hand id, so re-importing a file is harmless.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from sqlalchemy import delete, select

from ..db.models import Decision, HandRecord
from ..db.session import local_user_id, session
from ..poker.cards import CLASS_LABELS, COMBO_CLASS, card_from_str, combo_index
from ..strategies.loader import get_registry
from .hand_history import ParsedHand, parse_hand, parse_many
from .review import HandReview, ReviewedDecision, review_hand
from .srs import update_item

GRADE_ORDER = {"mistake": 3, "minor": 2, "good": 1, "best": 0}


def _decision_json(d: ReviewedDecision) -> dict[str, Any]:
    g = d.grade
    return {
        "street": d.street,
        "scenario": d.scenario,
        "hero_pos": d.hero_pos,
        "villain_pos": d.villain_pos,
        "action_text": d.action_text,
        "action": d.action,
        "scored": d.scored,
        "reason": d.reason,
        "spot_id": d.spot_id,
        "node": d.node,
        "board": d.board,
        "freqs": d.freqs,
        "labels": d.labels,
        "category": d.category,
        "pot_bb": d.pot_bb,
        "notes": d.notes,
        "grade": None
        if g is None
        else {
            "grade": g.grade,
            "chosen": g.chosen,
            "chosen_freq": g.chosen_freq,
            "best_action": g.best_action,
            "best_freq": g.best_freq,
            "freq_error": g.freq_error,
            "ev_loss": g.ev_loss,
            "explanation": g.explanation,
            "points": g.points,
        },
    }


def _store(rv: HandReview) -> int:
    h = rv.hand
    scored = [d for d in rv.decisions if d.scored and d.grade is not None]
    worst = max((d.grade.grade for d in scored if d.grade), key=GRADE_ORDER.__getitem__, default="")
    combo = (
        combo_index(*(card_from_str(c) for c in h.hero_cards)) if len(h.hero_cards) == 2 else None
    )
    hand_class = CLASS_LABELS[int(COMBO_CLASS[combo])] if combo is not None else ""
    with session() as s:
        uid = local_user_id(s)
        rec = HandRecord(
            user_id=uid,
            uid=h.uid,
            site=h.site,
            hand_id=h.hand_id,
            played_at=h.played_at,
            stakes=f"{h.sb:g}/{h.bb:g}",
            hero_cards="".join(h.hero_cards),
            hero_position=rv.hero_pos,
            board=" ".join(h.board),
            net_bb=rv.net_bb,
            effective_bb=rv.effective_bb,
            skip_reason=rv.skip_reason,
            postflop_note=rv.postflop_note,
            n_decisions=len(rv.decisions),
            n_scored=len(scored),
            worst_grade=worst,
            freq_error=sum(d.grade.freq_error for d in scored if d.grade),
            review_json=json.dumps([_decision_json(d) for d in rv.decisions]),
            raw=h.raw,
        )
        s.add(rec)
        s.flush()
        for d in scored:
            g = d.grade
            assert g is not None
            s.add(
                Decision(
                    user_id=uid,
                    mode="review",
                    format_id="6max_100bb",
                    spot_id=d.spot_id or "",
                    scenario=d.scenario,
                    hero_position=d.hero_pos,
                    villain_position=d.villain_pos,
                    street=d.street,
                    node=d.node,
                    board=d.board.replace(" ", ""),
                    hand="".join(h.hero_cards),
                    hand_class=hand_class,
                    hand_category=d.category,
                    chosen_action=g.chosen,
                    best_action=g.best_action,
                    chosen_freq=g.chosen_freq,
                    best_freq=g.best_freq,
                    freqs_json=json.dumps(d.freqs or {}),
                    grade=g.grade,
                    points=g.points,
                    freq_error=g.freq_error,
                    ev_loss_bb=g.ev_loss,
                    ev_loss_pct_pot=(g.ev_loss / d.pot_bb * 100)
                    if g.ev_loss is not None and d.pot_bb
                    else None,
                    source_type="solver" if d.street != "preflop" else "approximate",
                    reason="hh",
                    hand_ref=rec.id,
                )
            )
        rec_id = rec.id
    # Real-play errors become drill reviews (same spot + hand class).
    for d in scored:
        if d.grade and d.grade.grade in ("minor", "mistake") and d.spot_id:
            mode = "preflop" if d.street == "preflop" else "postflop"
            update_item(mode, "6max_100bb", d.spot_id, hand_class, d.grade.grade, node=d.node)
    return rec_id


def import_text(text: str) -> dict[str, Any]:
    hands, errors = parse_many(text)
    reg = get_registry()
    with session() as s:
        uid = local_user_id(s)
        known = set(s.scalars(select(HandRecord.uid).where(HandRecord.user_id == uid)).all())
    report: dict[str, Any] = {
        "parsed": len(hands),
        "imported": 0,
        "duplicates": 0,
        "parse_errors": [{"hand": a, "error": b} for a, b in errors],
        "skipped_hands": Counter(),
        "unscored": Counter(),
        "scored_decisions": 0,
        "flagged_decisions": 0,
    }
    seen: set[str] = set()
    for h in hands:
        if h.uid in known or h.uid in seen:
            report["duplicates"] += 1
            continue
        seen.add(h.uid)
        rv = review_hand(h, reg)
        _store(rv)
        report["imported"] += 1
        if rv.skip_reason:
            report["skipped_hands"][rv.skip_reason] += 1
        for d in rv.decisions:
            if d.scored and d.grade:
                report["scored_decisions"] += 1
                report["flagged_decisions"] += int(d.grade.grade in ("minor", "mistake"))
            else:
                report["unscored"][d.reason] += 1
        if rv.postflop_note and "not scored" in rv.postflop_note:
            reason = rv.postflop_note.split(" (solved:")[0]
            report["unscored"][reason.replace("postflop not scored: ", "postflop: ")] += 1
    report["skipped_hands"] = dict(report["skipped_hands"])
    report["unscored"] = dict(report["unscored"])
    return report


def _hand_row(r: HandRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "site": r.site,
        "hand_id": r.hand_id,
        "played_at": r.played_at.isoformat() if r.played_at else None,
        "stakes": r.stakes,
        "hero_cards": r.hero_cards,
        "hero_position": r.hero_position,
        "board": r.board,
        "net_bb": r.net_bb,
        "effective_bb": r.effective_bb,
        "skip_reason": r.skip_reason,
        "n_decisions": r.n_decisions,
        "n_scored": r.n_scored,
        "worst_grade": r.worst_grade,
        "freq_error": r.freq_error,
    }


def list_hands(flt: str = "all", limit: int = 500) -> list[dict[str, Any]]:
    with session() as s:
        uid = local_user_id(s)
        q = select(HandRecord).where(HandRecord.user_id == uid)
        if flt == "flagged":
            q = q.where(HandRecord.worst_grade.in_(["minor", "mistake"]))
        elif flt == "skipped":
            q = q.where(HandRecord.skip_reason.is_not(None))
        rows = s.scalars(
            q.order_by(HandRecord.played_at.desc(), HandRecord.id.desc()).limit(limit)
        ).all()
        return [_hand_row(r) for r in rows]


def deviations(limit: int = 20) -> list[dict[str, Any]]:
    """Biggest real-play deviations: mistakes before minors, then frequency error, then EV."""
    with session() as s:
        uid = local_user_id(s)
        rows = s.execute(
            select(Decision, HandRecord)
            .join(HandRecord, Decision.hand_ref == HandRecord.id)
            .where(
                Decision.user_id == uid,
                Decision.mode == "review",
                Decision.grade.in_(["minor", "mistake"]),
            )
        ).all()
    items = []
    for d, hr in rows:
        items.append(
            {
                "hand_ref": hr.id,
                "hand_id": hr.hand_id,
                "site": hr.site,
                "played_at": hr.played_at.isoformat() if hr.played_at else None,
                "hero_cards": hr.hero_cards,
                "hero_position": d.hero_position,
                "villain_position": d.villain_position,
                "street": d.street,
                "scenario": d.scenario,
                "spot_id": d.spot_id,
                "board": d.board,
                "chosen": d.chosen_action,
                "best": d.best_action,
                "chosen_freq": d.chosen_freq,
                "best_freq": d.best_freq,
                "grade": d.grade,
                "freq_error": d.freq_error,
                "ev_loss_bb": d.ev_loss_bb,
                "category": d.hand_category,
                "freqs": json.loads(d.freqs_json),
            }
        )
    # Frequency error is rounded so float noise (0.9995 vs 1.0) doesn't outrank a real EV gap.
    items.sort(
        key=lambda x: (
            -GRADE_ORDER[str(x["grade"])],
            -round(float(x["freq_error"]), 2),
            -(x["ev_loss_bb"] or 0.0),
        )
    )
    return items[:limit]


def timeline(h: ParsedHand, positions: dict[str, str]) -> list[dict[str, Any]]:
    """Readable action list in big blinds, with street markers and board cards."""
    out: list[dict[str, Any]] = []
    street = "preflop"
    boards = {"flop": h.board[:3], "turn": h.board[3:4], "river": h.board[4:5]}
    level: dict[str, float] = {}
    for a in h.actions:
        if a.street != street:
            street = a.street
            level = {}
            out.append({"kind": "street", "street": street, "cards": boards.get(street, [])})
        bb = h.bb
        if a.kind.startswith("post"):
            what = {"post_sb": "posts SB", "post_bb": "posts BB", "post_ante": "posts ante"}.get(
                a.kind, "posts"
            )
            text = f"{what} {a.amount / bb:g}bb"
        elif a.kind == "raise":
            text = f"raises to {(a.to or 0) / bb:g}bb"
        elif a.kind in ("bet", "call"):
            text = f"{a.kind}s {a.amount / bb:g}bb"
        else:
            text = f"{a.kind}s"
        if a.allin:
            text += " (all-in)"
        level[a.player] = level.get(a.player, 0.0) + a.amount
        out.append(
            {
                "kind": "action",
                "street": a.street,
                "player": a.player,
                "position": positions.get(a.player, ""),
                "hero": a.player == h.hero,
                "text": text,
            }
        )
    # streets dealt with no further action (all-in run-outs)
    shown = {x["street"] for x in out if x["kind"] == "street"}
    for st in ("flop", "turn", "river"):
        if boards[st] and st not in shown:
            out.append({"kind": "street", "street": st, "cards": boards[st]})
    return out


def hand_detail(hand_ref: int) -> dict[str, Any] | None:
    with session() as s:
        rec = s.get(HandRecord, hand_ref)
        if rec is None or rec.user_id != local_user_id(s):
            return None
        row = _hand_row(rec)
        raw, review = rec.raw, json.loads(rec.review_json)
        note = rec.postflop_note
    h = parse_hand(raw)
    from .review import SkipHand, assign_positions

    try:
        positions = assign_positions(h)
    except SkipHand:
        positions = {}
    return {
        **row,
        "postflop_note": note,
        "decisions": review,
        "raw": raw,
        "timeline": timeline(h, positions),
        "hero": h.hero,
    }


def clear_all() -> int:
    with session() as s:
        uid = local_user_id(s)
        n = len(s.scalars(select(HandRecord.id).where(HandRecord.user_id == uid)).all())
        s.execute(delete(Decision).where(Decision.user_id == uid, Decision.mode == "review"))
        s.execute(delete(HandRecord).where(HandRecord.user_id == uid))
    return n
