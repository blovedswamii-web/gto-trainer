"""Persisting decisions (the single source of truth for stats, leaks and SRS)."""

from __future__ import annotations

import json

from ..db.models import Decision
from ..db.session import local_user_id, session
from ..poker.cards import CLASS_LABELS, COMBO_CLASS, combo_to_str
from ..poker.categories import preflop_category
from ..strategies.loader import CompiledSpot
from .grading import GradeResult
from .srs import update_item


def record_preflop_decision(
    spot: CompiledSpot, combo: int, g: GradeResult, response_ms: int | None, reason: str = "new"
) -> int:
    label = CLASS_LABELS[int(COMBO_CLASS[combo])]
    freqs = spot.combo_freqs(combo)
    with session() as s:
        d = Decision(
            user_id=local_user_id(s),
            mode="preflop",
            format_id=spot.file.format_id,
            spot_id=spot.id,
            scenario=spot.file.scenario,
            hero_position=spot.file.hero,
            villain_position=spot.file.villain,
            street="preflop",
            hand=combo_to_str(combo),
            hand_class=label,
            hand_category=preflop_category(label),
            chosen_action=g.chosen,
            best_action=g.best_action,
            chosen_freq=g.chosen_freq,
            best_freq=g.best_freq,
            freqs_json=json.dumps(freqs),
            grade=g.grade,
            points=g.points,
            freq_error=g.freq_error,
            ev_loss_bb=g.ev_loss,
            ev_loss_pct_pot=(g.ev_loss / spot.ctx.pot_bb * 100) if g.ev_loss is not None else None,
            source_type=spot.source.type,
            response_ms=response_ms,
            reason=reason if reason in ("new", "review") else "new",
        )
        s.add(d)
        s.flush()
        did = d.id
    update_item("preflop", spot.file.format_id, spot.id, label, g.grade)
    return did
