"""Grading decisions against a *mixed* strategy.

Why frequencies, not "the right answer"
---------------------------------------
At equilibrium every action a hand takes with positive frequency has the same EV (that is
what makes mixing rational). So choosing a 40% action is not 60% wrong -- it is a perfectly
good play. What *is* wrong is choosing an action the solution essentially never takes,
because for that action there is no indifference and it loses EV.

Grades (thresholds in :class:`gto_trainer.config.GradingConfig`):

* ``best``    -- the highest-frequency action
* ``good``    -- any action played >= 30% (part of the intended mix)
* ``minor``   -- played 3-30%: legal but a deviation if you do it every time
* ``mistake`` -- played < 3%: not part of the strategy

Scores tracked per decision:

* ``freq_error = max_f - f(chosen)`` in [0, 1] -- a model-free distance from the mix that
  works for seed charts with no EVs. 0 for the top action, ~1 for a pure-strategy blunder.
* ``ev_loss = max EV - EV(chosen)`` -- only when the source provides per-action EVs
  (imported solver data, our own postflop solver). We never invent EVs for seed charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import GRADING, GradingConfig

Grade = Literal["best", "good", "minor", "mistake"]
GRADE_POINTS: dict[str, float] = {"best": 1.0, "good": 1.0, "minor": 0.5, "mistake": 0.0}


@dataclass(frozen=True)
class GradeResult:
    grade: Grade
    chosen: str
    chosen_freq: float
    best_action: str
    best_freq: float
    freq_error: float
    ev_loss: float | None
    explanation: str

    @property
    def points(self) -> float:
        return GRADE_POINTS[self.grade]


def grade_decision(
    freqs: dict[str, float],
    chosen: str,
    ev: dict[str, float] | None = None,
    labels: dict[str, str] | None = None,
    cfg: GradingConfig = GRADING,
) -> GradeResult:
    if chosen not in freqs:
        raise ValueError(f"action {chosen!r} is not available here ({sorted(freqs)})")
    total = sum(freqs.values())
    if total <= 0:
        raise ValueError("strategy has no mass for this hand")
    f = {a: v / total for a, v in freqs.items()}
    best = max(f, key=lambda a: f[a])
    fc, fb = f[chosen], f[best]
    name = (labels or {}).get

    if chosen == best or fc >= fb - 1e-9:
        grade: Grade = "best"
    elif fc >= cfg.good_freq:
        grade = "good"
    elif fc >= cfg.minor_freq:
        grade = "minor"
    else:
        grade = "mistake"

    ev_loss = None
    if ev is not None and chosen in ev:
        # EV of the best *available* action, not the most frequent one: with imperfect
        # convergence the most frequent action is not always the highest-EV one.
        ev_loss = max(0.0, max(ev.values()) - ev[chosen])

    chosen_l, best_l = name(chosen) or chosen, name(best) or best
    if grade == "best":
        text = f"{chosen_l} is the solution's main play here ({fc:.0%})."
    elif grade == "good":
        text = (
            f"{chosen_l} is part of the mix ({fc:.0%}); "
            f"the most frequent play is {best_l} ({fb:.0%})."
        )
    elif grade == "minor":
        text = (
            f"{chosen_l} is only a {fc:.0%} play here. Fine occasionally, a leak if you "
            f"always do it — the solution prefers {best_l} ({fb:.0%})."
        )
    else:
        text = (
            f"{chosen_l} is essentially never played here ({fc:.1%}). "
            f"The solution plays {best_l} ({fb:.0%})."
        )
    if ev_loss is not None:
        text += f" EV loss ≈ {ev_loss:.2f}bb."
    return GradeResult(grade, chosen, fc, best, fb, fb - fc, ev_loss, text)
