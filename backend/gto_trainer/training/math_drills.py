"""Betting-maths drills: pot odds, MDF and "how often must this bluff work".

Questions are generated from realistic bet sizes (fractions of pot) and graded with a
tolerance in percentage points; every answer is logged like any other decision (mode
"math") so it shows up in progress stats.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Literal

from ..db.models import Decision
from ..db.session import local_user_id, session
from ..poker import odds

Kind = Literal["pot_odds", "mdf", "bluff"]
KINDS: tuple[Kind, ...] = ("pot_odds", "mdf", "bluff")
SIZES = (0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0)
PROMPTS = {
    "pot_odds": "Villain bets {bet}bb into {pot}bb. What equity do you need to call?",
    "mdf": "Villain bets {bet}bb into {pot}bb. What is your minimum defence frequency?",
    "bluff": "You bet {bet}bb into {pot}bb as a pure bluff. How often must villain fold?",
}
EXPLAIN = {
    "pot_odds": "call / (pot + 2*bet) = {bet} / ({pot} + 2*{bet})",
    "mdf": "pot / (pot + bet) = {pot} / ({pot} + {bet})",
    "bluff": "bet / (pot + bet) = {bet} / ({pot} + {bet})",
}


@dataclass(frozen=True)
class MathQuestion:
    kind: Kind
    pot: float
    bet: float
    prompt: str
    answer: float  # fraction 0..1

    @property
    def explanation(self) -> str:
        return EXPLAIN[self.kind].format(pot=self.pot, bet=self.bet)


def make_question(kind: Kind | None = None, rng: random.Random | None = None) -> MathQuestion:
    r = rng or random.Random()
    k: Kind = kind if kind is not None else r.choice(KINDS)
    pot = float(r.choice([4, 6, 10, 12, 20, 24, 36, 50, 80, 100]))
    bet = round(pot * r.choice(SIZES), 1)
    fn = {
        "pot_odds": odds.pot_odds,
        "mdf": odds.minimum_defence_frequency,
        "bluff": odds.bluff_breakeven,
    }[k]
    return MathQuestion(k, pot, bet, PROMPTS[k].format(pot=pot, bet=bet), fn(pot, bet))


def grade_answer(q: MathQuestion, guess_pct: float, tolerance_pp: float = 3.0) -> dict[str, object]:
    err = abs(guess_pct - q.answer * 100)
    grade = "best" if err <= tolerance_pp else ("minor" if err <= 2 * tolerance_pp else "mistake")
    points = {"best": 1.0, "minor": 0.5, "mistake": 0.0}[grade]
    with session() as s:
        d = Decision(
            user_id=local_user_id(s),
            mode="math",
            format_id="-",
            spot_id=f"math.{q.kind}",
            scenario=q.kind,
            hero_position="-",
            street="-",
            hand="-",
            hand_class="-",
            hand_category=f"bet {q.bet / q.pot:.0%} pot",
            chosen_action=f"{guess_pct:.1f}",
            best_action=f"{q.answer * 100:.1f}",
            chosen_freq=0.0,
            best_freq=1.0,
            freqs_json=json.dumps({"pot": q.pot, "bet": q.bet}),
            grade=grade,
            points=points,
            freq_error=min(1.0, err / 100),
            source_type="exact",
        )
        s.add(d)
    return {
        "grade": grade,
        "correct_pct": q.answer * 100,
        "error_pp": err,
        "explanation": q.explanation,
        "points": points,
    }
