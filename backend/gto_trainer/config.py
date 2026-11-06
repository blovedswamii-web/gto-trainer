"""Game formats and app settings.

The game format is data, not code: every preflop sizing rule the spot builder and the
postflop solver need is derived from a :class:`GameFormat`, so adding 40bb / 200bb / heads-up
means adding an entry to :data:`FORMATS` plus strategy files for it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field

BACKEND_DIR: Final = Path(__file__).resolve().parent.parent
DATA_DIR: Final = Path(os.environ.get("GTO_DATA_DIR", BACKEND_DIR / "data"))
STRATEGY_DIRS: Final = [DATA_DIR / "strategies" / "seed", DATA_DIR / "strategies" / "imported"]
SOLUTIONS_DIR: Final = DATA_DIR / "solutions"
DB_PATH: Final = Path(os.environ.get("GTO_DB_PATH", DATA_DIR / "trainer.db"))

_POSITIONS: Final = {
    2: ["BTN", "BB"],  # heads-up: the button posts the small blind
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
}


class GameFormat(BaseModel):
    id: str
    name: str
    players: int = Field(ge=2, le=6)
    stack_bb: float = Field(gt=0)
    open_size_bb: float = 2.5
    sb_open_size_bb: float = 3.0
    three_bet_ip_mult: float = 3.0  # 3-bet size as a multiple of the open, in position
    three_bet_oop_mult: float = 4.0  # ... out of position (blinds 3-betting)
    four_bet_mult: float = 2.2  # 4-bet size as a multiple of the 3-bet
    rake_pct: float = 0.0  # ignored by the default format; kept so imports can declare it

    @property
    def positions(self) -> list[str]:
        """Preflop action order."""
        return list(_POSITIONS[self.players])

    @property
    def postflop_order(self) -> list[str]:
        """Postflop action order (first to act first)."""
        pos = self.positions
        if self.players == 2:
            return ["BB", "BTN"]
        return pos[-2:] + pos[:-2]

    def in_position(self, a: str, b: str) -> bool:
        """True if ``a`` acts after ``b`` postflop."""
        order = self.postflop_order
        return order.index(a) > order.index(b)

    def open_size(self, opener: str) -> float:
        if opener == "SB" or (self.players == 2 and opener == "BTN"):
            return self.sb_open_size_bb
        return self.open_size_bb

    def three_bet_size(self, opener: str, three_bettor: str) -> float:
        mult = (
            self.three_bet_ip_mult
            if self.in_position(three_bettor, opener)
            else (self.three_bet_oop_mult)
        )
        return round(self.open_size(opener) * mult, 1)

    def four_bet_size(self, opener: str, three_bettor: str) -> float:
        return round(self.three_bet_size(opener, three_bettor) * self.four_bet_mult, 1)


FORMATS: Final[dict[str, GameFormat]] = {
    "6max_100bb": GameFormat(
        id="6max_100bb", name="6-max NLHE cash, 100bb", players=6, stack_bb=100
    ),
    # Placeholders that become usable as soon as strategy files for them are added:
    "6max_40bb": GameFormat(id="6max_40bb", name="6-max NLHE, 40bb", players=6, stack_bb=40),
    "6max_200bb": GameFormat(
        id="6max_200bb", name="6-max NLHE cash, 200bb", players=6, stack_bb=200
    ),
    "hu_100bb": GameFormat(id="hu_100bb", name="Heads-up NLHE, 100bb", players=2, stack_bb=100),
}


class GradingConfig(BaseModel):
    """Mixed-strategy grading thresholds (fraction of the solution's frequency)."""

    good_freq: float = 0.30  # >= this: a legitimate part of the mix
    minor_freq: float = 0.03  # below this the action is treated as "not in the solution"


GRADING: Final = GradingConfig()
