"""On-disk and API schemas for strategies.

File format (JSON). A file is either one spot or a *pack* ``{"schema_version": 1, "spots": [...]}``.
Each spot entry::

    {
      "id": "6max_100bb.vs_open.BTN_vs_CO",        # optional, derived if omitted
      "format_id": "6max_100bb",
      "scenario": "vs_open",                        # rfi | vs_open | bb_defense | vs_3bet | vs_4bet
      "hero": "BTN", "villain": "CO",
      "source": {"type": "imported", "label": "GTO Wizard export 2026-10", "notes": "...",
                 "exploitability_pct": 0.3},
      // EITHER per-action range strings (fold = remainder of the reach):
      "ranges": {"raise": "TT+,AJs+,A5s:0.5", "call": "99-22,ATs:0.5"},
      // OR explicit frequencies per class ("AKs") or per combo ("AsKs", overrides its class):
      "strategy": {"AKs": {"raise": 0.8, "call": 0.2}, "AsKs": {"raise": 1.0}},
      "ev": {"AKs": {"raise": 3.1, "call": 2.9, "fold": 0.0}},   // optional, in bb
      "reach": "22+,A2s+,...",                      // optional: hands that arrive at the node
      "actions": [...]                               // optional: derived from the format
    }

``reach`` defaults to: every hand for first decisions (RFI / facing an open), and the hero's
relevant range from the earlier spot when that spot exists (e.g. vs_3bet reach = hero's RFI
raise range).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Scenario = Literal["rfi", "vs_open", "bb_defense", "vs_3bet", "vs_4bet"]
SourceType = Literal["approximate", "imported", "solver"]

SCENARIO_LABELS: dict[str, str] = {
    "rfi": "Raise first in",
    "vs_open": "Facing an open",
    "bb_defense": "BB defence",
    "vs_3bet": "Facing a 3-bet",
    "vs_4bet": "Facing a 4-bet",
}


class SourceTag(BaseModel):
    """Provenance and quality. Every strategy the API returns carries one of these."""

    type: SourceType
    label: str
    notes: str = ""
    exploitability_pct: float | None = None  # % of pot, when known

    @property
    def badge(self) -> str:
        if self.type == "solver" and self.exploitability_pct is not None:
            return f"solver-converged (exploitability {self.exploitability_pct:.2f}% pot)"
        return {
            "approximate": "approximate seed chart",
            "imported": "imported",
            "solver": "solver",
        }[self.type]


class ActionDef(BaseModel):
    id: str  # fold | check | call | raise | allin
    label: str
    size_bb: float | None = None  # total "raise to" size for raises


class HistoryStep(BaseModel):
    position: str
    action: str  # fold | post | raise | call | 3bet | 4bet
    size_bb: float | None = None


class SeatState(BaseModel):
    """One seat at the table, as rendered by the frontend."""

    position: str
    stack_bb: float  # behind, after chips committed
    committed_bb: float  # in front of the seat this street (blinds included)
    folded: bool = False
    is_hero: bool = False
    is_actor: bool = False  # seat whose turn it is
    is_villain: bool = False  # the opponent hero's decision is against
    last_action: str | None = None  # raise | 3bet | 4bet | call | check | bet | allin


class TableState(BaseModel):
    seats: list[SeatState]  # table order, first to act preflop first
    dealer: str
    pot_bb: float
    to_call_bb: float
    board: list[str] = Field(default_factory=list)


class SpotFile(BaseModel):
    """One spot as written in a JSON file (loose: most fields are derivable)."""

    id: str | None = None
    format_id: str = "6max_100bb"
    scenario: Scenario
    hero: str
    villain: str | None = None
    source: SourceTag
    actions: list[ActionDef] | None = None
    ranges: dict[str, str] | None = None
    strategy: dict[str, dict[str, float]] | None = None
    ev: dict[str, dict[str, float]] | None = None
    reach: str | None = None
    default_action: str = "fold"
    notes: str = ""

    @model_validator(mode="after")
    def _one_strategy_form(self) -> SpotFile:
        if (self.ranges is None) == (self.strategy is None):
            raise ValueError("provide exactly one of 'ranges' or 'strategy'")
        if self.scenario != "rfi" and self.villain is None:
            raise ValueError(f"scenario {self.scenario} needs a villain position")
        return self


class StrategyPack(BaseModel):
    schema_version: int = 1
    spots: list[SpotFile] = Field(default_factory=list)


# ----------------------------------------------------------------------------- API models


class SpotSummary(BaseModel):
    id: str
    format_id: str
    scenario: Scenario
    scenario_label: str
    hero: str
    villain: str | None
    title: str
    source: SourceTag


class GridCell(BaseModel):
    label: str
    combos: float  # combos of this class that reach the node (weighted)
    reach: float  # fraction of the class reaching the node
    freqs: dict[str, float]  # reach-weighted action frequencies
    ev: dict[str, float] | None = None


class ValueBluff(BaseModel):
    action: str
    value_combos: float
    bluff_combos: float
    ratio: float | None
    method: str


class SpotStats(BaseModel):
    reach_combos: float
    played_pct: float  # share of reaching combos that do not fold
    vpip_pct_of_all: float  # share of all 1326 combos that do not fold here
    combos_by_action: dict[str, float]
    pct_by_action: dict[str, float]
    value_bluff: list[ValueBluff]


class SpotDetail(BaseModel):
    summary: SpotSummary
    history: list[HistoryStep]
    pot_bb: float
    to_call_bb: float
    actions: list[ActionDef]
    grid: list[GridCell]
    stats: SpotStats
    notes: str
    villain_range_text: str | None
    table: TableState | None = None
