"""Derive preflop action history, pot and action menu for a spot from the game format.

Keeping this derivation in one place means strategy files only state *who* and *what*
(scenario, hero, villain, frequencies); sizes always agree with :class:`GameFormat`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import GameFormat
from .schemas import ActionDef, HistoryStep, Scenario


@dataclass(frozen=True)
class SpotContext:
    history: list[HistoryStep]
    pot_bb: float
    to_call_bb: float
    actions: list[ActionDef]
    title: str
    villain_spot: tuple[str, str, str | None, str] | None  # (scenario, hero, villain, action)
    reach_spot: tuple[str, str, str | None, str] | None
    # Villain's response spot to hero's aggression: (scenario, hero, villain, continuing actions)
    continue_spot: tuple[str, str, str | None, tuple[str, ...]] | None = None


def spot_id(fmt_id: str, scenario: Scenario, hero: str, villain: str | None) -> str:
    return f"{fmt_id}.{scenario}.{hero}" + (f"_vs_{villain}" if villain else "")


def build_context(
    fmt: GameFormat, scenario: Scenario, hero: str, villain: str | None
) -> SpotContext:
    pos = fmt.positions
    if hero not in pos or (villain is not None and villain not in pos):
        raise ValueError(f"unknown position for {fmt.id}: {hero}/{villain}")
    contrib = dict.fromkeys(pos, 0.0)
    contrib["SB" if "SB" in pos else "BTN"] = 0.5
    contrib["BB"] = 1.0
    hist: list[HistoryStep] = []

    folded: set[str] = set()

    def fold_between(start: str, stop: str) -> None:
        """Everyone still in the hand strictly between start and stop (wrapping) folds."""
        i = (pos.index(start) + 1) % len(pos)
        while pos[i] != stop:
            if pos[i] not in folded:
                hist.append(HistoryStep(position=pos[i], action="fold"))
                folded.add(pos[i])
            i = (i + 1) % len(pos)

    def fold_before(p: str) -> None:
        for q in pos[: pos.index(p)]:
            hist.append(HistoryStep(position=q, action="fold"))
            folded.add(q)

    def raise_to(p: str, size: float, kind: str) -> None:
        contrib[p] = size
        hist.append(HistoryStep(position=p, action=kind, size_bb=size))

    villain_spot: tuple[str, str, str | None, str] | None = None
    reach_spot: tuple[str, str, str | None, str] | None = None
    continue_spot: tuple[str, str, str | None, tuple[str, ...]] | None = None
    if scenario == "rfi":
        fold_before(hero)
        size = fmt.open_size(hero)
        actions = [
            ActionDef(id="fold", label="Fold"),
            ActionDef(id="raise", label=f"Raise {size:g}bb", size_bb=size),
        ]
        title = f"{hero} raise first in"
    elif scenario in ("vs_open", "bb_defense"):
        assert villain is not None
        if pos.index(villain) >= pos.index(hero):
            raise ValueError(f"{villain} cannot open before {hero} acts")
        fold_before(villain)
        o = fmt.open_size(villain)
        raise_to(villain, o, "raise")
        fold_between(villain, hero)
        tb = fmt.three_bet_size(villain, hero)
        actions = [
            ActionDef(id="fold", label="Fold"),
            ActionDef(id="call", label=f"Call {o - contrib[hero]:g}bb", size_bb=o),
            ActionDef(id="raise", label=f"3-bet to {tb:g}bb", size_bb=tb),
        ]
        title = f"{hero} vs {villain} open"
        villain_spot = ("rfi", villain, None, "raise")
        continue_spot = ("vs_3bet", villain, hero, ("call", "raise"))
    elif scenario == "vs_3bet":
        assert villain is not None
        fold_before(hero)
        o = fmt.open_size(hero)
        raise_to(hero, o, "raise")
        fold_between(hero, villain)
        tb = fmt.three_bet_size(hero, villain)
        raise_to(villain, tb, "3bet")
        fold_between(villain, hero)
        fb = fmt.four_bet_size(hero, villain)
        actions = [
            ActionDef(id="fold", label="Fold"),
            ActionDef(id="call", label=f"Call {tb - contrib[hero]:g}bb", size_bb=tb),
            ActionDef(id="raise", label=f"4-bet to {fb:g}bb", size_bb=fb),
        ]
        title = f"{hero} open, {villain} 3-bets"
        vs = "bb_defense" if villain == "BB" else "vs_open"
        villain_spot = (vs, villain, hero, "raise")
        reach_spot = ("rfi", hero, None, "raise")
        continue_spot = ("vs_4bet", villain, hero, ("call", "allin"))
    elif scenario == "vs_4bet":
        assert villain is not None
        fold_before(villain)
        o = fmt.open_size(villain)
        raise_to(villain, o, "raise")
        fold_between(villain, hero)
        tb = fmt.three_bet_size(villain, hero)
        raise_to(hero, tb, "3bet")
        fold_between(hero, villain)
        fb = fmt.four_bet_size(villain, hero)
        raise_to(villain, fb, "4bet")
        actions = [
            ActionDef(id="fold", label="Fold"),
            ActionDef(id="call", label=f"Call {fb - contrib[hero]:g}bb", size_bb=fb),
            ActionDef(id="allin", label=f"5-bet all-in ({fmt.stack_bb:g}bb)", size_bb=fmt.stack_bb),
        ]
        title = f"{hero} 3-bet vs {villain}, facing a 4-bet"
        villain_spot = ("vs_3bet", villain, hero, "raise")
        vs = "bb_defense" if hero == "BB" else "vs_open"
        reach_spot = (vs, hero, villain, "raise")
    else:  # pragma: no cover - Literal exhausts this
        raise ValueError(scenario)

    pot = sum(contrib.values())
    to_call = max(contrib.values()) - contrib[hero]
    return SpotContext(
        hist,
        round(pot, 2),
        round(to_call, 2),
        actions,
        title,
        villain_spot,
        reach_spot,
        continue_spot,
    )
