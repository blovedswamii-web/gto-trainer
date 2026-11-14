"""Load, validate and compile preflop strategy files into combo-level arrays.

Compiled form: ``freqs`` (1326, A) action frequencies per combo and ``reach`` (1326,) -- the
weight with which each combo arrives at the node. Everything the UI shows (grid cells,
summary stats, drill answers) is derived from those two arrays, so imported per-combo
solver data and class-level seed charts go through exactly the same code.

Precedence: files under ``data/strategies/imported`` replace seed spots with the same id.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ..config import FORMATS, STRATEGY_DIRS
from ..poker.cards import (
    CLASS_COMBOS,
    CLASS_LABELS,
    COMBO_CLASS,
    N_COMBOS,
    FloatArray,
    combo_from_str,
)
from ..poker.preflop_equity import class_equity_vs_range
from ..poker.ranges import RangeParseError, _token_classes, class_weights, format_range, parse_range
from .context import SpotContext, build_context, spot_id
from .schemas import (
    SCENARIO_LABELS,
    GridCell,
    SourceTag,
    SpotDetail,
    SpotFile,
    SpotStats,
    SpotSummary,
    StrategyPack,
    ValueBluff,
)

_COMBO_KEY = re.compile(r"^[2-9TJQKA][cdhs][2-9TJQKA][cdhs]$", re.IGNORECASE)
_SCENARIO_ORDER = {"rfi": 0, "vs_open": 1, "bb_defense": 1, "vs_3bet": 2, "vs_4bet": 3}
AGGRESSIVE = {"raise", "allin"}


class StrategyError(ValueError):
    pass


@dataclass
class CompiledSpot:
    id: str
    file: SpotFile
    ctx: SpotContext
    action_ids: list[str]
    freqs: FloatArray  # (1326, A)
    reach: FloatArray  # (1326,)
    ev: FloatArray | None  # (1326, A) bb, or None
    path: Path
    villain_range: FloatArray | None = None
    villain_continue: FloatArray | None = None  # villain's call+raise range vs hero's raise
    warnings: list[str] = field(default_factory=list)

    @property
    def source(self) -> SourceTag:
        return self.file.source

    def action_index(self, action: str) -> int:
        return self.action_ids.index(action)

    def combo_freqs(self, combo: int) -> dict[str, float]:
        return {a: float(self.freqs[combo, i]) for i, a in enumerate(self.action_ids)}

    def action_range(self, action: str) -> FloatArray:
        """Weights of combos that arrive here *and* take ``action``."""
        return self.reach * self.freqs[:, self.action_index(action)]

    def summary(self) -> SpotSummary:
        f = self.file
        return SpotSummary(
            id=self.id,
            format_id=f.format_id,
            scenario=f.scenario,
            scenario_label=SCENARIO_LABELS[f.scenario],
            hero=f.hero,
            villain=f.villain,
            title=self.ctx.title,
            source=f.source,
        )

    def class_view(self) -> tuple[FloatArray, FloatArray, FloatArray | None]:
        """(class combos (169,), class freqs (169, A), class ev (169, A) | None)."""
        combos = np.bincount(COMBO_CLASS, weights=self.reach, minlength=169)
        weighted = np.zeros((169, len(self.action_ids)))
        for i in range(len(self.action_ids)):
            weighted[:, i] = np.bincount(
                COMBO_CLASS, weights=self.reach * self.freqs[:, i], minlength=169
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            freqs = np.where(combos[:, None] > 0, weighted / combos[:, None], 0.0)
            ev = None
            if self.ev is not None:
                ev = np.zeros_like(freqs)
                for i in range(len(self.action_ids)):
                    ev[:, i] = np.bincount(
                        COMBO_CLASS, weights=self.reach * self.ev[:, i], minlength=169
                    )
                ev = np.where(combos[:, None] > 0, ev / combos[:, None], 0.0)
        return combos, freqs, ev

    def stats(self) -> SpotStats:
        total = float(self.reach.sum())
        by_action = {a: float(self.action_range(a).sum()) for a in self.action_ids}
        fold = by_action.get("fold", 0.0)
        vb: list[ValueBluff] = []
        # Value vs bluff, heuristically: a raise is "value" if it is ahead (>=50% all-in
        # equity) of the range that *continues* against it. If we have the villain's
        # response spot we use its call+raise range; otherwise villain's range at this node,
        # which over-counts value (weak hands that fold to the raise inflate our equity).
        target, what = self.villain_continue, "villain's continuing (call/raise) range"
        if target is None or target.sum() == 0:
            target, what = self.villain_range, "villain's whole range here (no response spot)"
        if target is not None and target.sum() > 0:
            eq_cls = class_equity_vs_range(class_weights(target))
            if eq_cls is not None:
                eq = eq_cls[COMBO_CLASS]
                for a in self.action_ids:
                    if a not in AGGRESSIVE:
                        continue
                    w = self.action_range(a)
                    value = float(w[eq >= 0.5].sum())
                    bluff = float(w[eq < 0.5].sum())
                    vb.append(
                        ValueBluff(
                            action=a,
                            value_combos=value,
                            bluff_combos=bluff,
                            ratio=value / bluff if bluff > 0 else None,
                            method=f"heuristic: value = >=50% all-in equity vs {what}",
                        )
                    )
        return SpotStats(
            reach_combos=total,
            played_pct=(total - fold) / total if total else 0.0,
            vpip_pct_of_all=(total - fold) / N_COMBOS,
            combos_by_action=by_action,
            pct_by_action={a: (v / total if total else 0.0) for a, v in by_action.items()},
            value_bluff=vb,
        )

    def detail(self) -> SpotDetail:
        combos, freqs, ev = self.class_view()
        grid = []
        for k in range(169):
            n = float(combos[k])
            grid.append(
                GridCell(
                    label=CLASS_LABELS[k],
                    combos=n,
                    reach=n / len(CLASS_COMBOS[k]),
                    freqs={a: float(freqs[k, i]) for i, a in enumerate(self.action_ids)}
                    if n
                    else {},
                    ev=(
                        {a: float(ev[k, i]) for i, a in enumerate(self.action_ids)}
                        if ev is not None and n
                        else None
                    ),
                )
            )
        notes = self.file.notes
        if self.warnings:
            notes = (notes + "\n" if notes else "") + "Loader warnings: " + "; ".join(self.warnings)
        return SpotDetail(
            summary=self.summary(),
            history=self.ctx.history,
            pot_bb=self.ctx.pot_bb,
            to_call_bb=self.ctx.to_call_bb,
            actions=self.ctx.actions,
            grid=grid,
            stats=self.stats(),
            notes=notes,
            villain_range_text=(
                format_range(self.villain_range) if self.villain_range is not None else None
            ),
            table=self.ctx.table,
        )


def _keys_to_combos(key: str) -> list[int]:
    if _COMBO_KEY.match(key):
        return [combo_from_str(key)]
    return [int(c) for cls in _token_classes(key) for c in CLASS_COMBOS[cls]]


def compile_spot(sf: SpotFile, path: Path) -> CompiledSpot:
    fmt = FORMATS.get(sf.format_id)
    if fmt is None:
        raise StrategyError(f"unknown format_id {sf.format_id!r}")
    ctx = build_context(fmt, sf.scenario, sf.hero, sf.villain)
    actions = sf.actions or ctx.actions
    if sf.actions:
        ctx = replace(ctx, actions=sf.actions, continue_spot=None)
    ids = [a.id for a in actions]
    if sf.default_action not in ids:
        raise StrategyError(f"default_action {sf.default_action!r} not in actions {ids}")
    sid = sf.id or spot_id(sf.format_id, sf.scenario, sf.hero, sf.villain)
    d = ids.index(sf.default_action)
    freqs = np.zeros((N_COMBOS, len(ids)))
    try:
        if sf.ranges is not None:
            for a, text in sf.ranges.items():
                if a not in ids:
                    raise StrategyError(f"{sid}: range given for unknown action {a!r}")
                if a == sf.default_action:
                    raise StrategyError(f"{sid}: the default action {a!r} is implied, omit it")
                freqs[:, ids.index(a)] = parse_range(text)
            over = freqs.sum(axis=1) > 1 + 1e-9
            if over.any():
                bad = sorted({CLASS_LABELS[COMBO_CLASS[i]] for i in np.flatnonzero(over)})
                raise StrategyError(f"{sid}: action weights exceed 100% for {', '.join(bad)}")
            freqs[:, d] = 1.0 - freqs.sum(axis=1)
        else:
            assert sf.strategy is not None
            freqs[:, d] = 1.0
            # classes/tokens first, explicit combos last so they override
            for key in sorted(sf.strategy, key=lambda k: bool(_COMBO_KEY.match(k))):
                row = np.zeros(len(ids))
                for a, f in sf.strategy[key].items():
                    if a not in ids:
                        raise StrategyError(f"{sid}: {key} uses unknown action {a!r}")
                    row[ids.index(a)] = f
                s = row.sum()
                if not 0.98 <= s <= 1.02:
                    raise StrategyError(f"{sid}: frequencies for {key} sum to {s:.3f}")
                freqs[_keys_to_combos(key)] = row / s
    except RangeParseError as e:
        raise StrategyError(f"{sid}: {e}") from e

    ev = None
    if sf.ev:
        ev = np.zeros((N_COMBOS, len(ids)))
        for key in sorted(sf.ev, key=lambda k: bool(_COMBO_KEY.match(k))):
            for a, v in sf.ev[key].items():
                if a in ids:
                    ev[_keys_to_combos(key), ids.index(a)] = v
    reach = parse_range(sf.reach) if sf.reach else np.ones(N_COMBOS)
    return CompiledSpot(
        id=sid, file=sf, ctx=ctx, action_ids=ids, freqs=freqs, reach=reach, ev=ev, path=path
    )


def read_file(path: Path) -> list[SpotFile]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "spots" in raw:
        return StrategyPack.model_validate(raw).spots
    return [SpotFile.model_validate(raw)]


@dataclass
class Registry:
    spots: dict[str, CompiledSpot] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def find(
        self, fmt_id: str, scenario: str, hero: str, villain: str | None
    ) -> CompiledSpot | None:
        return self.spots.get(spot_id(fmt_id, scenario, hero, villain))  # type: ignore[arg-type]

    def list(self, fmt_id: str | None = None, scenario: str | None = None) -> list[CompiledSpot]:
        out = [
            s
            for s in self.spots.values()
            if (fmt_id is None or s.file.format_id == fmt_id)
            and (scenario is None or s.file.scenario == scenario)
        ]
        pos_order = {p: i for i, p in enumerate(["UTG", "HJ", "CO", "BTN", "SB", "BB"])}
        return sorted(
            out,
            key=lambda s: (
                _SCENARIO_ORDER[s.file.scenario],
                pos_order.get(s.file.hero, 9),
                pos_order.get(s.file.villain or "", 9),
            ),
        )


def load_registry(dirs: list[Path] | None = None) -> Registry:
    reg = Registry()
    for d in dirs or STRATEGY_DIRS:
        if not d.exists():
            continue
        for path in sorted(d.rglob("*.json")):
            try:
                files = read_file(path)
            except (ValidationError, json.JSONDecodeError) as e:
                reg.errors.append(f"{path.name}: {e}")
                continue
            for sf in files:
                try:
                    cs = compile_spot(sf, path)
                except (StrategyError, ValueError) as e:
                    reg.errors.append(f"{path.name}: {e}")
                    continue
                if cs.id in reg.spots and cs.source.type == "approximate":
                    continue  # never let a seed chart shadow an import
                reg.spots[cs.id] = cs

    # Resolve cross-spot references in scenario order (rfi -> vs_open -> vs_3bet -> vs_4bet).
    for cs in sorted(reg.spots.values(), key=lambda s: _SCENARIO_ORDER[s.file.scenario]):
        fmt_id = cs.file.format_id
        if cs.file.reach is None and cs.ctx.reach_spot is not None:
            sc, h, v, a = cs.ctx.reach_spot
            ref = reg.find(fmt_id, sc, h, v)
            if ref is not None:
                cs.reach = ref.action_range(a)
            else:
                cs.warnings.append(f"no {sc} spot for {h} to derive reach; using all hands")
        if cs.ctx.villain_spot is not None:
            sc, h, v, a = cs.ctx.villain_spot
            ref = reg.find(fmt_id, sc, h, v)
            if ref is not None:
                cs.villain_range = ref.action_range(a)
        if cs.ctx.continue_spot is not None:
            sc, h, v, acts = cs.ctx.continue_spot
            ref = reg.find(fmt_id, sc, h, v)
            if ref is not None:
                cs.villain_continue = sum(
                    (ref.action_range(x) for x in acts if x in ref.action_ids),
                    start=np.zeros(N_COMBOS),
                )
    return reg


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    return load_registry()


def reload_registry() -> Registry:
    get_registry.cache_clear()
    return get_registry()


if __name__ == "__main__":  # python -m gto_trainer.strategies.loader  -> validate all files
    r = load_registry()
    for s in r.list():
        st = s.stats()
        print(
            f"{s.id:42s} reach {st.reach_combos:7.1f}  played {st.played_pct:6.1%}  "
            f"[{s.source.type}]"
        )
    for e in r.errors:
        print("ERROR", e)
    raise SystemExit(1 if r.errors else 0)
