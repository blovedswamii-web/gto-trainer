"""Import external strategy exports into the native JSON format.

Three input shapes cover the common solver/chart tools:

``csv``     one row per hand (class "AKs" or combo "AsKs"); columns = action ids holding
            frequencies (0-1 or 0-100), optional ``weight``/``reach`` column and optional
            ``ev_<action>`` columns in bb. (GTO Wizard / spreadsheet-style exports.)
``ranges``  a directory with one ``<action>.txt`` per action containing a weighted range
            string, e.g. ``raise.txt: "AA,KK,AKs,A5s:0.5"``. This is what PioSOLVER/GTO+ give
            you when you export each action's range: the weights are *reach x frequency*,
            so we normalise them back into frequencies. If ``fold.txt`` is missing, fold is
            the remainder of ``meta.reach`` (default: every hand).
``json``    an already-native file: validated and copied.

Every import needs a small ``meta.json``::

    {"format_id": "6max_100bb", "scenario": "vs_open", "hero": "BTN", "villain": "CO",
     "source": {"type": "imported", "label": "GTO Wizard 6max 100bb NL500", "notes": "...",
                "exploitability_pct": 0.25},
     "reach": "optional range string"}

Usage::

    python -m gto_trainer.strategies.importer csv  exports/btn_vs_co.csv --meta meta.json
    python -m gto_trainer.strategies.importer ranges exports/btn_vs_co/  --meta meta.json
    python -m gto_trainer.strategies.importer json  my_spot.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DATA_DIR, FORMATS
from ..poker.cards import CLASS_COMBOS, CLASS_LABELS, N_COMBOS, FloatArray, combo_to_str
from ..poker.ranges import format_range, parse_range
from .context import build_context, spot_id
from .loader import _keys_to_combos, compile_spot, read_file
from .schemas import SourceTag, SpotFile

IMPORT_DIR = DATA_DIR / "strategies" / "imported"


def _meta_spot(meta: dict[str, Any]) -> tuple[SourceTag, list[str], dict[str, Any]]:
    src = meta.get("source", {})
    src.setdefault("type", "imported")
    if src["type"] == "approximate":
        raise ValueError("imports cannot be tagged 'approximate'; use the seed directory")
    fmt = FORMATS[meta.get("format_id", "6max_100bb")]
    ctx = build_context(fmt, meta["scenario"], meta["hero"], meta.get("villain"))
    base: dict[str, Any] = {
        k: meta[k] for k in ("format_id", "scenario", "hero", "villain", "notes") if k in meta
    }
    return SourceTag.model_validate(src), [a.id for a in ctx.actions], base


def _compress(freqs: FloatArray, reach: FloatArray, ids: list[str]) -> dict[str, dict[str, float]]:
    """Per-combo frequencies -> class keys where the class is uniform, combo keys otherwise."""
    out: dict[str, dict[str, float]] = {}
    for k in range(169):
        combos = CLASS_COMBOS[k]
        live = combos[reach[combos] > 0]
        if len(live) == 0:
            continue
        rows = np.round(freqs[live], 4)
        if np.all(rows == rows[0]):
            out[CLASS_LABELS[k]] = {a: float(f) for a, f in zip(ids, rows[0], strict=True) if f > 0}
        else:
            for c, r in zip(live, rows, strict=True):
                out[combo_to_str(int(c))] = {
                    a: float(f) for a, f in zip(ids, r, strict=True) if f > 0
                }
    return out


def import_csv(path: Path, meta: dict[str, Any]) -> SpotFile:
    source, ids, base = _meta_spot(meta)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows")
    cols = {c.strip().lower(): c for c in rows[0]}
    hand_col = cols.get("hand") or cols.get("combo")
    if hand_col is None:
        raise ValueError("CSV needs a 'hand' (or 'combo') column")
    act_cols = {a: cols[a] for a in ids if a in cols}
    if not act_cols:
        raise ValueError(f"CSV has none of the action columns {ids}")
    pct_scale = (
        100.0 if any(float(r[c] or 0) > 1.0 for r in rows for c in act_cols.values()) else 1.0
    )
    weight_col = cols.get("weight") or cols.get("reach")
    strategy: dict[str, dict[str, float]] = {}
    ev: dict[str, dict[str, float]] = {}
    reach = np.zeros(N_COMBOS) if weight_col else None
    for r in rows:
        key = r[hand_col].strip()
        strategy[key] = {a: float(r[c] or 0) / pct_scale for a, c in act_cols.items()}
        evs = {
            a: float(r[cols[f"ev_{a}"]]) for a in ids if f"ev_{a}" in cols and r[cols[f"ev_{a}"]]
        }
        if evs:
            ev[key] = evs
        if reach is not None and weight_col:
            w = float(r[weight_col] or 0)
            reach[_keys_to_combos(key)] = w / 100.0 if w > 1 else w
    return SpotFile(
        **base,
        source=source,
        strategy=strategy,
        ev=ev or None,
        reach=format_range(reach) if reach is not None else meta.get("reach"),
    )


def import_range_files(directory: Path, meta: dict[str, Any]) -> SpotFile:
    source, ids, base = _meta_spot(meta)
    weights = {}
    for a in ids:
        f = directory / f"{a}.txt"
        if f.exists():
            weights[a] = parse_range(f.read_text().strip())
    if not weights:
        raise ValueError(f"no <action>.txt files for actions {ids} in {directory}")
    total = sum(weights.values(), start=np.zeros(N_COMBOS))
    if "fold" in ids and "fold" not in weights:
        reach = parse_range(meta["reach"]) if meta.get("reach") else np.ones(N_COMBOS)
        weights["fold"] = np.clip(reach - total, 0.0, None)
        total = total + weights["fold"]
    reach = np.clip(total, 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        freqs = np.stack(
            [np.where(total > 0, weights.get(a, np.zeros(N_COMBOS)) / total, 0.0) for a in ids],
            axis=1,
        )
    return SpotFile(
        **base, source=source, strategy=_compress(freqs, reach, ids), reach=format_range(reach)
    )


def write_spot(sf: SpotFile, out_dir: Path = IMPORT_DIR) -> Path:
    compile_spot(sf, Path("<import>"))  # raises with a readable message if invalid
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = sf.id or spot_id(sf.format_id, sf.scenario, sf.hero, sf.villain)
    path = out_dir / f"{sid}.json"
    path.write_text(json.dumps(sf.model_dump(exclude_none=True), indent=1))
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("kind", choices=["csv", "ranges", "json"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, default=IMPORT_DIR)
    args = ap.parse_args(argv)
    if args.kind == "json":
        spots = read_file(args.path)
    else:
        if args.meta is None:
            ap.error("--meta is required for csv/ranges imports")
        meta = json.loads(args.meta.read_text())
        spots = [
            import_csv(args.path, meta)
            if args.kind == "csv"
            else import_range_files(args.path, meta)
        ]
    for sf in spots:
        print(f"imported -> {write_spot(sf, args.out)}")


if __name__ == "__main__":
    main()
