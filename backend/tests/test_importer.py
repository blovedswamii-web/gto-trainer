"""Importing solver exports (CSV and per-action range files) into the native format."""

from __future__ import annotations

from pathlib import Path

import pytest

from gto_trainer.poker.cards import combo_from_str
from gto_trainer.strategies.importer import import_csv, import_range_files, write_spot
from gto_trainer.strategies.loader import compile_spot, load_registry

META = {
    "format_id": "6max_100bb",
    "scenario": "vs_open",
    "hero": "BTN",
    "villain": "CO",
    "source": {"type": "imported", "label": "unit-test export", "exploitability_pct": 0.3},
}


def test_csv_import_percent_and_ev(tmp_path: Path) -> None:
    f = tmp_path / "x.csv"
    f.write_text(
        "hand,fold,call,raise,ev_fold,ev_call,ev_raise\n"
        "AKs,0,25,75,0,3.1,3.2\nT9s,10,90,0,0,0.4,0.1\nAsKs,0,0,100,0,3.0,3.3\n"
    )
    sf = import_csv(f, META)
    cs = compile_spot(sf, f)
    assert cs.source.type == "imported"
    assert cs.combo_freqs(combo_from_str("AhKh"))["raise"] == pytest.approx(0.75)
    assert cs.combo_freqs(combo_from_str("AsKs"))["raise"] == pytest.approx(1.0)
    assert cs.combo_freqs(combo_from_str("9h8h"))["fold"] == 1.0  # absent hands fold
    assert cs.ev is not None


def test_range_files_normalise_reach_times_frequency(tmp_path: Path) -> None:
    # Pio-style exports: each file holds reach x frequency for that action.
    (tmp_path / "raise.txt").write_text("AA,KK,AKs:0.5")
    (tmp_path / "call.txt").write_text("AKs:0.5,QQ:0.8")
    sf = import_range_files(tmp_path, META)
    cs = compile_spot(sf, tmp_path)
    assert cs.combo_freqs(combo_from_str("AhKh")) == pytest.approx(
        {"fold": 0.0, "call": 0.5, "raise": 0.5}
    )
    assert cs.combo_freqs(combo_from_str("QhQd"))["call"] == pytest.approx(0.8)
    assert cs.combo_freqs(combo_from_str("QhQd"))["fold"] == pytest.approx(0.2)


def test_written_import_overrides_seed(tmp_path: Path) -> None:
    (tmp_path / "raise.txt").write_text("AA")
    out = write_spot(import_range_files(tmp_path, META), tmp_path / "imported")
    from gto_trainer.config import STRATEGY_DIRS

    reg = load_registry([STRATEGY_DIRS[0], out.parent])
    spot = reg.spots["6max_100bb.vs_open.BTN_vs_CO"]
    assert spot.source.type == "imported"
    assert spot.combo_freqs(combo_from_str("KhKd"))["fold"] == 1.0


def test_imports_cannot_claim_approximate(tmp_path: Path) -> None:
    (tmp_path / "raise.txt").write_text("AA")
    with pytest.raises(ValueError):
        import_range_files(tmp_path, {**META, "source": {"type": "approximate", "label": "x"}})
