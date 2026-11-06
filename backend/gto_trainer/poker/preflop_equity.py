"""169x169 preflop all-in equity table (class vs class), used for range-level heuristics.

``eq[a, b]`` is class ``a``'s all-in equity against class ``b``, averaged uniformly over the
*compatible* combo pairs (e.g. AKs vs AA only pairs combos that leave a card free).
``pairs[a, b]`` counts those compatible ordered combo pairs. Together they give an exact
class-level formula for equity against any range ``R`` given as per-class weights::

    eq(a vs R) = sum_b R[b] * pairs[a, b] * eq[a, b]  /  sum_b R[b] * pairs[a, b]

which is how the viewer splits aggression ranges into value vs bluffs without running a
fresh Monte Carlo per hand. The table itself is a Monte Carlo estimate (``samples`` boards
per class pair, standard error ~0.5/sqrt(samples)) built once by ``make precompute``.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from .cards import CLASS_COMBOS, COMBO_MASK, COMBOS, FloatArray, IntArray, class_ranks
from .evaluator import evaluate

TABLE_PATH = DATA_DIR / "preflop_equity.npz"


def _compat_pairs() -> IntArray:
    pairs = np.zeros((169, 169), dtype=np.int64)
    for a in range(169):
        ma = COMBO_MASK[CLASS_COMBOS[a]]
        for b in range(169):
            mb = COMBO_MASK[CLASS_COMBOS[b]]
            pairs[a, b] = int(((ma[:, None] & mb[None, :]) == 0).sum())
    return pairs


def build_table(samples: int = 3000, seed: int = 0, verbose: bool = True) -> dict[str, np.ndarray]:
    """Monte Carlo estimate over all 14,365 unordered class pairs (vectorised in chunks)."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    pairs = _compat_pairs()
    ua, ub = np.triu_indices(169)
    eq = np.zeros((169, 169))
    chunk_pairs = 200
    for s in range(0, len(ua), chunk_pairs):
        pa, pb = ua[s : s + chunk_pairs], ub[s : s + chunk_pairs]
        # Oversample to survive rejection of card-sharing draws (worst case AA vs AA: 1/6).
        draws = samples * 7
        pid = np.repeat(np.arange(len(pa)), draws)
        ca = np.array(
            [CLASS_COMBOS[a][rng.integers(len(CLASS_COMBOS[a]), size=draws)] for a in pa]
        ).ravel()
        cb = np.array(
            [CLASS_COMBOS[b][rng.integers(len(CLASS_COMBOS[b]), size=draws)] for b in pb]
        ).ravel()
        ok = (COMBO_MASK[ca] & COMBO_MASK[cb]) == 0
        # keep the first `samples` compatible draws of every pair
        rank_in_pair = np.cumsum(ok.reshape(len(pa), draws), axis=1).ravel()
        keep = ok & (rank_in_pair <= samples)
        pid, ca, cb = pid[keep], ca[keep], cb[keep]
        n = len(pid)
        ha, hb = COMBOS[ca], COMBOS[cb]
        keys = rng.random((n, 52))
        rows = np.arange(n)[:, None]
        keys[rows, ha] = 2.0
        keys[rows, hb] = 2.0
        board = np.argpartition(keys, 4, axis=1)[:, :5]
        sa = evaluate(np.concatenate([ha, board], axis=1))
        sb = evaluate(np.concatenate([hb, board], axis=1))
        out = np.where(sa > sb, 1.0, np.where(sa == sb, 0.5, 0.0))
        sums = np.bincount(pid, weights=out, minlength=len(pa))
        cnt = np.bincount(pid, minlength=len(pa))
        e = sums / np.maximum(cnt, 1)
        eq[pa, pb] = e
        eq[pb, pa] = 1.0 - e
        if verbose and (s // chunk_pairs) % 10 == 0:
            print(
                f"  preflop equity: {s + len(pa)}/{len(ua)} pairs "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    np.fill_diagonal(eq, 0.5)  # a class vs itself is symmetric by definition
    return {
        "eq": eq.astype(np.float32),
        "pairs": pairs.astype(np.int32),
        "samples": np.array(samples),
        "seed": np.array(seed),
    }


def save_table(path: Path = TABLE_PATH, samples: int = 3000, seed: int = 0) -> Path:
    table = build_table(samples=samples, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, eq=table["eq"], pairs=table["pairs"], samples=table["samples"], seed=table["seed"]
    )
    return path


@lru_cache(maxsize=1)
def load_table() -> tuple[FloatArray, FloatArray] | None:
    if not TABLE_PATH.exists():
        return None
    with np.load(TABLE_PATH) as z:
        return z["eq"].astype(np.float64), z["pairs"].astype(np.float64)


def class_equity_vs_range(villain_class_weights: FloatArray) -> FloatArray | None:
    """(169,) equity of every class against a villain range given as per-class weights."""
    table = load_table()
    if table is None:
        return None
    eq, pairs = table
    w = pairs * np.asarray(villain_class_weights)[None, :]
    z = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(z > 0, (w * eq).sum(axis=1) / z, np.nan)
    return out


def describe_class(cls: int) -> str:
    hi, lo, kind = class_ranks(cls)
    return f"{hi}/{lo}/{kind}"


if __name__ == "__main__":
    print(f"wrote {save_table()}")
