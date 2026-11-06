"""Offline precompute: preflop equity table + CFR solutions for the postflop trainer.

python -m gto_trainer.solver.precompute                 # everything missing
python -m gto_trainer.solver.precompute --force --iterations 600 --target 0.2
python -m gto_trainer.solver.precompute --spots srp_btn_bb_K72r --jobs 1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys

from ..config import SOLUTIONS_DIR
from ..poker import preflop_equity
from .postflop import SPOTS, save_solution, solve_spot


def _run(args: tuple[str, int, float]) -> str:
    spot_id, iterations, target = args
    d = SPOTS[spot_id]

    def say(msg: str) -> None:
        print(f"[{spot_id}] {msg}", flush=True)

    s, rep, hands = solve_spot(d, iterations=iterations, target_pct=target, progress=say)
    path = save_solution(d, s, rep, hands)
    return (
        f"{spot_id}: {rep.iterations} iterations, exploitability "
        f"{rep.exploitability_pct:.3f}% pot, {rep.seconds:.0f}s -> {path.name}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spots", nargs="*", default=list(SPOTS))
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--target", type=float, default=0.3, help="stop at this exploitability %% pot")
    ap.add_argument("--jobs", type=int, default=min(4, mp.cpu_count()))
    ap.add_argument("--force", action="store_true", help="re-solve spots that are cached")
    a = ap.parse_args(argv)

    if not preflop_equity.TABLE_PATH.exists():
        print("building preflop equity table ...", flush=True)
        preflop_equity.save_table()

    def cached(s: str) -> bool:  # older caches lack the full strategy file exploit mode needs
        return all(
            (SOLUTIONS_DIR / f"{s}{ext}").exists() for ext in (".json", ".npz", ".strategy.npz")
        )

    todo = [s for s in a.spots if a.force or not cached(s)]
    if not todo:
        print("all postflop solutions cached (use --force to re-solve)")
        return
    jobs = [(s, a.iterations, a.target) for s in todo]
    if a.jobs > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(a.jobs, len(jobs))) as pool:
            for line in pool.imap_unordered(_run, jobs):
                print(line, flush=True)
    else:
        for j in jobs:
            print(_run(j), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
