"""Pre-defined postflop spots: build from preflop ranges, solve, cache, load.

A spot = (preflop line, flop, bet-size tree, chance abstraction). Input ranges come from
the preflop strategy registry (so they inherit that data's quality tag: with the seed pack
they are *approximate* ranges, and the UI says so). The solution is cached as
``data/solutions/<id>.json`` (tree + metadata) and ``<id>.npz`` (per-node strategies, EVs
and reaches for flop and turn decision nodes).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..config import FORMATS, SOLUTIONS_DIR
from ..poker.cards import (
    CLASS_LABELS,
    COMBO_CLASS,
    blocked_combos,
    card_to_str,
    combo_to_str,
    parse_cards,
)
from ..strategies.loader import get_registry
from .batched import BSolver, build_bgame, chance_paths, node_paths
from .categories import classify_combos
from .cfr import DT, ChanceNode, Node, PlayerNode
from .tree import HandSet, StreetSizes, TreeConfig, action_label

RangeRef = tuple[str, str, str | None, str]  # (scenario, hero, villain, action) in the registry


@dataclass(frozen=True)
class PostflopSpotDef:
    id: str
    title: str
    pot_type: str  # "SRP" | "3BP"
    board: str
    oop: str
    ip: str
    oop_range: RangeRef
    ip_range: RangeRef
    tree: TreeConfig
    format_id: str = "6max_100bb"


# Bet trees. The batched engine keeps the cost per Python node independent of the number of
# runouts, so the defaults solve *every* turn card (no turn sampling) with 4 representative
# rivers per turn, and two sizes on every street.
_TURN = StreetSizes(bets=(0.5, 1.0), raises=(3.0,), max_raises=1)
_RIVER = StreetSizes(bets=(0.5, 1.0), raises=(), max_raises=0)


def _srp_tree(pot: float, stack: float) -> TreeConfig:
    return TreeConfig(
        starting_pot=pot,
        stack=stack,
        flop=StreetSizes(bets=(0.33, 0.75), oop_bets=(), raises=(3.0,), max_raises=1),
        turn=_TURN,
        river=_RIVER,
        turn_cards=None,
        river_cards=4,
    )


def _3bp_tree(pot: float, stack: float) -> TreeConfig:
    return TreeConfig(
        starting_pot=pot,
        stack=stack,
        flop=StreetSizes(bets=(0.33, 0.75), oop_bets=(0.33, 0.75), raises=(3.0,), max_raises=1),
        turn=_TURN,
        river=_RIVER,
        turn_cards=None,
        river_cards=4,
    )


def _pot_stack(scenario_pot: str) -> tuple[float, float]:
    fmt = FORMATS["6max_100bb"]
    if scenario_pot == "SRP_BTN_BB":
        o = fmt.open_size("BTN")
        return 2 * o + 0.5, fmt.stack_bb - o  # SB's 0.5 is dead money
    if scenario_pot == "3BP_SB_BTN":
        tb = fmt.three_bet_size("BTN", "SB")
        return 2 * tb + 1.0, fmt.stack_bb - tb  # BB's 1.0 is dead money
    raise ValueError(scenario_pot)


_SRP = _pot_stack("SRP_BTN_BB")
_3BP = _pot_stack("3BP_SB_BTN")
BTN_OPEN: RangeRef = ("rfi", "BTN", None, "raise")
BB_CALL: RangeRef = ("bb_defense", "BB", "BTN", "call")

SPOTS: dict[str, PostflopSpotDef] = {
    d.id: d
    for d in [
        PostflopSpotDef(
            "srp_btn_bb_K72r",
            "BTN vs BB · single-raised pot · K72 rainbow",
            "SRP",
            "Ks7d2c",
            "BB",
            "BTN",
            BB_CALL,
            BTN_OPEN,
            _srp_tree(*_SRP),
        ),
        PostflopSpotDef(
            "srp_btn_bb_A93ss",
            "BTN vs BB · single-raised pot · A93 two-tone",
            "SRP",
            "As9s3d",
            "BB",
            "BTN",
            BB_CALL,
            BTN_OPEN,
            _srp_tree(*_SRP),
        ),
        PostflopSpotDef(
            "srp_btn_bb_987tt",
            "BTN vs BB · single-raised pot · 987 two-tone",
            "SRP",
            "9h8h7c",
            "BB",
            "BTN",
            BB_CALL,
            BTN_OPEN,
            _srp_tree(*_SRP),
        ),
        PostflopSpotDef(
            "3bp_sb_btn_Q84r",
            "SB vs BTN · 3-bet pot · Q84 rainbow",
            "3BP",
            "Qc8d4s",
            "SB",
            "BTN",
            ("vs_open", "SB", "BTN", "raise"),
            ("vs_3bet", "BTN", "SB", "call"),
            _3bp_tree(*_3BP),
        ),
    ]
}


def _range_hands(ref: RangeRef, board: tuple[int, ...], fmt_id: str) -> tuple[HandSet, str]:
    reg = get_registry()
    spot = reg.find(fmt_id, ref[0], ref[1], ref[2])
    if spot is None:
        raise LookupError(f"preflop spot {ref} not loaded; cannot build postflop ranges")
    w = spot.action_range(ref[3]) * ~blocked_combos(board)
    idx = np.flatnonzero(w > 1e-9)
    return HandSet(idx, w[idx].astype(DT)), spot.source.type


@dataclass
class SolveReport:
    spot_id: str
    iterations: int
    seconds: float
    algo: str
    exploitability_pct: float
    history: list[tuple[int, float]] = field(default_factory=list)
    player_nodes: int = 0
    hands: tuple[int, int] = (0, 0)
    input_source: str = "approximate"
    tree: dict[str, Any] = field(default_factory=dict)  # the TreeConfig actually solved


def solve_spot(
    d: PostflopSpotDef,
    iterations: int = 300,
    check_every: int = 25,
    target_pct: float = 0.3,
    progress: Callable[[str], None] | None = None,
    tree: TreeConfig | None = None,
) -> tuple[BSolver, SolveReport, tuple[HandSet, HandSet]]:
    """Run DCFR until ``iterations`` or exploitability <= ``target_pct`` % of pot."""
    say = progress or (lambda _m: None)
    board = tuple(parse_cards(d.board))
    h0, src0 = _range_hands(d.oop_range, board, d.format_id)
    h1, src1 = _range_hands(d.ip_range, board, d.format_id)
    t0 = time.perf_counter()
    cfg = tree or d.tree
    game, tb = build_bgame(cfg, board, (h0, h1))
    s = BSolver(game, algo="dcfr")
    mem = sum(nd.regret.nbytes + nd.strat_sum.nbytes for nd in s.player_nodes) / 2**20
    say(
        f"{d.id}: {tb.n_player_nodes} decision nodes ({len(s.player_nodes)} batched), "
        f"{game.batches[-1].size} river boards, hands {game.n}, {mem:.0f} MB of regrets, "
        f"built in {time.perf_counter() - t0:.1f}s"
    )
    hist: list[tuple[int, float]] = []
    pct = float("nan")
    while s.t < iterations:
        s.iterate(min(check_every, iterations - s.t))
        pct = s.exploitability()[2] / cfg.starting_pot * 100
        hist.append((s.t, pct))
        say(f"  iter {s.t:4d}  exploitability {pct:.3f}% pot  ({time.perf_counter() - t0:.0f}s)")
        if pct <= target_pct:
            break
    s.evaluate()
    src = "approximate" if "approximate" in (src0, src1) else src0
    rep = SolveReport(
        d.id,
        s.t,
        time.perf_counter() - t0,
        "dcfr",
        pct,
        hist,
        tb.n_player_nodes,
        game.n,
        src,
        asdict(cfg),
    )
    return s, rep, (h0, h1)


# ---------------------------------------------------------------------------- persistence


def walk_player_nodes(root: Node) -> list[tuple[str, PlayerNode]]:
    """Every decision node with its path, in a deterministic depth-first order.

    The order depends only on the tree structure, so a game rebuilt from the same config and
    hands yields the same sequence -- that is how full strategies are stored and restored.
    """
    out: list[tuple[str, PlayerNode]] = []

    def go(node: Node, path: str) -> None:
        if isinstance(node, PlayerNode):
            out.append((path, node))
            for a, ch in zip(node.actions, node.children, strict=True):
                go(ch, f"{path}/{a}" if path else a)
        elif isinstance(node, ChanceNode):
            for br in node.branches:
                go(br.node, f"{path}/{card_to_str(br.card)}")

    go(root, "")
    return out


def tree_config_from_dict(d: dict[str, Any]) -> TreeConfig:
    def street(x: dict[str, Any]) -> StreetSizes:
        return StreetSizes(
            bets=tuple(x["bets"]),
            oop_bets=None if x["oop_bets"] is None else tuple(x["oop_bets"]),
            raises=tuple(x["raises"]),
            max_raises=x["max_raises"],
            allin=x["allin"],
        )

    rest = {k: v for k, v in d.items() if k not in ("flop", "turn", "river")}
    return TreeConfig(
        flop=street(d["flop"]), turn=street(d["turn"]), river=street(d["river"]), **rest
    )


def save_solution(
    d: PostflopSpotDef,
    s: BSolver,
    rep: SolveReport,
    hands: tuple[HandSet, HandSet],
    out_dir: Path = SOLUTIONS_DIR,
) -> Path:
    """Persist flop/turn decision nodes (one entry per board): average strategy, per-hand EV
    (bb) and reaches. Plus every node's full strategy (rivers included) for exploit mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta_nodes = []
    k = 0
    for nd in s.player_nodes:
        if not nd.record:
            continue
        assert nd.ev is not None and nd.reach is not None
        actor = nd.player
        strat = BSolver.avg(nd)
        # Normalise counterfactual values to "EV given I hold this hand and reach this node".
        mass = s.compat_sum(actor, nd.reach[1 - actor])
        ev = np.where(mass[:, :, None] > 1e-12, nd.ev / np.maximum(mass[:, :, None], 1e-12), 0.0)
        for b, path in enumerate(node_paths(nd)):
            arrays[f"s{k}"] = strat[b].astype(np.float16)
            arrays[f"e{k}"] = ev[b].astype(np.float16)
            arrays[f"r{k}"] = nd.reach[actor][b].astype(np.float16)
            arrays[f"o{k}"] = nd.reach[1 - actor][b].astype(np.float16)
            meta_nodes.append(
                {
                    "k": k,
                    "path": path,
                    "actor": actor,
                    "actions": nd.actions,
                    "street": nd.info["street"],
                    "board": [card_to_str(c) for c in nd.batch.boards[b]],
                    "pot": nd.info["pot"],
                    "to_call": nd.info["to_call"],
                    "stack": nd.info["stack"],
                }
            )
            k += 1
    meta = {
        "spot": {**asdict(d), "tree": rep.tree or asdict(d.tree)},
        "report": asdict(rep),
        "hands": [[combo_to_str(int(c)) for c in h.combos] for h in hands],
        "weights": [[float(x) for x in h.weights] for h in hands],
        "nodes": meta_nodes,
        "chance": chance_paths(s.g.root),
        "engine": "batched",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"{d.id}.json").write_text(json.dumps(meta))
    np.savez_compressed(out_dir / f"{d.id}.npz", **arrays)  # type: ignore[arg-type]
    # Full average strategy of every batched node (B, n, A), quantised to 1/255, in the
    # engine's deterministic node order. Only exploit mode needs it.
    full = {
        f"q{i}": np.round(BSolver.avg(nd) * 255).astype(np.uint8)
        for i, nd in enumerate(s.player_nodes)
    }
    np.savez_compressed(out_dir / f"{d.id}.strategy.npz", **full)  # type: ignore[arg-type]
    load_solution.cache_clear()
    return out_dir / f"{d.id}.json"


@dataclass
class NodeData:
    path: str
    actor: int
    actions: list[str]
    street: str
    board: list[str]
    pot: float
    to_call: float
    stack: float
    strategy: np.ndarray  # (n_actor, A)
    ev: np.ndarray  # (n_actor, A) bb
    reach: np.ndarray  # (n_actor,) own reach including initial weights
    opp_reach: np.ndarray

    @property
    def labels(self) -> list[str]:
        return [action_label(a) for a in self.actions]


@dataclass
class Solution:
    spot: dict[str, Any]
    report: dict[str, Any]
    hands: list[list[str]]
    combos: list[np.ndarray]  # combo ids per player
    nodes: dict[str, NodeData]
    chance: dict[str, list[str]]
    directory: Path = SOLUTIONS_DIR
    weights: list[list[float]] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)  # board key -> per-hand

    def positions(self) -> tuple[str, str]:
        return self.spot["oop"], self.spot["ip"]

    def hand_categories(self, actor: int, board: list[str]) -> list[str]:
        key = f"{actor}:{''.join(board)}"
        if key not in self.categories:
            self.categories[key] = classify_combos(self.combos[actor], parse_cards("".join(board)))
        return self.categories[key]

    def class_labels(self, actor: int) -> list[str]:
        return [CLASS_LABELS[int(COMBO_CLASS[c])] for c in self.combos[actor]]


@lru_cache(maxsize=16)
def load_solution(spot_id: str, directory: Path = SOLUTIONS_DIR) -> Solution | None:
    jp, npz = directory / f"{spot_id}.json", directory / f"{spot_id}.npz"
    if not jp.exists() or not npz.exists():
        return None
    meta = json.loads(jp.read_text())
    from ..poker.cards import combo_from_str

    combos = [np.array([combo_from_str(c) for c in hs], dtype=np.int64) for hs in meta["hands"]]
    nodes: dict[str, NodeData] = {}
    with np.load(npz) as z:
        for m in meta["nodes"]:
            k = m["k"]
            nodes[m["path"]] = NodeData(
                path=m["path"],
                actor=m["actor"],
                actions=m["actions"],
                street=m["street"],
                board=m["board"],
                pot=m["pot"],
                to_call=m["to_call"],
                stack=m["stack"],
                strategy=z[f"s{k}"].astype(np.float64),
                ev=z[f"e{k}"].astype(np.float64),
                reach=z[f"r{k}"].astype(np.float64),
                opp_reach=z[f"o{k}"].astype(np.float64),
            )
    return Solution(
        meta["spot"],
        meta["report"],
        meta["hands"],
        combos,
        nodes,
        meta["chance"],
        directory,
        meta["weights"],
    )


def available_solutions(directory: Path = SOLUTIONS_DIR) -> list[str]:
    return [sid for sid in SPOTS if (directory / f"{sid}.json").exists()]
