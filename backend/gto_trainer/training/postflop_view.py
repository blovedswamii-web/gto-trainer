"""Turning a cached CFR solution into what the trainer shows: node views, combo details,
drill deals and graded answers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..db.models import Decision
from ..db.session import local_user_id, session
from ..poker.cards import CLASS_LABELS, COMBO_CLASS, COMBOS, card_to_str, combo_from_str
from ..solver.categories import CATEGORIES
from ..solver.postflop import NodeData, Solution
from ..solver.tree import action_label
from ..strategies.schemas import SeatState, SourceTag, TableState
from .grading import GradeResult, grade_decision
from .srs import update_item


def source_tag(sol: Solution) -> SourceTag:
    r, t = sol.report, sol.spot["tree"]
    turn = t["turn_cards"] or "all"
    river = t["river_cards"] or "all"
    notes = (
        f"Own DCFR solver, {r['iterations']} iterations, {r['seconds']:.0f}s. Abstraction: "
        f"{turn} turn / {river} river representative cards (suit-isomorphic merging is "
        f"lossless), bet sizes {t['flop']['bets']} flop, {t['turn']['bets']} turn/river. "
        "Exploitability is measured inside this abstracted game."
    )
    if r.get("input_source") == "approximate":
        notes += " Input ranges come from the APPROXIMATE preflop seed charts."
    return SourceTag(
        type="solver",
        label=f"DCFR {r['iterations']} it",
        notes=notes,
        exploitability_pct=float(r["exploitability_pct"]),
    )


def _history(sol: Solution, path: str) -> list[dict[str, str]]:
    oop, ip = sol.positions()
    toks = path.split("/") if path else []
    out: list[dict[str, str]] = []
    prefix = ""
    for tok in toks:
        node = sol.nodes.get(prefix)
        if node is None:  # chance token
            out.append({"position": "", "action": tok, "kind": "card"})
        else:
            out.append(
                {
                    "position": oop if node.actor == 0 else ip,
                    "action": action_label(tok),
                    "kind": "action",
                }
            )
        prefix = f"{prefix}/{tok}" if prefix else tok
    return out


def strategy_view(
    sol: Solution,
    actor: int,
    board: list[str],
    actions: list[str],
    strategy: np.ndarray,
    reach: np.ndarray,
    ev: np.ndarray | None,
) -> dict[str, Any]:
    """Reach-weighted aggregation of a per-hand strategy: overall mix, hand categories and
    the 169-cell grid. Used for solutions and for exploit-mode comparisons alike."""
    combos = sol.combos[actor]
    cats = sol.hand_categories(actor, board)
    total = float(reach.sum())

    def mix(idx: np.ndarray | list[int], arr: np.ndarray) -> dict[str, float]:
        w = reach[idx]
        m = float(w.sum())
        return {a: float((w * arr[idx, i]).sum() / m) for i, a in enumerate(actions)}

    overall = mix(np.arange(len(reach)), strategy) if total > 1e-9 else dict.fromkeys(actions, 0.0)
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(cats):
        by_cat.setdefault(c, []).append(i)
    cat_rows = []
    for c in CATEGORIES:
        idx = by_cat.get(c)
        if not idx or float(reach[idx].sum()) <= 1e-9:
            continue
        cat_rows.append(
            {
                "category": c,
                "combos": float(reach[idx].sum()),
                "freqs": mix(idx, strategy),
                "ev": mix(idx, ev) if ev is not None else {},
            }
        )
    cls = np.array([COMBO_CLASS[c] for c in combos])
    grid = []
    for k, lab in enumerate(CLASS_LABELS):
        sel = np.flatnonzero(cls == k)
        m = float(reach[sel].sum()) if len(sel) else 0.0
        n_class = 6 if len(lab) == 2 else (4 if lab.endswith("s") else 12)
        live = m > 1e-9
        grid.append(
            {
                "label": lab,
                "combos": m,
                "reach": min(1.0, m / n_class),
                "freqs": mix(sel, strategy) if live else {},
                "ev": mix(sel, ev) if live and ev is not None else None,
            }
        )
    return {"overall": overall, "range_combos": total, "categories": cat_rows, "grid": grid}


def _seat_chips(sol: Solution, path: str) -> tuple[list[float], list[float], list[str | None]]:
    """Per seat (OOP, IP): stack behind, chips put in on this street, last action this street.

    A node only stores the actor's stack, the pot and the amount to call, so the opponent's
    side is recovered from the pot, and this street's chips from the node that opened the
    street (where both players have put in the same amount).
    """
    tree = sol.spot["tree"]
    start_stack, start_pot = float(tree["stack"]), float(tree["starting_pot"])
    nd = sol.nodes[path]
    a, o = nd.actor, 1 - nd.actor
    inv = [0.0, 0.0]
    inv[a] = start_stack - nd.stack
    inv[o] = nd.pot - start_pot - inv[a]

    last: list[str | None] = [None, None]
    street_root = prefix = ""
    for tok in path.split("/") if path else []:
        node = sol.nodes.get(prefix)
        prefix = f"{prefix}/{tok}" if prefix else tok
        if node is None:  # chance token: a new street starts after it
            last = [None, None]
            street_root = prefix
        else:
            last[node.actor] = tok.partition(":")[0]
    before = start_stack - sol.nodes[street_root].stack  # each player's chips in before this street
    return (
        [round(start_stack - x, 4) for x in inv],
        [round(max(0.0, x - before), 4) for x in inv],
        last,
    )


def node_structure(sol: Solution, path: str) -> dict[str, Any]:
    """Everything about a node except strategies: board, pot, history, navigation."""
    nd = sol.nodes[path]
    oop, ip = sol.positions()
    stacks, street_inv, last = _seat_chips(sol, path)
    table = TableState(
        seats=[
            SeatState(
                position=p,
                stack_bb=stacks[i],
                committed_bb=street_inv[i],
                is_hero=i == nd.actor,
                is_actor=i == nd.actor,
                is_villain=i != nd.actor,
                last_action=last[i],
            )
            for i, p in enumerate((oop, ip))
        ],
        dealer=ip,
        pot_bb=nd.pot,
        to_call_bb=nd.to_call,
        board=list(nd.board),
    ).model_dump()
    children: dict[str, Any] = {}
    for a in nd.actions:
        child = f"{path}/{a}" if path else a
        if child in sol.nodes:
            children[a] = {"kind": "node", "path": child}
        elif child in sol.chance:
            children[a] = {"kind": "chance", "cards": sol.chance[child], "path": child}
        else:
            children[a] = {"kind": "terminal"}
    return {
        "spot_id": sol.spot["id"],
        "title": sol.spot["title"],
        "pot_type": sol.spot["pot_type"],
        "path": path,
        "street": nd.street,
        "board": nd.board,
        "pot": nd.pot,
        "to_call": nd.to_call,
        "stack": nd.stack,
        "actor": "OOP" if nd.actor == 0 else "IP",
        "actor_position": oop if nd.actor == 0 else ip,
        "villain_position": ip if nd.actor == 0 else oop,
        "table": table,
        "history": _history(sol, path),
        "actions": [{"id": a, "label": action_label(a)} for a in nd.actions],
        "children": children,
        "source": source_tag(sol).model_dump(),
    }


def node_view(sol: Solution, path: str) -> dict[str, Any]:
    nd = sol.nodes[path]
    view = node_structure(sol, path)
    view.update(strategy_view(sol, nd.actor, nd.board, nd.actions, nd.strategy, nd.reach, nd.ev))
    return view


def combo_detail(sol: Solution, path: str, combo: int) -> dict[str, Any]:
    nd = sol.nodes[path]
    idx = _hand_index(sol, nd, combo)
    return {
        "combo": combo,
        "freqs": {a: float(nd.strategy[idx, i]) for i, a in enumerate(nd.actions)},
        "ev": {a: float(nd.ev[idx, i]) for i, a in enumerate(nd.actions)},
        "reach": float(nd.reach[idx]),
        "category": sol.hand_categories(nd.actor, nd.board)[idx],
    }


def _hand_index(sol: Solution, nd: NodeData, combo: int) -> int:
    hits = np.flatnonzero(sol.combos[nd.actor] == combo)
    if len(hits) == 0 or nd.reach[hits[0]] <= 0:
        raise KeyError("that hand does not reach this node")
    return int(hits[0])


@dataclass
class PostflopDeal:
    spot_id: str
    path: str
    combo: int


def deal(
    sols: list[Solution],
    rng: random.Random,
    streets: list[str] | None = None,
    role: str | None = None,
    forced: tuple[str, str, str] | None = None,
) -> PostflopDeal:
    """Pick a decision node (weighted by how often it is reached) and a hand in range."""
    if forced is not None:
        sid, path, hand_class = forced
        sol = next((s for s in sols if s.spot["id"] == sid), None)
        if sol is not None and path in sol.nodes:
            nd = sol.nodes[path]
            cand = [
                i
                for i, c in enumerate(sol.combos[nd.actor])
                if CLASS_LABELS[int(COMBO_CLASS[c])] == hand_class and nd.reach[i] > 1e-6
            ]
            if cand:
                return PostflopDeal(sid, path, int(sol.combos[nd.actor][rng.choice(cand)]))
    options: list[tuple[Solution, NodeData, float]] = []
    for sol in sols:
        for nd in sol.nodes.values():
            if len(nd.actions) < 2 or (streets and nd.street not in streets):
                continue
            if role and (("OOP" if nd.actor == 0 else "IP") != role):
                continue
            mass = float(nd.reach.sum()) * float(nd.opp_reach.sum())
            if mass > 1e-6:
                options.append((sol, nd, mass))
    if not options:
        raise LookupError("no solved decision nodes match the filters")
    # sqrt keeps rarer lines (check-raises, turn spots) in rotation without drowning in them
    weights = [m**0.5 for _, _, m in options]
    sol, nd, _ = rng.choices(options, weights=weights, k=1)[0]
    mixed = 1.0 - nd.strategy.max(axis=1)
    w = nd.reach * (0.25 + mixed)
    i = rng.choices(range(len(w)), weights=w.tolist(), k=1)[0]
    return PostflopDeal(sol.spot["id"], nd.path, int(sol.combos[nd.actor][i]))


def grade_and_record(
    sol: Solution, path: str, combo: int, action: str, response_ms: int | None, reason: str = "new"
) -> tuple[GradeResult, dict[str, Any]]:
    nd = sol.nodes[path]
    idx = _hand_index(sol, nd, combo)
    freqs = {a: float(nd.strategy[idx, i]) for i, a in enumerate(nd.actions)}
    ev = {a: float(nd.ev[idx, i]) for i, a in enumerate(nd.actions)}
    labels = {a: action_label(a) for a in nd.actions}
    g = grade_decision(freqs, action, ev, labels)
    oop, ip = sol.positions()
    hand_class = CLASS_LABELS[int(COMBO_CLASS[combo])]
    category = sol.hand_categories(nd.actor, nd.board)[idx]
    hi, lo = COMBOS[combo]
    with session() as s:
        d = Decision(
            user_id=local_user_id(s),
            mode="postflop",
            format_id=sol.spot["format_id"],
            spot_id=sol.spot["id"],
            scenario=sol.spot["pot_type"],
            hero_position=oop if nd.actor == 0 else ip,
            villain_position=ip if nd.actor == 0 else oop,
            street=nd.street,
            node=path,
            board="".join(nd.board),
            hand=card_to_str(int(hi)) + card_to_str(int(lo)),
            hand_class=hand_class,
            hand_category=category,
            chosen_action=action,
            best_action=g.best_action,
            chosen_freq=g.chosen_freq,
            best_freq=g.best_freq,
            freqs_json=json.dumps(freqs),
            grade=g.grade,
            points=g.points,
            freq_error=g.freq_error,
            ev_loss_bb=g.ev_loss,
            ev_loss_pct_pot=(g.ev_loss / nd.pot * 100) if g.ev_loss is not None else None,
            source_type="solver",
            response_ms=response_ms,
            reason=reason if reason in ("new", "review") else "new",
        )
        s.add(d)
        s.flush()
        did = d.id
    update_item("postflop", sol.spot["format_id"], sol.spot["id"], hand_class, g.grade, node=path)
    return g, {"decision_id": did, "freqs": freqs, "ev": ev, "category": category}


def parse_combo(text: str) -> int:
    return combo_from_str(text)
