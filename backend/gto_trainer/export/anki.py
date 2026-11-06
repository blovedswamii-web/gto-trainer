"""Export drills to Anki (.apkg) for spaced repetition outside the app.

Four deck types:

* ``leaks``    -- every (spot, decision node, hand class) you were graded *minor* or
  *mistake* on, in drills, real-hand reviews or push/fold; one card each, with how often
  you got it wrong and what you chose. Your personal error log as flashcards.
* ``preflop``  -- chart cards for chosen spots: "close decisions" (mixed or range-boundary
  hands) or every hand that reaches the spot.
* ``postflop`` -- solver strategy by hand category at each flop decision of solved spots.
* ``pushfold`` -- close shove/call decisions for one push/fold table setup.

Every answer side shows the solution's mix, the strategy's source tag (approximate seed
chart / imported / solver + exploitability) and the full range as an SVG grid. SVGs are
content-addressed media files shared by all cards of a spot, so decks stay small.

Note GUIDs are derived from stable keys (deck type + spot + node + hand), so re-exporting
and importing again *updates* existing cards (and keeps their review history) instead of
duplicating them.
"""

from __future__ import annotations

import hashlib
import html
import json
import tempfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import genanki
import numpy as np

from ..db.models import Decision
from ..db.session import local_user_id, session
from ..poker.cards import CLASS_INDEX, CLASS_LABELS, combo_from_str
from ..solver.postflop import Solution, available_solutions, load_solution
from ..solver.pushfold import POSITIONS, PushFoldResult, PushFoldSetup, solve_cached
from ..solver.tree import action_label
from ..strategies.loader import CompiledSpot, get_registry
from ..strategies.schemas import SourceTag
from ..training.postflop_view import _history, source_tag, strategy_view
from ..training.spots import _boundary_mask

MODEL_ID = 1_740_221_337  # fixed forever: Anki identifies the note type by it
DECK_NAMES: dict[str, str] = {
    "leaks": "GTO Trainer::My leaks",
    "preflop": "GTO Trainer::Preflop charts",
    "postflop": "GTO Trainer::Postflop",
    "pushfold": "GTO Trainer::Push-fold",
}
COLORS = {
    "fold": "#3987e5",
    "check": "#199e70",
    "call": "#199e70",
    "raise": "#d95926",
    "allin": "#9085e9",
    "push": "#9085e9",
    "bet_small": "#c98500",
    "bet_big": "#d55181",
}

CSS = """
.card { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif; font-size: 18px;
  line-height: 1.45; color: #1d2330; background: #f6f6f3; text-align: left; padding: 6px; }
.nightMode.card, .night_mode .card { color: #eef0f3; background: #161a21; }
.kind { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; opacity: .6; }
.title { font-weight: 650; margin: 2px 0 6px; }
.ctx { font-size: 14px; opacity: .8; }
.hand { font-size: 30px; font-weight: 700; margin: 10px 0 4px; }
.q { font-size: 16px; margin-top: 8px; }
.s-s { color: #1d2330; } .nightMode .s-s, .night_mode .s-s { color: #eef0f3; }
.s-h { color: #d03b3b; } .s-d { color: #2a78d6; } .s-c { color: #128a52; }
table.mix { border-collapse: collapse; margin: 6px 0; font-size: 15px; }
table.mix td { padding: 2px 8px 2px 0; vertical-align: middle; }
table.mix tr.me td { font-weight: 700; }
.bar { display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; }
.note { font-size: 14px; margin: 6px 0; }
.src { font-size: 12px; opacity: .65; margin-top: 8px; }
.grid img { max-width: 360px; width: 100%; margin-top: 8px; }
"""

MODEL = genanki.Model(
    MODEL_ID,
    "GTO Trainer card",
    fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Key"}],
    templates=[
        {"name": "Card 1", "qfmt": "{{Front}}", "afmt": "{{FrontSide}}<hr id=answer>{{Back}}"}
    ],
    css=CSS,
)


@dataclass
class Card:
    key: str
    front: str
    back: str
    tags: list[str] = field(default_factory=list)


@dataclass
class DeckBuild:
    kind: str
    cards: list[Card] = field(default_factory=list)
    media: dict[str, str] = field(default_factory=dict)  # filename -> SVG text
    skipped: Counter[str] = field(default_factory=Counter)

    @property
    def name(self) -> str:
        return DECK_NAMES[self.kind]


# ---------------------------------------------------------------------------- HTML bits

_SUIT = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def cards_html(cards: str) -> str:
    parts = [cards[i : i + 2] for i in range(0, len(cards), 2)]
    return " ".join(
        f'<span class="s-{c[1]}">{c[0]}{_SUIT.get(c[1], "")}</span>' for c in parts if len(c) == 2
    )


def _color(action: str) -> str:
    kind, _, size = action.partition(":")
    if kind == "bet":
        return COLORS["bet_small" if float(size) <= 0.5 else "bet_big"]
    return COLORS.get(kind, "#8a919e")


def mix_html(
    freqs: dict[str, float],
    labels: dict[str, str],
    mine: str | None = None,
    ev: dict[str, float] | None = None,
    units: str = "bb",
) -> str:
    rows = []
    for a, f in freqs.items():
        ev_txt = f"<td>{ev[a]:+.2f}{units}</td>" if ev and a in ev else ""
        rows.append(
            f'<tr class="{"me" if a == mine else ""}"><td>{html.escape(labels.get(a, a))}</td>'
            f'<td><span class="bar" style="width:{max(f * 140, 2 if f > 0 else 0):.0f}px;'
            f'background:{_color(a)}"></span></td><td>{f:.0%}</td>{ev_txt}</tr>'
        )
    return f'<table class="mix">{"".join(rows)}</table>'


def source_html(src: SourceTag) -> str:
    return f'<div class="src">Source: {html.escape(src.badge)} · {html.escape(src.label)}</div>'


def svg_grid(cells: Sequence[tuple[str, dict[str, float], float]], order: Sequence[str]) -> str:
    """13x13 range grid as standalone SVG: segment widths = frequencies, height = reach."""
    size, gap = 26.0, 1.5
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {13 * size:.0f} '
        f'{13 * size:.0f}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{13 * size:.0f}" height="{13 * size:.0f}" fill="#161a21"/>',
    ]
    for k, (label, freqs, reach) in enumerate(cells):
        x0, y0 = (k % 13) * size, (k // 13) * size
        w = size - gap
        out.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{w:.1f}" fill="#2a303b"/>'
        )
        h = w * max(0.0, min(1.0, reach))
        x = x0
        for a in order:
            f = freqs.get(a, 0.0)
            if f < 0.005 or h <= 0:
                continue
            out.append(
                f'<rect x="{x:.1f}" y="{y0 + w - h:.1f}" width="{w * f:.1f}" '
                f'height="{h:.1f}" fill="{_color(a)}"/>'
            )
            x += w * f
        out.append(
            f'<text x="{x0 + 2:.1f}" y="{y0 + 9:.1f}" font-size="7.5" fill="#fff" '
            f'font-weight="600">{label}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def _media(build: DeckBuild, svg: str) -> str:
    name = f"gto_{hashlib.sha1(svg.encode()).hexdigest()[:14]}.svg"
    build.media[name] = svg
    return f'<div class="grid"><img src="{name}"></div>'


def _tag(*parts: str) -> list[str]:
    return ["gto_trainer", *[p.replace(" ", "_").replace("/", "-") for p in parts if p]]


# ---------------------------------------------------------------------------- preflop


def _history_text(spot: CompiledSpot) -> str:
    steps = []
    for h in spot.ctx.history:
        if h.action == "fold":
            steps.append(f"{h.position} folds")
        else:
            verb = {"raise": "raises to", "3bet": "3-bets to", "4bet": "4-bets to"}.get(
                h.action, h.action
            )
            steps.append(f"{h.position} {verb} {h.size_bb:g}bb")
    steps.append(f"<b>{spot.file.hero} to act</b>")
    return ", ".join(steps)


def _spot_svg(build: DeckBuild, spot: CompiledSpot) -> str:
    combos, freqs, _ = spot.class_view()
    cells = [
        (
            CLASS_LABELS[k],
            {a: float(freqs[k, i]) for i, a in enumerate(spot.action_ids)},
            float(combos[k])
            / [6, 4, 12][
                0 if len(CLASS_LABELS[k]) == 2 else (1 if CLASS_LABELS[k].endswith("s") else 2)
            ],
        )
        for k in range(169)
    ]
    return _media(build, svg_grid(cells, ["fold", "call", "raise", "allin"]))


def _preflop_front(spot: CompiledSpot, hand_html: str, kind: str = "Preflop") -> str:
    labels = " / ".join(a.label.split(" ")[0] for a in spot.ctx.actions)
    return (
        f'<div class="kind">{kind} · {html.escape(spot.summary().scenario_label)} · '
        f'100bb</div><div class="title">{html.escape(spot.ctx.title)}</div>'
        f'<div class="ctx">{_history_text(spot)} · pot {spot.ctx.pot_bb:g}bb</div>'
        f'<div class="hand">{hand_html}</div><div class="q">{labels}?</div>'
    )


def build_preflop(
    scenarios: Sequence[str] = (),
    spot_ids: Sequence[str] = (),
    hands: Literal["close", "all"] = "close",
) -> DeckBuild:
    b = DeckBuild("preflop")
    reg = get_registry()
    for spot in reg.list("6max_100bb"):
        if scenarios and spot.file.scenario not in scenarios:
            continue
        if spot_ids and spot.id not in spot_ids:
            continue
        combos, freqs, ev = spot.class_view()
        mixed = 1 - freqs.max(axis=1)
        edge = _boundary_mask(spot)
        grid = _spot_svg(b, spot)
        labels = {a.id: a.label for a in spot.ctx.actions}
        for k in range(169):
            if combos[k] <= 0 or (hands == "close" and mixed[k] < 0.05 and not edge[k]):
                continue
            lab = CLASS_LABELS[k]
            f = {a: float(freqs[k, i]) for i, a in enumerate(spot.action_ids)}
            evd = (
                {a: float(ev[k, i]) for i, a in enumerate(spot.action_ids)}
                if ev is not None
                else None
            )
            back = mix_html(f, labels, ev=evd) + source_html(spot.source) + grid
            b.cards.append(
                Card(
                    f"preflop|{spot.id}|{lab}",
                    _preflop_front(spot, lab),
                    back,
                    _tag("preflop", spot.file.scenario, spot.file.hero),
                )
            )
    return b


# ---------------------------------------------------------------------------- postflop


def _node_svg(build: DeckBuild, sol: Solution, path: str) -> str:
    nd = sol.nodes[path]
    view = strategy_view(sol, nd.actor, nd.board, nd.actions, nd.strategy, nd.reach, nd.ev)
    cells = [(c["label"], c["freqs"], float(c["reach"])) for c in view["grid"]]
    order = sorted(
        nd.actions,
        key=lambda a: ["fold", "check", "call", "bet", "raise", "allin"].index(a.split(":")[0]),
    )
    return _media(build, svg_grid(cells, order))


def _postflop_front(sol: Solution, path: str, hand_html: str, board: list[str]) -> str:
    nd = sol.nodes[path]
    oop, ip = sol.positions()
    actor = oop if nd.actor == 0 else ip
    hist = (
        " · ".join(
            f"{h['position']} {h['action'].lower()}" if h["kind"] == "action" else h["action"]
            for h in _history(sol, path)
        )
        or "first to act"
    )
    opts = " / ".join(action_label(a) for a in nd.actions)
    return (
        f'<div class="kind">Postflop · {sol.spot["pot_type"]} · {nd.street}</div>'
        f'<div class="title">{html.escape(sol.spot["title"])}</div>'
        f'<div class="ctx">Board {cards_html("".join(board))} · {html.escape(hist)} · '
        f"pot {nd.pot:.1f}bb · <b>{actor} to act</b></div>"
        f'<div class="hand">{hand_html}</div><div class="q">{opts}?</div>'
    )


def build_postflop(spot_ids: Sequence[str] = (), min_combos: float = 2.0) -> DeckBuild:
    b = DeckBuild("postflop")
    for sid in spot_ids or available_solutions():
        sol = load_solution(sid)
        if sol is None:
            b.skipped["spot not solved"] += 1
            continue
        src = source_tag(sol)
        for path, nd in sol.nodes.items():
            if nd.street != "flop" or len(nd.actions) < 2:
                continue
            view = strategy_view(sol, nd.actor, nd.board, nd.actions, nd.strategy, nd.reach, nd.ev)
            if max(view["overall"].values()) >= 0.98:
                b.skipped["flop node where the whole range takes one action"] += 1
                continue
            grid = _node_svg(b, sol, path)
            labels = {a: action_label(a) for a in nd.actions}
            for row in view["categories"]:
                if row["combos"] < min_combos:
                    continue
                front = _postflop_front(sol, path, html.escape(row["category"]), nd.board)
                back = (
                    mix_html(row["freqs"], labels, ev=row["ev"])
                    + f'<div class="note">{row["combos"]:.0f} combos of this category reach '
                    f"here; EV in bb (each player owns half the starting pot).</div>"
                    + source_html(src)
                    + grid
                )
                b.cards.append(
                    Card(
                        f"postflop|{sid}|{path}|{row['category']}",
                        front,
                        back,
                        _tag("postflop", sid, row["category"]),
                    )
                )
    return b


# ---------------------------------------------------------------------------- push/fold


def _pushfold_context(st: PushFoldSetup) -> str:
    pos = st.positions
    stacks = " · ".join(f"{p} {x:g}" for p, x in zip(pos, st.stacks, strict=True))
    pay = ("payouts " + "/".join(f"{p:g}" for p in st.payouts)) if st.icm else "chip EV"
    ante = f" · ante {st.ante:g}" if st.ante else ""
    return f"Stacks (bb): {stacks} · {pay}{ante}"


def _pushfold_card(
    b: DeckBuild,
    r: PushFoldResult,
    kind: str,
    i: int,
    j: int | None,
    cls: int,
    hand_html: str,
    key: str,
    extra: str = "",
) -> Card:
    st = r.setup
    pos = st.positions
    act = "push" if kind == "push" else "call"
    freqs_arr = r.push[i] if j is None else r.call[(i, j)]
    margin = (r.push_margin[i] if j is None else r.call_margin[(i, j)])[cls]
    hero = i if j is None else j
    situation = (
        f"Folded to you in the <b>{pos[i]}</b>"
        if j is None
        else f"<b>{pos[i]}</b> shoves {st.stacks[i]:g}bb; you are in the <b>{pos[j]}</b>"
    )
    front = (
        f'<div class="kind">Push/fold · {"ICM" if st.icm else "chip EV"}</div>'
        f'<div class="ctx">{_pushfold_context(st)}</div>'
        f'<div class="title">{situation} ({st.stacks[hero]:g}bb)</div>'
        f'<div class="hand">{hand_html}</div>'
        f'<div class="q">{"Shove" if act == "push" else "Call"} or fold?</div>'
    )
    f = float(freqs_arr[cls])
    unit = "bb" if not st.icm else "% of the prize pool"
    cells = [
        (CLASS_LABELS[k], {act: float(freqs_arr[k]), "fold": 1 - float(freqs_arr[k])}, 1.0)
        for k in range(169)
    ]
    back = (
        mix_html({act: f, "fold": 1 - f}, {"push": "Shove", "call": "Call", "fold": "Fold"})
        + f'<div class="note">EV({"shove" if act == "push" else "call"}) − EV(fold) = '
        f"{margin:+.3f} {unit}</div>{extra}"
        f'<div class="src">Source: push/fold equilibrium (fictitious play, '
        f"exploitability {r.exploitability:.4f} {r.units})</div>"
        + _media(b, svg_grid(cells, ["fold", act]))
    )
    return Card(key, front, back, _tag("pushfold", kind, pos[hero]))


def build_pushfold(setup: PushFoldSetup, hands: Literal["close", "all"] = "close") -> DeckBuild:
    b = DeckBuild("pushfold")
    r = solve_cached(setup)
    decisions: list[tuple[str, int, int | None]] = [
        ("push", i, None) for i in range(len(setup.stacks) - 1)
    ]
    decisions += [("call", i, j) for (i, j) in r.call]
    for kind, i, j in decisions:
        freqs = r.push[i] if j is None else r.call[(i, j)]
        margin = r.push_margin[i] if j is None else r.call_margin[(i, j)]
        scale = float(np.abs(margin).max()) or 1.0
        for k in range(169):
            close = 0.02 < freqs[k] < 0.98 or abs(margin[k]) < 0.15 * scale
            if hands == "close" and not close:
                continue
            key = f"pushfold|{setup.code()}|{kind}|{i}|{j}|{CLASS_LABELS[k]}"
            b.cards.append(_pushfold_card(b, r, kind, i, j, k, CLASS_LABELS[k], key))
    return b


# ---------------------------------------------------------------------------- leaks


def build_leaks(days: int | None = None, include_minor: bool = True) -> DeckBuild:
    from datetime import timedelta

    from sqlalchemy import select

    from ..db.models import utcnow

    b = DeckBuild("leaks")
    grades = ["mistake", "minor"] if include_minor else ["mistake"]
    with session() as s:
        q = select(Decision).where(
            Decision.user_id == local_user_id(s),
            Decision.grade.in_(grades),
            Decision.mode.in_(["preflop", "postflop", "review", "pushfold"]),
        )
        if days:
            q = q.where(Decision.created_at >= utcnow() - timedelta(days=days))
        rows = list(s.scalars(q.order_by(Decision.created_at)).all())
    groups: dict[tuple[str, str, str, str], list[Decision]] = defaultdict(list)
    for d in rows:
        kind = (
            "pushfold"
            if d.mode == "pushfold"
            else ("preflop" if d.street == "preflop" else "postflop")
        )
        groups[(kind, d.spot_id, d.node, d.hand_class)].append(d)
    reg = get_registry()
    for (kind, spot_id, node, hand_class), ds in groups.items():
        last = ds[-1]
        tally = Counter(d.chosen_action for d in ds)
        history = (
            f'<div class="note">Flagged {len(ds)}× — you chose '
            + ", ".join(f"{a} ×{n}" for a, n in tally.most_common())
            + "</div>"
        )
        key = f"leak|{kind}|{spot_id}|{node}|{hand_class}"
        hand_html = f"{cards_html(last.hand)} <span class='ctx'>({hand_class})</span>"
        tags = _tag("leak", kind, last.hero_position, last.hand_category)
        if kind == "preflop":
            spot = reg.spots.get(spot_id)
            if spot is None:
                b.skipped["preflop spot no longer loaded"] += 1
                continue
            combo = combo_from_str(last.hand)
            labels = {a.id: a.label for a in spot.ctx.actions}
            back = (
                mix_html(spot.combo_freqs(combo), labels)
                + history
                + source_html(spot.source)
                + _spot_svg(b, spot)
            )
            source = "real hand" if last.mode == "review" else "drill"
            b.cards.append(
                Card(key, _preflop_front(spot, hand_html, f"Leak · {source}"), back, tags)
            )
        elif kind == "postflop":
            sol = load_solution(spot_id)
            if sol is None or node not in sol.nodes:
                b.skipped["postflop solution missing"] += 1
                continue
            freqs = json.loads(last.freqs_json)
            labels = {a: action_label(a) for a in freqs}
            board = [last.board[i : i + 2] for i in range(0, len(last.board), 2)] or sol.nodes[
                node
            ].board
            back = (
                mix_html(freqs, labels, mine=last.chosen_action)
                + history
                + f'<div class="note">{html.escape(last.hand_category)}</div>'
                + source_html(source_tag(sol))
                + _node_svg(b, sol, node)
            )
            b.cards.append(Card(key, _postflop_front(sol, node, hand_html, board), back, tags))
        else:
            if not node:
                b.skipped["push/fold decision recorded before setups were stored"] += 1
                continue
            setup = PushFoldSetup.from_code(node)
            r = solve_cached(setup)
            parts = spot_id.split(".")
            act, who = parts[-2], parts[-1]
            pos = POSITIONS[len(setup.stacks)]
            if act == "push":
                i, j = pos.index(who), None
            else:
                hero_p, _, vil_p = who.partition("_vs_")
                i, j = pos.index(vil_p), pos.index(hero_p)
            b.cards.append(
                _pushfold_card(b, r, act, i, j, CLASS_INDEX[hand_class], hand_html, key, history)
            )
    return b


# ---------------------------------------------------------------------------- package


def deck_id(kind: str) -> int:
    return 1_900_000_000 + zlib.crc32(DECK_NAMES[kind].encode()) % 100_000_000


def write_apkg(build: DeckBuild, path: Path) -> Path:
    deck = genanki.Deck(
        deck_id(build.kind),
        build.name,
        description="Exported from GTO Trainer. Strategy sources are shown "
        "on every card; approximate seed charts are not GTO.",
    )
    for c in build.cards:
        deck.add_note(
            genanki.Note(
                model=MODEL,
                fields=[c.front, c.back, c.key],
                tags=c.tags,
                guid=genanki.guid_for(c.key),
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, svg in build.media.items():
            f = Path(tmp) / name
            f.write_text(svg)
            files.append(str(f))
        pkg = genanki.Package(deck)
        pkg.media_files = files
        pkg.write_to_file(str(path))
    return path
