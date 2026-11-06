"""Hand-history review: map each hero decision onto a modelled spot and grade it.

Preflop: a small state machine replays the action and classifies every hero decision:

* no raise yet, no limpers                    -> ``rfi``
* one open, nobody else in                     -> ``vs_open`` (``bb_defense`` for the BB)
* hero opened, one player 3-bet, nobody else   -> ``vs_3bet``
* hero 3-bet the opener, the opener 4-bet      -> ``vs_4bet`` (hero raising = 5-bet jam)

Everything else (limped pots, cold callers / squeezes, 5-bet+ wars, heads-up or 7+ handed
tables, very short stacks, tournaments) is reported as *not scored*, with the reason. We'd
rather show "no solution for this" than grade against a chart that doesn't describe it.

Positions come from the seat order relative to the button, named the way the 6-max format
names them for that many players (5-handed has no UTG: first in is "HJ"), which keeps the
"number of players behind" -- the thing an opening range depends on -- correct.

Postflop: if the hand went heads-up to a flop that is a suit permutation of a solved flop
(same pot type and positions), the line is walked through the solved tree. Bet sizes map to
the nearest modelled size (log-distance) and suits are relabelled through the permutation,
so hero's hand is graded in the isomorphic solved spot. Turn cards must be one of the
solver's representative turn cards (up to the flop's suit symmetries); rivers aren't stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import FORMATS, GRADING
from ..poker.cards import (
    CLASS_LABELS,
    COMBO_CLASS,
    card_from_str,
    card_to_str,
    combo_index,
    parse_cards,
)
from ..poker.categories import preflop_category
from ..solver.abstraction import SUIT_PERMS, Perm, permute_card, stabilizer
from ..solver.postflop import SPOTS, Solution, load_solution
from ..solver.tree import action_label
from ..strategies.loader import Registry
from .grading import GradeResult, grade_decision
from .hand_history import HHAction, ParsedHand

FORMAT = FORMATS["6max_100bb"]
_NAMES = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"],
}


class SkipHand(Exception):  # noqa: N818 - control flow, not an error
    pass


@dataclass
class ReviewedDecision:
    street: str
    scenario: str
    hero_pos: str
    villain_pos: str | None
    action_text: str  # what hero did, in words
    action: str | None  # model action id it maps to
    scored: bool
    reason: str = ""  # why not scored
    spot_id: str | None = None
    node: str = ""
    board: str = ""
    freqs: dict[str, float] | None = None
    labels: dict[str, str] = field(default_factory=dict)
    grade: GradeResult | None = None
    category: str = ""
    pot_bb: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class HandReview:
    hand: ParsedHand
    skip_reason: str | None = None
    positions: dict[str, str] = field(default_factory=dict)
    hero_pos: str = ""
    effective_bb: float = 0.0
    net_bb: float = 0.0
    decisions: list[ReviewedDecision] = field(default_factory=list)
    postflop_note: str | None = None

    @property
    def hand_class(self) -> str:
        if len(self.hand.hero_cards) != 2:
            return ""
        c = combo_index(*(card_from_str(x) for x in self.hand.hero_cards))
        return CLASS_LABELS[int(COMBO_CLASS[c])]


def assign_positions(h: ParsedHand) -> dict[str, str]:
    """Name -> position, clockwise from the small blind."""
    active = {a.player for a in h.actions}
    seats = sorted(
        (s, n) for s, (n, _) in h.seats.items() if n in active and n not in h.sitting_out
    )
    n = len(seats)
    if n < 3:
        raise SkipHand("heads-up hands are not modelled (6-max data)")
    if n > 6:
        raise SkipHand(f"{n}-handed table; the strategy data is 6-max")
    if h.button_seat is None:
        raise SkipHand("no button seat in the hand history")
    after = [x for x in seats if x[0] > h.button_seat] + [x for x in seats if x[0] <= h.button_seat]
    names = _NAMES[n]
    order = names[-2:] + names[:-2]  # SB, BB, first-in ..., BTN
    pos = {name: order[i] for i, (_, name) in enumerate(after)}
    sb = next((a.player for a in h.actions if a.kind == "post_sb"), None)
    bb = next((a.player for a in h.actions if a.kind == "post_bb"), None)
    if bb is None or pos.get(bb) != "BB" or (sb is not None and pos.get(sb) != "SB"):
        raise SkipHand("irregular blinds (dead/missing blind); positions are ambiguous")
    return pos


def _combo(cards: list[str]) -> int:
    return combo_index(card_from_str(cards[0]), card_from_str(cards[1]))


def _grade(
    freqs: dict[str, float], action: str, labels: dict[str, str], ev: dict[str, float] | None = None
) -> GradeResult:
    if action not in freqs:
        freqs = {**freqs, action: 0.0}
    return grade_decision(freqs, action, ev, labels, GRADING)


# ---------------------------------------------------------------------------- preflop


@dataclass
class _Raise:
    player: str
    to: float


def review_hand(h: ParsedHand, registry: Registry) -> HandReview:
    rv = HandReview(hand=h)
    try:
        if h.tournament:
            raise SkipHand("tournament hand (cash-game data only)")
        if not h.game_ok:
            raise SkipHand("not No-Limit Hold'em")
        if h.hero is None or len(h.hero_cards) != 2:
            raise SkipHand("no hero hole cards in the history")
        if h.bb <= 0:
            raise SkipHand("could not read the blinds")
        rv.positions = assign_positions(h)
    except SkipHand as e:
        rv.skip_reason = str(e)
        return rv
    hero, bb = h.hero, h.bb
    rv.hero_pos = rv.positions.get(hero, "")
    rv.net_bb = h.net(hero) / bb
    others = [n for n in rv.positions if n != hero]
    rv.effective_bb = min(h.stack_of(hero), max(h.stack_of(o) for o in others)) / bb
    combo = _combo(h.hero_cards)

    raises: list[_Raise] = []
    callers: set[str] = set()  # called since the first raise
    limpers: set[str] = set()
    folded: set[str] = set()
    for a in (x for x in h.actions if x.street == "preflop"):
        if a.kind.startswith("post"):
            continue
        if a.player == hero:
            rv.decisions.append(_preflop_decision(rv, registry, a, raises, callers, limpers, combo))
        if a.kind == "raise":
            raises.append(_Raise(a.player, a.to or a.amount))
        elif a.kind == "call":
            (callers if raises else limpers).add(a.player)
        elif a.kind == "fold":
            folded.add(a.player)

    in_hand = [n for n in rv.positions if n not in folded]
    if hero in in_hand and any(a.street == "flop" for a in h.actions) and len(h.board) >= 3:
        rv.decisions.extend(_review_postflop(rv, raises, in_hand))
    return rv


def _preflop_decision(
    rv: HandReview,
    reg: Registry,
    a: HHAction,
    raises: list[_Raise],
    callers: set[str],
    limpers: set[str],
    combo: int,
) -> ReviewedDecision:
    h, bb, pos = rv.hand, rv.hand.bb, rv.positions
    hero_pos = rv.hero_pos
    text = {
        "fold": "folds",
        "check": "checks",
        "call": "calls",
        "raise": f"raises to {(a.to or 0) / bb:g}bb",
    }.get(a.kind, a.kind)
    if a.allin:
        text += " (all-in)"
    d = ReviewedDecision("preflop", "", hero_pos, None, text, None, False)

    def unscored(reason: str) -> ReviewedDecision:
        d.reason = reason
        return d

    n = len(raises)
    if n == 0:
        if limpers:
            return unscored("limped pot: not modelled")
        d.scenario = "rfi"
        if a.kind == "call" and hero_pos == "SB":
            return unscored("SB limp: the seed SB chart is raise-or-fold")
        if a.kind == "check":
            return unscored("BB option in a limped pot: not modelled")
        action = {"fold": "fold", "raise": "raise", "call": "call"}[a.kind]
    elif n == 1:
        opener = raises[0].player
        if callers or limpers:
            return unscored("multiway: someone called/limped before you (squeeze spot)")
        d.villain_pos = pos[opener]
        d.scenario = "bb_defense" if hero_pos == "BB" else "vs_open"
        action = {"fold": "fold", "call": "call", "raise": "raise"}.get(a.kind, "")
        if raises[0].to / bb > FORMAT.open_size(pos[opener]) * 1.3:
            d.notes.append(
                f"villain opened to {raises[0].to / bb:g}bb "
                f"(model: {FORMAT.open_size(pos[opener]):g}bb)"
            )
    elif n == 2:
        if raises[0].player != rv.hand.hero or callers or limpers:
            return unscored("3-bet pot you didn't open, or multiway")
        d.villain_pos = pos[raises[1].player]
        d.scenario = "vs_3bet"
        action = {"fold": "fold", "call": "call", "raise": "raise"}.get(a.kind, "")
    elif n == 3:
        if (
            raises[1].player != rv.hand.hero
            or raises[2].player != raises[0].player
            or callers
            or limpers
        ):
            return unscored("4-bet pot outside the modelled lines")
        d.villain_pos = pos[raises[0].player]
        d.scenario = "vs_4bet"
        action = {"fold": "fold", "call": "call", "raise": "allin"}.get(a.kind, "")
        if a.kind == "raise" and not a.allin:
            d.notes.append("your 5-bet wasn't all-in; graded as the model's 5-bet jam")
    else:
        return unscored("5-bet+ pot: not modelled")

    eff = (
        rv.effective_bb
        if d.villain_pos is None
        else min(
            h.stack_of(rv.hand.hero or ""),
            h.stack_of(next(p for p, q in pos.items() if q == d.villain_pos)),
        )
        / bb
    )
    if eff < 40:
        return unscored(f"effective stack {eff:.0f}bb: push/fold depth, data is 100bb")
    if not 70 <= eff <= 140:
        d.notes.append(f"effective stack {eff:.0f}bb (model: 100bb)")

    spot = reg.find(FORMAT.id, d.scenario, hero_pos, d.villain_pos)
    if spot is None:
        who = f"{hero_pos} vs {d.villain_pos}" if d.villain_pos else hero_pos
        return unscored(f"no strategy loaded for {d.scenario} {who}")
    d.spot_id = spot.id
    if spot.reach[combo] <= 0:
        return unscored("your hand isn't in the modelled range that reaches this spot")
    d.action = action
    d.labels = {x.id: x.label for x in spot.ctx.actions}
    if action == "call" and d.scenario == "rfi":
        d.labels["call"] = "Limp"
    d.freqs = spot.combo_freqs(combo)
    ev = None
    if spot.ev is not None:
        ev = {x: float(spot.ev[combo, i]) for i, x in enumerate(spot.action_ids)}
    d.grade = _grade(d.freqs, action, d.labels, ev)
    model = next((x.size_bb for x in spot.ctx.actions if x.id == action), None)
    if a.kind == "raise" and model and a.to and abs(a.to / bb - model) / model > 0.25:
        d.notes.append(f"you raised to {a.to / bb:g}bb (model size {model:g}bb)")
    d.category = preflop_category(CLASS_LABELS[int(COMBO_CLASS[combo])])
    d.pot_bb = spot.ctx.pot_bb
    d.scored = True
    return d


# ---------------------------------------------------------------------------- postflop


def _find_perm(src: list[int], dst: list[int]) -> Perm | None:
    target = set(dst)
    for p in SUIT_PERMS:
        if {permute_card(c, p) for c in src} == target:
            return p
    return None


def _compose(outer: Perm, inner: Perm) -> Perm:
    """Suit map applying ``inner`` first, then ``outer``."""
    return tuple(outer[inner[s]] for s in range(4))


def _matching_spot(
    rv: HandReview, raises: list[_Raise], in_hand: list[str]
) -> tuple[Solution | None, str, Perm | None]:
    pos = rv.positions
    names = {pos[n] for n in in_hand}
    flop = parse_cards("".join(rv.hand.board[:3]))
    for d in SPOTS.values():
        if names != {d.oop, d.ip}:
            continue
        if d.pot_type == "SRP":
            ok = len(raises) == 1 and pos[raises[0].player] == d.ip
        else:
            ok = (
                len(raises) == 2
                and pos[raises[0].player] == d.ip
                and pos[raises[1].player] == d.oop
            )
        if not ok:
            continue
        perm = _find_perm(flop, parse_cards(d.board))
        if perm is None:
            continue
        sol = load_solution(d.id)
        if sol is None:
            return None, f"postflop: {d.id} is not solved yet (run make precompute)", None
        return sol, "", perm
    solved = ", ".join(
        sorted({f"{d.board} ({d.ip} vs {d.oop} {d.pot_type})" for d in SPOTS.values()})
    )
    return None, f"postflop not scored: no solved spot for this pot/flop (solved: {solved})", None


def _nearest(tokens: list[str], kind: str, value: float) -> tuple[str | None, float]:
    best, dist = None, math.inf
    for t in tokens:
        if t.startswith(kind + ":"):
            v = float(t.split(":")[1])
            dd = abs(math.log(max(value, 1e-6) / v))
            if dd < dist:
                best, dist = t, dd
    return best, dist


def _review_postflop(
    rv: HandReview, raises: list[_Raise], in_hand: list[str]
) -> list[ReviewedDecision]:
    if len(in_hand) != 2:
        rv.postflop_note = "postflop not scored: multiway pot"
        return []
    sol, note, perm = _matching_spot(rv, raises, in_hand)
    if sol is None or perm is None:
        rv.postflop_note = note
        return []
    h, bb, hero = rv.hand, rv.hand.bb, rv.hand.hero or ""
    oop_pos, ip_pos = sol.positions()
    actor_name = {
        0: next(n for n in in_hand if rv.positions[n] == oop_pos),
        1: next(n for n in in_hand if rv.positions[n] == ip_pos),
    }
    pot = sum(h.invested_on(n, "preflop") for n in rv.positions) / bb
    if abs(pot - sol.spot["tree"]["starting_pot"]) / sol.spot["tree"]["starting_pot"] > 0.15:
        rv.postflop_note = (
            f"pot {pot:.1f}bb vs solved {sol.spot['tree']['starting_pot']:g}bb: "
            "sizes mapped by pot fraction"
        )
    hero_cards = [card_from_str(c) for c in h.hero_cards]
    solved_flop = parse_cards(sol.spot["board"])
    board = [card_from_str(c) for c in h.board]
    out: list[ReviewedDecision] = []
    path, street = "", "flop"
    street_inv: dict[str, float] = {}

    def stop(msg: str) -> list[ReviewedDecision]:
        rv.postflop_note = msg
        return out

    for a in (x for x in h.actions if x.street in ("flop", "turn", "river")):
        if a.street != street:
            if a.street == "river":
                return stop("river decisions aren't stored in the cached solution")
            if path not in sol.chance:
                return stop("this line doesn't continue to the turn in the solved tree")
            turn = permute_card(board[3], perm)
            reps = sol.chance[path]
            if card_to_str(turn) not in reps:
                alt = next(
                    (
                        q
                        for q in stabilizer(tuple(solved_flop))
                        if card_to_str(permute_card(turn, q)) in reps
                    ),
                    None,
                )
                if alt is None:
                    return stop(
                        f"turn {h.board[3]} isn't one of the solver's representative "
                        "turn cards; turn play not scored"
                    )
                perm = _compose(alt, perm)
                turn = permute_card(board[3], perm)
            path = f"{path}/{card_to_str(turn)}"
            street = a.street
            street_inv = {}
        nd = sol.nodes.get(path)
        if nd is None:
            return stop("the line left the stored part of the solved tree")
        if a.player != actor_name[nd.actor]:
            return stop("action order doesn't match the solved tree")
        prev = street_inv.get(a.player, 0.0)
        facing = max(street_inv.values(), default=0.0)
        if a.kind in ("fold", "check", "call"):
            token: str | None = a.kind
        elif a.kind == "bet":
            frac = a.amount / bb / pot
            token, dist = _nearest(nd.actions, "bet", frac)
            if (a.allin or token is None) and "allin" in nd.actions:
                token, dist = "allin", 0.0
            if token and dist > math.log(1.5):
                rv.postflop_note = f"bet of {frac:.0%} pot mapped to {action_label(token)}"
        else:  # raise
            to_bb = (a.to or 0.0) / bb
            mult = to_bb / max(facing / bb, 1e-9)
            token, _ = _nearest(nd.actions, "raise", mult)
            if (a.allin or token is None) and "allin" in nd.actions:
                token = "allin"
        if token not in nd.actions:
            return stop(
                f"{a.player}'s {a.kind} isn't in the solved tree here "
                f"(options: {', '.join(action_label(x) for x in nd.actions)})"
            )
        assert token is not None
        if a.player == hero:
            combo = combo_index(*(permute_card(c, perm) for c in hero_cards))
            hits = [i for i, c in enumerate(sol.combos[nd.actor]) if int(c) == combo]
            d = ReviewedDecision(
                street,
                sol.spot["pot_type"],
                rv.positions[hero],
                rv.positions[actor_name[1 - nd.actor]],
                f"{a.kind}s" if a.kind != "check" else "checks",
                token,
                False,
                spot_id=sol.spot["id"],
                node=path,
                board=" ".join(h.board[: 3 if street == "flop" else 4]),
                pot_bb=nd.pot,
            )
            if a.kind in ("bet", "raise"):
                d.action_text += (
                    f" {a.amount / bb:g}bb" if a.kind == "bet" else f" to {(a.to or 0) / bb:g}bb"
                )
            if not hits or nd.reach[hits[0]] <= 1e-6:
                d.reason = "your hand isn't in the modelled range at this node"
            else:
                i = hits[0]
                d.freqs = {x: float(nd.strategy[i, k]) for k, x in enumerate(nd.actions)}
                d.labels = {x: action_label(x) for x in nd.actions}
                ev = {x: float(nd.ev[i, k]) for k, x in enumerate(nd.actions)}
                d.grade = _grade(d.freqs, token, d.labels, ev)
                d.category = sol.hand_categories(nd.actor, nd.board)[i]
                d.scored = True
                if perm != (0, 1, 2, 3):
                    d.notes.append(
                        f"graded in the suit-isomorphic solved spot ({' '.join(nd.board)})"
                    )
            out.append(d)
        # bookkeeping
        if a.kind == "raise":
            add = (a.to or 0.0) - prev
            street_inv[a.player] = a.to or 0.0
        else:
            add = a.amount
            street_inv[a.player] = prev + a.amount
        pot += add / bb
        path = f"{path}/{token}" if path else token
        if token == "fold":
            break
    return out
