"""Parser for PokerStars and GGPoker No-Limit Hold'em hand histories (cash games).

Both sites use the same skeleton, so one line-oriented parser covers them::

    PokerStars Hand #245678901234:  Hold'em No Limit ($0.25/$0.50 USD) - 2026/10/01 20:15:03 ET
    Poker Hand #HD12345678: Hold'em No Limit ($0.25/$0.5) - 2026/10/01 20:15:03      <- GGPoker
    Table 'Alcor II' 6-max Seat #3 is the button
    Seat 1: Villain1 ($50.00 in chips)
    Villain4: posts small blind $0.25
    *** HOLE CARDS ***
    Dealt to Hero [Ah Kd]
    Villain1: raises $0.75 to $1.25          <- "to" = total street investment
    Hero: calls $1.25
    *** FLOP *** [Ks 7d 2c]
    Villain1: bets $2.75 and is all-in
    Uncalled bet ($2.75) returned to Hero
    Hero collected $8.25 from pot

Unknown lines (chat, "is disconnected", jackpot/insurance lines, summary details) are
ignored. Amounts are converted to big blinds by the caller. Tournaments and non-NLHE games
are parsed but flagged so the review can skip them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

_HEADER = re.compile(r"^(PokerStars(?: Zoom)? Hand|Poker Hand) #(\w+):\s*(.*)$")
_BLINDS = re.compile(r"\(([^\d\s(]?)([\d.,]+)/([^\d\s]?)([\d.,]+)")
_DATE = re.compile(r"(\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2})")
_TABLE = re.compile(r"^Table '([^']*)' (\d+)-max .*?Seat #(\d+) is the button")
_SEAT = re.compile(r"^Seat (\d+): (.+?) \(([^\d\s]?)([\d.,]+) in chips[^)]*\)(.*)$")
_POST = re.compile(
    r"^(.+?): posts (small blind|big blind|the ante|small & big blinds|missed blind|straddle)"
    r" ([^\d\s]?)([\d.,]+)"
)
_DEALT = re.compile(r"^Dealt to (.+?) \[(\w\w) (\w\w)\]")
_STREET = re.compile(r"^\*\*\* (HOLE CARDS|FLOP|TURN|RIVER|SHOW ?DOWN|SUMMARY) \*\*\*(.*)$")
_ACTION = re.compile(
    r"^(.+?): (folds|checks|calls|bets|raises)"
    r"(?: ([^\d\s]?)([\d.,]+))?(?: to ([^\d\s]?)([\d.,]+))?( and is all-in)?"
)
_UNCALLED = re.compile(r"^Uncalled bet \(([^\d\s]?)([\d.,]+)\) returned to (.+)$")
_COLLECTED = re.compile(r"^(.+?) collected ([^\d\s]?)([\d.,]+) from")


def _num(text: str) -> float:
    return float(text.replace(",", ""))


@dataclass
class HHAction:
    street: str  # preflop | flop | turn | river
    player: str
    kind: str  # post_sb, post_bb, post_ante, post_other, fold, check, call, bet, raise
    amount: float  # chips put in by this action
    to: float | None = None  # raises: total street investment afterwards
    allin: bool = False


@dataclass
class ParsedHand:
    site: str  # pokerstars | ggpoker
    hand_id: str
    raw: str
    played_at: datetime | None = None
    tournament: bool = False
    game_ok: bool = True  # No-Limit Hold'em
    sb: float = 0.0
    bb: float = 0.0
    table_max: int | None = None
    button_seat: int | None = None
    seats: dict[int, tuple[str, float]] = field(default_factory=dict)  # seat -> (name, stack)
    sitting_out: set[str] = field(default_factory=set)
    hero: str | None = None
    hero_cards: list[str] = field(default_factory=list)
    board: list[str] = field(default_factory=list)
    actions: list[HHAction] = field(default_factory=list)
    uncalled: dict[str, float] = field(default_factory=dict)
    collected: dict[str, float] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.site}:{self.hand_id}"

    def stack_of(self, name: str) -> float:
        return next((st for n, st in self.seats.values() if n == name), 0.0)

    def invested(self, name: str) -> float:
        """Total chips ``name`` put in, net of uncalled bets returned."""
        street_inv: dict[str, float] = {}
        total = 0.0
        for a in self.actions:
            if a.player != name:
                continue
            key = a.street
            if a.kind == "raise" and a.to is not None:
                add = a.to - street_inv.get(key, 0.0)
                street_inv[key] = a.to
            else:
                add = a.amount
                if a.kind != "post_ante":
                    street_inv[key] = street_inv.get(key, 0.0) + a.amount
            total += add
        return total - self.uncalled.get(name, 0.0)

    def invested_on(self, name: str, street: str) -> float:
        """Chips ``name`` put in on one street (antes included, uncalled bets not deducted)."""
        level, total = 0.0, 0.0
        for a in self.actions:
            if a.player != name or a.street != street:
                continue
            if a.kind == "raise" and a.to is not None:
                total += a.to - level
                level = a.to
            else:
                total += a.amount
                if a.kind != "post_ante":
                    level += a.amount
        return total

    def net(self, name: str) -> float:
        return self.collected.get(name, 0.0) - self.invested(name)


def split_hands(text: str) -> list[str]:
    """Split a file/paste into individual hand texts (header line starts a hand)."""
    chunks: list[list[str]] = []
    for line in text.replace("\r\n", "\n").replace("﻿", "").split("\n"):
        if _HEADER.match(line.strip()):
            chunks.append([])
        if chunks:
            chunks[-1].append(line.rstrip())
    return ["\n".join(c).strip() for c in chunks if any(x.strip() for x in c)]


def parse_hand(text: str) -> ParsedHand:
    lines = [ln.strip() for ln in text.strip().split("\n")]
    m = _HEADER.match(lines[0])
    if not m:
        raise ValueError("not a PokerStars/GGPoker hand history")
    site = "pokerstars" if m.group(1).startswith("PokerStars") else "ggpoker"
    rest = m.group(3)
    h = ParsedHand(site=site, hand_id=m.group(2), raw=text)
    h.tournament = "Tournament" in rest
    h.game_ok = "Hold'em No Limit" in rest or "Holdem No Limit" in rest
    if bm := _BLINDS.search(rest):
        h.sb, h.bb = _num(bm.group(2)), _num(bm.group(4))
    if dm := _DATE.search(rest):
        try:
            h.played_at = datetime.strptime(dm.group(1), "%Y/%m/%d %H:%M:%S")
        except ValueError:
            h.played_at = None

    street = "preflop"
    for line in lines[1:]:
        if not line:
            continue
        if tm := _TABLE.match(line):
            h.table_max, h.button_seat = int(tm.group(2)), int(tm.group(3))
            continue
        if street == "preflop" and (sm := _SEAT.match(line)):
            name = sm.group(2)
            h.seats[int(sm.group(1))] = (name, _num(sm.group(4)))
            if "sitting out" in sm.group(5):
                h.sitting_out.add(name)
            continue
        if st := _STREET.match(line):
            tag = st.group(1)
            if tag == "SUMMARY":
                break
            street = {
                "HOLE CARDS": "preflop",
                "FLOP": "flop",
                "TURN": "turn",
                "RIVER": "river",
            }.get(tag, "showdown")
            cards = re.findall(r"\[([^\]]+)\]", st.group(2))
            if cards:  # "[Ks 7d 2c] [9h]" -> the last bracket holds the new card(s)
                h.board.extend(cards[-1].split())
            continue
        if pm := _POST.match(line):
            kind = {"small blind": "post_sb", "big blind": "post_bb", "the ante": "post_ante"}.get(
                pm.group(2), "post_other"
            )
            h.actions.append(HHAction("preflop", pm.group(1), kind, _num(pm.group(4))))
            continue
        if dm2 := _DEALT.match(line):
            h.hero = dm2.group(1)
            h.hero_cards = [dm2.group(2), dm2.group(3)]
            continue
        if um := _UNCALLED.match(line):
            h.uncalled[um.group(3)] = h.uncalled.get(um.group(3), 0.0) + _num(um.group(2))
            continue
        if cm := _COLLECTED.match(line):
            h.collected[cm.group(1)] = h.collected.get(cm.group(1), 0.0) + _num(cm.group(3))
            continue
        if street in ("preflop", "flop", "turn", "river") and (am := _ACTION.match(line)):
            verb = am.group(2)
            kind = {
                "folds": "fold",
                "checks": "check",
                "calls": "call",
                "bets": "bet",
                "raises": "raise",
            }[verb]
            amount = _num(am.group(4)) if am.group(4) else 0.0
            to = _num(am.group(6)) if am.group(6) else None
            h.actions.append(HHAction(street, am.group(1), kind, amount, to, bool(am.group(7))))
    return h


def parse_many(text: str) -> tuple[list[ParsedHand], list[tuple[str, str]]]:
    """Parse every hand in ``text``. Returns (hands, [(snippet, error)])."""
    hands, errors = [], []
    for chunk in split_hands(text):
        try:
            hands.append(parse_hand(chunk))
        except (ValueError, IndexError) as e:
            errors.append((chunk.split("\n", 1)[0][:80], str(e)))
    return hands, errors
