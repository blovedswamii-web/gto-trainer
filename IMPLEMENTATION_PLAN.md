# Implementation plan: visual poker table and drill UI

Scope: replace the text-only game state in the drills with a visual table, make
betting controls part of that table, and tidy the surrounding UI. Solver, grading,
ranges and strategy data are not touched. No new dependencies (React 19, Tailwind 4,
inline SVG only).

Each phase is independently shippable. Phases 1–3 deliver the preflop table; 4–5
extend it to the other drills; 6–7 are polish.

## Ground rules

- Backend changes are **additive**: new optional fields on existing responses. No
  existing field is renamed or removed, and no solver/grading code changes.
- The table is presentational. It never computes poker state; it renders a
  `TableState` it is given.
- Reuse `PlayingCard`, `Panel`, `Pill`, `Board`, `GradeCard`, `HistoryLine`, the
  action colours from `lib/actions.ts`, and the existing hotkeys (`keyToAction`).

## The contract: `TableState`

```ts
interface SeatState {
  position: string;      // "UTG" … "BB"
  stack_bb: number;      // behind, after chips committed
  committed_bb: number;  // chips in front of the seat this street (blinds included)
  folded: boolean;
  is_hero: boolean;
  is_actor: boolean;     // seat whose turn it is
  is_villain: boolean;   // the opponent hero's decision is against
  last_action: string | null; // "raise", "3bet", "call", "check", "bet", "allin"
}
interface TableState {
  seats: SeatState[];    // in table order, starting from first to act preflop
  dealer: string;        // position holding the button
  pot_bb: number;        // total pot including committed chips
  to_call_bb: number;
  board: string[];       // [] preflop
}
```

Invariant (tested): `sum(committed_bb) + dead money == pot_bb` preflop, and exactly
one seat has `is_hero`, exactly one has `is_actor`.

---

## Phase 1 — Backend: expose seat state (Low)

| File | Change |
|---|---|
| `backend/gto_trainer/strategies/schemas.py` | Add `SeatState` and `TableState` pydantic models. |
| `backend/gto_trainer/strategies/context.py` | `build_context` already tracks `contrib` and `folded`; add a `table: TableState` field to `SpotContext` built from them (stack = `fmt.stack_bb - contrib`, dealer = `BTN`). No change to history, pot or action derivation. |
| `backend/gto_trainer/api/routes_drill.py`, `routes_spots.py` | Add optional `table` to the preflop deal and spot-detail responses. |
| `backend/gto_trainer/training/postflop_view.py` | In the node-base dict (around line 135) add `table`: two live seats from `actor_position` / `villain_position`, the rest folded; `stack_bb = nd.stack`; villain `committed_bb = nd.to_call`; `pot_bb = nd.pot`; `board`. |
| `backend/tests/test_drill_api.py`, `test_postflop.py` | Assert the invariant above for every scenario (`rfi`, `vs_open`, `bb_defense`, `vs_3bet`, `vs_4bet`) and for a flop and a turn node. |

Acceptance
- `make test-fast` and `make lint` pass; existing response fields are byte-identical.
- For `BTN vs CO open`: CO committed = open size, SB 0.5, BB 1, UTG/HJ folded, BTN is
  hero and actor, `pot_bb` equals the existing `pot_bb`.

## Phase 2 — `PokerTable` component (Medium)

| File | Change |
|---|---|
| `frontend/src/lib/types.ts` | Add `SeatState`, `TableState`; add optional `table` to `PreflopDeal`, `SpotDetail`, `PfNodeBase`. |
| `frontend/src/components/PokerTable.tsx` (new) | The component below. |
| `frontend/src/index.css` | Tokens: `--color-felt`, `--color-felt-rim`, chip colours; a `seat-active` ring keyframe wrapped in `prefers-reduced-motion`. |

Design
- Container with fixed aspect ratio (~16:9); an inline SVG ellipse for felt and rim;
  seats, chips and cards are absolutely positioned HTML over it.
- Seat coordinates come from one pure function `seatLayout(n)` (2–9 players) that
  returns percentage positions around the ellipse. Seats are **rotated so the hero is
  always bottom-centre**; relative order is preserved.
- Seat: position badge, stack in bb, two face-down card backs (hero shows
  `heroCards` via `PlayingCard`). Folded = 40% opacity and no cards. Villain = accent
  border. Actor = animated ring.
- Dealer button chip next to the `dealer` seat. `committed_bb > 0` renders a chip
  stack with a `num` label, placed 35% of the way from seat to centre.
- Centre: five board slots (empty outlines preflop, filled via `Board`), pot total
  beneath, and "pot odds X%" when `to_call_bb > 0`.
- Props: `{ table, heroCards?, children? (overlay slot under hero), dimmed? }`.
- Accessibility: `role="img"` with an `aria-label` summarising the state in the same
  words `HistoryLine` uses; all colour signals also carry text or shape.

Acceptance
- Renders correctly for 2, 6 and 9 seats with no overlap at 360 px and 1400 px width.
- Hero is bottom-centre for every hero position; SB/BB posts and the button sit at
  the right seats.
- `npm run typecheck` passes; no new packages in `package.json`.

## Phase 3 — Preflop drill on the table, with betting controls (Low–Medium)

| File | Change |
|---|---|
| `frontend/src/components/ActionBar.tsx` (new) | Extract the duplicated action-button grid: buttons coloured by `ActionMeta`, hotkey hint, bet size as the primary label ("3-bet to 9bb"), disabled while busy or after answering. Hovering or focusing a button previews the hero's chips for that size on the table. |
| `frontend/src/pages/PreflopDrill.tsx` | Replace the cards + "Pot / To call" block (lines ~157–190) with `<PokerTable>` and `<ActionBar>` docked beneath the hero seat. Keep `HistoryLine` as a one-line caption above. Fall back to the current layout if `deal.table` is absent. |

Acceptance
- A user can tell hero position, who raised, how much, and the pot without reading the
  caption.
- Keys `1–3`, `F/C/R/A`, `Space/Enter/N` behave exactly as today.
- Grades and logged decisions for a fixed spot/hand/action match the current build
  (request payloads unchanged).

## Phase 4 — Shared drill loop (Low–Medium)

| File | Change |
|---|---|
| `frontend/src/lib/useDrill.ts` (new) | `useDrill<Deal, Feedback>({ fetchNext, submit, actionsOf })` owning `deal`, `fb`, `busy`, `error`, session stats, response timing, and the keyboard handler (with the input/textarea focus guard). |
| `frontend/src/pages/PreflopDrill.tsx`, `PostflopTrainer.tsx`, `PushFold.tsx` | Replace the three copy-pasted loops with the hook. This also fixes the missing focus guard in the postflop key handler. |
| `frontend/src/components/Postflop.tsx` | Remove the dead ternary at line 49. |

Acceptance
- No behaviour change in any drill; net line count of the three pages goes down.
- Typing in any input never triggers an answer.

## Phase 5 — Postflop and push/fold on the table (Medium)

| File | Change |
|---|---|
| `frontend/src/pages/PostflopTrainer.tsx` | Drill tab: `<PokerTable table={deal.table} heroCards=…>` replaces `Board` + text block (lines ~228–245); `ActionBar` in a 2-column layout for the larger action menu. Explorer tab: show the same table read-only above `NodeStrategy`. |
| `frontend/src/pages/PushFold.tsx` | Build a `TableState` in the page from data it already has (`positions`, `deal.stacks`, `deal.hero`, `deal.shover`, ante); render the table in `Drill` instead of the sentence at lines ~113–121. No backend change. |

Acceptance
- Postflop: board, pot, villain's bet and effective stack are all visible on the
  table; turn nodes show four cards.
- Push/fold: works for 2–9 players with unequal stacks; the shover's all-in chips are
  shown.

## Phase 6 — Feedback on the felt (Medium)

| File | Change |
|---|---|
| `frontend/src/components/PokerTable.tsx` | Optional `result` prop: after answering, hero's chosen action moves chips in (CSS transition), and a `SegmentedBar` of the solution mix appears at the hero seat. |
| `frontend/src/components/Feedback.tsx` | `GradeCard` gains a compact variant shown beside the pot (grade, EV loss when available); the full card stays below. |
| `PreflopDrill.tsx`, `PostflopTrainer.tsx` | Pass `result` after feedback arrives. |

Acceptance
- Within one glance after answering: what you did, what the solution does, and the
  grade. Animations are skipped under `prefers-reduced-motion`.

## Phase 7 — Navigation and layout polish (Low)

| File | Change |
|---|---|
| `frontend/src/App.tsx` | Group the nine links: **Train** (Preflop, Postflop, Push/fold), **Study** (Ranges, Range editor, Equity & odds), **Review** (Hand review, Progress, Anki). Routes unchanged. |
| `frontend/src/pages/PreflopDrill.tsx`, `PostflopTrainer.tsx` | Collapse the filter row into a "Filters" disclosure on narrow screens so the table stays above the fold. |
| `frontend/src/pages/PreflopDrill.tsx` | Take the position filter list from the format (`/formats`) instead of the hard-coded `POSITIONS`. |

Acceptance
- All existing URLs still resolve; table and action bar are visible without scrolling
  at 1280×720.

---

## Out of scope (candidates for later)

- "Play the hand through" postflop mode (chained flop → turn decisions).
- Table replay in Hand review (`HandReview.tsx` timeline → `TableState` per step).
- Frontend test setup (would add a dependency such as Vitest).

## Verification for every phase

- `make lint` (ruff, mypy `--strict`, `tsc --noEmit`) and `make test-fast`.
- Manual pass in `make dev`: one hand of each preflop scenario, one flop and one turn
  postflop spot, one push/fold deal, at desktop and phone widths.
