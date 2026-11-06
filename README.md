# GTO Trainer — local-first Texas Hold'em decision trainer

Drill preflop and postflop decisions against mixed strategies, see ranges on a 13×13 grid,
and track your leaks over time. Everything runs on your machine, with no accounts and no
external APIs.

> **Training tool only.** This is not a bot and not a real-time assistant. Do not use it
> during live play.

---

## Quick start

Requirements: Python 3.12 (with [uv](https://github.com/astral-sh/uv), or plain `pip`),
Node 20+ and `make`.

```bash
make install      # backend venv + frontend node_modules
make precompute   # preflop equity table + solve the 4 postflop spots (~6 min, 4 processes)
make dev          # API on :8000 + UI on http://localhost:5173
```

Other targets:

| target | what |
|---|---|
| `make test` | full pytest suite (~1 min; includes the exhaustive evaluator check) |
| `make test-fast` | skips tests marked `slow` |
| `make lint` | ruff (lint + format check), mypy `--strict`, `tsc --noEmit` |
| `make backend` / `make frontend` | run one side only |

The SQLite database lives at `backend/data/trainer.db`. Delete it to reset your progress.

## Pages

| Page | What it does |
|---|---|
| **Ranges** | 13×13 grid per spot (RFI, vs open, BB defence, vs 3-bet, vs 4-bet). Each cell is a split-colour bar: widths are the action frequencies, height is the share of the cell that reaches the node. Hover shows exact combos and frequencies; click shows each suit combo. The side panel shows % played, combos per action and value:bluff for raises. A table view is available. |
| **Preflop drill** | Deals a spot and a hand, you pick fold/call/raise (keys `1-3`, `F/C/R/A`), then you get a grade, the solution's mix for your exact hand, your hand's all-in equity vs villain's range, and the full grid. Filter by spot type or position. Choose "close decisions" (mixed and boundary hands) or "realistic" dealing. Mistakes come back through spaced repetition. |
| **Postflop** | *Drill*: a decision node from a solved spot plus a hand in range. Graded on frequency **and EV loss** (bb and % pot) from the solver. *Explore*: walk the solved tree, with the strategy by **hand category** (sets, top pair good kicker, flush draws…), the reach-weighted grid, and EV per category. *Exploit mode*: give villain tendencies (over-folds to c-bets, calling station, never check-raises …) and see your best response, its EV gain over GTO, and what it costs if villain adapts. |
| **Equity & odds** | Hand/range vs range equity (exact enumeration when cheap, Monte Carlo with standard error otherwise). Pot odds, MDF, bluff break-even and river value:bluff. A pot-odds / MDF / bluff drill that is logged with your stats. |
| **Progress** | Accuracy, frequency error and EV loss. Rolling charts. Accuracy by position, street, spot type and hand class. **Top leaks** in plain English (e.g. *"You over-fold BB vs BTN open with suited gappers — 4 of 5 decisions off; the solution mostly plays call (100%)"*). Recent decisions. |
| **Range editor** | Paint a range with 100/75/50/25% brushes or type notation, then compare it against any chart action. You get overlap, Jaccard, over- and under-included hands, and a diverging difference heatmap. Quiz mode hides the answer. Ranges can be saved. |
| **Hand review** | Import PokerStars/GGPoker cash histories (paste or `.txt` files; a synthetic sample is built in). Every hero decision is mapped to a spot and graded, and the biggest deviations are listed first. The hand view shows the action timeline, the solution's mix for each decision, and the full range. Real-play decisions feed the progress page ("Real hands") and leak mining, and mistakes are queued for spaced-repetition drills. |

---

## What is and isn't GTO here

Read this before trusting any number in the app. Every strategy on screen carries a
**source badge**.

| Badge | Meaning | Where |
|---|---|---|
| ⚠ **Approximate seed chart · not solver output** | Hand-built approximations of commonly published 6-max 100bb charts. **Not GTO.** They're good enough to practise the shape of ranges, but wrong in the details. They have no EVs, so preflop drills grade by frequency only. | `backend/data/strategies/seed/` |
| ↧ **Imported** | Whatever you import (GTO Wizard, PioSOLVER, GTO+ exports …). Exactly as good as its source. | `backend/data/strategies/imported/` |
| ✓ **Solver-converged · exploitability X% pot** | Solved locally by the built-in DCFR solver. Exploitability is measured **inside the abstracted game** (see below). The postflop spots use the *approximate* preflop ranges as inputs, and the badge notes say so. | `backend/data/solutions/` |

Concretely:

- **The preflop charts are approximations.** Import real solver output to replace them:
  imports automatically override a seed spot with the same id.
- **The SB plays raise-or-fold** in the seed data, with no limping. Real solutions limp a lot,
  so the BB-vs-SB spots built on it are especially rough.
- **Postflop solutions are real equilibria of a smaller game.** The bet sizes are restricted:
  - flop: 33% / 75% / all-in, plus one 3× raise and then all-in
  - turn and river: 75% / all-in, with all-in as the only raise
  - no donk bets in single-raised pots

  Only **8 representative turn cards and 5 river cards per turn** are kept (see *Solver
  design*). The strategy is converged *for that game*, not for real NLHE with every size
  and runout. All-in equities are always exact over every runout.
- **Rake is ignored.** Each format has a `rake_pct` field so imported data can declare it.

---

## Architecture

```
frontend/ (React 19 + TypeScript + Vite + Tailwind 4)
  pages/       RangeViewer · PreflopDrill · PostflopTrainer · EquityTools · Progress · RangeEditor
  components/  HandGrid (13×13 split bars + tooltips) · SpotView · Postflop · Charts · Feedback
      │  JSON over /api  (Vite proxies to :8000)
backend/gto_trainer/ (Python 3.12, FastAPI, SQLAlchemy 2, NumPy)
  poker/       cards & combos · vectorised evaluator · equity (exact + MC) · range notation
               odds · preflop 169×169 equity table · preflop hand categories
  strategies/  pydantic schemas · format-derived spot context · loader/registry · importer CLI
  solver/      vector-form DCFR/CFR+ engine · Kuhn · suit isomorphism + runout sampling
               postflop tree builder · hand categories · spot definitions · precompute CLI
  training/    spot dealing · grading · SM-2 spaced repetition · progress/leak mining
               postflop node views · odds drills
  db/          models (decisions, srs_items, custom_ranges, users) · session
  api/         routers: spots, drill, postflop, stats, tools
backend/data/
  strategies/seed/*.json      approximate seed pack (23 spots)
  strategies/imported/*.json  your imports (override seed)
  preflop_equity.npz          169×169 class-vs-class equity (Monte Carlo, ±~1%)
  solutions/<spot>.{json,npz} cached postflop solutions (flop/turn nodes, committed)
  solutions/<spot>.strategy.npz  full per-node strategies for exploit mode (gitignored)
```

`GameFormat` in `gto_trainer/config.py` makes the game format **data**: player count, stack
depth, open sizes, 3-bet multipliers (in and out of position) and 4-bet multiplier. History,
pot and action menus for every spot are derived from it. `6max_40bb`, `6max_200bb` and
`hu_100bb` already exist as formats and become usable once strategy files for them are added.

### Hand evaluator

`poker/evaluator.py` is a **batched NumPy lookup-table evaluator** (5, 6 or 7 cards). It
runs at about 3M seven-card hands/sec on an M1, in pure NumPy.

**Why not treys/eval7/phevaluator?** Everything here evaluates *batches*: Monte Carlo
draws, and every combo on every runout for CFR showdown matrices.
- Per-hand Python APIs spend most of their time in call overhead.
- eval7 is unmaintained, with patchy 3.12 wheels.

A vectorised evaluator costs ~15 array operations per batch regardless of batch size, needs
no compiled extension, and is easy to verify exhaustively.

**How it works.**
1. Rank counts and per-suit 13-bit rank masks come from `bincount`.
2. Flushes and straights are resolved through two 8192-entry tables: highest straight in a
   mask (wheel-aware), and the top-5 ranks packed as nibbles.
3. Paired hands are resolved by sorting `count·16 + rank`.
4. The score packs into `category<<20 | kickers`, so comparing two hands is just comparing
   two ints.

**Verification** (`tests/test_evaluator.py`):
- All 2,598,960 five-card hands give the textbook category counts and exactly **7,462**
  distinct ranks.
- Every 7-card score equals the best of its 21 five-card subsets.
- 1,500 random showdowns agree with treys.
- Plus targeted cases: wheels, steel wheels, counterfeits, three pairs, two sets, board-plays
  splits.

### Equity

`poker/equity.py` treats a single hand as a one-hot range, so hand-vs-hand, hand-vs-range
and range-vs-range share one code path.

- **Exact enumeration** is used when `n_hero·n_villain·runouts` and the evaluation count are
  small. Examples: any hand vs hand preflop (1.7M boards, ~1.3s), hand vs range on the flop,
  and ranges on the turn. Cards held by every combo of a range are removed from the deck,
  and combos that conflict with every opposing combo are dropped first.
- **Monte Carlo** draws combos by weight and discards card-sharing pairs. Rejection is
  exactly conditioning on compatibility. It completes boards with a vectorised
  `argpartition` over random keys, and reports the standard error.

Known values are tested: AKs vs QQ = 46.2% exact, AA vs KK ≈ 81.3%, AKo vs 22 = 47.0%.

---

## Solver design

**Full 6-max postflop solving in Python is infeasible**: multiway trees, every bet size,
every runout. What *is* feasible, and what's implemented, is **heads-up postflop subgames**
with real combos, a restricted bet tree and an abstracted set of runouts. The pieces are
below.

### Engine: vector-form CFR (`solver/cfr.py`)

**Tree.** The tree contains only **public** states: betting history and dealt cards. Each
player node stores `regret` and `strat_sum` arrays of shape `(n_hands_of_actor, n_actions)`.

**Traversal.** A recursive pass sends **reach vectors** down and **counterfactual value
vectors** back up, updating every hand at once. At terminals, values are matrix-vector
products against the opponent's reach:

```
fold:      v_p =  ±(P0/2 + invested_by_folder) · C_p @ reach_opp
showdown:  v_p =   (P0/2 + invested)           · E_p @ reach_opp
```

- `C` is the card-compatibility matrix.
- `E[i,j] = P(win) − P(lose)`: an exact river comparison, or an exact all-in equity over
  every remaining runout before the river.
- Both are zero for hands that share a card, so **card removal is exact**.
- Treating the starting pot as owned half by each player makes the game zero-sum.

**Algorithm.**
- **DCFR** (Brown & Sandholm 2019; α=1.5, β=0, γ=2) with alternating updates.
- **CFR+** is selectable.
- On Kuhn poker the engine reaches exploitability ~5e-5 in 3k iterations, with the exact
  game value −1/18 and the textbook bet/bluff ratio (`tests/test_solver.py`).

### Card abstraction (`solver/abstraction.py`)

**Suit isomorphism (lossless).** Suit permutations that fix the board (its *stabiliser*)
partition the next cards into orbits:

| Board | Turn orbits |
|---|---|
| K72r | 49 (no reduction) |
| A93ss | 36 |
| 987hhh | 23 |

Only one representative per orbit is solved. The others are recovered by permuting hand
indices, `V_{φ(c)}(h) = V_c(φ⁻¹h)`, so blockers stay exact.

A test checks that merged and fully-explicit trees agree on the game value and on both
best-response values for a fixed strategy.

**Runout sampling (lossy; this is the actual abstraction).**
- Keep `turn_cards` (default 8) and `river_cards` (default 5) representative orbits per
  chance node.
- Orbits are sorted by suit relation to the board and by rank, then picked at equal
  **card-mass quantiles**, so flush cards, overcards, bricks and pairing cards are
  represented in proportion.
- Each kept orbit carries the probability mass of its stratum.

**Hand-strength bucketing is deliberately *not* used for solving.** I'm diverging from the
spec here. With ~600 combos per player, the vector engine and per-board showdown matrices
are fast enough on exact combos. Bucketing would add abstraction error for little speed-up.

Hand categories (`solver/categories.py`) are used only to **display** strategies ("bet/check
by hand category") and to aggregate leaks.

### Exploitability

A best-response pass against the average strategy gives
`(BR₀ + BR₁)/2 / starting pot × 100` = **% of pot**. This is the number on the source badge.

A typical `make precompute` run:

| Spot | Exploitability | Iterations | Time |
|---|---|---|---|
| SB vs BTN 3-bet pot, Q84r | 0.28% pot | 150 | ~40s |
| BTN vs BB single-raised pots (K72r, A93ss, 987tt) | ~0.4–0.5% pot | 400 | ~6 min (in parallel) |

Rerun with more iterations or a tighter target:

```bash
cd backend && .venv/bin/python -m gto_trainer.solver.precompute --force --iterations 800 --target 0.2
```

### Exploit mode (`solver/exploit.py`)

How the best response shifts when villain is *not* playing equilibrium:

1. Rebuild the solved game from the cache. It's deterministic: the same config, board and
   hand lists give the same tree and matrices. Then load every node's equilibrium
   strategy, rivers included, from `<spot>.strategy.npz`.
2. Edit villain's strategy with **tendencies**. Each one moves a fraction `x` of each hand's
   probability mass between actions at matching nodes, filtered by street:

   | Tendency | Mass moved |
   |---|---|
   | `fold_more` | call/raise → fold |
   | `call_more` | fold → call |
   | `raise_less` | raise → call |
   | `bet_less` | bet → check |
   | `bet_more` | check → smallest bet |

   Rows stay probability distributions, and every other node stays at equilibrium.

   Decisions **facing an all-in are excluded by default** (there's an opt-in per
   tendency). A uniform "calls too much" applied to 20×-pot jams makes villain call off
   his stack with air. The resulting "best response" just jams the nuts and checks
   everything else, which is technically correct and practically useless.
3. Run the solver's **best-response pass** for hero, recording per-hand action values at
   every hero node. The BR strategy is their argmax. Re-evaluating with that strategy
   reproduces the BR value exactly, and that is checked on every run.

What it reports, as hero EV per hand:

| Value | What it means |
|---|---|
| **Equilibrium** | GTO vs GTO |
| **GTO vs this villain** | What his leaks give you without adjusting |
| **Best response** | The maximum. The difference from the line above is what adjusting is worth. |
| **If villain adapts** | Villain's best response to *your* BR. A pure exploit is maximally greedy and maximally brittle, and this number shows how brittle. |

You also get the biggest adjustments (weighted by how often you reach the node) and a
GTO-vs-BR comparison by hand category and grid at any node.

It runs in ~1s per query, plus a few seconds the first time a spot is rebuilt. Caveats:
- It runs inside the same abstracted game as the solution.
- It assumes villain plays *exactly* the edited strategy.
- The full strategy files (~17 MB) are gitignored. `make precompute` creates them, and
  re-solves any spot that lacks one.

### Adding or changing postflop spots

Edit `SPOTS` in `solver/postflop.py`. Each spot sets:
- the board
- the preflop range references (scenario, hero, villain, action), resolved from the
  strategy registry, so imported preflop data automatically feeds the postflop ranges
- a `TreeConfig`: bet sizes per street and per player, raise multipliers, all-in threshold,
  and turn/river sample counts (`None` = all orbits = no sampling)

Then run `make precompute`.

---

## Strategy data & import format

A strategy file is JSON: either one spot, or a pack `{"schema_version": 1, "spots": [...]}`.

```jsonc
{
  "format_id": "6max_100bb",
  "scenario": "vs_open",            // rfi | vs_open | bb_defense | vs_3bet | vs_4bet
  "hero": "BTN", "villain": "CO",
  "source": {"type": "imported", "label": "GTO Wizard NL500 export", "notes": "...",
             "exploitability_pct": 0.3},

  // EITHER per-action range strings — fold is the remainder of each hand's reach:
  "ranges": {"raise": "TT+,AJs+,A5s:0.5", "call": "99-22,ATs:0.5"},
  // OR explicit frequencies per class ("AKs") or combo ("AsKs" overrides its class):
  "strategy": {"AKs": {"raise": 0.8, "call": 0.2}, "AsKs": {"raise": 1.0}},

  "ev":    {"AKs": {"raise": 3.1, "call": 2.9, "fold": 0}},  // optional, bb → enables EV-loss grading
  "reach": "22+,A2s+,..."   // optional; default = all hands, or the hero's range from the
                            // previous spot (vs_3bet reach = hero's RFI raise range)
}
```

Action ids are `fold`, `call`, `raise` (with `allin` for the 5-bet jam in `vs_4bet`).
Sizes, history and pot are derived from the `GameFormat`. Validate everything with:

```bash
cd backend && .venv/bin/python -m gto_trainer.strategies.loader
```

This prints every spot's reach and % played, and exits non-zero on errors (for example
"action weights exceed 100% for KK").

### Importer CLI

```bash
cd backend
# 1. CSV: one row per hand or combo; action columns hold freqs (0-1 or 0-100);
#    optional weight/reach and ev_<action> columns
.venv/bin/python -m gto_trainer.strategies.importer csv export.csv --meta meta.json
# 2. One <action>.txt per action, each a weighted range (PioSOLVER/GTO+ "export range").
#    Weights are reach × frequency and get normalised back into frequencies.
.venv/bin/python -m gto_trainer.strategies.importer ranges exports/btn_vs_co/ --meta meta.json
# 3. Already-native JSON: validated and copied
.venv/bin/python -m gto_trainer.strategies.importer json my_spot.json
```

`meta.json` carries `format_id`, `scenario`, `hero`, `villain`, `source` and an optional
`reach`. Imports cannot be tagged `approximate`. Reload with `POST /api/spots/reload` or
restart the API.

Range notation everywhere: `77+`, `99-66`, `AKs`, `AK`, `ATs+`, `A5s-A2s`, `T9s-54s`,
`AhKh`, `random`. Weights are written `AKs:0.5`, `:50` or `:50%`, and later tokens override
earlier ones.

---

## Hand-history review

The parser lives in `training/hand_history.py` and the review in `training/review.py`.
PokerStars and GGPoker share one line-oriented parser, and amounts are converted to big
blinds. Positions are assigned clockwise from the small blind and named the way the 6-max
format names that many players. For example, 5-handed has no UTG and first-in is "HJ".
This keeps the number of players behind correct, which is what an opening range depends on.

**Preflop.** A state machine classifies each hero decision:

| Situation | Spot |
|---|---|
| Unopened pot | RFI |
| Single open | vs open, or BB defence |
| Hero opened and faces a 3-bet | vs 3-bet |
| Hero 3-bet and faces a 4-bet | vs 4-bet (raising = the 5-bet jam) |

The decision is graded exactly like a drill. Everything else is listed as **not scored, with
the reason**: limped and multiway pots, squeezes, 5-bet wars, heads-up or 7+ handed tables,
tournaments, irregular blinds, effective stacks under 40bb, and spots with no loaded
strategy. Off-model sizes and stack depths are graded but annotated.

**Postflop.** A hand is graded only if it went heads-up to a flop that is a **suit
permutation** of a solved spot, with the same pot type and positions:
- The line is walked through the solved tree, with bet sizes mapped to the nearest
  modelled size by pot fraction.
- Hero's hand is relabelled through the permutation.
- The turn must be one of the solver's representative turn cards, up to the flop's suit
  symmetries. River play isn't stored.

With four solved flops this is rare for real hands. The review says so rather than
guessing.

Imported hands are deduplicated by site + hand id. Scored decisions are logged with
`mode="review"`; on an existing database the new `hand_ref` column is added automatically.

## Grading, scores, spaced repetition, leaks

**Mixed-strategy grading** (`training/grading.py`). At equilibrium, every action with
positive frequency has equal EV, so picking a 40% action is *fine*.

| Grade | Rule |
|---|---|
| **best** | top action |
| **good** | ≥ 30% |
| **minor** | 3–30% |
| **mistake** | < 3% |

The thresholds live in `GradingConfig`. Each decision also stores:
- the **frequency error**, `max_f − f(chosen)`, a model-free proxy that also works for
  seed charts
- the **EV loss**, `max EV − EV(chosen)` in bb and % pot, *only* when the source has EVs
  (solver and imports). EVs are never invented for seed charts.

**Spaced repetition** (`training/srs.py`). This is an SM-2 variant over
(spot, node, hand class) items. Items are created on minor and mistake answers.

| Answer | Next review |
|---|---|
| mistake | 2 minutes (comes back this session) |
| minor | ≥ 10 minutes |
| correct | 1 day → 3 days → ×ease |

About 60% of deals are reviews when items are due.

**Leaks** (`training/leaks.py`). Decisions are grouped by (spot, hand category, chosen
action, solution's preferred action). Groups are ranked by summed frequency error (or EV
loss) with small-sample shrinkage `n/(n+2)`, so one misclick ranks below a consistent
pattern. The "close decisions" dealer weights hands by reach × (mixedness + boundary-cell
bonus), so practice concentrates where mistakes actually happen.

---

## Quality

- `make test`: about 130 pytest tests. Coverage:
  - evaluator (exhaustive), equity vs known values and brute force
  - range parsing and formatting round-trip, combo counting with blockers
  - strategy loading, importer
  - grading thresholds, the drill, SRS and leak flows end-to-end through the API
  - Kuhn convergence (exploitability → 0, value −1/18), river-subgame convergence
  - isomorphism exactness, the postflop solve→save→load→view→drill pipeline
- `make lint`: ruff (lint + format), **mypy --strict** clean, TypeScript `strict` +
  `noUncheckedIndexedAccess`.
- Action colours were validated with a colour-vision-deficiency palette checker on the dark
  surface. Grid segments are separated by 2px gaps, and every colour is backed by a label
  or tooltip.

## Roadmap (stretch goals, not built)

- ICM / push-fold charts.
- Anki export of drill items.
- Solver speed-ups: batch river subtrees across boards and Numba-compile the hot loops,
  which would allow more runouts and turn/river raise sizes.
