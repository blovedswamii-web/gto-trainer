import type { ReactNode } from "react";
import type { ActionMeta } from "../lib/actions";
import { pct } from "../lib/actions";
import type { SeatState, TableState } from "../lib/types";
import { Board } from "./Postflop";
import { PlayingCard } from "./ui";

/** Percentage coordinates for `n` seats around the felt; slot 0 is bottom-centre (the hero),
 *  and slots run clockwise in table order. Pure: no poker state involved. */
export function seatLayout(n: number): { x: number; y: number }[] {
  return Array.from({ length: n }, (_, i) => {
    const a = Math.PI / 2 + (2 * Math.PI * i) / n;
    return { x: 50 + 41 * Math.cos(a), y: 46 + 36 * Math.sin(a) };
  });
}

const fmt = (x: number) => `${Number(x.toFixed(2))}`;
const ACTION_WORD: Record<string, string> = {
  raise: "raised",
  "3bet": "3-bet",
  "4bet": "4-bet",
  call: "called",
  check: "checked",
  bet: "bet",
  allin: "all-in",
  post: "posted",
};

/** Same wording as HistoryLine, flattened for screen readers. */
export function describeTable(t: TableState, heroCards?: string[]): string {
  const folded = t.seats.filter((s) => s.folded && s.committed_bb === 0).map((s) => s.position);
  const acts = t.seats
    .filter((s) => !s.folded && s.last_action && s.last_action !== "post")
    .map(
      (s) =>
        `${s.position} ${ACTION_WORD[s.last_action!] ?? s.last_action}` +
        (s.committed_bb > 0 ? ` ${fmt(s.committed_bb)}bb` : ""),
    );
  const hero = t.seats.find((s) => s.is_hero);
  return [
    folded.length ? `${folded.join(", ")} fold.` : "",
    acts.length ? `${acts.join(", ")}.` : "",
    `${hero?.position ?? "Hero"} to act${heroCards ? ` with ${heroCards.join(" ")}` : ""}.`,
    t.board.length ? `Board ${t.board.join(" ")}.` : "",
    `Pot ${fmt(t.pot_bb)}bb.`,
    t.to_call_bb > 0 ? `${fmt(t.to_call_bb)}bb to call.` : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function ChipStack({ bb, label }: { bb: number; label?: string }) {
  const n = Math.min(4, Math.max(1, Math.ceil(Math.log2(bb + 1))));
  return (
    <span className="inline-flex flex-col items-center">
      <span className="relative block h-3.5 w-4">
        {Array.from({ length: n }, (_, i) => (
          <span
            key={i}
            className="absolute left-0 h-2 w-4 rounded-full border"
            style={{
              bottom: i * 2.5,
              background: "var(--color-chip)",
              borderColor: "var(--color-chip-edge)",
            }}
          />
        ))}
      </span>
      <span className="num mt-0.5 rounded bg-surface-0/80 px-1 text-[10px] font-semibold leading-tight text-ink-1">
        {label ?? `${fmt(bb)}`}
      </span>
    </span>
  );
}

function CardBack() {
  return (
    <span
      aria-hidden
      className="block h-6 w-4 rounded-sm border border-surface-3"
      style={{ background: "repeating-linear-gradient(45deg,#2b3a63,#2b3a63 3px,#1f2a4a 3px,#1f2a4a 6px)" }}
    />
  );
}

function Seat({ seat, heroCards, dealer }: { seat: SeatState; heroCards?: string[]; dealer: boolean }) {
  const showHero = seat.is_hero && heroCards && !seat.folded;
  return (
    <div
      className={`flex flex-col items-center gap-0.5 ${seat.folded ? "opacity-40" : ""}`}
      data-seat={seat.position}
    >
      {!seat.folded && (
        <div className="flex gap-0.5">
          {showHero ? (
            heroCards.map((c) => <PlayingCard key={c} card={c} size="sm" />)
          ) : (
            <>
              <CardBack />
              <CardBack />
            </>
          )}
        </div>
      )}
      <div
        className={[
          "num relative min-w-[3.25rem] rounded-md border bg-surface-1 px-1.5 py-0.5 text-center leading-tight",
          seat.is_actor ? "seat-active" : "",
          seat.is_villain ? "border-act-raise" : "border-surface-3",
        ].join(" ")}
      >
        {dealer && (
          // Corner badge: clear of the stack text, the chips and the neighbouring seats.
          <span
            className="absolute -left-2.5 -top-2 flex h-4 w-4 items-center justify-center rounded-full bg-ink-1 text-[9px] font-bold text-surface-0 opacity-100"
            title="Dealer button"
          >
            D
          </span>
        )}
        <div className="text-[11px] font-semibold text-ink-1">
          {seat.position}
          {seat.is_hero && <span className="sr-only"> (hero)</span>}
          {seat.is_villain && <span aria-hidden className="ml-0.5 text-act-raise">▲</span>}
        </div>
        <div className="text-[10px] text-ink-3">
          {seat.folded ? "fold" : seat.stack_bb <= 0 ? "all-in" : `${fmt(seat.stack_bb)}bb`}
        </div>
      </div>
    </div>
  );
}

export interface TableResult {
  /** Action the hero took, as chips: amount committed after acting. */
  chosenBb: number | null;
  chosen: ActionMeta;
  /** Solution mix for the hero's hand. */
  mix: { action: ActionMeta; freq: number }[];
}

export function PokerTable({
  table,
  heroCards,
  children,
  dimmed = false,
  heroBet = null,
  result = null,
  centre,
  canCall = true,
}: {
  table: TableState;
  heroCards?: string[];
  /** Slot under the hero seat (action bar). */
  children?: ReactNode;
  dimmed?: boolean;
  /** Preview: hero's total committed chips if they took the hovered action. */
  heroBet?: number | null;
  result?: TableResult | null;
  /** Extra content beside the pot (compact grade). */
  centre?: ReactNode;
  /** Whether calling is one of the hero's options; pot odds are only shown when it is. */
  canCall?: boolean;
}) {
  const n = table.seats.length;
  const h = Math.max(0, table.seats.findIndex((s) => s.is_hero));
  const layout = seatLayout(n);
  const place = (i: number) => layout[(i - h + n) % n]!;
  const heroAfter = result?.chosenBb ?? heroBet;
  const dealerIdx = table.seats.findIndex((s) => s.position === table.dealer);
  const showPot = table.pot_bb + (heroAfter != null ? Math.max(0, heroAfter - (table.seats[h]?.committed_bb ?? 0)) : 0);

  return (
    <figure className={`m-0 ${dimmed ? "opacity-60" : ""}`}>
      <div
        role="img"
        aria-label={describeTable(table, heroCards)}
        className="relative mx-auto aspect-[11/10] w-full max-w-[760px] sm:aspect-[16/9]"
      >
        <svg viewBox="0 0 160 90" preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden>
          <ellipse cx="80" cy="45" rx="76" ry="41" fill="var(--color-felt-rim)" />
          <ellipse cx="80" cy="45" rx="72" ry="37.5" fill="var(--color-felt)" />
          <ellipse cx="80" cy="45" rx="64" ry="31" fill="none" stroke="#ffffff14" strokeWidth="0.4" />
        </svg>

        {/* centre: board slots, pot, pot odds. Anchored by its top edge so it grows away from the far seat's chips. */}
        <div data-centre className="absolute left-1/2 top-[31%] flex -translate-x-1/2 flex-col items-center gap-1">
          {table.board.length > 0 ? (
            <Board cards={table.board} size="sm" />
          ) : (
            <div className="hidden gap-1 sm:flex" aria-hidden>
              {Array.from({ length: 5 }, (_, i) => (
                <span key={i} className="h-11 w-8 rounded-md border border-dashed border-white/15" />
              ))}
            </div>
          )}
          <div className="num rounded-xl bg-surface-0/70 px-2.5 py-0.5 text-center text-xs text-ink-1">
            Pot <b>{fmt(showPot)}bb</b>
            {canCall && table.to_call_bb > 0 && (
              <span className="block text-ink-2 sm:ml-1.5 sm:inline">
                <span className="hidden sm:inline">· </span>odds {pct(table.to_call_bb / (table.pot_bb + table.to_call_bb), 0)}
              </span>
            )}
          </div>
          {centre}
        </div>

        {table.seats.map((s, i) => {
          const p = place(i);
          const isHero = i === h;
          const committed = isHero && heroAfter != null ? heroAfter : s.committed_bb;
          // The hero's hole cards sit above the seat, so their chips go beside the cards;
          // everyone else's go part-way towards the pot.
          const chipPos = isHero
            ? { left: `calc(${p.x}% - 56px)`, top: `calc(${p.y}% - 14px)` }
            : { left: `${50 + (p.x - 50) * 0.62}%`, top: `${50 + (p.y - 50) * 0.62}%` };
          return (
            <div key={s.position}>
              <div
                className="absolute -translate-x-1/2 -translate-y-1/2"
                style={{ left: `${p.x}%`, top: `${p.y}%` }}
              >
                <Seat seat={s} heroCards={heroCards} dealer={i === dealerIdx} />
              </div>
              {committed > 0 && (
                <div
                  className="chip-move absolute -translate-x-1/2 -translate-y-1/2"
                  style={{ ...chipPos, opacity: isHero && heroBet != null && !result ? 0.7 : 1 }}
                  data-chips={s.position}
                >
                  <ChipStack bb={committed} />
                </div>
              )}
            </div>
          );
        })}

      </div>
      {(children || result) && (
        <figcaption className="mx-auto mt-9 max-w-[760px] sm:mt-6">
          {/* Solution mix sits under the hero seat, off the felt, so it never covers chips or cards. */}
          {result && (
            <div className="mx-auto mb-2 w-56">
              <div className="flex h-2.5 gap-0.5 overflow-hidden rounded" aria-hidden>
                {result.mix
                  .filter((m) => m.freq > 0)
                  .map((m) => (
                    <span key={m.action.id} style={{ width: `${m.freq * 100}%`, background: m.action.color }} />
                  ))}
              </div>
              <div className="num mt-0.5 text-center text-[10px] text-ink-2">
                GTO:{" "}
                {result.mix
                  .filter((m) => m.freq >= 0.01)
                  .map((m) => `${m.action.label.split(" ")[0]} ${pct(m.freq)}`)
                  .join(" · ")}
              </div>
            </div>
          )}
          {children}
        </figcaption>
      )}
    </figure>
  );
}
