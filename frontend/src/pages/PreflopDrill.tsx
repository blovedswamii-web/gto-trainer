import { useCallback, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { GradeCard, SessionBar } from "../components/Feedback";
import { PokerTable, type TableResult } from "../components/PokerTable";
import { SpotView } from "../components/SpotView";
import { ErrorNote, FilterDisclosure, HistoryLine, Panel, Pill, PlayingCard, SourceBadge } from "../components/ui";
import { pct, toActionMetas } from "../lib/actions";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { FormatOut, PreflopDeal, PreflopFeedback, Scenario } from "../lib/types";
import { useDrill } from "../lib/useDrill";

const SCENARIOS: { id: Scenario; label: string }[] = [
  { id: "rfi", label: "RFI" },
  { id: "vs_open", label: "vs open" },
  { id: "bb_defense", label: "BB defence" },
  { id: "vs_3bet", label: "vs 3-bet" },
  { id: "vs_4bet", label: "vs 4-bet" },
];
const FALLBACK_POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];
const DRILL_FORMAT = "6max_100bb";

function toggle<T>(xs: T[], x: T): T[] {
  return xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x];
}

export default function PreflopDrill() {
  const formats = useApi<FormatOut[]>("/formats");
  const positionList =
    formats.data?.find((f) => f.format.id === DRILL_FORMAT)?.positions ?? FALLBACK_POSITIONS;
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [positions, setPositions] = useState<string[]>([]);
  const [focus, setFocus] = useState<"boundary" | "uniform">("boundary");
  const [useSrs, setUseSrs] = useState(true);
  const [preview, setPreview] = useState<number | null>(null);

  const fetchNext = useCallback(
    () =>
      api.post<PreflopDeal>("/drill/preflop/next", {
        scenarios,
        positions,
        focus,
        use_srs: useSrs,
      }),
    [scenarios, positions, focus, useSrs],
  );
  const submit = useCallback(
    (deal: PreflopDeal, action: string, responseMs: number) =>
      api.post<PreflopFeedback>("/drill/preflop/answer", {
        spot_id: deal.spot.id,
        combo: deal.combo,
        action,
        response_ms: responseMs,
        reason: deal.reason,
      }),
    [],
  );
  const { deal, fb, busy, error, stats, actions, next, answer } = useDrill<PreflopDeal, PreflopFeedback>({
    fetchNext,
    submit,
    actionsOf: useCallback((d: PreflopDeal) => toActionMetas(d.actions), []),
    refetchOn: [scenarios, positions, focus], // not useSrs: toggling it applies from the next deal
  });

  const sizeOf = (id: string) =>
    id === "fold" ? null : (deal?.actions.find((a) => a.id === id)?.size_bb ?? null);
  const result: TableResult | null =
    deal && fb
      ? {
          chosen: actions.find((a) => a.id === fb.grade.chosen) ?? actions[0]!,
          chosenBb: sizeOf(fb.grade.chosen),
          mix: actions.map((a) => ({ action: a, freq: fb.combo_freqs[a.id] ?? 0 })),
        }
      : null;

  return (
    <div className="space-y-4">
      <FilterDisclosure right={<SessionBar {...stats} />}>
        <div className="flex flex-wrap items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Spots</span>
          {SCENARIOS.map((s) => (
            <Pill key={s.id} active={scenarios.includes(s.id)} onClick={() => setScenarios((x) => toggle(x, s.id))}>
              {s.label}
            </Pill>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Hero</span>
          {positionList.map((p) => (
            <Pill key={p} active={positions.includes(p)} onClick={() => setPositions((x) => toggle(x, p))}>
              {p}
            </Pill>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Hands</span>
          <Pill active={focus === "boundary"} onClick={() => setFocus("boundary")}>
            Close decisions
          </Pill>
          <Pill active={focus === "uniform"} onClick={() => setFocus("uniform")}>
            Realistic
          </Pill>
        </div>
        <label className="flex items-center gap-1.5 text-xs text-ink-2">
          <input type="checkbox" checked={useSrs} onChange={(e) => setUseSrs(e.target.checked)} />
          Review mistakes (spaced repetition)
        </label>
      </FilterDisclosure>
      <ErrorNote error={error} />

      {deal && (
        <div className="grid gap-4 xl:grid-cols-[560px_minmax(0,1fr)]">
          <div className="space-y-4">
            <Panel
              title={
                <span className="text-ink-1">
                  {deal.spot.title}
                  {deal.reason === "review" && (
                    <span className="ml-2 rounded bg-warning/15 px-1.5 py-0.5 text-[11px] text-warning">
                      ↻ review
                    </span>
                  )}
                </span>
              }
              right={<SourceBadge source={deal.spot.source} compact />}
            >
              <HistoryLine history={deal.history} hero={deal.spot.hero} />
              {deal.table ? (
                <div className="mt-3">
                  <PokerTable
                    table={deal.table}
                    heroCards={deal.hero_cards}
                    heroBet={fb ? null : preview}
                    result={result}
                    canCall={actions.some((a) => a.id === "call")}
                    centre={fb && <GradeCard compact grade={fb.grade} freqs={fb.combo_freqs} actions={actions} />}
                  >
                    <ActionBar
                      actions={actions}
                      disabled={!!fb || busy}
                      onPick={(id) => void answer(id)}
                      sizeOf={sizeOf}
                      onPreview={setPreview}
                    />
                  </PokerTable>
                </div>
              ) : (
                <>
                  <div className="my-6 flex items-center justify-center gap-6">
                    <div className="flex gap-2">
                      {deal.hero_cards.map((c) => (
                        <PlayingCard key={c} card={c} size="lg" />
                      ))}
                    </div>
                    <div className="num text-sm text-ink-2">
                      <div>
                        Pot <span className="text-ink-1">{deal.pot_bb}bb</span>
                      </div>
                      <div>
                        To call <span className="text-ink-1">{deal.to_call_bb}bb</span>
                      </div>
                      <div className="text-ink-3">100bb effective</div>
                    </div>
                  </div>
                  <ActionBar actions={actions} disabled={!!fb || busy} onPick={(id) => void answer(id)} />
                </>
              )}
            </Panel>

            {fb && (
              <>
                <GradeCard
                  grade={fb.grade}
                  freqs={fb.combo_freqs}
                  actions={actions}
                  extra={
                    fb.equity_vs_villain !== null && (
                      <>
                        {deal.hand_class} has {pct(fb.equity_vs_villain, 1)} all-in equity vs{" "}
                        {deal.spot.villain}'s range here (preflop table, ±1%).
                      </>
                    )
                  }
                />
                <button
                  onClick={() => void next()}
                  className="w-full rounded-lg bg-ink-1 py-2.5 text-sm font-semibold text-surface-0 hover:bg-ink-2"
                >
                  Next hand <span className="font-normal">(Space)</span>
                </button>
              </>
            )}
          </div>

          <div>
            {fb ? (
              <SpotView detail={fb.detail} highlight={deal.hand_class} compact />
            ) : (
              <Panel>
                <div className="flex h-64 items-center justify-center text-center text-sm text-ink-3">
                  Choose an action — the full range for this spot appears after you answer.
                  <br />
                  Keys: 1-3 or F / C / R / A.
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
