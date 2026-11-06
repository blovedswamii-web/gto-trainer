import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GradeCard, SessionBar, keyToAction } from "../components/Feedback";
import { SpotView } from "../components/SpotView";
import { ErrorNote, HistoryLine, Panel, Pill, PlayingCard, SourceBadge } from "../components/ui";
import { pct, toActionMetas } from "../lib/actions";
import { api } from "../lib/api";
import type { PreflopDeal, PreflopFeedback, Scenario } from "../lib/types";

const SCENARIOS: { id: Scenario; label: string }[] = [
  { id: "rfi", label: "RFI" },
  { id: "vs_open", label: "vs open" },
  { id: "bb_defense", label: "BB defence" },
  { id: "vs_3bet", label: "vs 3-bet" },
  { id: "vs_4bet", label: "vs 4-bet" },
];
const POSITIONS = ["UTG", "HJ", "CO", "BTN", "SB", "BB"];

function toggle<T>(xs: T[], x: T): T[] {
  return xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x];
}

export default function PreflopDrill() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [positions, setPositions] = useState<string[]>([]);
  const [focus, setFocus] = useState<"boundary" | "uniform">("boundary");
  const [useSrs, setUseSrs] = useState(true);
  const [deal, setDeal] = useState<PreflopDeal | null>(null);
  const [fb, setFb] = useState<PreflopFeedback | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState({ n: 0, points: 0, freqErr: 0 });
  const shownAt = useRef(0);

  const next = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const d = await api.post<PreflopDeal>("/drill/preflop/next", {
        scenarios,
        positions,
        focus,
        use_srs: useSrs,
      });
      setDeal(d);
      setFb(null);
      shownAt.current = performance.now();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [scenarios, positions, focus, useSrs]);

  const answer = useCallback(
    async (action: string) => {
      if (!deal || fb || busy) return;
      setBusy(true);
      try {
        const r = await api.post<PreflopFeedback>("/drill/preflop/answer", {
          spot_id: deal.spot.id,
          combo: deal.combo,
          action,
          response_ms: Math.round(performance.now() - shownAt.current),
          reason: deal.reason,
        });
        setFb(r);
        setStats((s) => ({
          n: s.n + 1,
          points: s.points + r.grade.points,
          freqErr: s.freqErr + r.grade.freq_error,
        }));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [deal, fb, busy],
  );

  useEffect(() => {
    void next();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refetch only when filters change
  }, [scenarios, positions, focus]);

  const actions = useMemo(() => (deal ? toActionMetas(deal.actions) : []), [deal]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (fb && (e.key === " " || e.key === "Enter" || e.key.toLowerCase() === "n")) {
        e.preventDefault();
        void next();
        return;
      }
      if (!fb && deal) {
        const a = keyToAction(e.key, actions);
        if (a) void answer(a);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fb, deal, actions, next, answer]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
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
          {POSITIONS.map((p) => (
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
        <div className="ml-auto">
          <SessionBar {...stats} />
        </div>
      </div>
      <ErrorNote error={error} />

      {deal && (
        <div className="grid gap-4 xl:grid-cols-[440px_minmax(0,1fr)]">
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
              <div className="grid grid-cols-3 gap-2">
                {actions.map((a, i) => (
                  <button
                    key={a.id}
                    disabled={!!fb || busy}
                    onClick={() => void answer(a.id)}
                    className="rounded-lg border-2 bg-surface-2 px-3 py-3 text-sm font-semibold transition hover:bg-surface-3 disabled:opacity-50"
                    style={{ borderColor: a.color }}
                  >
                    {a.label}
                    <span className="ml-1.5 text-[10px] font-normal text-ink-3">[{i + 1}]</span>
                  </button>
                ))}
              </div>
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
