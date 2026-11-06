import { useState } from "react";
import { Link } from "react-router-dom";
import { BarList, LineChart } from "../components/Charts";
import { ErrorNote, Panel, Pill, Stat } from "../components/ui";
import { pct } from "../lib/actions";
import { useApi } from "../lib/hooks";

interface Breakdown {
  key: string;
  n: number;
  accuracy: number;
  avg_freq_error: number;
  avg_ev_loss_bb: number | null;
}
interface Summary {
  total: Breakdown;
  grades: Record<string, number>;
  by_position: Breakdown[];
  by_street: Breakdown[];
  by_scenario: Breakdown[];
  by_category: Breakdown[];
  by_mode: Breakdown[];
  srs: { items: number; due: number; learning: number };
}
interface RollPoint {
  i: number;
  t: number;
  freq_error: number;
  accuracy: number;
  ev_loss_pct_pot: number | null;
}
interface Leak {
  mode: string;
  spot_id: string;
  scenario: string;
  hero: string;
  category: string;
  chosen: string;
  best: string;
  errors: number;
  cell_decisions: number;
  avg_freq_error: number;
  ev_loss_bb_total: number | null;
  severity: number;
  hands: string[];
  text: string;
}
interface Recent {
  id: number;
  t: number;
  mode: string;
  spot: string;
  board: string;
  hand: string;
  chosen: string;
  best: string;
  chosen_freq: number;
  grade: string;
  freq_error: number;
  ev_loss_bb: number | null;
  source: string;
}

const MODES = [
  { id: "", label: "All" },
  { id: "preflop", label: "Preflop" },
  { id: "postflop", label: "Postflop" },
  { id: "math", label: "Odds drills" },
  { id: "review", label: "Real hands" },
];
const GRADE_ICON: Record<string, string> = { best: "✓", good: "✓", minor: "⚠", mistake: "✕" };

function bars(rows: Breakdown[] | undefined) {
  return (rows ?? []).map((r) => ({
    key: r.key,
    value: r.accuracy,
    sub: `n=${r.n}`,
    tooltip: `${r.key}: ${pct(r.accuracy)} accuracy over ${r.n} decisions, avg freq error ${r.avg_freq_error.toFixed(2)}`,
  }));
}

export default function Progress() {
  const [mode, setMode] = useState("");
  const [days, setDays] = useState<number | null>(null);
  const q = `mode=${mode}${days ? `&days=${days}` : ""}`;
  const summary = useApi<Summary>(`/stats/summary?${q}`);
  const rolling = useApi<RollPoint[]>(`/stats/rolling?mode=${mode}&window=25`);
  const leaks = useApi<Leak[]>(`/stats/leaks?mode=${mode}&limit=8`);
  const recent = useApi<Recent[]>(`/stats/recent?limit=25`);
  const s = summary.data;
  const ev = (rolling.data ?? []).filter((p) => p.ev_loss_pct_pot !== null);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {MODES.map((m) => (
          <Pill key={m.id} active={mode === m.id} onClick={() => setMode(m.id)}>
            {m.label}
          </Pill>
        ))}
        <span className="mx-2 h-4 w-px bg-surface-3" />
        {[null, 7, 30].map((d) => (
          <Pill key={String(d)} active={days === d} onClick={() => setDays(d)}>
            {d ? `Last ${d} days` : "All time"}
          </Pill>
        ))}
      </div>
      <ErrorNote error={summary.error} />

      {s && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
          <Panel><Stat label="Decisions" value={s.total.n} /></Panel>
          <Panel><Stat label="Accuracy" value={pct(s.total.accuracy as number, 1)} sub="best/good = 1, minor = ½" /></Panel>
          <Panel><Stat label="Avg frequency error" value={(s.total.avg_freq_error as number).toFixed(3)} sub="0 = always the top action" /></Panel>
          <Panel>
            <Stat
              label="Avg EV loss"
              value={s.total.avg_ev_loss_bb !== null ? `${(s.total.avg_ev_loss_bb as number).toFixed(2)}bb` : "—"}
              sub="only where the source has EVs"
            />
          </Panel>
          <Panel><Stat label="Reviews due" value={s.srs.due} sub={`${s.srs.items} items tracked`} /></Panel>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Rolling frequency error (last 25 decisions) — lower is better">
          <LineChart
            ariaLabel="Rolling frequency error"
            points={(rolling.data ?? []).map((p) => ({ x: p.i, y: p.freq_error }))}
            yMax={1}
          />
        </Panel>
        <Panel title="Rolling EV loss, % of pot (solver/imported spots only)">
          <LineChart
            ariaLabel="Rolling EV loss as percent of pot"
            points={ev.map((p) => ({ x: p.i, y: p.ev_loss_pct_pot }))}
            yMax={Math.max(5, ...ev.map((p) => p.ev_loss_pct_pot ?? 0)) * 1.1}
            yFormat={(v) => `${v.toFixed(1)}%`}
          />
        </Panel>
      </div>

      <Panel title="Top leaks">
        {leaks.data && leaks.data.length === 0 && (
          <p className="text-sm text-ink-3">
            No systematic leaks yet — a leak needs at least two errors of the same kind in the same spot.
          </p>
        )}
        <ol className="space-y-2">
          {(leaks.data ?? []).map((l, i) => (
            <li key={i} className="flex flex-wrap items-start justify-between gap-3 rounded-md bg-surface-2 p-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm text-ink-1">{l.text}</div>
                <div className="num mt-1 text-xs text-ink-3">
                  severity {l.severity.toFixed(2)} · avg freq error {l.avg_freq_error.toFixed(2)}
                  {l.ev_loss_bb_total !== null && ` · EV lost ${l.ev_loss_bb_total.toFixed(2)}bb`} · hands:{" "}
                  {l.hands.join(" ")}
                </div>
              </div>
              {l.mode === "preflop" && (
                <div className="flex gap-2 text-xs">
                  <Link className="rounded bg-surface-3 px-2 py-1 hover:text-ink-1" to={`/?spot=${encodeURIComponent(l.spot_id)}&scenario=${l.scenario}`}>
                    View range
                  </Link>
                  <Link className="rounded bg-surface-3 px-2 py-1 hover:text-ink-1" to="/drill">
                    Drill
                  </Link>
                </div>
              )}
            </li>
          ))}
        </ol>
      </Panel>

      <div className="grid gap-4 md:grid-cols-2">
        <Panel title="Accuracy by position"><BarList rows={bars(s?.by_position)} /></Panel>
        <Panel title="Accuracy by street"><BarList rows={bars(s?.by_street)} /></Panel>
        <Panel title="Accuracy by spot type"><BarList rows={bars(s?.by_scenario)} /></Panel>
        <Panel title="Accuracy by hand class"><BarList rows={bars(s?.by_category)} /></Panel>
      </div>

      <Panel title="Recent decisions">
        <div className="max-h-96 overflow-auto">
          <table className="num w-full text-xs">
            <thead className="sticky top-0 bg-surface-1 text-left text-ink-3">
              <tr>
                <th className="py-1 font-normal">Spot</th>
                <th className="font-normal">Hand</th>
                <th className="font-normal">You</th>
                <th className="font-normal">Solution</th>
                <th className="font-normal">Grade</th>
                <th className="text-right font-normal">Freq err</th>
                <th className="text-right font-normal">EV loss</th>
              </tr>
            </thead>
            <tbody>
              {(recent.data ?? []).map((r) => (
                <tr key={r.id} className="border-t border-surface-3">
                  <td className="py-1 text-ink-2">{r.spot}{r.board && <span className="ml-1 text-ink-3">[{r.board}]</span>}</td>
                  <td className="font-mono">{r.hand}</td>
                  <td>{r.chosen} <span className="text-ink-3">({pct(r.chosen_freq)})</span></td>
                  <td>{r.best}</td>
                  <td>{GRADE_ICON[r.grade]} {r.grade}</td>
                  <td className="text-right">{r.freq_error.toFixed(2)}</td>
                  <td className="text-right">{r.ev_loss_bb !== null ? `${r.ev_loss_bb.toFixed(2)}bb` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
