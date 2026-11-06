import { useEffect, useMemo, useRef, useState } from "react";
import { HandGrid } from "../components/HandGrid";
import { ErrorNote, Panel, Pill, SourceBadge, Stat } from "../components/ui";
import { pct } from "../lib/actions";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { CellData, ParsedRange, SourceTag, SpotDetail, SpotSummary } from "../lib/types";

const RANKS = "AKQJT98765432";
const LABELS: string[] = [];
for (let r = 0; r < 13; r++)
  for (let c = 0; c < 13; c++) {
    const a = RANKS[Math.min(r, c)]!;
    const b = RANKS[Math.max(r, c)]!;
    LABELS.push(r === c ? a + b : a + b + (r < c ? "s" : "o"));
  }
const N_COMBOS = LABELS.map((l) => (l.length === 2 ? 6 : l.endsWith("s") ? 4 : 12));

interface CompareOut {
  overlap_combos: number;
  user_combos: number;
  reference_combos: number;
  jaccard: number;
  over_included: [string, number][];
  under_included: [string, number][];
  cell_diff: number[];
  reference_class_weights: number[];
  reference_text: string;
  user_text: string;
  source: SourceTag;
  title: string;
}
interface Saved {
  id: number;
  name: string;
  notation: string;
  class_weights: number[];
}

// Diverging encoding for the difference heatmap: red = you include it more than the
// reference, blue = you are missing it; neutral grey midpoint (validated dataviz pair).
const OVER = "#e66767";
const UNDER = "#3987e5";
const MID = "#383835";
const PAINT = "#199e70";

function diffColor(d: number): string {
  const p = Math.round(Math.min(1, Math.abs(d)) * 100);
  return `color-mix(in oklab, ${d > 0 ? OVER : UNDER} ${p}%, ${MID})`;
}

export default function RangeEditor() {
  const spots = useApi<SpotSummary[]>("/spots?format_id=6max_100bb");
  const [spotId, setSpotId] = useState("6max_100bb.rfi.BTN");
  const detail = useApi<SpotDetail>(`/spots/${encodeURIComponent(spotId)}`);
  const [actions, setActions] = useState<string[]>(["raise"]);
  const [weights, setWeights] = useState<number[]>(() => Array(169).fill(0));
  const [brush, setBrush] = useState(1);
  const [text, setText] = useState("");
  const [cmp, setCmp] = useState<CompareOut | null>(null);
  const [view, setView] = useState<"paint" | "diff" | "reference">("paint");
  const [quiz, setQuiz] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState("");
  const saved = useApi<Saved[]>("/ranges/custom");
  const paintMode = useRef(1);

  const nonFold = useMemo(() => (detail.data?.actions ?? []).filter((a) => a.id !== "fold"), [detail.data]);
  useEffect(() => {
    if (nonFold.length && !actions.every((a) => nonFold.some((n) => n.id === a))) {
      setActions([nonFold[nonFold.length - 1]!.id]);
    }
  }, [nonFold, actions]);
  useEffect(() => setCmp(null), [spotId, actions]);

  const userCombos = weights.reduce((s, w, i) => s + w * N_COMBOS[i]!, 0);

  const paint = (label: string, start: boolean) => {
    const i = LABELS.indexOf(label);
    setWeights((ws) => {
      if (start) paintMode.current = Math.abs(ws[i]! - brush) < 1e-9 ? 0 : brush;
      const next = [...ws];
      next[i] = paintMode.current;
      return next;
    });
    setView("paint");
  };

  const applyText = async () => {
    setErr(null);
    try {
      const r = await api.post<ParsedRange>("/ranges/parse", { text });
      setWeights(r.class_weights);
      setText(r.normalized);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const compare = async () => {
    setErr(null);
    try {
      const r = await api.post<CompareOut>("/ranges/compare", { class_weights: weights, spot_id: spotId, actions });
      setCmp(r);
      setText(r.user_text);
      setView("diff");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const cells: CellData[] = LABELS.map((label, i) => {
    const w = view === "reference" && cmp ? cmp.reference_class_weights[i]! : weights[i]!;
    return { label, combos: w * N_COMBOS[i]!, reach: w, freqs: {} };
  });

  const renderCell = (cell: CellData) => {
    const i = LABELS.indexOf(cell.label);
    if (view === "diff" && cmp) {
      return <div className="absolute inset-0" style={{ background: diffColor(cmp.cell_diff[i]!) }} />;
    }
    const w = cell.reach;
    return w > 0 ? (
      <div
        className="absolute inset-x-0 bottom-0"
        style={{ height: `${w * 100}%`, background: view === "reference" ? "var(--color-act-raise)" : PAINT }}
      />
    ) : null;
  };

  const tooltip = (cell: CellData) => {
    const i = LABELS.indexOf(cell.label);
    return (
      <div className="num text-xs">
        <div className="mb-1 text-sm font-semibold">{cell.label}</div>
        <div>You: {pct(weights[i]!)} ({(weights[i]! * N_COMBOS[i]!).toFixed(1)} combos)</div>
        {cmp && !quiz && <div>Reference: {pct(cmp.reference_class_weights[i]!)}</div>}
        {cmp && <div>Difference: {cmp.cell_diff[i]! > 0 ? "+" : ""}{pct(cmp.cell_diff[i]!)}</div>}
      </div>
    );
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
      <Panel
        title={<span className="text-ink-1">Paint a range{cmp ? ` · compared with ${cmp.title}` : ""}</span>}
        right={
          <div className="flex gap-1">
            <Pill active={view === "paint"} onClick={() => setView("paint")}>Your range</Pill>
            <Pill active={view === "diff"} disabled={!cmp} onClick={() => setView("diff")}>Difference</Pill>
            <Pill active={view === "reference"} disabled={!cmp || quiz} onClick={() => setView("reference")}>Reference</Pill>
          </div>
        }
      >
        <div className="mb-3 flex flex-wrap items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Brush</span>
          {[1, 0.75, 0.5, 0.25, 0].map((b) => (
            <Pill key={b} active={brush === b} onClick={() => setBrush(b)}>
              {b === 0 ? "Erase" : pct(b)}
            </Pill>
          ))}
          <button className="ml-2 text-xs text-ink-3 hover:text-ink-1" onClick={() => setWeights(Array(169).fill(0))}>
            Clear
          </button>
          <span className="num ml-auto text-xs text-ink-3">
            {userCombos.toFixed(1)} combos · {pct(userCombos / 1326, 1)} of hands
          </span>
        </div>
        <div className="mx-auto max-w-[640px]">
          <HandGrid cells={cells} actions={[]} onPaint={paint} renderCell={renderCell} renderTooltip={tooltip} ariaLabel="Range painter" />
          {view === "diff" && (
            <div className="mt-2 flex items-center gap-4 text-xs text-ink-2">
              <span className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-sm" style={{ background: OVER }} />you include more</span>
              <span className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-sm" style={{ background: MID }} />match</span>
              <span className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-sm" style={{ background: UNDER }} />you are missing</span>
            </div>
          )}
        </div>
        <div className="mt-4 flex gap-2">
          <input className="flex-1 rounded-md border border-surface-3 bg-surface-2 px-2.5 py-1.5 font-mono text-xs" value={text} placeholder="Range notation, e.g. 22+,A2s+,KTo+" onChange={(e) => setText(e.target.value)} />
          <button className="rounded-md bg-surface-3 px-3 text-xs" onClick={() => void applyText()}>Apply text</button>
        </div>
        <ErrorNote error={err} />
      </Panel>

      <div className="space-y-4">
        <Panel title="Compare against a chart">
          <select
            className="w-full rounded-md border border-surface-3 bg-surface-2 px-2 py-1.5 text-sm"
            value={spotId}
            onChange={(e) => setSpotId(e.target.value)}
            aria-label="Reference spot"
          >
            {(spots.data ?? []).map((s) => (
              <option key={s.id} value={s.id}>{s.title} ({s.scenario_label})</option>
            ))}
          </select>
          <div className="mt-2 flex flex-wrap items-center gap-1">
            <span className="mr-1 text-xs text-ink-3">Reference =</span>
            {nonFold.map((a) => (
              <Pill key={a.id} active={actions.includes(a.id)} onClick={() => setActions((x) => (x.includes(a.id) ? x.filter((y) => y !== a.id) : [...x, a.id]))}>
                {a.label}
              </Pill>
            ))}
          </div>
          <label className="mt-2 flex items-center gap-1.5 text-xs text-ink-2">
            <input type="checkbox" checked={quiz} onChange={(e) => setQuiz(e.target.checked)} />
            Quiz mode (hide the reference; show only the difference)
          </label>
          {detail.data && <div className="mt-2"><SourceBadge source={detail.data.summary.source} compact /></div>}
          <button onClick={() => void compare()} className="mt-3 w-full rounded-md bg-ink-1 py-2 text-sm font-semibold text-surface-0">
            Compare
          </button>
        </Panel>

        {cmp && (
          <Panel title="Result">
            <div className="grid grid-cols-3 gap-3">
              <Stat label="Match (Jaccard)" value={pct(cmp.jaccard)} />
              <Stat label="Overlap" value={cmp.overlap_combos.toFixed(0)} sub="combos" />
              <Stat label="Size" value={`${cmp.user_combos.toFixed(0)} / ${cmp.reference_combos.toFixed(0)}`} sub="you / ref" />
            </div>
            <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
              <div>
                <div className="mb-1 text-ink-3">Over-included ({cmp.over_included.length})</div>
                <div className="flex flex-wrap gap-1">
                  {cmp.over_included.slice(0, 30).map(([l, d]) => (
                    <span key={l} className="rounded bg-surface-2 px-1 font-mono" title={`+${pct(d)}`}>{l}</span>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-1 text-ink-3">Missing ({cmp.under_included.length})</div>
                <div className="flex flex-wrap gap-1">
                  {cmp.under_included.slice(0, 30).map(([l, d]) => (
                    <span key={l} className="rounded bg-surface-2 px-1 font-mono" title={`-${pct(d)}`}>{l}</span>
                  ))}
                </div>
              </div>
            </div>
            {!quiz && <p className="mt-3 break-words font-mono text-[11px] text-ink-3">Reference: {cmp.reference_text}</p>}
          </Panel>
        )}

        <Panel title="Saved ranges">
          <div className="flex gap-2">
            <input className="flex-1 rounded-md border border-surface-3 bg-surface-2 px-2 py-1 text-sm" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
            <button
              className="rounded-md bg-surface-3 px-3 text-xs disabled:opacity-40"
              disabled={!name || userCombos === 0}
              onClick={() => void api.post("/ranges/custom", { name, class_weights: weights, spot_id: spotId }).then(() => { setName(""); saved.reload(); })}
            >
              Save
            </button>
          </div>
          <ul className="mt-2 space-y-1 text-xs">
            {(saved.data ?? []).map((r) => (
              <li key={r.id} className="flex items-center justify-between gap-2 rounded bg-surface-2 px-2 py-1">
                <button className="truncate text-left hover:text-ink-1" title={r.notation} onClick={() => { setWeights(r.class_weights); setText(r.notation); setView("paint"); }}>
                  {r.name}
                </button>
                <button className="text-ink-3 hover:text-critical" onClick={() => void api.del(`/ranges/custom/${r.id}`).then(() => saved.reload())} aria-label={`Delete ${r.name}`}>
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
