import { useCallback, useMemo, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { GradeCard, SessionBar } from "../components/Feedback";
import { PokerTable, type TableResult } from "../components/PokerTable";
import { HandGrid, Legend, totalCombos } from "../components/HandGrid";
import { ErrorNote, Panel, Pill, Stat } from "../components/ui";
import { type ActionMeta, pct } from "../lib/actions";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { CellData, GradeOut, TableState } from "../lib/types";
import { useDrill } from "../lib/useDrill";

interface Preset {
  id: string;
  label: string;
  stacks: number[];
  payouts: number[];
  ante: number;
}
interface RangeOut {
  position?: string;
  shover?: string;
  caller?: string;
  freqs: number[];
  margin: number[];
  pct: number;
}
interface SolveOut {
  key: string;
  labels: string[];
  positions: string[];
  stacks: number[];
  chip_pct: number[];
  equity: number[];
  icm: boolean;
  units: string;
  iterations: number;
  exploitability: number;
  warnings: string[];
  push: RangeOut[];
  call: RangeOut[];
  bubble_factors: (number | null)[][];
}
interface Deal {
  kind: "push" | "call";
  shover: number;
  caller: number | null;
  hero: number;
  positions: string[];
  stacks: number[];
  hand: string;
  hand_class: string;
  actions: string[];
}
interface Answer {
  grade: GradeOut;
  freqs: Record<string, number>;
  margin: number;
  units: string;
  ev_loss: number | null;
}

const SHOVE: ActionMeta = { id: "push", label: "Shove", color: "var(--color-act-allin)" };
const CALL: ActionMeta = { id: "call", label: "Call", color: "var(--color-act-call)" };
const FOLD: ActionMeta = { id: "fold", label: "Fold", color: "var(--color-act-fold)" };
const LABEL: Record<string, string> = { push: "Shove", call: "Call", fold: "Fold" };
const input = "w-full rounded-md border border-surface-3 bg-surface-2 px-2 py-1 text-sm num";

function fmtUnits(x: number, units: string) {
  return units === "bb" ? `${x >= 0 ? "+" : ""}${x.toFixed(2)}bb` : `${x >= 0 ? "+" : ""}${x.toFixed(2)}% of prize pool`;
}

function cells(labels: string[], r: RangeOut, act: string): CellData[] {
  return labels.map((label, k) => ({
    label,
    combos: totalCombos(label),
    reach: 1,
    freqs: { [act]: r.freqs[k]!, fold: 1 - r.freqs[k]! },
    ev: { [act]: r.margin[k]!, fold: 0 },
  }));
}

/** Table for a push/fold deal, built from data the deal already carries. */
function pushFoldTable(positions: string[], stacks: number[], ante: number, deal: Deal): TableState {
  const n = positions.length;
  const blind = (i: number) => (positions[i] === "BB" ? 1 : positions[i] === "SB" || (n === 2 && positions[i] === "BTN") ? 0.5 : 0);
  const shover = deal.kind === "push" ? deal.hero : deal.shover;
  const committed = positions.map((_, i) => (i === shover && deal.kind === "call" ? stacks[i]! : blind(i)));
  const toCall =
    deal.kind === "call"
      ? Math.min(stacks[deal.hero]!, stacks[shover]!) - blind(deal.hero)
      : Math.max(0, 1 - blind(deal.hero));
  return {
    seats: positions.map((p, i) => ({
      position: p,
      stack_bb: Math.max(0, stacks[i]! - committed[i]! - (i === shover && deal.kind === "call" ? 0 : ante)),
      committed_bb: committed[i]!,
      folded: i < deal.hero && i !== shover,
      is_hero: i === deal.hero,
      is_actor: i === deal.hero,
      is_villain: deal.kind === "call" && i === shover,
      last_action: deal.kind === "call" && i === shover ? "allin" : null,
    })),
    dealer: "BTN",
    pot_bb: committed.reduce((x, y) => x + y, 0) + ante * n,
    to_call_bb: toCall,
    board: [],
  };
}

function Drill({ setup, positions }: { setup: { stacks: number[]; payouts: number[]; ante: number }; positions: string[] }) {
  const [preview, setPreview] = useState<number | null>(null);
  const fetchNext = useCallback(() => api.post<Deal>("/pushfold/drill/next", setup), [setup]);
  const submit = useCallback(
    (deal: Deal, action: string) =>
      api.post<Answer>("/pushfold/drill/answer", {
        ...setup, kind: deal.kind, shover: deal.shover, caller: deal.caller, hand: deal.hand, hand_class: deal.hand_class, action,
      }),
    [setup],
  );
  const { deal, fb: ans, error: err, stats, actions: metas, next, answer } = useDrill<Deal, Answer>({
    fetchNext,
    submit,
    actionsOf: useCallback((d: Deal) => [d.kind === "push" ? SHOVE : CALL, FOLD], []),
    refetchOn: [setup],
  });
  const heroStack = deal ? deal.stacks[deal.hero]! : 0;
  const sizeOf = (id: string) =>
    !deal || id === "fold" ? null : id === "push" ? heroStack : Math.min(heroStack, deal.stacks[deal.shover]!);
  const result: TableResult | null =
    deal && ans
      ? {
          chosen: metas.find((a) => a.id === ans.grade.chosen) ?? metas[0]!,
          chosenBb: sizeOf(ans.grade.chosen),
          mix: metas.map((a) => ({ action: a, freq: ans.freqs[a.id] ?? 0 })),
        }
      : null;
  return (
    <Panel title="Drill this table" right={<SessionBar {...stats} />}>
      <ErrorNote error={err} />
      {deal && (
        <>
          <div className="mb-3 text-sm">
            {deal.kind === "push" ? (
              <>Folded to you in the <b>{positions[deal.hero]}</b>.</>
            ) : (
              <>
                <b>{positions[deal.shover]}</b> shoves {deal.stacks[deal.shover]}bb
                {deal.caller !== null && deal.caller - deal.shover > 1 && ", folded to you"}. You are in the{" "}
                <b>{positions[deal.hero]}</b>.
              </>
            )}{" "}
            <span className="text-ink-3">Your stack {deal.stacks[deal.hero]}bb.</span>
          </div>
          <div className="mb-3">
            <PokerTable
              table={pushFoldTable(positions, deal.stacks, setup.ante, deal)}
              heroCards={[deal.hand.slice(0, 2), deal.hand.slice(2, 4)]}
              heroBet={ans ? null : preview}
              result={result}
              canCall={metas.some((a) => a.id === "call")}
              centre={ans && <GradeCard compact grade={ans.grade} freqs={ans.freqs} actions={metas} />}
            >
              <ActionBar
                actions={metas}
                columns={2}
                disabled={!!ans}
                onPick={(id) => void answer(id)}
                sizeOf={sizeOf}
                onPreview={setPreview}
              />
            </PokerTable>
          </div>
          {ans && (
            <div className="space-y-2">
              <GradeCard
                grade={ans.grade}
                freqs={ans.freqs}
                actions={metas}
                extra={
                  <>
                    {deal.hand_class}: {LABEL[deal.kind]} − fold = {fmtUnits(ans.margin, ans.units)}
                    {ans.units !== "bb" && ans.ev_loss !== null && ans.ev_loss > 0 && (
                      <> · ICM loss {ans.ev_loss.toFixed(2)}% of the prize pool</>
                    )}
                  </>
                }
              />
              <button onClick={() => void next()} className="w-full rounded-lg bg-ink-1 py-2 text-sm font-semibold text-surface-0">
                Next hand
              </button>
            </div>
          )}
        </>
      )}
    </Panel>
  );
}

export default function PushFold() {
  const presets = useApi<Preset[]>("/pushfold/presets");
  const [stacks, setStacks] = useState<number[]>([12, 8, 15, 10]);
  const [payoutText, setPayoutText] = useState("50, 30, 20");
  const [chipEv, setChipEv] = useState(false);
  const [ante, setAnte] = useState(0);
  const [res, setRes] = useState<SolveOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<"push" | "call">("push");
  const [shover, setShover] = useState(0);
  const [caller, setCaller] = useState(1);

  const payouts = useMemo(
    () => (chipEv ? [] : payoutText.split(/[,\s]+/).filter(Boolean).map(Number).filter((x) => !Number.isNaN(x))),
    [chipEv, payoutText],
  );
  const setup = useMemo(() => ({ stacks, payouts, ante }), [stacks, payouts, ante]);
  const positionsFor = (n: number) =>
    ({
      2: ["BTN", "BB"], 3: ["BTN", "SB", "BB"], 4: ["CO", "BTN", "SB", "BB"], 5: ["HJ", "CO", "BTN", "SB", "BB"],
      6: ["UTG", "HJ", "CO", "BTN", "SB", "BB"], 7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
      8: ["UTG", "UTG1", "LJ", "HJ", "CO", "BTN", "SB", "BB"], 9: ["UTG", "UTG1", "UTG2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    })[n] ?? [];

  const solve = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.post<SolveOut>("/pushfold/solve", setup);
      setRes(r);
      setShover(0);
      setCaller(1);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const loadPreset = (p: Preset) => {
    setStacks(p.stacks);
    setChipEv(p.payouts.length === 0);
    if (p.payouts.length) setPayoutText(p.payouts.join(", "));
    setAnte(p.ante);
    setRes(null);
  };

  const pos = positionsFor(stacks.length);
  const callRange = res?.call.find((c) => c.shover === res.positions[shover] && c.caller === res.positions[caller]);
  const shown = res ? (tab === "push" ? res.push[shover] : callRange) : undefined;
  const act = tab === "push" ? SHOVE : CALL;
  const tooltip = (cell: CellData) => (
    <div className="num text-xs">
      <div className="mb-1 text-sm font-semibold">{cell.label}</div>
      <div>{act.label} {pct(cell.freqs[act.id] ?? 0)}</div>
      {res && <div className="text-ink-2">EV({act.label.toLowerCase()}) − EV(fold): {fmtUnits(cell.ev?.[act.id] ?? 0, res.units)}</div>}
    </div>
  );

  return (
    <div className="space-y-4">
      <Panel title="Table setup (stacks in big blinds, preflop action order)">
        <div className="mb-3 flex flex-wrap items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Presets</span>
          {(presets.data ?? []).map((p) => (
            <Pill key={p.id} onClick={() => loadPreset(p)}>{p.label}</Pill>
          ))}
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-ink-3">
            Players
            <select
              className={input}
              value={stacks.length}
              onChange={(e) => {
                const n = Number(e.target.value);
                setStacks((s) => Array.from({ length: n }, (_, i) => s[i] ?? 10));
                setRes(null);
              }}
            >
              {[2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>
          {stacks.map((s, i) => (
            <label key={i} className="w-16 text-xs text-ink-3">
              {pos[i]}
              <input
                type="number"
                min={2}
                step={0.5}
                className={input}
                value={s}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setStacks((xs) => xs.map((x, j) => (j === i ? v : x)));
                  setRes(null);
                }}
              />
            </label>
          ))}
          <label className="w-20 text-xs text-ink-3">
            Ante (bb)
            <input type="number" min={0} step={0.025} className={input} value={ante} onChange={(e) => { setAnte(Number(e.target.value)); setRes(null); }} />
          </label>
          <label className="w-48 text-xs text-ink-3">
            Payouts (% per place)
            <input className={input} value={payoutText} disabled={chipEv} onChange={(e) => { setPayoutText(e.target.value); setRes(null); }} />
          </label>
          <label className="flex items-center gap-1.5 pb-1.5 text-xs text-ink-2">
            <input type="checkbox" checked={chipEv} onChange={(e) => { setChipEv(e.target.checked); setRes(null); }} />
            Chip EV (no ICM)
          </label>
          <button
            onClick={() => void solve()}
            disabled={busy}
            className="ml-auto rounded-md bg-ink-1 px-4 py-2 text-sm font-semibold text-surface-0 disabled:opacity-50"
          >
            {busy ? "Solving…" : "Solve push/fold"}
          </button>
        </div>
        <p className="mt-2 text-[11px] text-ink-3">
          Model: first-in players shove or fold; facing a shove, later players call or fold; one caller at most (no overcalls).
          Payoffs are ICM equity (Malmuth–Harville) of the end-of-hand stacks, or chips. Equities come from the 169×169 preflop
          table (±~1%). Solved by fictitious play.
        </p>
        <ErrorNote error={err} />
      </Panel>

      {res && (
        <>
          {res.warnings.map((w) => (
            <div key={w} className="rounded-md border border-warning/50 bg-warning/10 px-3 py-2 text-sm">⚠ {w}</div>
          ))}
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title={res.icm ? "Stacks and ICM equity" : "Stacks (chip EV)"}
              right={<span className="num text-xs text-ink-3">{res.iterations} iterations · exploitability {res.exploitability.toFixed(4)} {res.units}</span>}
            >
              <table className="num w-full text-sm">
                <thead className="text-left text-[11px] uppercase tracking-wider text-ink-3">
                  <tr>
                    <th className="pb-1 font-normal">Pos</th>
                    <th className="pb-1 text-right font-normal">Stack</th>
                    <th className="pb-1 text-right font-normal">Chips</th>
                    {res.icm && <th className="pb-1 text-right font-normal">ICM equity</th>}
                    <th className="pb-1 text-right font-normal">Shoves</th>
                  </tr>
                </thead>
                <tbody>
                  {res.positions.map((p, i) => (
                    <tr key={p} className="border-t border-surface-3">
                      <td className="py-1 font-semibold">{p}</td>
                      <td className="text-right">{res.stacks[i]}bb</td>
                      <td className="text-right text-ink-2">{res.chip_pct[i]!.toFixed(1)}%</td>
                      {res.icm && <td className="text-right">{res.equity[i]!.toFixed(1)}%</td>}
                      <td className="text-right">{i < res.push.length ? pct(res.push[i]!.pct, 1) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {res.icm && (
                <div className="mt-4">
                  <div className="mb-1 text-xs text-ink-3">
                    Bubble factors (row = you, column = opponent): equity lost by losing a full stack-off ÷ equity won by winning
                    it. Above 1, you need more than pot odds to call.
                  </div>
                  <table className="num w-full text-xs">
                    <thead>
                      <tr>
                        <th />
                        {res.positions.map((p) => (
                          <th key={p} className="px-1 text-right font-normal text-ink-3">{p}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {res.bubble_factors.map((row, i) => (
                        <tr key={i}>
                          <td className="pr-2 font-semibold">{res.positions[i]}</td>
                          {row.map((b, j) => (
                            <td key={j} className={`px-1 text-right ${b !== null && b > 1.5 ? "text-warning" : "text-ink-2"}`}>
                              {b === null ? "·" : b.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Panel>
            <Drill key={res.key} setup={setup} positions={res.positions} />
          </div>

          <Panel
            title={tab === "push" ? `${res.positions[shover]} shoving range when folded to` : `${res.positions[caller]} calling range vs ${res.positions[shover]} shove`}
            right={
              <div className="flex gap-1">
                <Pill active={tab === "push"} onClick={() => setTab("push")}>Shove charts</Pill>
                <Pill active={tab === "call"} onClick={() => { setTab("call"); setCaller((c) => Math.max(c, shover + 1)); }}>Call charts</Pill>
              </div>
            }
          >
            <div className="mb-3 flex flex-wrap items-center gap-1 text-xs">
              <span className="mr-1 text-ink-3">{tab === "push" ? "Shover" : "Shover"}</span>
              {res.push.map((p, i) => (
                <Pill key={p.position} active={shover === i} onClick={() => { setShover(i); setCaller((c) => (c <= i ? i + 1 : c)); }}>
                  {p.position} · {pct(p.pct)}
                </Pill>
              ))}
              {tab === "call" && (
                <>
                  <span className="ml-3 mr-1 text-ink-3">Caller</span>
                  {res.positions.map((p, j) =>
                    j > shover ? (
                      <Pill key={p} active={caller === j} onClick={() => setCaller(j)}>
                        {p} · {pct(res.call.find((c) => c.shover === res.positions[shover] && c.caller === p)?.pct ?? 0)}
                      </Pill>
                    ) : null,
                  )}
                </>
              )}
            </div>
            {shown && (
              <div className="mx-auto max-w-[600px]">
                <HandGrid cells={cells(res.labels, shown, act.id)} actions={[FOLD, act]} renderTooltip={tooltip} />
                <div className="mt-2 flex items-center justify-between text-xs text-ink-3">
                  <Legend actions={[FOLD, act]} />
                  <span>{pct(shown.pct, 1)} of hands · hover for the EV margin</span>
                </div>
              </div>
            )}
          </Panel>
          {res.icm && (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <Panel><Stat label="Players" value={res.positions.length} /></Panel>
              <Panel><Stat label="Paid places" value={payouts.filter((p) => p > 0).length} /></Panel>
              <Panel><Stat label="Biggest stack" value={`${Math.max(...res.stacks)}bb`} /></Panel>
              <Panel><Stat label="Shortest stack" value={`${Math.min(...res.stacks)}bb`} /></Panel>
            </div>
          )}
        </>
      )}
    </div>
  );
}
