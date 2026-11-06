import { useMemo, useState } from "react";
import { pct, toActionMetas } from "../lib/actions";
import { useApi } from "../lib/hooks";
import type { ComboFreq, SpotDetail } from "../lib/types";
import { HandGrid, Legend } from "./HandGrid";
import { HistoryLine, Panel, SourceBadge, Stat } from "./ui";

/** Grid + stats for one preflop spot. Shared by the range viewer and the drill feedback. */
export function SpotView({
  detail,
  highlight,
  compact = false,
}: {
  detail: SpotDetail;
  highlight?: string | null;
  compact?: boolean;
}) {
  const actions = useMemo(() => toActionMetas(detail.actions), [detail.actions]);
  const [selected, setSelected] = useState<string | null>(null);
  const [table, setTable] = useState(false);
  const cell = useApi<ComboFreq[]>(
    selected ? `/spots/${encodeURIComponent(detail.summary.id)}/cell/${selected}` : null,
  );
  const st = detail.stats;

  return (
    <div className={compact ? "space-y-3" : "grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]"}>
      <Panel
        title={
          <span className="text-base text-ink-1">
            {detail.summary.title}{" "}
            <span className="text-ink-3">· {detail.summary.scenario_label}</span>
          </span>
        }
        right={<SourceBadge source={detail.summary.source} compact={compact} />}
      >
        <div className="mb-3 space-y-1">
          <HistoryLine history={detail.history} hero={detail.summary.hero} />
          <div className="num text-xs text-ink-3">
            Pot {detail.pot_bb}bb · to call {detail.to_call_bb}bb
          </div>
        </div>
        <div className="mx-auto max-w-[640px]">
          {table ? (
            <StrategyTable detail={detail} />
          ) : (
            <HandGrid
              cells={detail.grid}
              actions={actions}
              highlight={highlight}
              selected={selected}
              onCellClick={(l) => setSelected(l === selected ? null : l)}
              ariaLabel={`${detail.summary.title} strategy grid`}
            />
          )}
          <div className="mt-3 flex items-center justify-between gap-2">
            <Legend actions={actions} shares={st.pct_by_action} />
            <button
              className="text-xs text-ink-3 underline-offset-2 hover:text-ink-1 hover:underline"
              onClick={() => setTable((t) => !t)}
            >
              {table ? "Grid view" : "Table view"}
            </button>
          </div>
        </div>
      </Panel>

      <div className="space-y-4">
        <Panel title="Summary">
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Played" value={pct(st.played_pct, 1)} sub="of hands reaching" />
            <Stat label="Of all hands" value={pct(st.vpip_pct_of_all, 1)} sub="non-fold" />
            <Stat label="Reach" value={st.reach_combos.toFixed(0)} sub="combos" />
          </div>
          <table className="num mt-4 w-full text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wider text-ink-3">
                <th className="pb-1 font-normal">Action</th>
                <th className="pb-1 text-right font-normal">Combos</th>
                <th className="pb-1 text-right font-normal">Share</th>
              </tr>
            </thead>
            <tbody>
              {actions.map((a) => (
                <tr key={a.id}>
                  <td className="py-0.5">
                    <span className="mr-2 inline-block size-2.5 rounded-sm" style={{ background: a.color }} />
                    {a.label}
                  </td>
                  <td className="text-right">{(st.combos_by_action[a.id] ?? 0).toFixed(1)}</td>
                  <td className="text-right text-ink-2">{pct(st.pct_by_action[a.id] ?? 0, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {st.value_bluff.map((vb) => (
            <div key={vb.action} className="mt-4 rounded-md bg-surface-2 p-3 text-sm">
              <div className="mb-1 font-medium">
                {actions.find((a) => a.id === vb.action)?.label ?? vb.action}: value vs bluffs
              </div>
              <div className="num text-ink-2">
                {vb.value_combos.toFixed(1)} value : {vb.bluff_combos.toFixed(1)} bluff
                {vb.ratio !== null && <span className="text-ink-1"> = {vb.ratio.toFixed(2)} : 1</span>}
              </div>
              <div className="mt-1 text-[11px] text-ink-3">{vb.method}</div>
            </div>
          ))}
        </Panel>

        {selected && (
          <Panel title={`${selected} by combo`} right={<button className="text-xs text-ink-3" onClick={() => setSelected(null)}>close</button>}>
            {cell.data ? (
              <table className="num w-full text-xs">
                <tbody>
                  {cell.data.map((c) => (
                    <tr key={c.combo} className={c.reach > 0 ? "" : "text-ink-3"}>
                      <td className="py-0.5 font-mono">{c.combo}</td>
                      <td className="text-right text-ink-3">reach {pct(c.reach)}</td>
                      {actions.map((a) => (
                        <td key={a.id} className="pl-2 text-right">
                          {pct(c.freqs[a.id] ?? 0)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-xs text-ink-3">loading…</div>
            )}
          </Panel>
        )}

        {!compact && (detail.notes || detail.villain_range_text) && (
          <Panel title="Notes">
            {detail.notes && <p className="whitespace-pre-line text-xs text-ink-2">{detail.notes}</p>}
            {detail.villain_range_text && (
              <details className="mt-2 text-xs">
                <summary className="cursor-pointer text-ink-2">Villain range at this node</summary>
                <p className="mt-1 break-words font-mono text-ink-3">{detail.villain_range_text}</p>
              </details>
            )}
          </Panel>
        )}
      </div>
    </div>
  );
}

function StrategyTable({ detail }: { detail: SpotDetail }) {
  const actions = toActionMetas(detail.actions);
  const rows = detail.grid.filter((c) => c.combos > 0);
  return (
    <div className="max-h-[560px] overflow-auto rounded-md border border-surface-3">
      <table className="num w-full text-xs">
        <thead className="sticky top-0 bg-surface-2">
          <tr>
            <th className="px-2 py-1 text-left">Hand</th>
            <th className="px-2 py-1 text-right">Combos</th>
            {actions.map((a) => (
              <th key={a.id} className="px-2 py-1 text-right">{a.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.label} className="border-t border-surface-3">
              <td className="px-2 py-0.5 font-semibold">{c.label}</td>
              <td className="px-2 text-right text-ink-3">{c.combos.toFixed(1)}</td>
              {actions.map((a) => (
                <td key={a.id} className="px-2 text-right">{pct(c.freqs[a.id] ?? 0)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
