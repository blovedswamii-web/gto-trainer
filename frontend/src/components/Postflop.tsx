import { useMemo } from "react";
import { pct, toActionMetas, type ActionMeta } from "../lib/actions";
import type { PfNode, PfNodeBase } from "../lib/types";
import { HandGrid, Legend } from "./HandGrid";
import { Panel, PlayingCard, SourceBadge } from "./ui";

/** One whole split into parts: each part its own block with a 2px gap (no baseline). */
export function SegmentedBar({
  freqs,
  actions,
  height = 14,
}: {
  freqs: Record<string, number>;
  actions: ActionMeta[];
  height?: number;
}) {
  return (
    <div className="flex w-full gap-[2px]" style={{ height }}>
      {actions.map((a) => {
        const f = freqs[a.id] ?? 0;
        if (f < 0.005) return null;
        return (
          <div
            key={a.id}
            title={`${a.label}: ${pct(f, 1)}`}
            className="rounded-[3px]"
            style={{ flexGrow: f, flexBasis: 0, background: a.color }}
          />
        );
      })}
    </div>
  );
}

export function Board({ cards, size = "md" }: { cards: string[]; size?: "sm" | "md" | "lg" }) {
  return (
    <div className="flex gap-1.5">
      {cards.map((c) => (
        <PlayingCard key={c} card={c} size={size} />
      ))}
    </div>
  );
}

export function PfHistory({ node }: { node: PfNodeBase }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-sm">
      <span className="text-ink-3">{node.pot_type} ·</span>
      {node.history.map((h, i) =>
        h.kind === "card" ? (
          <span key={i} className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-xs">
            {h.action}
          </span>
        ) : (
          <span key={i} className="rounded bg-surface-2 px-1.5 py-0.5">
            <span className="font-semibold">{h.position}</span>{" "}
            <span className="text-ink-2">{h.action.toLowerCase()}</span>
          </span>
        ),
      )}
      <span className="rounded bg-act-fold/20 px-1.5 py-0.5 font-semibold">
        {node.actor_position} to act
      </span>
    </div>
  );
}

/** Range-vs-range strategy at a node: overall mix, by hand category, and the 169 grid. */
export function NodeStrategy({
  node,
  highlightClass,
  highlightCategory,
}: {
  node: PfNode;
  highlightClass?: string | null;
  highlightCategory?: string | null;
}) {
  const actions = useMemo(() => toActionMetas(node.actions), [node.actions]);
  return (
    <div className="grid gap-4 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Panel
        title={`${node.actor_position} strategy by hand category`}
        right={<SourceBadge source={node.source} compact />}
      >
        <div className="mb-3">
          <div className="mb-1 flex items-center justify-between text-xs text-ink-3">
            <span>Whole range · {node.range_combos.toFixed(0)} combos</span>
            <Legend actions={actions} shares={node.overall} />
          </div>
          <SegmentedBar freqs={node.overall} actions={actions} height={18} />
        </div>
        <table className="num w-full text-xs">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wider text-ink-3">
              <th className="pb-1 font-normal">Category</th>
              <th className="pb-1 text-right font-normal">Combos</th>
              <th className="w-1/2 pb-1 pl-3 font-normal">Strategy</th>
              <th className="pb-1 text-right font-normal">Best EV</th>
            </tr>
          </thead>
          <tbody>
            {node.categories.map((c) => {
              const best = Object.entries(c.ev).sort((a, b) => b[1] - a[1])[0];
              const hi = c.category === highlightCategory;
              return (
                <tr key={c.category} className={hi ? "bg-warning/10" : ""}>
                  <td className={`py-1 pr-2 ${hi ? "font-semibold text-ink-1" : "text-ink-2"}`}>
                    {hi && "▸ "}
                    {c.category}
                  </td>
                  <td className="text-right text-ink-3">{c.combos.toFixed(1)}</td>
                  <td className="pl-3" title={actions.map((a) => `${a.label} ${pct(c.freqs[a.id] ?? 0)}`).join(" · ")}>
                    <SegmentedBar freqs={c.freqs} actions={actions} />
                  </td>
                  <td className="text-right text-ink-2">
                    {best ? `${best[1] >= 0 ? "+" : ""}${best[1].toFixed(1)}bb` : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-2 text-[11px] text-ink-3">
          EV is in bb relative to the start of the hand's postflop play, each player owning half
          the starting pot. Hover the bars for exact frequencies.
        </p>
      </Panel>
      <Panel title={`${node.actor_position} strategy by starting hand (reach-weighted)`}>
        <div className="mx-auto max-w-[560px]">
          <HandGrid cells={node.grid} actions={actions} highlight={highlightClass} />
          <div className="mt-2">
            <Legend actions={actions} />
          </div>
        </div>
      </Panel>
    </div>
  );
}
