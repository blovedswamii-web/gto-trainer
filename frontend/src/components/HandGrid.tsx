import { useCallback, useRef, useState, type ReactNode } from "react";
import type { ActionMeta } from "../lib/actions";
import { pct } from "../lib/actions";
import type { CellData } from "../lib/types";

/**
 * The 13x13 starting-hand grid.
 *
 * Strategy mode (default): each cell is a split-colour bar. Segment *widths* are the action
 * frequencies (reach-weighted); the bar *height* is the fraction of that cell's combos that
 * reach the node, so a hand that is opened 50% of the time in the previous spot shows as a
 * half-height bar. Segments are separated by 2px surface gaps so adjacent colours never
 * touch. A custom `renderCell` turns the same grid into a weight painter or diff heatmap.
 */
export interface HandGridProps {
  cells: CellData[];
  actions: ActionMeta[];
  highlight?: string | null;
  selected?: string | null;
  onCellClick?: (label: string) => void;
  onPaint?: (label: string, start: boolean) => void;
  renderCell?: (cell: CellData) => ReactNode;
  renderTooltip?: (cell: CellData) => ReactNode;
  ariaLabel?: string;
}

const TOTAL_COMBOS: Record<string, number> = { pair: 6, suited: 4, offsuit: 12 };

export function totalCombos(label: string): number {
  if (label.length === 2) return TOTAL_COMBOS.pair!;
  return label.endsWith("s") ? TOTAL_COMBOS.suited! : TOTAL_COMBOS.offsuit!;
}

export function StrategyTooltip({ cell, actions }: { cell: CellData; actions: ActionMeta[] }) {
  const total = totalCombos(cell.label);
  return (
    <div className="min-w-44">
      <div className="mb-1 flex items-baseline justify-between gap-4">
        <span className="text-sm font-semibold">{cell.label}</span>
        <span className="num text-xs text-ink-3">
          {cell.combos.toFixed(cell.combos % 1 ? 2 : 0)} / {total} combos
        </span>
      </div>
      {cell.combos <= 0 ? (
        <div className="text-xs text-ink-3">Not in range at this node</div>
      ) : (
        <table className="num w-full text-xs">
          <tbody>
            {actions.map((a) => {
              const f = cell.freqs[a.id] ?? 0;
              return (
                <tr key={a.id} className={f < 0.005 ? "text-ink-3" : "text-ink-1"}>
                  <td className="pr-2">
                    <span
                      className="mr-1.5 inline-block size-2.5 rounded-sm align-middle"
                      style={{ background: a.color }}
                    />
                    {a.label}
                  </td>
                  <td className="pr-2 text-right">{pct(f, 1)}</td>
                  <td className="text-right text-ink-3">{(f * cell.combos).toFixed(2)}c</td>
                  {cell.ev && (
                    <td className="pl-2 text-right text-ink-2">
                      {(cell.ev[a.id] ?? 0) >= 0 ? "+" : ""}
                      {(cell.ev[a.id] ?? 0).toFixed(2)}bb
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function StrategyBar({ cell, actions }: { cell: CellData; actions: ActionMeta[] }) {
  if (cell.combos <= 0) return null;
  const h = Math.max(0, Math.min(1, cell.reach));
  return (
    <div className="absolute inset-x-0 bottom-0 flex gap-[2px]" style={{ height: `${h * 100}%` }}>
      {actions.map((a) => {
        const f = cell.freqs[a.id] ?? 0;
        if (f < 0.005) return null;
        return <div key={a.id} style={{ flexGrow: f, flexBasis: 0, background: a.color }} />;
      })}
    </div>
  );
}

export function HandGrid({
  cells,
  actions,
  highlight,
  selected,
  onCellClick,
  onPaint,
  renderCell,
  renderTooltip,
  ariaLabel = "Starting hand grid",
}: HandGridProps) {
  const [hover, setHover] = useState<{ cell: CellData; x: number; y: number } | null>(null);
  const painting = useRef(false);

  const move = useCallback((cell: CellData, e: React.MouseEvent) => {
    setHover({ cell, x: e.clientX, y: e.clientY });
  }, []);

  return (
    <div
      className="relative w-full select-none"
      onMouseLeave={() => {
        setHover(null);
        painting.current = false;
      }}
      onMouseUp={() => (painting.current = false)}
    >
      <div
        role="grid"
        aria-label={ariaLabel}
        className="grid w-full gap-[2px]"
        style={{ gridTemplateColumns: "repeat(13, minmax(0, 1fr))" }}
      >
        {cells.map((cell) => {
          const isHi = highlight === cell.label;
          const isSel = selected === cell.label;
          const freqText = Object.entries(cell.freqs)
            .filter(([, f]) => f > 0.005)
            .map(([a, f]) => `${a} ${pct(f)}`)
            .join(", ");
          return (
            <div
              key={cell.label}
              role="gridcell"
              aria-label={`${cell.label}: ${cell.combos > 0 ? freqText || "no data" : "not in range"}`}
              className={[
                "relative aspect-square overflow-hidden rounded-[3px] bg-surface-2",
                onCellClick || onPaint ? "cursor-pointer" : "",
                isHi ? "ring-2 ring-warning ring-offset-1 ring-offset-surface-1 z-10" : "",
                isSel ? "ring-2 ring-ink-1 z-10" : "hover:ring-1 hover:ring-ink-2",
              ].join(" ")}
              onMouseEnter={(e) => {
                move(cell, e);
                if (painting.current && onPaint) onPaint(cell.label, false);
              }}
              onMouseMove={(e) => move(cell, e)}
              onMouseDown={(e) => {
                if (onPaint) {
                  e.preventDefault();
                  painting.current = true;
                  onPaint(cell.label, true);
                }
              }}
              onClick={() => onCellClick?.(cell.label)}
            >
              {renderCell ? renderCell(cell) : <StrategyBar cell={cell} actions={actions} />}
              <span
                className={[
                  "pointer-events-none absolute left-[3px] top-[1px] font-semibold leading-tight",
                  cell.combos > 0 || renderCell ? "text-ink-1" : "text-ink-3",
                ].join(" ")}
                style={{
                  fontSize: "clamp(8px, 1.15vw, 12px)",
                  textShadow: "0 1px 2px rgba(0,0,0,.85)",
                }}
              >
                {cell.label}
              </span>
            </div>
          );
        })}
      </div>
      {hover && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-surface-3 bg-surface-1/95 p-2.5 shadow-xl backdrop-blur"
          style={{
            left: Math.min(hover.x + 14, window.innerWidth - 260),
            top: Math.min(hover.y + 14, window.innerHeight - 180),
          }}
        >
          {renderTooltip ? (
            renderTooltip(hover.cell)
          ) : (
            <StrategyTooltip cell={hover.cell} actions={actions} />
          )}
        </div>
      )}
    </div>
  );
}

export function Legend({
  actions,
  shares,
}: {
  actions: ActionMeta[];
  shares?: Record<string, number>;
}) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {actions.map((a) => (
        <span key={a.id} className="flex items-center gap-1.5">
          <span className="inline-block size-3 rounded-sm" style={{ background: a.color }} />
          {a.label}
          {shares && <span className="num text-ink-3">{pct(shares[a.id] ?? 0, 1)}</span>}
        </span>
      ))}
    </div>
  );
}
