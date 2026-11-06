import { useRef, useState } from "react";

// Hand-rolled SVG charts (no chart dependency). Mark specs follow the dataviz rules:
// 2px lines, hairline solid gridlines, <=24px bars with a 4px rounded data-end and a square
// baseline, values in ink colours (never the series colour), hover tooltips on every mark.

const SERIES = "#3987e5";
const GRID = "#2a303b";

export interface LinePoint {
  x: number;
  y: number | null;
  label?: string;
}

export function LineChart({
  points,
  yMax = 1,
  yFormat = (v: number) => v.toFixed(2),
  height = 180,
  ariaLabel,
  xLabel = "decision #",
}: {
  points: LinePoint[];
  yMax?: number;
  yFormat?: (v: number) => string;
  height?: number;
  ariaLabel: string;
  xLabel?: string;
}) {
  const W = 640;
  const H = height;
  const pad = { l: 40, r: 12, t: 10, b: 24 };
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const valid = points.filter((p) => p.y !== null);
  if (valid.length === 0) {
    return <div className="flex h-32 items-center justify-center text-sm text-ink-3">No data yet</div>;
  }
  const xMin = points[0]!.x;
  const xMax = Math.max(points[points.length - 1]!.x, xMin + 1);
  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin)) * (W - pad.l - pad.r);
  const sy = (y: number) => pad.t + (1 - y / yMax) * (H - pad.t - pad.b);
  const path = valid.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y!).toFixed(1)}`).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax);
  const last = valid[valid.length - 1]!;
  const hp = hover !== null ? points[hover] : null;

  const onMove = (e: React.MouseEvent) => {
    const svg = ref.current;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const xv = xMin + ((x - pad.l) / (W - pad.l - pad.r)) * (xMax - xMin);
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(p.x - xv) < Math.abs(points[best]!.x - xv)) best = i;
    });
    setHover(best);
  };

  return (
    <div className="relative">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={ariaLabel}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="#8a919e">
              {yFormat(t)}
            </text>
          </g>
        ))}
        <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize={10} fill="#8a919e">
          {xLabel}
        </text>
        <path d={path} fill="none" stroke={SERIES} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={sx(last.x)} cy={sy(last.y!)} r={4} fill={SERIES} stroke="#161a21" strokeWidth={2} />
        <text x={sx(last.x) - 8} y={sy(last.y!) - 8} textAnchor="end" fontSize={11} fill="#eef0f3">
          {yFormat(last.y!)}
        </text>
        {hp && hp.y !== null && (
          <g>
            <line x1={sx(hp.x)} x2={sx(hp.x)} y1={pad.t} y2={H - pad.b} stroke="#8a919e" strokeWidth={1} />
            <circle cx={sx(hp.x)} cy={sy(hp.y)} r={4} fill={SERIES} stroke="#161a21" strokeWidth={2} />
          </g>
        )}
      </svg>
      {hp && hp.y !== null && (
        <div
          className="pointer-events-none absolute top-0 rounded-md border border-surface-3 bg-surface-1/95 px-2 py-1 text-xs shadow"
          style={{ left: `${(sx(hp.x) / W) * 100}%`, transform: "translateX(-50%)" }}
        >
          <span className="num font-semibold text-ink-1">{yFormat(hp.y)}</span>
          <span className="ml-1.5 text-ink-3">{hp.label ?? `#${hp.x}`}</span>
        </div>
      )}
    </div>
  );
}

export interface BarRow {
  key: string;
  value: number; // 0..1
  sub?: string;
  tooltip?: string;
}

/** Horizontal bars sharing one 0..1 scale. Label left, value right in ink colour. */
export function BarList({ rows, format = (v: number) => `${Math.round(v * 100)}%` }: {
  rows: BarRow[];
  format?: (v: number) => string;
}) {
  if (rows.length === 0) return <div className="text-sm text-ink-3">No data yet</div>;
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.key} className="group grid grid-cols-[150px_1fr_96px] items-center gap-2 text-sm" title={r.tooltip}>
          <span className="truncate text-ink-2">{r.key}</span>
          <span className="relative h-[14px]">
            <span className="absolute inset-y-0 left-0 right-0 border-l border-surface-3" />
            <span
              className="absolute inset-y-0 left-0 rounded-r-[4px] group-hover:opacity-80"
              style={{ width: `${Math.max(0, Math.min(1, r.value)) * 100}%`, background: SERIES }}
            />
          </span>
          <span className="num text-right text-ink-1">
            {format(r.value)}
            {r.sub && <span className="ml-1 text-[11px] text-ink-3">{r.sub}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}
