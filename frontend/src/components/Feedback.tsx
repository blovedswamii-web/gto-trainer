import type { ActionMeta } from "../lib/actions";
import { pct } from "../lib/actions";
import type { GradeOut } from "../lib/types";

const GRADE_STYLE: Record<string, { icon: string; label: string; cls: string }> = {
  best: { icon: "✓", label: "Best", cls: "border-good/60 bg-good/10" },
  good: { icon: "✓", label: "Good — part of the mix", cls: "border-good/60 bg-good/10" },
  minor: { icon: "⚠", label: "Minor deviation", cls: "border-warning/60 bg-warning/10" },
  mistake: { icon: "✕", label: "Mistake", cls: "border-critical/60 bg-critical/10" },
};

/** Grade banner + the solution's frequency for each action, chosen action marked. */
export function GradeCard({
  grade,
  freqs,
  actions,
  extra,
  compact = false,
}: {
  grade: GradeOut;
  freqs: Record<string, number>;
  actions: ActionMeta[];
  extra?: React.ReactNode;
  /** One-line badge for the table centre; the full card stays below. */
  compact?: boolean;
}) {
  const g = GRADE_STYLE[grade.grade]!;
  if (compact) {
    return (
      <div className={`num flex items-center gap-2 whitespace-nowrap rounded-full border px-3 py-0.5 text-xs ${g.cls}`}>
        <span aria-hidden>{g.icon}</span>
        <span className="font-semibold">{g.label}</span>
        {grade.ev_loss !== null && <span className="text-ink-2">−{grade.ev_loss.toFixed(2)}bb</span>}
      </div>
    );
  }
  return (
    <div className={`rounded-xl border p-4 ${g.cls}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-lg font-semibold">
          <span aria-hidden>{g.icon}</span>
          {g.label}
        </div>
        <div className="num text-xs text-ink-2">
          freq error {grade.freq_error.toFixed(2)}
          {grade.ev_loss !== null && ` · EV loss ${grade.ev_loss.toFixed(2)}bb`}
        </div>
      </div>
      <p className="mt-1 text-sm text-ink-2">{grade.explanation}</p>
      <div className="mt-3 space-y-1.5">
        {actions.map((a) => {
          const f = freqs[a.id] ?? 0;
          const chosen = a.id === grade.chosen;
          return (
            <div key={a.id} className="grid grid-cols-[150px_1fr_52px] items-center gap-2 text-sm">
              <span className={chosen ? "font-semibold text-ink-1" : "text-ink-2"}>
                {chosen ? "▸ " : ""}
                {a.label}
              </span>
              <span className="h-3 overflow-hidden rounded bg-surface-2">
                <span
                  className="block h-full rounded"
                  style={{ width: `${Math.max(f * 100, f > 0 ? 1.5 : 0)}%`, background: a.color }}
                />
              </span>
              <span className="num text-right text-ink-1">{pct(f, f > 0 && f < 0.1 ? 1 : 0)}</span>
            </div>
          );
        })}
      </div>
      {extra && <div className="mt-3 text-xs text-ink-2">{extra}</div>}
    </div>
  );
}

export function SessionBar({ n, points, freqErr }: { n: number; points: number; freqErr: number }) {
  return (
    <div className="num flex gap-4 text-xs text-ink-3">
      <span>
        hands <span className="text-ink-1">{n}</span>
      </span>
      <span>
        accuracy <span className="text-ink-1">{n ? pct(points / n) : "—"}</span>
      </span>
      <span>
        avg freq error <span className="text-ink-1">{n ? (freqErr / n).toFixed(2) : "—"}</span>
      </span>
    </div>
  );
}

/** Map keyboard keys to actions: first letter of the id, or 1..n by order. */
export function keyToAction(key: string, actions: { id: string }[]): string | null {
  const k = key.toLowerCase();
  const byNum = Number(k);
  if (byNum >= 1 && byNum <= actions.length) return actions[byNum - 1]!.id;
  const letter: Record<string, string> = { f: "fold", c: "call", r: "raise", a: "allin", x: "check" };
  const id = letter[k];
  if (id && actions.some((a) => a.id === id)) return id;
  if (k === "c" && actions.some((a) => a.id === "check")) return "check";
  if (k === "b") return actions.find((a) => a.id.startsWith("bet"))?.id ?? null;
  return null;
}
