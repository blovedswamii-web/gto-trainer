import type { ReactNode } from "react";
import type { HistoryStep, SourceTag } from "../lib/types";

/** Every strategy on screen carries its provenance. Approximate data is never styled as truth. */
export function SourceBadge({ source, compact = false }: { source: SourceTag; compact?: boolean }) {
  const styles: Record<string, string> = {
    approximate: "border-warning/60 bg-warning/10 text-warning",
    imported: "border-act-fold/60 bg-act-fold/10 text-ink-1",
    solver: "border-good/60 bg-good/10 text-ink-1",
  };
  const text =
    source.type === "solver"
      ? `Solver-converged${
          source.exploitability_pct !== null
            ? ` · exploitability ${source.exploitability_pct.toFixed(2)}% pot`
            : ""
        }`
      : source.type === "imported"
        ? "Imported"
        : "Approximate seed chart · not solver output";
  return (
    <span
      title={[source.label, source.notes].filter(Boolean).join(" — ")}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${styles[source.type]}`}
    >
      <span aria-hidden>{source.type === "approximate" ? "⚠" : source.type === "solver" ? "✓" : "↧"}</span>
      {text}
      {!compact && source.type !== "approximate" && (
        <span className="text-ink-3">· {source.label}</span>
      )}
    </span>
  );
}

export function Panel({
  title,
  children,
  right,
  className = "",
}: {
  title?: ReactNode;
  children: ReactNode;
  right?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card-surface p-4 ${className}`}>
      {(title || right) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold tracking-wide text-ink-2">{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Pill({
  active,
  onClick,
  children,
  disabled,
}: {
  active?: boolean;
  onClick?: () => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={[
        "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? "bg-ink-1 text-surface-0"
          : "bg-surface-2 text-ink-2 hover:bg-surface-3 hover:text-ink-1",
        disabled ? "cursor-not-allowed opacity-40" : "",
      ].join(" ")}
    >
      {children}
    </button>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-ink-3">{label}</div>
      <div className="num text-xl font-semibold text-ink-1">{value}</div>
      {sub && <div className="text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

const ACTION_WORD: Record<string, string> = {
  fold: "folds",
  raise: "raises to",
  call: "calls",
  "3bet": "3-bets to",
  "4bet": "4-bets to",
};

export function HistoryLine({ history, hero }: { history: HistoryStep[]; hero: string }) {
  const steps = history.filter((h) => h.action !== "fold");
  const folds = history.filter((h) => h.action === "fold").map((h) => h.position);
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-sm">
      {folds.length > 0 && (
        <span className="text-ink-3">{folds.join(", ")} fold ·</span>
      )}
      {steps.map((h, i) => (
        <span key={i} className="rounded bg-surface-2 px-1.5 py-0.5">
          <span className="font-semibold">{h.position}</span>{" "}
          <span className="text-ink-2">
            {ACTION_WORD[h.action] ?? h.action}
            {h.size_bb ? ` ${h.size_bb}bb` : ""}
          </span>
        </span>
      ))}
      <span className="rounded bg-act-fold/20 px-1.5 py-0.5 font-semibold">{hero} to act</span>
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <div className="rounded-md border border-critical/50 bg-critical/10 px-3 py-2 text-sm text-ink-1">
      ⚠ {error}
    </div>
  );
}

export function PlayingCard({ card, size = "md" }: { card: string; size?: "sm" | "md" | "lg" }) {
  const rank = card[0] === "T" ? "10" : card[0];
  const suit = card[1] ?? "";
  const sym: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };
  // Four-colour deck: suit identity is also carried by the symbol, never colour alone.
  const col: Record<string, string> = { s: "#e9ecf1", h: "#ef5b5b", d: "#4c9df0", c: "#3fbf7f" };
  const dims = { sm: "w-8 h-11 text-sm", md: "w-12 h-16 text-xl", lg: "w-16 h-22 text-3xl" }[size];
  return (
    <span
      className={`${dims} inline-flex flex-col items-center justify-center rounded-md border border-surface-3 bg-surface-2 font-bold leading-none shadow`}
      style={{ color: col[suit] }}
      aria-label={card}
    >
      <span>{rank}</span>
      <span className="text-[0.8em]">{sym[suit]}</span>
    </span>
  );
}
