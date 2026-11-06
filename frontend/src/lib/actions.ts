// Action identity -> colour and display order. Colours come from the validated palette in
// index.css; text never uses these colours (labels stay in ink tokens).

export interface ActionMeta {
  id: string;
  label: string;
  color: string;
}

const COLOR: Record<string, string> = {
  fold: "var(--color-act-fold)",
  check: "var(--color-act-call)",
  call: "var(--color-act-call)",
  raise: "var(--color-act-raise)",
  allin: "var(--color-act-allin)",
};

/** Postflop ids look like "bet:0.33", "raise:3.0"; preflop ids are plain words. */
export function actionColor(id: string): string {
  if (COLOR[id]) return COLOR[id];
  const [kind, size] = id.split(":");
  if (kind === "bet") {
    return Number(size) <= 0.5 ? "var(--color-act-bet-small)" : "var(--color-act-bet-big)";
  }
  if (kind === "raise") return COLOR.raise!;
  return "var(--color-ink-3)";
}

const ORDER = ["fold", "check", "call", "bet", "raise", "allin"];

export function actionRank(id: string): number {
  const [kind, size] = id.split(":");
  const base = ORDER.indexOf(kind ?? id);
  return (base < 0 ? 99 : base) * 100 + (size ? Number(size) : 0);
}

export function toActionMetas(actions: { id: string; label: string }[]): ActionMeta[] {
  return [...actions]
    .sort((a, b) => actionRank(a.id) - actionRank(b.id))
    .map((a) => ({ id: a.id, label: a.label, color: actionColor(a.id) }));
}

export function pct(x: number, digits = 0): string {
  return `${(x * 100).toFixed(digits)}%`;
}
