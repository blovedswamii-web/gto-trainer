import { useEffect } from "react";
import type { ActionMeta } from "../lib/actions";

/** Action buttons coloured by ActionMeta, hotkey hint, bet size as the primary label.
 *  Hover/focus reports the previewed action so the table can show the chips. */
export function ActionBar({
  actions,
  onPick,
  disabled,
  columns = 3,
  sizeOf,
  onPreview,
}: {
  actions: ActionMeta[];
  onPick: (id: string) => void;
  disabled?: boolean;
  columns?: 2 | 3;
  /** Total chips committed if this action is taken, for the table preview. */
  sizeOf?: (id: string) => number | null;
  onPreview?: (bb: number | null) => void;
}) {
  // A disabled button never fires mouseleave, so a hover preview would otherwise survive
  // the answer and show up as phantom chips on the next deal.
  useEffect(() => {
    if (disabled) onPreview?.(null);
  }, [disabled, onPreview]);

  return (
    <div className={`grid gap-2 ${columns === 2 ? "grid-cols-2" : "grid-cols-3"}`}>
      {actions.map((a, i) => {
        const show = () => onPreview?.(sizeOf?.(a.id) ?? null);
        const hide = () => onPreview?.(null);
        return (
          <button
            key={a.id}
            disabled={disabled}
            onClick={() => onPick(a.id)}
            onMouseEnter={show}
            onMouseLeave={hide}
            onFocus={show}
            onBlur={hide}
            className="rounded-lg border-2 bg-surface-2 px-3 py-2.5 text-sm font-semibold transition hover:bg-surface-3 disabled:opacity-50"
            style={{ borderColor: a.color }}
          >
            {a.label}
            <span className="ml-1.5 text-[10px] font-normal text-ink-3">[{i + 1}]</span>
          </button>
        );
      })}
    </div>
  );
}
