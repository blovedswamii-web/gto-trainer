import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keyToAction } from "../components/Feedback";
import type { ActionMeta } from "./actions";

const DEFAULT_NEXT_KEYS = [" ", "Enter", "n"];

interface Scored {
  grade: { points: number; freq_error: number };
}

/** The deal -> answer -> feedback loop shared by every drill: owns the current deal,
 *  feedback, busy/error state, session stats, response timing and the keyboard handler. */
export function useDrill<Deal, Feedback extends Scored>({
  fetchNext,
  submit,
  actionsOf,
  refetchOn,
  enabled = true,
  nextKeys = DEFAULT_NEXT_KEYS,
}: {
  fetchNext: () => Promise<Deal>;
  submit: (deal: Deal, action: string, responseMs: number) => Promise<Feedback>;
  actionsOf: (deal: Deal) => ActionMeta[];
  /** Fetch a fresh deal whenever one of these changes (and on mount). */
  refetchOn: unknown[];
  enabled?: boolean;
  nextKeys?: string[];
}) {
  const [deal, setDeal] = useState<Deal | null>(null);
  const [fb, setFb] = useState<Feedback | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState({ n: 0, points: 0, freqErr: 0 });
  const shownAt = useRef(0);

  const next = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setDeal(await fetchNext());
      setFb(null);
      shownAt.current = performance.now();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [fetchNext]);

  const answer = useCallback(
    async (action: string) => {
      if (!deal || fb || busy) return;
      setBusy(true);
      try {
        const r = await submit(deal, action, Math.round(performance.now() - shownAt.current));
        setFb(r);
        setStats((s) => ({
          n: s.n + 1,
          points: s.points + r.grade.points,
          freqErr: s.freqErr + r.grade.freq_error,
        }));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [deal, fb, busy, submit],
  );

  useEffect(() => {
    if (enabled) void next();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refetch only when the caller's keys change
  }, [enabled, ...refetchOn]);

  const actions = useMemo(() => (deal ? actionsOf(deal) : []), [deal, actionsOf]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (fb && nextKeys.includes(e.key.length === 1 ? e.key.toLowerCase() : e.key)) {
        e.preventDefault();
        void next();
        return;
      }
      if (!fb && deal) {
        const a = keyToAction(e.key, actions);
        if (a) void answer(a);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fb, deal, actions, next, answer, nextKeys]);

  return { deal, fb, busy, error, setError, stats, actions, next, answer };
}
