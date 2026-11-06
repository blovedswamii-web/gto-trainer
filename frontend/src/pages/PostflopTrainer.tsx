import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GradeCard, SessionBar, keyToAction } from "../components/Feedback";
import { Board, NodeStrategy, PfHistory } from "../components/Postflop";
import { ErrorNote, Panel, Pill, PlayingCard, SourceBadge } from "../components/ui";
import { pct, toActionMetas } from "../lib/actions";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { PfDeal, PfFeedback, PfNode, PfSpot } from "../lib/types";

function classOf(cards: string[]): string {
  const order = "23456789TJQKA";
  const [a, b] = cards as [string, string];
  const [hi, lo] = order.indexOf(a[0]!) >= order.indexOf(b[0]!) ? [a, b] : [b, a];
  if (hi[0] === lo[0]) return hi[0]! + lo[0]!;
  return hi[0]! + lo[0]! + (hi[1] === lo[1] ? "s" : "o");
}

function NotSolved() {
  return (
    <Panel>
      <p className="text-sm text-ink-2">
        No postflop solutions cached yet. Run <code className="rounded bg-surface-2 px-1">make precompute</code>{" "}
        (a few minutes) to solve the pre-defined spots with the built-in DCFR solver.
      </p>
    </Panel>
  );
}

function Explorer({ spots }: { spots: PfSpot[] }) {
  const solved = spots.filter((s) => s.solved);
  const [spotId, setSpotId] = useState(solved[0]?.id ?? "");
  const [path, setPath] = useState<string | null>(null);
  const root = useApi<PfNode>(spotId && path === null ? `/postflop/spots/${spotId}/root` : null);
  const nodeQ = useApi<PfNode>(
    spotId && path !== null ? `/postflop/spots/${spotId}/node?path=${encodeURIComponent(path)}` : null,
  );
  const node = path === null ? root.data : nodeQ.data;
  if (solved.length === 0) return <NotSolved />;
  const spot = solved.find((s) => s.id === spotId);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1">
        {solved.map((s) => (
          <Pill key={s.id} active={s.id === spotId} onClick={() => { setSpotId(s.id); setPath(null); }}>
            {s.title}
          </Pill>
        ))}
      </div>
      <ErrorNote error={root.error ?? nodeQ.error} />
      {node && spot && (
        <>
          <Panel
            title={<span className="text-ink-1">{node.title}</span>}
            right={<SourceBadge source={node.source} />}
          >
            <div className="flex flex-wrap items-center gap-6">
              <Board cards={node.board} />
              <div className="num text-sm text-ink-2">
                <div>Pot <span className="text-ink-1">{node.pot.toFixed(1)}bb</span></div>
                <div>To call <span className="text-ink-1">{node.to_call.toFixed(1)}bb</span></div>
                <div>Behind <span className="text-ink-1">{node.stack.toFixed(1)}bb</span></div>
              </div>
              <div className="min-w-0 flex-1 space-y-2">
                <PfHistory node={node} />
                <div className="flex flex-wrap gap-1.5">
                  <button className="rounded bg-surface-2 px-2 py-1 text-xs text-ink-3 hover:text-ink-1" onClick={() => setPath(null)}>
                    ⟲ start
                  </button>
                  {node.actions.map((a) => {
                    const ch = node.children[a.id];
                    if (!ch || ch.kind === "terminal") {
                      return (
                        <span key={a.id} className="rounded bg-surface-2 px-2 py-1 text-xs text-ink-3" title="ends the hand or goes to the river (not stored)">
                          {a.label} · {pct(node.overall[a.id] ?? 0)}
                        </span>
                      );
                    }
                    if (ch.kind === "node") {
                      return (
                        <button key={a.id} className="rounded bg-surface-3 px-2 py-1 text-xs hover:bg-ink-3/40" onClick={() => setPath(ch.path!)}>
                          {a.label} · {pct(node.overall[a.id] ?? 0)} →
                        </button>
                      );
                    }
                    return (
                      <span key={a.id} className="flex flex-wrap items-center gap-1 rounded bg-surface-2 px-2 py-1 text-xs">
                        {a.label} · {pct(node.overall[a.id] ?? 0)} → turn:
                        {ch.cards!.map((c) => (
                          <button key={c} className="rounded bg-surface-3 px-1 font-mono hover:bg-ink-3/40" onClick={() => setPath(`${ch.path}/${c}`)}>
                            {c}
                          </button>
                        ))}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </Panel>
          <NodeStrategy node={node} />
        </>
      )}
    </div>
  );
}

function Drill({ spots }: { spots: PfSpot[] }) {
  const solved = spots.filter((s) => s.solved);
  const [spotIds, setSpotIds] = useState<string[]>([]);
  const [streets, setStreets] = useState<string[]>([]);
  const [role, setRole] = useState<string | null>(null);
  const [deal, setDeal] = useState<PfDeal | null>(null);
  const [fb, setFb] = useState<PfFeedback | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState({ n: 0, points: 0, freqErr: 0 });
  const shownAt = useRef(0);

  const next = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const d = await api.post<PfDeal>("/postflop/drill/next", { spot_ids: spotIds, streets, role });
      setDeal(d);
      setFb(null);
      shownAt.current = performance.now();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [spotIds, streets, role]);

  const answer = useCallback(
    async (action: string) => {
      if (!deal || fb || busy) return;
      setBusy(true);
      try {
        const r = await api.post<PfFeedback>("/postflop/drill/answer", {
          spot_id: deal.spot_id,
          path: deal.path,
          combo: deal.combo,
          action,
          response_ms: Math.round(performance.now() - shownAt.current),
          reason: deal.reason,
        });
        setFb(r);
        setStats((s) => ({ n: s.n + 1, points: s.points + r.grade.points, freqErr: s.freqErr + r.grade.freq_error }));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [deal, fb, busy],
  );

  useEffect(() => {
    if (solved.length) void next();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refetch when filters change
  }, [spotIds, streets, role, solved.length]);

  const actions = useMemo(() => (deal ? toActionMetas(deal.actions) : []), [deal]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (fb && (e.key === " " || e.key === "Enter")) {
        e.preventDefault();
        void next();
      } else if (!fb && deal) {
        const a = keyToAction(e.key, actions);
        if (a) void answer(a);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fb, deal, actions, next, answer]);

  if (solved.length === 0) return <NotSolved />;
  const toggle = (xs: string[], x: string) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <div className="flex flex-wrap items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Spots</span>
          {solved.map((s) => (
            <Pill key={s.id} active={spotIds.includes(s.id)} onClick={() => setSpotIds((x) => toggle(x, s.id))}>
              {s.board} {s.pot_type}
            </Pill>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Street</span>
          {["flop", "turn"].map((st) => (
            <Pill key={st} active={streets.includes(st)} onClick={() => setStreets((x) => toggle(x, st))}>
              {st}
            </Pill>
          ))}
        </div>
        <div className="flex items-center gap-1">
          <span className="mr-1 text-xs text-ink-3">Hero</span>
          {["OOP", "IP"].map((r) => (
            <Pill key={r} active={role === r} onClick={() => setRole(role === r ? null : r)}>
              {r}
            </Pill>
          ))}
        </div>
        <div className="ml-auto">
          <SessionBar {...stats} />
        </div>
      </div>
      <ErrorNote error={error} />
      {deal && (
        <div className="grid gap-4 xl:grid-cols-[420px_minmax(0,1fr)]">
          <div className="space-y-4">
            <Panel
              title={
                <span className="text-ink-1">
                  {deal.title}
                  {deal.reason === "review" && (
                    <span className="ml-2 rounded bg-warning/15 px-1.5 py-0.5 text-[11px] text-warning">↻ review</span>
                  )}
                </span>
              }
            >
              <PfHistory node={deal} />
              <div className="my-5 flex flex-col items-center gap-4">
                <Board cards={deal.board} />
                <div className="flex items-center gap-5">
                  <div className="flex gap-1.5">
                    {deal.hero_cards.map((c) => (
                      <PlayingCard key={c} card={c} size="lg" />
                    ))}
                  </div>
                  <div className="num text-sm text-ink-2">
                    <div>Pot <span className="text-ink-1">{deal.pot.toFixed(1)}bb</span></div>
                    <div>To call <span className="text-ink-1">{deal.to_call.toFixed(1)}bb</span></div>
                    <div>Behind <span className="text-ink-1">{deal.stack.toFixed(1)}bb</span></div>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {actions.map((a, i) => (
                  <button
                    key={a.id}
                    disabled={!!fb || busy}
                    onClick={() => void answer(a.id)}
                    className="rounded-lg border-2 bg-surface-2 px-3 py-2.5 text-sm font-semibold hover:bg-surface-3 disabled:opacity-50"
                    style={{ borderColor: a.color }}
                  >
                    {a.label}
                    <span className="ml-1.5 text-[10px] font-normal text-ink-3">[{i + 1}]</span>
                  </button>
                ))}
              </div>
            </Panel>
            {fb && (
              <>
                <GradeCard
                  grade={fb.grade}
                  freqs={fb.freqs}
                  actions={actions}
                  extra={
                    <div className="space-y-1">
                      <div>
                        Your hand: <span className="text-ink-1">{fb.category}</span>
                        {fb.ev_loss_pct_pot !== null && (
                          <> · EV loss {fb.ev_loss_pct_pot.toFixed(1)}% of pot</>
                        )}
                      </div>
                      <div className="num flex flex-wrap gap-x-3">
                        {actions.map((a) => (
                          <span key={a.id}>
                            EV({a.label}) = {(fb.ev[a.id] ?? 0).toFixed(2)}bb
                          </span>
                        ))}
                      </div>
                    </div>
                  }
                />
                <button onClick={() => void next()} className="w-full rounded-lg bg-ink-1 py-2.5 text-sm font-semibold text-surface-0 hover:bg-ink-2">
                  Next spot <span className="font-normal">(Space)</span>
                </button>
              </>
            )}
          </div>
          <div>
            {fb ? (
              <NodeStrategy node={fb.node} highlightClass={classOf(deal.hero_cards)} highlightCategory={fb.category} />
            ) : (
              <Panel>
                <div className="flex h-64 items-center justify-center text-center text-sm text-ink-3">
                  Choose an action — the range-vs-range strategy for this node appears after you answer.
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function PostflopTrainer() {
  const spots = useApi<PfSpot[]>("/postflop/spots");
  const [tab, setTab] = useState<"drill" | "explore">("drill");
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-1">
        <Pill active={tab === "drill"} onClick={() => setTab("drill")}>Drill</Pill>
        <Pill active={tab === "explore"} onClick={() => setTab("explore")}>Explore solutions</Pill>
        <span className="ml-3 text-xs text-ink-3">
          Heads-up subgames solved locally with DCFR; see the source badge for exploitability and abstraction.
        </span>
      </div>
      <ErrorNote error={spots.error} />
      {spots.data && (tab === "drill" ? <Drill spots={spots.data} /> : <Explorer spots={spots.data} />)}
    </div>
  );
}
