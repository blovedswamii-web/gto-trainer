import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { SpotView } from "../components/SpotView";
import { ErrorNote, Panel, Pill, PlayingCard, Stat } from "../components/ui";
import { pct, toActionMetas } from "../lib/actions";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { Deviation, HandDetail, HandRow, ImportReport, ReviewDecision, SpotDetail } from "../lib/types";

const GRADE: Record<string, { icon: string; label: string; cls: string }> = {
  best: { icon: "✓", label: "best", cls: "text-good" },
  good: { icon: "✓", label: "good", cls: "text-good" },
  minor: { icon: "⚠", label: "minor", cls: "text-warning" },
  mistake: { icon: "✕", label: "mistake", cls: "text-critical" },
};

function GradeTag({ grade }: { grade: string }) {
  const g = GRADE[grade];
  if (!g) return <span className="text-ink-3">—</span>;
  return (
    <span className={`whitespace-nowrap font-medium ${g.cls}`}>
      <span aria-hidden>{g.icon}</span> {g.label}
    </span>
  );
}

function Cards({ cards }: { cards: string }) {
  const list = cards.match(/.{2}/g) ?? [];
  return (
    <span className="inline-flex gap-0.5">
      {list.map((c) => (
        <PlayingCard key={c} card={c} size="sm" />
      ))}
    </span>
  );
}

function label(id: string, labels: Record<string, string>) {
  return labels[id] ?? id;
}

function FreqBars({ d }: { d: ReviewDecision }) {
  if (!d.freqs) return null;
  const actions = toActionMetas(Object.keys(d.freqs).map((id) => ({ id, label: label(id, d.labels) })));
  return (
    <div className="mt-2 space-y-1">
      {actions.map((a) => {
        const f = d.freqs![a.id] ?? 0;
        const mine = a.id === d.action;
        return (
          <div key={a.id} className="grid grid-cols-[130px_1fr_44px] items-center gap-2 text-xs">
            <span className={mine ? "font-semibold text-ink-1" : "text-ink-2"}>
              {mine ? "▸ " : ""}
              {a.label}
            </span>
            <span className="h-2.5 overflow-hidden rounded bg-surface-2">
              <span className="block h-full rounded" style={{ width: `${f * 100}%`, background: a.color }} />
            </span>
            <span className="num text-right">{pct(f)}</span>
          </div>
        );
      })}
    </div>
  );
}

function DecisionCard({ d, onSpot }: { d: ReviewDecision; onSpot: (id: string) => void }) {
  return (
    <div className="rounded-lg border border-surface-3 bg-surface-2/60 p-3 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <span className="mr-2 rounded bg-surface-3 px-1.5 py-0.5 text-[11px] uppercase tracking-wide">{d.street}</span>
          <span className="text-ink-2">
            {d.hero_pos}
            {d.villain_pos ? ` vs ${d.villain_pos}` : ""} {d.scenario && `· ${d.scenario.replace("_", " ")}`}
            {d.board && <span className="ml-1 font-mono text-ink-3">[{d.board}]</span>}
          </span>
        </div>
        {d.grade ? <GradeTag grade={d.grade.grade} /> : <span className="text-xs text-ink-3">not scored</span>}
      </div>
      <div className="mt-1">
        You <span className="font-semibold">{d.action_text}</span>
        {d.category && <span className="text-ink-3"> · {d.category}</span>}
      </div>
      {d.grade ? (
        <>
          <p className="mt-1 text-xs text-ink-2">{d.grade.explanation}</p>
          <FreqBars d={d} />
        </>
      ) : (
        <p className="mt-1 text-xs text-ink-3">{d.reason}</p>
      )}
      {d.notes.map((n) => (
        <p key={n} className="mt-1 text-[11px] text-ink-3">· {n}</p>
      ))}
      {d.street === "preflop" && d.spot_id && (
        <button className="mt-2 text-xs text-ink-3 underline-offset-2 hover:text-ink-1 hover:underline" onClick={() => onSpot(d.spot_id!)}>
          Show the full range for this spot
        </button>
      )}
    </div>
  );
}

function HandView({ id }: { id: number }) {
  const hand = useApi<HandDetail>(`/review/hands/${id}`);
  const [spotId, setSpotId] = useState<string | null>(null);
  const spot = useApi<SpotDetail>(spotId ? `/spots/${encodeURIComponent(spotId)}` : null);
  const [raw, setRaw] = useState(false);
  const h = hand.data;
  if (!h) return <ErrorNote error={hand.error} />;
  const cls = (() => {
    const order = "23456789TJQKA";
    const [a, b] = [h.hero_cards.slice(0, 2), h.hero_cards.slice(2, 4)];
    const [hi, lo] = order.indexOf(a[0]!) >= order.indexOf(b[0]!) ? [a, b] : [b, a];
    return hi[0] === lo[0] ? hi[0]! + lo[0]! : hi[0]! + lo[0]! + (hi[1] === lo[1] ? "s" : "o");
  })();

  return (
    <div className="space-y-4">
      <Panel
        title={
          <span className="text-ink-1">
            Hand #{h.hand_id} <span className="text-ink-3">· {h.site} · {h.stakes} · {h.played_at?.replace("T", " ")}</span>
          </span>
        }
        right={
          <button className="text-xs text-ink-3 hover:text-ink-1" onClick={() => setRaw((r) => !r)}>
            {raw ? "Hide" : "Show"} raw text
          </button>
        }
      >
        <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
          <Cards cards={h.hero_cards} />
          <span>
            {h.hero_position} · {h.effective_bb.toFixed(0)}bb effective ·{" "}
            <span className={h.net_bb >= 0 ? "text-good" : "text-critical"}>{h.net_bb >= 0 ? "+" : ""}{h.net_bb.toFixed(1)}bb</span>
          </span>
        </div>
        {h.skip_reason && <p className="mb-2 text-sm text-warning">⚠ Not reviewed: {h.skip_reason}</p>}
        <div className="grid gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
          <ol className="space-y-0.5 text-sm">
            {h.timeline.map((t, i) =>
              t.kind === "street" ? (
                <li key={i} className="pt-2 text-[11px] uppercase tracking-wider text-ink-3">
                  {t.street} {t.cards && t.cards.length > 0 && <span className="font-mono normal-case text-ink-2">[{t.cards.join(" ")}]</span>}
                </li>
              ) : (
                <li key={i} className={t.hero ? "font-semibold text-ink-1" : "text-ink-2"}>
                  <span className="inline-block w-10 text-ink-3">{t.position}</span>
                  {t.hero ? "Hero" : t.player} {t.text}
                </li>
              ),
            )}
          </ol>
          <div className="space-y-2">
            {h.decisions.map((d, i) => (
              <DecisionCard key={i} d={d} onSpot={setSpotId} />
            ))}
            {h.postflop_note && <p className="text-xs text-ink-3">ⓘ {h.postflop_note}</p>}
          </div>
        </div>
        {raw && <pre className="mt-3 max-h-80 overflow-auto rounded bg-surface-0 p-3 text-[11px] text-ink-2">{h.raw}</pre>}
      </Panel>
      {spot.data && <SpotView key={spot.data.summary.id} detail={spot.data} highlight={cls} compact />}
    </div>
  );
}

export default function HandReview() {
  const [text, setText] = useState("");
  const [report, setReport] = useState<ImportReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState<"all" | "flagged" | "skipped">("flagged");
  const [selected, setSelected] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const summary = useApi<{
    hands: number;
    skipped: number;
    flagged_hands: number;
    net_bb: number;
    decisions: { n: number; accuracy: number; avg_freq_error: number; avg_ev_loss_bb: number | null };
    grades: Record<string, number>;
  }>(`/review/summary?t=${tick}`);
  const devs = useApi<Deviation[]>(`/review/deviations?limit=15&t=${tick}`);
  const hands = useApi<HandRow[]>(`/review/hands?filter=${filter}&t=${tick}`);

  const doImport = async (body: string) => {
    setBusy(true);
    setErr(null);
    try {
      setReport(await api.post<ImportReport>("/review/import", { text: body }));
      setText("");
      setTick((t) => t + 1);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const onFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    const parts = await Promise.all([...files].map((f) => f.text()));
    await doImport(parts.join("\n\n\n"));
  };
  const s = summary.data;
  const unscored = useMemo(() => Object.entries(report?.unscored ?? {}).sort((a, b) => b[1] - a[1]), [report]);

  return (
    <div className="space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Panel title="Import hand histories (PokerStars / GGPoker, cash NLHE)">
          <textarea
            className="h-28 w-full rounded-md border border-surface-3 bg-surface-2 p-2 font-mono text-xs"
            placeholder="Paste hand histories here, or choose .txt files below…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              className="rounded-md bg-ink-1 px-4 py-1.5 text-sm font-semibold text-surface-0 disabled:opacity-40"
              disabled={busy || !text.trim()}
              onClick={() => void doImport(text)}
            >
              {busy ? "Reviewing…" : "Import & review"}
            </button>
            <label className="cursor-pointer rounded-md bg-surface-3 px-3 py-1.5 text-sm hover:text-ink-1">
              Choose files…
              <input type="file" accept=".txt" multiple className="hidden" onChange={(e) => void onFiles(e.target.files)} />
            </label>
            <button
              className="rounded-md bg-surface-3 px-3 py-1.5 text-sm hover:text-ink-1"
              onClick={() => void api.get<{ text: string }>("/review/sample").then((r) => setText(r.text))}
            >
              Load synthetic sample hands
            </button>
            {s && s.hands > 0 && (
              <button
                className="ml-auto text-xs text-ink-3 hover:text-critical"
                onClick={() => {
                  if (window.confirm(`Delete all ${s.hands} imported hands and their review stats?`)) {
                    void api.del("/review/hands").then(() => { setSelected(null); setReport(null); setTick((t) => t + 1); });
                  }
                }}
              >
                Delete all imported hands
              </button>
            )}
          </div>
          <p className="mt-2 text-[11px] text-ink-3">
            Hands are parsed locally. Preflop decisions are graded against the loaded charts (approximate seed charts unless you imported
            solver data). Postflop decisions are graded only when the flop is a suit-permutation of a solved spot. Re-importing a file
            skips hands already stored.
          </p>
          <ErrorNote error={err} />
          {report && (
            <div className="mt-3 rounded-md bg-surface-2 p-3 text-xs">
              <div className="num text-ink-1">
                {report.parsed} hands parsed · {report.imported} new · {report.duplicates} duplicates · {report.scored_decisions} decisions
                graded · {report.flagged_decisions} flagged
              </div>
              {unscored.length > 0 && (
                <ul className="mt-1 text-ink-3">
                  {unscored.map(([r, n]) => (
                    <li key={r}>not scored ×{n}: {r}</li>
                  ))}
                </ul>
              )}
              {Object.entries(report.skipped_hands).map(([r, n]) => (
                <div key={r} className="text-warning">⚠ skipped ×{n}: {r}</div>
              ))}
              {report.parse_errors.map((p) => (
                <div key={p.hand} className="text-critical">✕ {p.hand}: {p.error}</div>
              ))}
            </div>
          )}
        </Panel>
        {s && (
          <div className="grid grid-cols-2 gap-4">
            <Panel><Stat label="Hands imported" value={s.hands} sub={`${s.skipped} not reviewable`} /></Panel>
            <Panel><Stat label="Decisions graded" value={s.decisions.n} sub={`${(s.grades.minor ?? 0) + (s.grades.mistake ?? 0)} flagged`} /></Panel>
            <Panel><Stat label="Accuracy" value={s.decisions.n ? pct(s.decisions.accuracy, 1) : "—"} sub="vs the loaded strategies" /></Panel>
            <Panel>
              <Stat label="Net result" value={`${s.net_bb >= 0 ? "+" : ""}${s.net_bb.toFixed(1)}bb`} sub="actual results: mostly variance" />
            </Panel>
          </div>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-[420px_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="Biggest deviations">
            {devs.data && devs.data.length === 0 && <p className="text-sm text-ink-3">Nothing flagged yet.</p>}
            <ol className="space-y-1.5">
              {(devs.data ?? []).map((d, i) => (
                <li key={i}>
                  <button
                    className={`w-full rounded-md px-2 py-1.5 text-left text-xs hover:bg-surface-3 ${selected === d.hand_ref ? "bg-surface-3" : "bg-surface-2"}`}
                    onClick={() => setSelected(d.hand_ref)}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span>
                        <span className="font-mono text-ink-1">{d.hero_cards}</span> · {d.hero_position}
                        {d.villain_position ? ` vs ${d.villain_position}` : ""} · {d.street}
                        {d.board && <span className="font-mono text-ink-3"> [{d.board}]</span>}
                      </span>
                      <GradeTag grade={d.grade} />
                    </div>
                    <div className="num text-ink-2">
                      you {d.chosen} ({pct(d.chosen_freq)}) · solution {d.best} ({pct(d.best_freq)})
                      {d.ev_loss_bb !== null && ` · −${d.ev_loss_bb.toFixed(2)}bb EV`}
                    </div>
                  </button>
                </li>
              ))}
            </ol>
            <Link to="/progress" className="mt-2 inline-block text-xs text-ink-3 hover:text-ink-1">
              Leak analysis of real hands → Progress, "Real hands"
            </Link>
          </Panel>
          <Panel
            title="Hands"
            right={
              <div className="flex gap-1">
                {(["flagged", "all", "skipped"] as const).map((f) => (
                  <Pill key={f} active={filter === f} onClick={() => setFilter(f)}>{f}</Pill>
                ))}
              </div>
            }
          >
            <div className="max-h-[420px] overflow-auto">
              <table className="num w-full text-xs">
                <tbody>
                  {(hands.data ?? []).map((h) => (
                    <tr
                      key={h.id}
                      className={`cursor-pointer border-t border-surface-3 hover:bg-surface-2 ${selected === h.id ? "bg-surface-2" : ""}`}
                      onClick={() => setSelected(h.id)}
                    >
                      <td className="py-1 font-mono">{h.hero_cards}</td>
                      <td>{h.hero_position}</td>
                      <td className="font-mono text-ink-3">{h.board}</td>
                      <td className={`text-right ${h.net_bb >= 0 ? "text-good" : "text-critical"}`}>{h.net_bb.toFixed(1)}</td>
                      <td className="pl-2 text-right">{h.skip_reason ? <span className="text-ink-3">skipped</span> : <GradeTag grade={h.worst_grade} />}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {hands.data && hands.data.length === 0 && <p className="text-sm text-ink-3">No hands here yet.</p>}
            </div>
          </Panel>
        </div>
        <div>
          {selected ? (
            <HandView key={selected} id={selected} />
          ) : (
            <Panel>
              <div className="flex h-48 items-center justify-center text-sm text-ink-3">
                Select a deviation or a hand to see the action, your decisions and the solution.
              </div>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}
