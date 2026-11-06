import { useState } from "react";
import { ErrorNote, Panel, Pill } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { PfSpot } from "../lib/types";

type Kind = "leaks" | "preflop" | "postflop" | "pushfold";

interface Preview {
  deck: string;
  cards: number;
  media: number;
  skipped: Record<string, number>;
  samples: { front: string; back: string; tags: string[] }[];
  css: string;
}
interface Preset {
  id: string;
  label: string;
  stacks: number[];
  payouts: number[];
  ante: number;
}

const KINDS: { id: Kind; title: string; text: string }[] = [
  { id: "leaks", title: "My leaks", text: "Every spot + hand you were graded minor/mistake on — drills, real hands, push/fold. One card each, with your past choices." },
  { id: "preflop", title: "Preflop charts", text: "One card per hand class per spot: just the close decisions (mixed or boundary hands), or the whole range." },
  { id: "postflop", title: "Postflop strategy", text: "Solver mix by hand category at every flop decision of the solved spots." },
  { id: "pushfold", title: "Push/fold", text: "Close shove and call decisions for one table setup (ICM or chip EV)." },
];
const SCENARIOS = [
  ["rfi", "RFI"], ["vs_open", "vs open"], ["bb_defense", "BB defence"], ["vs_3bet", "vs 3-bet"], ["vs_4bet", "vs 4-bet"],
] as const;

export default function AnkiExport() {
  const [kind, setKind] = useState<Kind>("leaks");
  const [days, setDays] = useState<number | null>(null);
  const [minor, setMinor] = useState(true);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [hands, setHands] = useState<"close" | "all">("close");
  const [spotIds, setSpotIds] = useState<string[]>([]);
  const [preset, setPreset] = useState("sng_bubble");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [flip, setFlip] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pf = useApi<PfSpot[]>("/postflop/spots");
  const presets = useApi<Preset[]>("/pushfold/presets");

  const body = () => {
    const p = presets.data?.find((x) => x.id === preset);
    return {
      kind, days, include_minor: minor, scenarios, hands,
      spot_ids: kind === "postflop" ? spotIds : [],
      stacks: p?.stacks ?? [], payouts: p?.payouts ?? [], ante: p?.ante ?? 0,
    };
  };
  const toggle = (xs: string[], x: string) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);

  const doPreview = async () => {
    setBusy(true);
    setErr(null);
    try {
      setPreview(await api.post<Preview>("/export/anki/preview", body()));
      setFlip(0);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const download = async () => {
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch("/api/export/anki", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
      });
      if (!res.ok) throw new Error(((await res.json()) as { detail?: string }).detail ?? res.statusText);
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = `gto-trainer-${kind}.apkg`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const sample = preview?.samples[flip];
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_480px]">
      <div className="space-y-4">
        <Panel title="Export drills to Anki (.apkg)">
          <div className="grid gap-2 sm:grid-cols-2">
            {KINDS.map((k) => (
              <button
                key={k.id}
                onClick={() => { setKind(k.id); setPreview(null); }}
                className={`rounded-lg border p-3 text-left ${kind === k.id ? "border-ink-1 bg-surface-2" : "border-surface-3 hover:bg-surface-2"}`}
              >
                <div className="text-sm font-semibold">{k.title}</div>
                <div className="mt-0.5 text-xs text-ink-3">{k.text}</div>
              </button>
            ))}
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
            {kind === "leaks" && (
              <>
                <span className="text-ink-3">Period</span>
                {[null, 30, 7].map((d) => (
                  <Pill key={String(d)} active={days === d} onClick={() => { setDays(d); setPreview(null); }}>
                    {d ? `last ${d} days` : "all time"}
                  </Pill>
                ))}
                <label className="ml-3 flex items-center gap-1.5 text-ink-2">
                  <input type="checkbox" checked={minor} onChange={(e) => { setMinor(e.target.checked); setPreview(null); }} />
                  include minor deviations
                </label>
              </>
            )}
            {kind === "preflop" && (
              <>
                <span className="text-ink-3">Spots</span>
                {SCENARIOS.map(([id, label]) => (
                  <Pill key={id} active={scenarios.includes(id)} onClick={() => { setScenarios((x) => toggle(x, id)); setPreview(null); }}>
                    {label}
                  </Pill>
                ))}
                <span className="ml-3 text-ink-3">Hands</span>
                <Pill active={hands === "close"} onClick={() => { setHands("close"); setPreview(null); }}>close decisions</Pill>
                <Pill active={hands === "all"} onClick={() => { setHands("all"); setPreview(null); }}>whole range</Pill>
              </>
            )}
            {kind === "postflop" && (
              <>
                <span className="text-ink-3">Spots</span>
                {(pf.data ?? []).filter((s) => s.solved).map((s) => (
                  <Pill key={s.id} active={spotIds.includes(s.id)} onClick={() => { setSpotIds((x) => toggle(x, s.id)); setPreview(null); }}>
                    {s.board} {s.pot_type}
                  </Pill>
                ))}
                <span className="text-ink-3">(none selected = all)</span>
              </>
            )}
            {kind === "pushfold" && (
              <>
                <span className="text-ink-3">Table</span>
                <select
                  className="rounded-md border border-surface-3 bg-surface-2 px-2 py-1"
                  value={preset}
                  onChange={(e) => { setPreset(e.target.value); setPreview(null); }}
                  aria-label="Push/fold table"
                >
                  {(presets.data ?? []).map((p) => (
                    <option key={p.id} value={p.id}>{p.label}</option>
                  ))}
                </select>
                <Pill active={hands === "close"} onClick={() => { setHands("close"); setPreview(null); }}>close decisions</Pill>
                <Pill active={hands === "all"} onClick={() => { setHands("all"); setPreview(null); }}>all hands</Pill>
              </>
            )}
          </div>

          <div className="mt-4 flex items-center gap-2">
            <button onClick={() => void doPreview()} disabled={busy} className="rounded-md bg-surface-3 px-4 py-2 text-sm hover:text-ink-1 disabled:opacity-50">
              Preview
            </button>
            <button onClick={() => void download()} disabled={busy} className="rounded-md bg-ink-1 px-4 py-2 text-sm font-semibold text-surface-0 disabled:opacity-50">
              {busy ? "Working…" : "Download .apkg"}
            </button>
            {preview && (
              <span className="num text-xs text-ink-2">
                {preview.deck}: {preview.cards} cards, {preview.media} range images
              </span>
            )}
          </div>
          {preview && Object.keys(preview.skipped).length > 0 && (
            <ul className="mt-2 text-xs text-ink-3">
              {Object.entries(preview.skipped).map(([r, n]) => (
                <li key={r}>skipped ×{n}: {r}</li>
              ))}
            </ul>
          )}
          {preview && preview.cards === 0 && (
            <p className="mt-2 text-sm text-ink-3">
              Nothing to export yet{kind === "leaks" ? " — drill a bit (or import hands) and your mistakes will show up here." : "."}
            </p>
          )}
          <ErrorNote error={err} />
        </Panel>
        <Panel title="Importing into Anki">
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            <li>Desktop: double-click the .apkg (or File → Import). AnkiDroid: ⋮ → Import. AnkiMobile: open the file from Files/share sheet.</li>
            <li>Each deck type goes to its own sub-deck under <span className="font-mono">GTO Trainer::</span>.</li>
            <li>Re-exporting later and importing again <b>updates</b> existing cards (stable note IDs) and keeps your review history.</li>
            <li>Every answer shows where the strategy comes from — approximate seed charts are labelled as such, solver cards show exploitability.</li>
          </ul>
        </Panel>
      </div>

      <Panel
        title="Sample card"
        right={
          preview && preview.samples.length > 1 ? (
            <div className="flex gap-1">
              {preview.samples.map((_, i) => (
                <Pill key={i} active={flip === i} onClick={() => setFlip(i)}>{i + 1}</Pill>
              ))}
            </div>
          ) : undefined
        }
      >
        {sample && preview ? (
          <div className="overflow-hidden rounded-lg border border-surface-3">
            <style>{preview.css}</style>
            <div className="nightMode card" dangerouslySetInnerHTML={{ __html: sample.front }} />
            <hr className="border-surface-3" />
            <div className="nightMode card" dangerouslySetInnerHTML={{ __html: sample.back }} />
            <div className="px-2 py-1 text-[11px] text-ink-3">tags: {sample.tags.join(" ")}</div>
          </div>
        ) : (
          <div className="flex h-48 items-center justify-center text-sm text-ink-3">Preview a deck to see a card.</div>
        )}
      </Panel>
    </div>
  );
}
