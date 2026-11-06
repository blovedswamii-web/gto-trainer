import { useEffect, useState } from "react";
import { ErrorNote, Panel, Pill, Stat } from "../components/ui";
import { pct } from "../lib/actions";
import { api } from "../lib/api";

interface EquityOut {
  equity: number;
  win: number;
  tie: number;
  lose: number;
  method: string;
  samples: number;
  std_error: number;
  matchups: number;
  hero_combos: number;
  villain_combos: number;
  per_combo: { combo: string; equity: number }[];
}
interface OddsOut {
  pot_odds: number;
  mdf: number;
  bluff_breakeven: number;
  optimal_bluff_fraction: number;
  value_to_bluff: number;
}
interface Question {
  kind: "pot_odds" | "mdf" | "bluff";
  pot: number;
  bet: number;
  prompt: string;
}
interface MathResult {
  grade: string;
  correct_pct: number;
  error_pp: number;
  explanation: string;
}

const input =
  "w-full rounded-md border border-surface-3 bg-surface-2 px-2.5 py-1.5 font-mono text-sm text-ink-1 placeholder:text-ink-3";

function EquityCalc() {
  const [hero, setHero] = useState("AhKh");
  const [villain, setVillain] = useState("QQ+,AKs,AKo,A5s");
  const [board, setBoard] = useState("");
  const [dead, setDead] = useState("");
  const [method, setMethod] = useState("auto");
  const [res, setRes] = useState<EquityOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      setRes(await api.post<EquityOut>("/equity", { hero, villain, board, dead, method }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Panel title="Equity calculator (hand or range vs range)">
      <div className="grid gap-2 sm:grid-cols-2">
        <label className="text-xs text-ink-3">
          Hero (combo or range)
          <input className={input} value={hero} onChange={(e) => setHero(e.target.value)} />
        </label>
        <label className="text-xs text-ink-3">
          Villain range
          <input className={input} value={villain} onChange={(e) => setVillain(e.target.value)} />
        </label>
        <label className="text-xs text-ink-3">
          Board (0, 3, 4 or 5 cards)
          <input className={input} value={board} placeholder="e.g. Kd7c2s" onChange={(e) => setBoard(e.target.value)} />
        </label>
        <label className="text-xs text-ink-3">
          Dead cards
          <input className={input} value={dead} onChange={(e) => setDead(e.target.value)} />
        </label>
      </div>
      <div className="mt-3 flex items-center gap-2">
        {["auto", "exact", "monte_carlo"].map((m) => (
          <Pill key={m} active={method === m} onClick={() => setMethod(m)}>
            {m === "monte_carlo" ? "Monte Carlo" : m}
          </Pill>
        ))}
        <button
          onClick={() => void run()}
          disabled={busy}
          className="ml-auto rounded-md bg-ink-1 px-4 py-1.5 text-sm font-semibold text-surface-0 disabled:opacity-50"
        >
          {busy ? "Calculating…" : "Calculate"}
        </button>
      </div>
      <p className="mt-2 text-[11px] text-ink-3">
        Notation: 77+, ATs+, A5s-A2s, T9s-54s, KQ, AhKh, weights like AKs:0.5. "auto" enumerates exactly
        when cheap (e.g. any hand vs hand, ranges on the turn) and uses Monte Carlo otherwise.
      </p>
      <ErrorNote error={err} />
      {res && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-4 gap-3">
            <Stat label="Hero equity" value={pct(res.equity, 2)} sub={res.std_error ? `± ${(res.std_error * 100).toFixed(2)}% (1 SE)` : "exact"} />
            <Stat label="Win" value={pct(res.win, 2)} />
            <Stat label="Tie" value={pct(res.tie, 2)} />
            <Stat label="Lose" value={pct(res.lose, 2)} />
          </div>
          <div className="num text-xs text-ink-3">
            {res.method === "exact"
              ? `Exact enumeration: ${res.samples.toLocaleString()} runouts per matchup, ${res.matchups} compatible combo pairs.`
              : `Monte Carlo: ${res.samples.toLocaleString()} samples.`}{" "}
            Hero {res.hero_combos.toFixed(0)} combos vs villain {res.villain_combos.toFixed(0)} combos (before blockers).
          </div>
          {res.per_combo.length > 1 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-ink-2">Equity of each hero combo</summary>
              <div className="num mt-2 grid grid-cols-3 gap-x-4 gap-y-0.5 sm:grid-cols-5">
                {res.per_combo.map((c) => (
                  <span key={c.combo}>
                    <span className="font-mono">{c.combo}</span> <span className="text-ink-2">{pct(c.equity, 1)}</span>
                  </span>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </Panel>
  );
}

function OddsCalc() {
  const [pot, setPot] = useState(10);
  const [bet, setBet] = useState(7.5);
  const [o, setO] = useState<OddsOut | null>(null);
  useEffect(() => {
    if (pot > 0 && bet > 0) void api.post<OddsOut>("/odds", { pot, bet }).then(setO).catch(() => setO(null));
  }, [pot, bet]);
  return (
    <Panel title="Pot odds & MDF">
      <div className="grid grid-cols-2 gap-2">
        <label className="text-xs text-ink-3">
          Pot before the bet (bb)
          <input type="number" className={input} value={pot} min={0.1} step={0.5} onChange={(e) => setPot(Number(e.target.value))} />
        </label>
        <label className="text-xs text-ink-3">
          Bet (bb)
          <input type="number" className={input} value={bet} min={0.1} step={0.5} onChange={(e) => setBet(Number(e.target.value))} />
        </label>
      </div>
      {o && (
        <div className="mt-4 grid grid-cols-2 gap-4">
          <Stat label="Equity to call" value={pct(o.pot_odds, 1)} sub="bet / (pot + 2·bet)" />
          <Stat label="Minimum defence" value={pct(o.mdf, 1)} sub="pot / (pot + bet)" />
          <Stat label="Bluff must work" value={pct(o.bluff_breakeven, 1)} sub="bet / (pot + bet)" />
          <Stat label="River value : bluff" value={`${o.value_to_bluff.toFixed(2)} : 1`} sub={`bluffs = ${pct(o.optimal_bluff_fraction, 1)} of bets`} />
        </div>
      )}
    </Panel>
  );
}

function OddsDrill() {
  const [kind, setKind] = useState<Question["kind"] | "">("");
  const [q, setQ] = useState<Question | null>(null);
  const [guess, setGuess] = useState("");
  const [res, setRes] = useState<MathResult | null>(null);
  const [score, setScore] = useState({ n: 0, ok: 0 });
  const next = async () => {
    setRes(null);
    setGuess("");
    setQ(await api.get<Question>(`/math/question${kind ? `?kind=${kind}` : ""}`));
  };
  useEffect(() => {
    void next();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind]);
  const submit = async () => {
    if (!q || guess === "") return;
    const r = await api.post<MathResult>("/math/answer", { ...q, guess_pct: Number(guess) });
    setRes(r);
    setScore((s) => ({ n: s.n + 1, ok: s.ok + (r.grade === "best" ? 1 : 0) }));
  };
  return (
    <Panel
      title="Odds drill"
      right={<span className="num text-xs text-ink-3">{score.ok}/{score.n} within ±3pp</span>}
    >
      <div className="mb-3 flex gap-1">
        {[["", "Mixed"], ["pot_odds", "Pot odds"], ["mdf", "MDF"], ["bluff", "Bluff break-even"]].map(([k, l]) => (
          <Pill key={k} active={kind === k} onClick={() => setKind(k as Question["kind"] | "")}>{l}</Pill>
        ))}
      </div>
      {q && (
        <form
          className="space-y-3"
          onSubmit={(e) => {
            e.preventDefault();
            if (res) void next();
            else void submit();
          }}
        >
          <p className="text-base text-ink-1">{q.prompt}</p>
          <div className="flex items-center gap-2">
            <input autoFocus className={`${input} w-32`} value={guess} inputMode="decimal" placeholder="%" onChange={(e) => setGuess(e.target.value)} disabled={!!res} />
            <span className="text-ink-3">%</span>
            <button className="rounded-md bg-ink-1 px-4 py-1.5 text-sm font-semibold text-surface-0">{res ? "Next (Enter)" : "Check"}</button>
          </div>
          {res && (
            <div className={`rounded-md border p-3 text-sm ${res.grade === "best" ? "border-good/60 bg-good/10" : res.grade === "minor" ? "border-warning/60 bg-warning/10" : "border-critical/60 bg-critical/10"}`}>
              <span aria-hidden>{res.grade === "best" ? "✓ " : res.grade === "minor" ? "⚠ " : "✕ "}</span>
              Answer: <span className="num font-semibold">{res.correct_pct.toFixed(1)}%</span> (you were off by{" "}
              {res.error_pp.toFixed(1)}pp). <span className="font-mono text-xs text-ink-2">{res.explanation}</span>
            </div>
          )}
        </form>
      )}
    </Panel>
  );
}

export default function EquityTools() {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <EquityCalc />
      <div className="space-y-4">
        <OddsCalc />
        <OddsDrill />
      </div>
    </div>
  );
}
