import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { SpotView } from "../components/SpotView";
import { ErrorNote, Pill } from "../components/ui";
import { useApi } from "../lib/hooks";
import type { FormatOut, Scenario, SpotDetail, SpotSummary } from "../lib/types";

const SCENARIOS: { id: Scenario; label: string }[] = [
  { id: "rfi", label: "RFI" },
  { id: "vs_open", label: "vs open" },
  { id: "bb_defense", label: "BB defence" },
  { id: "vs_3bet", label: "vs 3-bet" },
  { id: "vs_4bet", label: "vs 4-bet" },
];

export default function RangeViewer() {
  const [params, setParams] = useSearchParams();
  const formats = useApi<FormatOut[]>("/formats");
  const formatId = params.get("format") ?? "6max_100bb";
  const spots = useApi<SpotSummary[]>(`/spots?format_id=${formatId}`);
  const spotId = params.get("spot");
  const current = spots.data?.find((s) => s.id === spotId) ?? null;
  const scenario = (params.get("scenario") as Scenario | null) ?? current?.scenario ?? "rfi";
  const inScenario = useMemo(
    () => (spots.data ?? []).filter((s) => s.scenario === scenario),
    [spots.data, scenario],
  );

  // Default to the first spot of the scenario.
  useEffect(() => {
    if (!spots.data) return;
    if (!current || current.scenario !== scenario) {
      const first = inScenario[0];
      if (first) setParams({ format: formatId, scenario, spot: first.id }, { replace: true });
    }
  }, [spots.data, current, scenario, inScenario, formatId, setParams]);

  const detail = useApi<SpotDetail>(
    current && current.scenario === scenario ? `/spots/${encodeURIComponent(current.id)}` : null,
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="rounded-md border border-surface-3 bg-surface-1 px-2 py-1.5 text-sm"
          value={formatId}
          onChange={(e) => setParams({ format: e.target.value })}
          aria-label="Game format"
        >
          {(formats.data ?? []).map((f) => (
            <option key={f.format.id} value={f.format.id} disabled={f.spot_count === 0}>
              {f.format.name}
              {f.spot_count === 0 ? " (no data yet)" : ""}
            </option>
          ))}
        </select>
        <div className="flex gap-1" role="tablist" aria-label="Scenario">
          {SCENARIOS.map((s) => (
            <Pill
              key={s.id}
              active={s.id === scenario}
              onClick={() => setParams({ format: formatId, scenario: s.id })}
            >
              {s.label}
            </Pill>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-1">
        {inScenario.map((s) => (
          <Pill
            key={s.id}
            active={s.id === current?.id}
            onClick={() => setParams({ format: formatId, scenario, spot: s.id })}
          >
            {s.villain ? `${s.hero} vs ${s.villain}` : s.hero}
          </Pill>
        ))}
        {spots.data && inScenario.length === 0 && (
          <span className="text-sm text-ink-3">No spots for this scenario in this format yet.</span>
        )}
      </div>
      <ErrorNote error={spots.error ?? detail.error} />
      {detail.data && <SpotView key={detail.data.summary.id} detail={detail.data} />}
    </div>
  );
}
