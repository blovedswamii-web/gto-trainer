// Mirrors backend pydantic models (gto_trainer/strategies/schemas.py and api/*).

export type SourceType = "approximate" | "imported" | "solver";

export interface SourceTag {
  type: SourceType;
  label: string;
  notes: string;
  exploitability_pct: number | null;
}

export interface ActionDef {
  id: string;
  label: string;
  size_bb: number | null;
}

export interface HistoryStep {
  position: string;
  action: string;
  size_bb: number | null;
}

export type Scenario = "rfi" | "vs_open" | "bb_defense" | "vs_3bet" | "vs_4bet";

export interface SpotSummary {
  id: string;
  format_id: string;
  scenario: Scenario;
  scenario_label: string;
  hero: string;
  villain: string | null;
  title: string;
  source: SourceTag;
}

export interface CellData {
  label: string;
  combos: number;
  reach: number;
  freqs: Record<string, number>;
  ev?: Record<string, number> | null;
}

export interface ValueBluff {
  action: string;
  value_combos: number;
  bluff_combos: number;
  ratio: number | null;
  method: string;
}

export interface SpotStats {
  reach_combos: number;
  played_pct: number;
  vpip_pct_of_all: number;
  combos_by_action: Record<string, number>;
  pct_by_action: Record<string, number>;
  value_bluff: ValueBluff[];
}

export interface SpotDetail {
  summary: SpotSummary;
  history: HistoryStep[];
  pot_bb: number;
  to_call_bb: number;
  actions: ActionDef[];
  grid: CellData[];
  stats: SpotStats;
  notes: string;
  villain_range_text: string | null;
}

export interface GameFormat {
  id: string;
  name: string;
  players: number;
  stack_bb: number;
  open_size_bb: number;
  sb_open_size_bb: number;
}

export interface FormatOut {
  format: GameFormat;
  positions: string[];
  spot_count: number;
}

export interface ComboFreq {
  combo: string;
  reach: number;
  freqs: Record<string, number>;
}

export interface ParsedRange {
  class_weights: number[];
  combos: number;
  percent: number;
  normalized: string;
}
