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

export interface PreflopDeal {
  spot: SpotSummary;
  history: HistoryStep[];
  pot_bb: number;
  to_call_bb: number;
  actions: ActionDef[];
  hero_cards: string[];
  hand_class: string;
  combo: string;
  reason: string;
}

export type GradeName = "best" | "good" | "minor" | "mistake";

export interface GradeOut {
  grade: GradeName;
  chosen: string;
  chosen_freq: number;
  best_action: string;
  best_freq: number;
  freq_error: number;
  ev_loss: number | null;
  explanation: string;
  points: number;
}

export interface PreflopFeedback {
  grade: GradeOut;
  combo_freqs: Record<string, number>;
  equity_vs_villain: number | null;
  detail: SpotDetail;
  decision_id: number | null;
}

export interface PfSpot {
  id: string;
  title: string;
  board: string;
  pot_type: string;
  oop: string;
  ip: string;
  solved: boolean;
  source?: SourceTag;
  pot?: number;
  stack?: number;
}

export interface PfCategory {
  category: string;
  combos: number;
  freqs: Record<string, number>;
  ev: Record<string, number>;
}

export interface PfChild {
  kind: "node" | "chance" | "terminal";
  path?: string;
  cards?: string[];
}

export interface PfNodeBase {
  spot_id: string;
  title: string;
  pot_type: string;
  path: string;
  street: string;
  board: string[];
  pot: number;
  to_call: number;
  stack: number;
  actor: "OOP" | "IP";
  actor_position: string;
  villain_position: string;
  history: { position: string; action: string; kind: "card" | "action" }[];
  actions: { id: string; label: string }[];
  children: Record<string, PfChild>;
  source: SourceTag;
}

export interface PfNode extends PfNodeBase {
  overall: Record<string, number>;
  range_combos: number;
  categories: PfCategory[];
  grid: CellData[];
}

export interface PfDeal extends PfNodeBase {
  hero_cards: string[];
  combo: string;
  reason: string;
}

export interface PfFeedback {
  grade: GradeOut;
  ev_loss_pct_pot: number | null;
  decision_id: number;
  freqs: Record<string, number>;
  ev: Record<string, number>;
  category: string;
  node: PfNode;
}

export interface StrategyAgg {
  overall: Record<string, number>;
  range_combos: number;
  categories: PfCategory[];
  grid: CellData[];
}

export interface ExploitShift {
  path: string;
  street: string;
  mass: number;
  action: string;
  label: string;
  gto_freq: number;
  exploit_freq: number;
  score: number;
}

export interface ExploitRun {
  key: string;
  hero_position: string;
  villain_position: string;
  starting_pot: number;
  values: Record<string, number>;
  values_pct_pot: Record<string, number>;
  modified_nodes: Record<string, number>;
  shifts: ExploitShift[];
  start_path: string;
}

export interface ExploitNode extends PfNodeBase {
  role: "hero" | "villain";
  gto: StrategyAgg;
  exploit: StrategyAgg;
  reached_in_exploit: boolean;
}

export interface ReviewGrade {
  grade: GradeName;
  chosen: string;
  chosen_freq: number;
  best_action: string;
  best_freq: number;
  freq_error: number;
  ev_loss: number | null;
  explanation: string;
  points: number;
}

export interface ReviewDecision {
  street: string;
  scenario: string;
  hero_pos: string;
  villain_pos: string | null;
  action_text: string;
  action: string | null;
  scored: boolean;
  reason: string;
  spot_id: string | null;
  node: string;
  board: string;
  freqs: Record<string, number> | null;
  labels: Record<string, string>;
  category: string;
  pot_bb: number;
  notes: string[];
  grade: ReviewGrade | null;
}

export interface HandRow {
  id: number;
  site: string;
  hand_id: string;
  played_at: string | null;
  stakes: string;
  hero_cards: string;
  hero_position: string;
  board: string;
  net_bb: number;
  effective_bb: number;
  skip_reason: string | null;
  n_decisions: number;
  n_scored: number;
  worst_grade: string;
  freq_error: number;
}

export interface TimelineItem {
  kind: "street" | "action";
  street: string;
  cards?: string[];
  player?: string;
  position?: string;
  hero?: boolean;
  text?: string;
}

export interface HandDetail extends HandRow {
  postflop_note: string | null;
  decisions: ReviewDecision[];
  raw: string;
  timeline: TimelineItem[];
  hero: string | null;
}

export interface Deviation {
  hand_ref: number;
  hand_id: string;
  site: string;
  played_at: string | null;
  hero_cards: string;
  hero_position: string;
  villain_position: string | null;
  street: string;
  scenario: string;
  spot_id: string;
  board: string;
  chosen: string;
  best: string;
  chosen_freq: number;
  best_freq: number;
  grade: GradeName;
  freq_error: number;
  ev_loss_bb: number | null;
  category: string;
  freqs: Record<string, number>;
}

export interface ImportReport {
  parsed: number;
  imported: number;
  duplicates: number;
  parse_errors: { hand: string; error: string }[];
  skipped_hands: Record<string, number>;
  unscored: Record<string, number>;
  scored_decisions: number;
  flagged_decisions: number;
}
