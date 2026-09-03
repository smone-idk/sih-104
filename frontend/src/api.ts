// Typed API boundary. Mirrors the backend Pydantic/dataclass payloads.

export type DetectorKind = "trained" | "pretrained" | "heuristic" | "simulated";
export type Band = "LOW" | "MEDIUM" | "HIGH";

export interface DetectorInfo {
  name: string;
  kind: DetectorKind;
  feeds: string | null;
  contributes: boolean;
  fusion_weight: number;
  available: boolean;
  device: string;
  load_error: string;
  model_id?: string;
  training_data?: string;
  licence?: string;
  note?: string;
  warmup_ms?: number | null;
}

export interface InventoryReport {
  device: string;
  offline: boolean;
  model_cache_dir: string;
  db_path: string;
  torch: { torch: string; cuda_available: string; cuda_device: string };
  detectors: DetectorInfo[];
  tier_a_ok: boolean;
}

export interface Health {
  status: string;
  version: string;
  device: string;
  offline: boolean;
  detectors_loaded: number;
  detectors_total: number;
}

export interface Scenario {
  id: string;
  title: string;
  description: string;
  tier: "genuine" | "synthetic" | "cloned";
  expected: string;
  relpath: string;
  caller_claims: string;
  language: string;
  available: boolean;
  duration_s: number | null;
}

export interface DetectorResult {
  name: string;
  kind: DetectorKind;
  score: number | null;
  available: boolean;
  detail: Record<string, unknown>;
  latency_ms: number;
  note: string;
}

export interface WindowMsg {
  type: "window";
  index: number;
  t_start: number;
  t_end: number;
  is_speech: boolean;
  raw_score: number | null;
  ema_score: number | null;
  rms_energy: number;
  spectrum: number[];
  detectors: Record<string, DetectorResult>;
  latency_ms: Record<string, number>;
}

export interface SessionMsg {
  type: "session";
  session_id: string;
  source: string;
  channel: string;
  scenario: string | null;
  profile_name: string | null;
  telephony_degraded: boolean;
  sample_rate: number;
  window_seconds: number;
  hop_seconds: number;
  ema_alpha: number;
  bands: { low_max: number; high_min: number };
  fusion_weights: Record<string, number>;
  detectors: DetectorInfo[];
  device: string;
  scenario_meta?: Scenario;
}

export interface Finding {
  code: string;
  label: string;
  severity: "info" | "warning" | "critical";
  value: number | null;
  kind: DetectorKind | "";
  evidence: Record<string, unknown>;
}

export interface AppliedFloor {
  code: string;
  description: string;
  from_band: Band;
  to_band: Band;
}

export interface Component {
  name: string;
  raw_value: number | null;
  weight: number;
  effective_weight: number;
  contribution_points: number;
  available: boolean;
  note: string;
  kind: string;
  detail: Record<string, unknown>;
}

export interface FinalMsg {
  type: "final";
  session_id: string;
  score: number;
  band: Band;
  voice_verdict: string;
  findings: Finding[];
  fusion: {
    score: number;
    band: Band;
    band_from_score: Band;
    floors_applied: AppliedFloor[];
    redistributed: boolean;
    available_components: string[];
    unavailable_components: string[];
    components: Component[];
    disclaimer: string;
  };
  detector_means: Record<string, number | null>;
  baseline_detectors: Record<string, number>;
  latency_ms: Record<string, number>;
  n_windows: number;
  n_speech_windows: number;
  duration_s: number;
  warnings: string[];
}

export type StreamMsg =
  | SessionMsg
  | WindowMsg
  | FinalMsg
  | { type: "error"; error: string };

const base = "/api/v1";

export async function getHealth(): Promise<Health> {
  return (await fetch(`${base}/health`)).json();
}
export async function getInventory(): Promise<InventoryReport> {
  return (await fetch(`${base}/inventory`)).json();
}
export async function getScenarios(): Promise<{ scenarios: Scenario[]; note: string }> {
  return (await fetch(`${base}/scenarios`)).json();
}

export interface StartOpts {
  scenario: string;
  telephony?: boolean;
  snr_db?: number | null;
  use_profile?: boolean;
  realtime?: boolean;
  speed?: number;
}

/** Opens the analysis WebSocket. Returns a closer. */
export function openStream(
  opts: StartOpts,
  onMsg: (m: StreamMsg) => void,
  onClose?: (why: string) => void,
): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}${base}/stream`);
  ws.onopen = () => ws.send(JSON.stringify(opts));
  ws.onmessage = (e) => {
    try {
      onMsg(JSON.parse(e.data) as StreamMsg);
    } catch {
      /* ignore malformed frame */
    }
  };
  ws.onerror = () => onClose?.("connection error");
  ws.onclose = () => onClose?.("closed");
  return () => {
    try {
      ws.close();
    } catch {
      /* already closed */
    }
  };
}
