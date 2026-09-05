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

export interface ContextQuote {
  quote: string;
  start: number;
  end: number;
  rule: string;
  t_start: number | null;
  t_end: number | null;
  utterance_index: number | null;
  signal: string;
}

export interface SignalData {
  name: string;
  value: number;
  kind: DetectorKind;
  matches: Omit<ContextQuote, "signal">[];
  detail: Record<string, unknown>;
}

export interface TranscriptSegmentData {
  index: number;
  t_start: number;
  t_end: number;
  text: string;
  language: string;
  no_speech_prob: number;
}

export interface TranscriptData {
  text: string;
  segments: TranscriptSegmentData[];
  n_segments: number;
  duration_s: number;
}

export interface ContextData {
  signals: Record<string, SignalData>;
  components: Record<string, { value: number; available: boolean; note: string }>;
  directory: Record<string, unknown> | null;
  transcript_chars: number;
  kind: DetectorKind;
  note: string;
}

export interface ContextMsg {
  type: "context";
  session_id: string;
  transcript: TranscriptData;
  context: ContextData | null;
  quotes: ContextQuote[];
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
  transcript: TranscriptData;
  context_quotes: ContextQuote[];
  context: ContextData | null;
  warnings: string[];
}

export type StreamMsg =
  | SessionMsg
  | WindowMsg
  | ContextMsg
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

// ---------------------------------------------------------------- Phase 4
export interface PolicyDecision {
  action: "ALLOW" | "VERIFY" | "ESCALATE";
  allowed: boolean;
  reason: string;
  band: Band | null;
  score: number | null;
  error_code: string;
  detail: Record<string, unknown>;
}

export interface Approval {
  id: string;
  created_at: string;
  description: string;
  amount: number;
  currency: string;
  state: "pending" | "blocked" | "approved" | "rejected";
  session_id: string | null;
  unlocked_by: string | null;
  policy: PolicyDecision;
  session: { id: string; final_band: Band; final_score: number; scenario: string | null } | null;
}

export interface Verification {
  id: string;
  created_at: string;
  session_id: string | null;
  method: string;
  challenge_phrase: string | null;
  result: "pending" | "passed" | "failed";
  detail: Record<string, unknown>;
}

export interface Incident {
  id: string;
  created_at: string;
  session_id: string | null;
  band: string;
  score: number;
  action: string;
  summary: string;
  payload: Record<string, unknown>;
}

export async function getApprovals(): Promise<{ approvals: Approval[]; note: string }> {
  return (await fetch(`${base}/approvals`)).json();
}

export async function linkSession(approvalId: string, sessionId: string) {
  return (await fetch(`${base}/approvals/${approvalId}/link`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  })).json();
}

/** Returns the HTTP status too — a 403 here is the demo's centrepiece. */
export async function approve(
  approvalId: string,
  body: Record<string, unknown> = {},
): Promise<{ status: number; body: PolicyDecision & { approved: boolean; error?: string } }> {
  const r = await fetch(`${base}/approvals/${approvalId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: r.status, body: await r.json() };
}

export async function resetApproval(approvalId: string) {
  return (await fetch(`${base}/approvals/${approvalId}/reset`, { method: "POST" })).json();
}

export async function issueChallenge(sessionId: string | null, approvalId: string) {
  return (await fetch(`${base}/verification/challenge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, approval_id: approvalId }),
  })).json();
}

export async function respondToChallenge(verificationId: string, file: File) {
  const fd = new FormData();
  fd.append("audio", file);
  return (await fetch(`${base}/verification/${verificationId}/respond`, {
    method: "POST",
    body: fd,
  })).json();
}

export async function simulatedVerification(
  sessionId: string | null,
  method: string,
  outcome = "passed",
) {
  return (await fetch(`${base}/verification/simulated`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, method, outcome }),
  })).json();
}

export async function getVerifications(sessionId?: string): Promise<{ verifications: Verification[] }> {
  const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
  return (await fetch(`${base}/verification${q}`)).json();
}

export async function getIncidents(band?: string): Promise<{ incidents: Incident[] }> {
  const q = band ? `?band=${band}` : "";
  return (await fetch(`${base}/incidents${q}`)).json();
}

// ---------------------------------------------------------------- Phase 5
export interface VoiceProfile {
  id: string;
  display_name: string;
  role: string;
  embedding_dim: number;
  n_enroll_clips: number;
  created_at: string;
  source_note: string;
}

export async function getProfiles(): Promise<{ profiles: VoiceProfile[]; note: string }> {
  return (await fetch(`${base}/profiles`)).json();
}

export async function enrolProfile(name: string, role: string, files: File[]) {
  const fd = new FormData();
  fd.append("display_name", name);
  fd.append("role", role);
  files.forEach((f) => fd.append("audio", f));
  const r = await fetch(`${base}/profiles`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
  return r.json();
}

export async function deleteProfile(id: string) {
  const r = await fetch(`${base}/profiles/${id}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export interface AnalyzeResult {
  score: number;
  band: Band;
  voice_verdict: string;
  findings: Finding[];
  fusion: FinalMsg["fusion"];
  transcript: TranscriptData;
  context_quotes: ContextQuote[];
  context: ContextData | null;
  detector_means: Record<string, number | null>;
  latency_ms: Record<string, number>;
  duration_s: number;
  n_windows: number;
  n_speech_windows: number;
  warnings: string[];
  windows: WindowMsg[];
}

export async function analyzeUpload(
  file: File,
  opts: { useProfile?: boolean; compareTelephony?: boolean; snrDb?: number | null } = {},
): Promise<{
  clean: AnalyzeResult;
  telephony?: AnalyzeResult;
  delta?: number;
  telephony_config?: Record<string, unknown>;
  filename: string;
  privacy_note: string;
}> {
  const fd = new FormData();
  fd.append("audio", file);
  fd.append("use_profile", String(opts.useProfile ?? true));
  fd.append("compare_telephony", String(opts.compareTelephony ?? false));
  if (opts.snrDb != null) fd.append("snr_db", String(opts.snrDb));
  const r = await fetch(`${base}/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
  return r.json();
}
