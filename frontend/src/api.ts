// Typed API boundary. Regenerate/extend as backend Pydantic models grow.

export type DetectorKind = "trained" | "pretrained" | "heuristic" | "simulated";

export interface DetectorInfo {
  name: string;
  kind: DetectorKind;
  feeds: string;
  available: boolean;
  device: string;
  load_error: string;
  warmup_ms: number | null;
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

const base = "/api/v1";

export async function getHealth(): Promise<Health> {
  return (await fetch(`${base}/health`)).json();
}
export async function getInventory(): Promise<InventoryReport> {
  return (await fetch(`${base}/inventory`)).json();
}
