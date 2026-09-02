"""Startup detector inventory (§1, §17 Phase 0 gate).

`python -m voiceshield.inventory` — loads every model once and prints an
accurate table of detector name / kind / status / device. Also called from the
FastAPI startup hook.

Exit code is non-zero if a Tier-A detector failed to load, so `make inventory`
in CI / the acceptance checklist fails loudly rather than silently degrading.
"""
from __future__ import annotations

import sys

from .config import get_settings
from .ml.registry import get_registry
from .store.db import init_db

# Tier A detectors (see §2) — these MUST load or the prototype is not honest.
_TIER_A = {"speaker_consistency", "prosody_anomaly"}


def _torch_report() -> dict[str, str]:
    try:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "-",
        }
    except Exception as exc:
        return {"torch": f"NOT INSTALLED ({exc})", "cuda_available": "False",
                "cuda_device": "-"}


def _timed_warmup(reg) -> dict[str, float]:
    """One dummy inference per detector — a first, rough latency reading (§1)."""
    import time

    import numpy as np

    dummy = np.zeros(int(4.0 * 16000), dtype=np.float32)
    out: dict[str, float] = {}
    for d in reg.all():
        if not d.available:
            continue
        t0 = time.perf_counter()
        try:
            d.analyze(dummy, 16000, {})
        except Exception:
            pass
        out[d.name] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def build_report() -> dict:
    s = get_settings()
    reg = get_registry()
    warm = _timed_warmup(reg)
    inv = reg.inventory()
    for d in inv:
        d["warmup_ms"] = warm.get(d["name"])
    tier_a_ok = all(
        d["available"] for d in inv if d["name"] in _TIER_A
    ) and _TIER_A.issubset({d["name"] for d in inv})
    return {
        "device": s.device,
        "offline": s.offline,
        "model_cache_dir": str(s.model_cache_dir),
        "db_path": str(s.db_path),
        "torch": _torch_report(),
        "detectors": inv,
        "tier_a_ok": tier_a_ok,
    }


def print_inventory() -> bool:
    rep = build_report()
    try:
        from rich.console import Console
        from rich.table import Table

        c = Console()
        c.rule("[bold]VoiceShield — detector inventory")
        c.print(
            f"device=[bold cyan]{rep['device']}[/]  offline={rep['offline']}  "
            f"torch={rep['torch']['torch']}  cuda={rep['torch']['cuda_available']} "
            f"({rep['torch']['cuda_device']})"
        )
        t = Table(show_lines=False)
        for col in ("detector", "kind", "feeds", "status", "device", "warmup", "note"):
            t.add_column(col)
        for d in rep["detectors"]:
            status = "[green]loaded[/]" if d["available"] else "[red]UNAVAILABLE[/]"
            wm = f"{d['warmup_ms']:.0f} ms" if d.get("warmup_ms") is not None else "-"
            t.add_row(d["name"], d["kind"].upper(), d["feeds"], status,
                      d["device"], wm, (d["load_error"] or "")[:40])
        c.print(t)
        c.print(f"model cache: {rep['model_cache_dir']}")
        c.print(f"db:          {rep['db_path']}")
        if rep["tier_a_ok"]:
            c.print("[green]Tier-A detectors OK[/]")
        else:
            c.print("[red]Tier-A detector(s) missing — run scripts/fetch_models.py[/]")
    except Exception:  # rich not installed — plain fallback
        print("=== VoiceShield detector inventory ===")
        print(f"device={rep['device']} offline={rep['offline']} torch={rep['torch']}")
        for d in rep["detectors"]:
            print(f"  {d['name']:<20} {d['kind']:<10} "
                  f"{'loaded' if d['available'] else 'UNAVAILABLE':<12} "
                  f"{d['device']:<5} {d['load_error']}")
        print("tier_a_ok:", rep["tier_a_ok"])
    return rep["tier_a_ok"]


def main() -> int:
    init_db()
    ok = print_inventory()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
