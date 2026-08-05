"""Phase 4 gate measurement: audit wall-clock against the spec budgets.

Two corpora, matching the spec's two budgets:
  A. traces-heavy: about 1 GB of JSONL with the minimum vector count,
     so the audit time is trace-attributable. Budget: 300 s.
  B. vector-heavy: 1M x 768 fp32 vectors with a minimal trace stream,
     so the audit time is vector-attributable. Budget: 600 s.

Every number is measured in-process and written to
benchmarks/results/phase4.json. A failed part (for example OOM on a
small-RAM machine) is recorded with its error, never hidden.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _machine import library_versions, machine_info

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results"
SCRATCH = REPO / "scratch" / "phase4"

SEED = 1337
DIM = 768


def run_part(name: str, gb: float, n_vectors: int, budget_s: float) -> dict:
    """Generate one corpus, audit it, and time the whole pipeline."""
    from density.audit.runner import run_audit
    from density.bench.synth import generate

    raw = SCRATCH / f"{name}_raw"
    out = SCRATCH / f"{name}_report.html"
    part: dict = {"name": name, "gb": gb, "n_vectors": n_vectors, "budget_s": budget_s}
    try:
        t0 = time.perf_counter()
        stats = generate(gb, raw, seed=SEED, dim=DIM, n_vectors=n_vectors)
        part["synth_seconds"] = round(time.perf_counter() - t0, 2)
        part["raw_trace_bytes"] = int(stats.to_dict()["bytes_traces"])
        t1 = time.perf_counter()
        result = run_audit(str(raw), out=str(out), seed=SEED)
        wall = time.perf_counter() - t1
        d = result.to_dict()
        part["audit_wall_seconds"] = round(wall, 2)
        part["stage_seconds"] = d["stage_seconds"]
        part["peak_rss_mb"] = d["peak_rss_mb"]
        part["honesty_flags"] = d["honesty"]
        part["tier_recall_pass"] = {
            t: r["recall_pass"] for t, r in d["tier_results"].items()
        }
        part["pass"] = wall < budget_s
    except MemoryError:
        part["error"] = "MemoryError: corpus exceeds this machine's RAM"
        part["pass"] = False
    except Exception as exc:  # recorded, never hidden
        part["error"] = f"{type(exc).__name__}: {exc}"
        part["traceback_tail"] = traceback.format_exc()[-800:]
        part["pass"] = False
    return part


def main() -> None:
    global SCRATCH
    if len(sys.argv) > 1:
        SCRATCH = Path(sys.argv[1])
    SCRATCH.mkdir(parents=True, exist_ok=True)
    from density.engine import _accel

    doc: dict = {
        "phase": 4,
        "seed": SEED,
        "dim": DIM,
        "accel_active": bool(_accel.ACCEL_ACTIVE),
        "machine": machine_info(),
        "versions": library_versions(),
        "parts": [],
    }
    # Part A: about 1 GB of traces. n_vectors floor keeps vector work tiny.
    doc["parts"].append(run_part("traces_1gb", 1.03, 5000, 300.0))
    # Part B: 1M x 768 vectors. gb sized to leave only a sliver of traces.
    doc["parts"].append(run_part("vectors_1m", 3.15, 1_000_000, 600.0))
    doc["gates"] = [
        {
            "gate": f"audit_{p['name']}_under_{int(p['budget_s'])}s",
            "target": p["budget_s"],
            "measured": p.get("audit_wall_seconds", p.get("error", "failed")),
            "op": "<",
            "pass": bool(p["pass"]),
        }
        for p in doc["parts"]
    ]
    doc["all_pass"] = all(g["pass"] for g in doc["gates"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "phase4.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps(doc["gates"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
