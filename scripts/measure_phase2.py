"""Phase 2 gate measurement: trace engine numeric targets.

Measures on synth corpus traces, seed 1337:
- compression >= 12x vs raw JSONL (COLD settings: dicts, level 19, dedup)
- compression >= 2.5x vs whole-file zstd -19 of the same bytes
- lossless: byte equality over 100 percent of lines after round trip

Writes benchmarks/results/phase2.json. Never edit the numbers by hand.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import zstandard

sys.path.insert(0, str(Path(__file__).parent))
from _machine import machine_info

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337


def whole_file_zstd19_bytes(paths: list[Path]) -> int:
    """The honest baseline: each raw file through plain zstd level 19."""
    total = 0
    cctx = zstandard.ZstdCompressor(level=19)
    for p in paths:
        with p.open("rb") as fh:
            reader = cctx.stream_reader(fh) if False else None
        # Stream compress without holding the file in memory.
        with p.open("rb") as src:
            compressor = cctx.compressobj()
            while chunk := src.read(1 << 20):
                total += len(compressor.compress(chunk))
            total += len(compressor.flush())
    return total


def main() -> int:
    from density.bench.synth import generate
    from density.engine.trace.shred import shred_events, unshred
    from density.engine.trace.zdict import LEVEL_COLD
    from density.ingest.traces import iter_events, iter_raw_lines

    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratch" / "phase2_corpus"
    if not (corpus / "traces").exists():
        print(f"generating 0.5 GB corpus at {corpus} ...")
        t0 = time.perf_counter()
        generate(0.5, corpus, seed=SEED)
        print(f"  synth took {time.perf_counter() - t0:.1f}s")

    trace_dir = corpus / "traces"
    trace_files = sorted(trace_dir.glob("*.jsonl"))
    raw_bytes = sum(p.stat().st_size for p in trace_files)

    print("shredding at COLD settings ...")
    out_dir = corpus / "shred_cold"
    t0 = time.perf_counter()
    shred_events(iter_events(trace_dir), out_dir, level=LEVEL_COLD, seed=SEED)
    shred_seconds = time.perf_counter() - t0
    shredded_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())

    print("baseline: whole-file zstd -19 ...")
    t0 = time.perf_counter()
    baseline_bytes = whole_file_zstd19_bytes(trace_files)
    baseline_seconds = time.perf_counter() - t0

    print("verifying lossless round trip on 100 percent of lines ...")
    t0 = time.perf_counter()
    mismatches = 0
    total_lines = 0
    original = iter_raw_lines(trace_dir)
    reconstructed = unshred(out_dir)
    for (of, oi, ob), (rf, ri, rb) in zip(original, reconstructed, strict=True):
        total_lines += 1
        if (of, oi, ob) != (rf, ri, rb):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH {of}:{oi} vs {rf}:{ri}")
    roundtrip_seconds = time.perf_counter() - t0

    gates = {
        "raw_bytes": raw_bytes,
        "shredded_bytes": shredded_bytes,
        "baseline_zstd19_bytes": baseline_bytes,
        "ratio_vs_raw": raw_bytes / shredded_bytes,
        "ratio_vs_zstd19": baseline_bytes / shredded_bytes,
        "lines": total_lines,
        "mismatches": mismatches,
    }
    passes = {
        "ge_12x_vs_raw": gates["ratio_vs_raw"] >= 12.0,
        "ge_2.5x_vs_zstd19": gates["ratio_vs_zstd19"] >= 2.5,
        "lossless_100pct": mismatches == 0 and total_lines > 0,
    }
    payload = {
        "phase": 2,
        "seed": SEED,
        "machine": machine_info(),
        "timings_s": {
            "shred": round(shred_seconds, 2),
            "baseline_zstd19": round(baseline_seconds, 2),
            "roundtrip_verify": round(roundtrip_seconds, 2),
        },
        "gates": gates,
        "passes": passes,
        "all_pass": all(passes.values()),
    }
    out = ROOT / "benchmarks" / "results" / "phase2.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"gates": gates, "passes": passes}, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
