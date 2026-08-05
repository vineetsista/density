"""Phase 2 gate measurement: trace engine numeric targets.

Measures on the synth trace corpus (seed 1337, dim 768) at two sizes: the
gate point at 1 GB and a 0.2 GB sanity point. The gates only bind at the
1 GB point: interleaving depth grows with corpus size (a bigger platform
has more concurrent sessions), and the whole-file zstd baseline only
loses its 8 MiB window advantage once same-conversation repeats land
farther apart than that, which is the realistic regime for archives.

Per corpus:
- raw JSONL bytes of the trace shards.
- naive baseline: ONE zstd level 19 frame over the concatenated shards
  (the strongest honest whole-file baseline: the window may match across
  shard boundaries).
- COLD bundle: shred_events at LEVEL_COLD over iter_events. Bundle bytes
  are structured.parquet plus every block store, exactly as ShredResult
  accounts them; the on-disk total including index files is also
  recorded.
- FULL round-trip: unshred output must equal every original line
  byte-for-byte, over the entire corpus, zero sampling.
- dedup: find_clusters over the content bodies of all parsed events
  (near-duplicate clusters, exact groups, bytes saved by interning).

Writes benchmarks/results/phase2.json. Never edit the numbers by hand.

Usage: measure_phase2.py [work_dir]
work_dir holds the generated corpora and bundles (default: repo
scratch/phase2). Corpora already present under work_dir are reused, so
an interrupted run resumes without regenerating.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import zstandard

sys.path.insert(0, str(Path(__file__).parent))
from _machine import machine_info

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337
DIM = 768
FULL_GB = 1.0
SANITY_GB = 0.2

GATES = (
    ("bundle_vs_raw_x", 12.0),
    ("bundle_vs_naive_x", 2.5),
)


def naive_zstd19_bytes(paths: list[Path]) -> int:
    """Whole-file baseline: one zstd-19 frame over the concatenated shards.

    Streaming through one compressobj keeps memory flat and lets the
    window match across shard boundaries, which is the most favorable
    honest treatment of the naive approach.
    """
    compressor = zstandard.ZstdCompressor(level=19).compressobj()
    total = 0
    for p in paths:
        with p.open("rb") as src:
            while chunk := src.read(1 << 20):
                total += len(compressor.compress(chunk))
    total += len(compressor.flush())
    return total


def _content_bodies(trace_dir: Path) -> Iterator[str]:
    """Content body of every parsed event, in stream order."""
    from density.ingest.traces import iter_events

    for parsed in iter_events(trace_dir):
        if parsed.event is not None:
            yield parsed.event.content


def measure_corpus(work_dir: Path, gb: float) -> dict:
    """Generate (or reuse) one corpus and run every Phase 2 measurement."""
    from density.bench.synth import generate
    from density.engine.trace.dedup import find_clusters
    from density.engine.trace.shred import shred_events, unshred
    from density.engine.trace.zdict import LEVEL_COLD
    from density.ingest.traces import iter_events, iter_raw_lines

    tag = f"{gb:g}gb"
    corpus = work_dir / f"corpus_{tag}"
    timings: dict[str, float] = {}
    trace_dir = corpus / "traces"

    if not trace_dir.exists():
        print(f"[{tag}] generating {gb} GB corpus at {corpus} ...", flush=True)
        t0 = time.perf_counter()
        stats = generate(gb, corpus, seed=SEED, dim=DIM)
        timings["generate"] = round(time.perf_counter() - t0, 2)
        synth_stats = stats.to_dict()
    else:
        print(f"[{tag}] reusing corpus at {corpus}", flush=True)
        synth_stats = None

    trace_files = sorted(trace_dir.glob("part-*.jsonl"))
    raw_bytes = sum(p.stat().st_size for p in trace_files)

    print(f"[{tag}] shredding at COLD settings ...", flush=True)
    bundle_dir = work_dir / f"shred_cold_{tag}"
    t0 = time.perf_counter()
    result = shred_events(iter_events(trace_dir), bundle_dir, level=LEVEL_COLD, seed=SEED)
    timings["shred"] = round(time.perf_counter() - t0, 2)
    # The gate accounting is the ShredResult's own: parquet plus block
    # stores. The on-disk total additionally counts the per-store index
    # json (random-access metadata), recorded for honesty.
    bundle_bytes = result.structured_bytes + sum(result.store_bytes.values())
    disk_bytes = sum(p.stat().st_size for p in bundle_dir.rglob("*") if p.is_file())

    print(f"[{tag}] naive baseline: single zstd-19 frame ...", flush=True)
    t0 = time.perf_counter()
    naive_bytes = naive_zstd19_bytes(trace_files)
    timings["naive_zstd19"] = round(time.perf_counter() - t0, 2)

    print(f"[{tag}] FULL round-trip byte equality ...", flush=True)
    t0 = time.perf_counter()
    lines = 0
    mismatches = 0
    for orig, rebuilt in zip(iter_raw_lines(trace_dir), unshred(bundle_dir), strict=True):
        lines += 1
        if orig != rebuilt:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH {orig[0]}:{orig[1]} vs {rebuilt[0]}:{rebuilt[1]}")
    timings["roundtrip_verify"] = round(time.perf_counter() - t0, 2)

    print(f"[{tag}] dedup over content bodies ...", flush=True)
    t0 = time.perf_counter()
    dd = find_clusters(_content_bodies(trace_dir), seed=SEED)
    timings["dedup"] = round(time.perf_counter() - t0, 2)

    ratios = {
        "bundle_vs_raw_x": round(raw_bytes / bundle_bytes, 4),
        "naive_vs_raw_x": round(raw_bytes / naive_bytes, 4),
        "bundle_vs_naive_x": round(naive_bytes / bundle_bytes, 4),
    }
    print(f"[{tag}] ratios: {ratios}", flush=True)
    return {
        "gb": gb,
        "synth": synth_stats,
        "raw_jsonl_bytes": raw_bytes,
        "naive_zstd19_bytes": naive_bytes,
        "bundle": {
            "structured_parquet_bytes": result.structured_bytes,
            "store_bytes": result.store_bytes,
            "bundle_bytes": bundle_bytes,
            "disk_total_bytes": disk_bytes,
            "lines": result.lines,
            "events": result.events,
            "malformed": result.malformed,
            "residual_lines": result.residual_lines,
        },
        "ratios": ratios,
        "roundtrip": {
            "lines": lines,
            "mismatches": mismatches,
            "byte_equal": mismatches == 0 and lines > 0,
        },
        "dedup": {
            "near_dup_clusters": len(dd.clusters),
            "exact_dup_groups": len(dd.exact_dup_groups),
            "exact_dup_bytes_saved": dd.stats.exact_dup_bytes_saved,
            "unique_texts": dd.stats.unique_texts,
            "skipped_over_cap": dd.stats.skipped_over_cap,
            "minhash_truncated": dd.stats.minhash_truncated,
            "seconds": timings["dedup"],
        },
        "timings_s": timings,
    }


def main() -> int:
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratch" / "phase2"
    work_dir.mkdir(parents=True, exist_ok=True)

    full = measure_corpus(work_dir, FULL_GB)
    sanity = measure_corpus(work_dir, SANITY_GB)

    gates = [
        {
            "gate": name,
            "op": ">=",
            "target": target,
            "measured": full["ratios"][name],
            "pass": bool(full["ratios"][name] >= target),
        }
        for name, target in GATES
    ]
    # Losslessness is a hard gate at both sizes: a single mismatched byte
    # anywhere fails the phase regardless of the ratios.
    gates.append(
        {
            "gate": "roundtrip_byte_equality",
            "op": "==",
            "target": True,
            "measured": bool(
                full["roundtrip"]["byte_equal"] and sanity["roundtrip"]["byte_equal"]
            ),
            "pass": bool(
                full["roundtrip"]["byte_equal"] and sanity["roundtrip"]["byte_equal"]
            ),
        }
    )
    payload = {
        "phase": 2,
        "seed": SEED,
        "dim": DIM,
        "machine": machine_info(),
        "corpora": {"full_1gb": full, "sanity_0.2gb": sanity},
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates),
    }
    out = ROOT / "benchmarks" / "results" / "phase2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    width = max(len(g["gate"]) for g in gates)
    print()
    for g in gates:
        status = "PASS" if g["pass"] else "FAIL"
        print(f"{g['gate']:<{width}}  {g['op']} {g['target']}  measured {g['measured']}  {status}")
    print(f"wrote {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
