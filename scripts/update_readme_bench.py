#!/usr/bin/env python3
"""Inject the measured benchmark table into README.md.

Reads the newest bench_full.json (falling back to bench_quick.json) from
benchmarks/results/, renders a compact markdown table, and replaces
everything between the <!-- BENCH:START --> and <!-- BENCH:END --> markers
in README.md, leaving every other byte of the file untouched. Benchmark
tables are never typed by hand: when no results JSON exists the script
refuses to run instead of inventing content.

Modes:
    python scripts/update_readme_bench.py            # inject
    python scripts/update_readme_bench.py --check    # CI honesty guard:
        exit nonzero when the README table does not match the newest
        results, so a stale or hand-edited table fails the build.

Exit codes: 0 success (or --check match), 1 --check mismatch, 2 refusal
(no results JSON, or missing markers in the README).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = ROOT / "README.md"
DEFAULT_RESULTS = ROOT / "benchmarks" / "results"

START = "<!-- BENCH:START -->"
END = "<!-- BENCH:END -->"

# Display order for the codec matrix: warm before cold, coarser rerank
# configs last, matching the tier story. Config keys not in this list
# (a future harness addition) are appended in sorted order rather than
# dropped, so the table never silently hides a measured row.
_ROW_ORDER = ("sq8", "pq_adc", "pq_rerank", "binary_rerank")

# Human labels per config key. m and rerank depth are printed from the
# measured entry itself, never from these labels.
_TIER_LABEL = {"warm": "WARM", "cold": "COLD", "hot": "HOT"}


def newest_results(results_dir: Path) -> Path | None:
    """The newest full-mode results file, else the newest quick one.

    Full results always beat quick results regardless of age: the full
    matrix is the public claim, quick is a smoke check. Globs allow
    timestamped variants (bench_full-*.json) next to the canonical names.
    """
    for pattern in ("bench_full*.json", "bench_quick*.json"):
        candidates = sorted(
            results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            return candidates[0]
    return None


def _codec_label(key: str, row: dict[str, Any]) -> str:
    tier = _TIER_LABEL.get(str(row.get("tier", "")), str(row.get("tier", "?")))
    codec = str(row.get("codec", key))
    detail = ""
    if codec == "pq":
        detail = f" (ADC, m={row.get('m')})" if row.get("rerank_depth", 0) == 0 else f" (m={row.get('m')})"
    return f"{tier} / {codec}{detail}"


def _accel_line(payload: dict[str, Any]) -> str:
    """One line: ACCEL_ACTIVE state plus per-kernel speedups when present."""
    accel = payload.get("accel", {})
    active = bool(accel.get("accel_active", payload.get("accel_active", False)))
    mk = accel.get("measure_kernels", {})
    kernels = mk.get("kernels", {})
    parts: list[str] = []
    for name in ("sq8_scores", "pq_adc_scan", "hamming_scan"):
        entry = kernels.get(name, {})
        speedup = entry.get("speedup")
        if speedup is not None:
            parts.append(f"{name} {float(speedup):.1f}x")
    if parts:
        return (
            f"Accel: ACCEL_ACTIVE={active}; compiled kernel speedups "
            f"{', '.join(parts)} (n={mk.get('n')})."
        )
    return (
        f"Accel: ACCEL_ACTIVE={active}; compiled kernels unavailable, "
        "numpy fallback timings only."
    )


def render(payload: dict[str, Any], source_name: str) -> str:
    """Render the between-markers markdown from one results payload.

    The table shows the largest measured dim (the headline corpus); the
    full per-dim matrix stays in the JSON, which the caption names so
    every number is traceable to its measurement.
    """
    vectors = payload["vectors"]
    by_dim = vectors["by_dim"]
    dim_key = str(max(int(k) for k in by_dim))
    dim_block = by_dim[dim_key]
    codecs = dim_block["codecs"]

    lines: list[str] = []
    lines.append(
        f"Source: {source_name} (mode {payload.get('mode')}, seed "
        f"{payload.get('seed')}, machine: {payload.get('machine', {}).get('cpu_model') or 'unknown cpu'})."
    )
    lines.append(
        f"Vector corpus: {vectors['n_vectors']} synth vectors at dim "
        f"{dim_block['dim']}, {vectors['n_queries']} held-out queries, "
        "recall measured by the recall verifier against exact ground truth."
    )
    lines.append("")
    lines.append("| Tier / codec | Reduction vs fp32 | recall@10 | Rerank depth | Search seconds |")
    lines.append("|---|---:|---:|---:|---:|")
    ordered = [k for k in _ROW_ORDER if k in codecs]
    ordered += sorted(k for k in codecs if k not in _ROW_ORDER)
    for key in ordered:
        row = codecs[key]
        lines.append(
            f"| {_codec_label(key, row)} "
            f"| {float(row['reduction_x']):.1f}x "
            f"| {float(row['recall']['recall@10']):.4f} "
            f"| {int(row['rerank_depth'])} "
            f"| {float(row['search_seconds']):.3f} |"
        )
    lines.append("")

    traces = payload["traces"]
    lines.append(
        f"Traces: {float(traces['ratio_vs_raw']):.1f}x vs raw JSONL, "
        f"{float(traces['ratio_vs_naive_zstd19']):.2f}x vs whole-file zstd "
        f"level {traces['zstd_level']} "
        f"({float(traces['corpus']['gb']):g} GB synth corpus, COLD settings)."
    )
    lines.append(_accel_line(payload))
    return "\n".join(lines)


def _split_readme(raw: str, readme_path: Path) -> tuple[str, str, str] | None:
    """Split README text into (head incl START, between, tail incl END).

    Returns None (after printing why) when the markers are missing or out
    of order; the caller refuses rather than guessing where the table goes.
    """
    start = raw.find(START)
    end = raw.find(END)
    if start == -1 or end == -1 or end < start:
        print(
            f"refusing: {readme_path} is missing the {START} / {END} "
            "markers (or they are out of order)",
            file=sys.stderr,
        )
        return None
    head = raw[: start + len(START)]
    between = raw[start + len(START) : end]
    tail = raw[end:]
    return head, between, tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the README table matches the newest results; write nothing",
    )
    parser.add_argument(
        "--readme", type=Path, default=DEFAULT_README, help="README path"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS, help="results directory"
    )
    args = parser.parse_args(argv)

    source = newest_results(args.results_dir) if args.results_dir.is_dir() else None
    if source is None:
        print(
            f"refusing: no bench_full.json or bench_quick.json under "
            f"{args.results_dir}. Run `density bench` first; benchmark "
            "tables are never typed by hand.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        body = render(payload, source.name)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"refusing: cannot render {source}: {exc}", file=sys.stderr)
        return 2

    # Bytes, not text mode: read_text's universal newlines would rewrite
    # line endings and break the byte-identical-outside-markers contract.
    raw = args.readme.read_bytes().decode("utf-8")
    parts = _split_readme(raw, args.readme)
    if parts is None:
        return 2
    head, between, tail = parts
    expected = "\n" + body + "\n"

    if args.check:
        if between != expected:
            print(
                f"README benchmark table is stale or hand-edited: run "
                f"`python scripts/update_readme_bench.py` to regenerate it "
                f"from {source.name}.",
                file=sys.stderr,
            )
            return 1
        print(f"README benchmark table matches {source.name}.")
        return 0

    args.readme.write_bytes((head + expected + tail).encode("utf-8"))
    print(f"README benchmark table updated from {source.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
