"""The public benchmark harness: one JSON file of measured numbers.

run_bench executes the whole reproducible benchmark matrix in-process and
writes a single results file (bench_quick.json or bench_full.json) into
out_dir. Every number in the file is measured here, never typed by hand;
the README table is later rendered from this file by
scripts/update_readme_bench.py, which is what keeps the public numbers
honest by construction.

Three sections:

- vectors: for each dim, the codec matrix (sq8, pq ADC at the default
  m = d / 4, pq at m = 128 with rerank 200, binary with rerank 500)
  measured by the recall verifier against exact ground truth on the
  synth embedding distribution: recall@1/10/100, bytes per vector, and
  fit/add/search wall seconds.
- traces: a synth trace corpus shredded at COLD settings versus the raw
  bytes and versus naive whole-file zstd level 19, the same two
  baselines the Phase 2 gate uses.
- accel: ACCEL_ACTIVE plus measure_kernels speedups of the compiled
  kernels over the numpy fallback.

Determinism: every stochastic step derives from the seed argument, so two
runs with the same seed differ only in the wall-clock fields (every key
ending in _seconds) and the machine section.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zstandard

from density.bench.synth import generate, sample_embeddings
from density.engine.embed.binary import BinaryCodec
from density.engine.embed.pq import PQ
from density.engine.embed.rerank import Reranker, SQ8Reranker
from density.engine.embed.sq8 import SQ8
from density.engine.trace.shred import shred_events
from density.engine.trace.zdict import LEVEL_COLD
from density.errors import DensityError
from density.ingest.traces import iter_events
from density.recall.verifier import verify_tiers

__all__ = ["run_bench", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

# Vector matrix sizing. Full mode is the standard corpus of the phase
# gates (100k vectors, 1k held-out queries) at the two common embedding
# widths; quick mode shrinks everything so the whole bench lands well
# under two minutes on a laptop CPU.
_FULL_DIMS: tuple[int, ...] = (384, 768)
_QUICK_DIMS: tuple[int, ...] = (64,)
_FULL_N_VECTORS = 100_000
_QUICK_N_VECTORS = 5_000
_FULL_N_QUERIES = 1_000
_QUICK_N_QUERIES = 200
_SAMPLE_CAP = 200_000

# Trace corpus sizing. The trace section measures trace compression, so
# the synth byte budget goes to traces: the corpus embeds a deliberately
# tiny vector stub (dim 64, 1000 vectors, about 270 KB) and everything
# else is JSONL. Full reuses a cached corpus across runs because 0.2 GB
# of seeded synth is identical every time; see _trace_corpus.
_FULL_TRACE_GB = 0.2
_QUICK_TRACE_GB = 0.01
_TRACE_STUB_DIM = 64
_TRACE_STUB_VECTORS = 1_000

# Accel kernel measurement rows (measure_kernels defaults for d and m).
_FULL_KERNEL_N = 1_000_000
_QUICK_KERNEL_N = 200_000

# Cold rerank depths per policy (tiers/policy.py): pq 200, binary 500.
_PQ_RERANK_DEPTH = 200
_BINARY_RERANK_DEPTH = 500

_READ_CHUNK = 1 << 20


def _rerank_m(dim: int) -> int:
    """Subquantizer count for the pq-with-rerank config.

    The contract point is m = 128 (DECISIONS.md item 8: the COLD gate is
    measured at m = 128 for dim 768). 128 divides both full-mode dims, so
    they use it verbatim; quick mode's dim 64 cannot hold 128 subspaces,
    so it scales to d / 2, which always divides d and preserves the
    intent of a finer-than-default code for the rerank config.
    """
    if dim % 128 == 0:
        return 128
    return max(2, dim // 2)


def _machine_info() -> dict[str, Any]:
    """Machine fingerprint, field-for-field the scripts/_machine.py set.

    Mirrored rather than imported because scripts/ is not a package: the
    library cannot depend on repo-layout files that a wheel install does
    not ship.
    """
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }


def _library_versions() -> dict[str, str]:
    """Library versions, the same package list scripts/measure_phase1.py records."""
    from importlib import metadata

    versions: dict[str, str] = {}
    for pkg in ("density", "numpy", "pyarrow", "zstandard", "pydantic"):
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def _instrument(codec: Any, sink: dict[str, float], seed: int) -> Any:
    """Wrap fit and add on one codec instance to record wall seconds.

    The verifier owns the fit/add call sites, so timing them from outside
    (instance-attribute wrappers, same trick as scripts/measure_phase1.py)
    is what keeps the referee module free of bench concerns.
    """
    orig_fit, orig_add = codec.fit, codec.add

    def fit(X: np.ndarray, seed: int = seed) -> Any:
        t0 = time.perf_counter()
        out = orig_fit(X, seed=seed)
        sink["fit_seconds"] = time.perf_counter() - t0
        return out

    def add(X: np.ndarray) -> None:
        t0 = time.perf_counter()
        orig_add(X)
        sink["add_seconds"] = time.perf_counter() - t0

    codec.fit = fit
    codec.add = add
    return codec


def _sq8_reranker_factory(
    seed: int, sink: dict[str, float]
) -> Callable[[np.ndarray], Reranker]:
    """A caching factory building one SQ8Reranker per verify call.

    COLD recall is defined with rerank against the WARM int8 vectors
    (DECISIONS.md item 6), so the harness reranks through SQ8 fitted on
    the leave-out database, exactly like the audit runner: rescoring with
    exact fp32 would flatter the cold tier.
    """
    cache: list[Reranker] = []

    def factory(x_db: np.ndarray) -> Reranker:
        if not cache:
            t0 = time.perf_counter()
            sq = SQ8().fit(x_db, seed=seed)
            sq.add(x_db)
            cache.append(SQ8Reranker(sq))
            sink["reranker_build_seconds"] = time.perf_counter() - t0
        return cache[0]

    return factory


def _measure_vectors(
    dims: tuple[int, ...], n_vectors: int, n_queries: int, seed: int
) -> dict[str, Any]:
    """The codec matrix, one verify_tiers run per dim.

    Corpus is sample_embeddings(n_vectors, dim, seed): the single source
    of truth for the synth embedding geometry, so the public bench and
    the corpus users generate can never drift apart.
    """
    by_dim: dict[str, Any] = {}
    for dim in dims:
        X = sample_embeddings(n_vectors, dim, seed=seed)
        m_rerank = _rerank_m(dim)
        # (codec instance, rerank depth, tier the config represents,
        # recorded m). sq8 is the WARM contract codec; the three cold
        # configs are pq ADC standalone, pq with rerank, binary with
        # rerank, matching the tier policy and the Phase 1 gate rows.
        configs: dict[str, tuple[Any, int, str, int | None]] = {
            "sq8": (SQ8(), 0, "warm", None),
            "pq_adc": (PQ(), 0, "cold", dim // 4),
            "pq_rerank": (PQ(m=m_rerank), _PQ_RERANK_DEPTH, "cold", m_rerank),
            "binary_rerank": (BinaryCodec(), _BINARY_RERANK_DEPTH, "cold", None),
        }
        stages: dict[str, dict[str, float]] = {name: {} for name in configs}
        rerank_sink: dict[str, float] = {}
        result = verify_tiers(
            X,
            {
                name: _instrument(codec, stages[name], seed)
                for name, (codec, _, _, _) in configs.items()
            },
            seed=seed,
            n_queries=n_queries,
            sample_cap=_SAMPLE_CAP,
            rerank_depths={name: depth for name, (_, depth, _, _) in configs.items()},
            reranker_factory=_sq8_reranker_factory(seed, rerank_sink),
        ).to_dict()
        del X

        fp32_bpv = dim * 4
        n_database = int(result["n_database"])
        rows: dict[str, Any] = {}
        for name, (_codec, _depth, tier, m) in configs.items():
            rep = result["codecs"][name]
            codes_bpv = rep["bytes"]["encoded"] / n_database
            total_bpv = float(rep["bytes_per_vector"])  # codes plus aux
            rows[name] = {
                "codec": {"sq8": "sq8", "pq_adc": "pq", "pq_rerank": "pq",
                          "binary_rerank": "binary"}[name],
                "tier": tier,
                "m": m,
                "rerank_depth": int(rep["rerank_depth"]),
                "recall": {k: float(v) for k, v in rep["recall"].items()},
                "bytes": {
                    "raw": int(rep["bytes"]["raw"]),
                    "encoded": int(rep["bytes"]["encoded"]),
                    "aux": int(rep["bytes"]["aux"]),
                },
                "bytes_per_vector": total_bpv,
                "bytes_per_vector_codes_only": float(codes_bpv),
                "reduction_x": fp32_bpv / total_bpv,
                "reduction_x_codes_only": fp32_bpv / codes_bpv,
                "fit_seconds": float(stages[name].get("fit_seconds", 0.0)),
                "add_seconds": float(stages[name].get("add_seconds", 0.0)),
                "search_seconds": float(rep["search_seconds"]),
            }
        by_dim[str(dim)] = {
            "dim": dim,
            "fp32_bytes_per_vector": fp32_bpv,
            "n_database": n_database,
            "sample_size": int(result["sample_size"]),
            "reranker_build_seconds": float(
                rerank_sink.get("reranker_build_seconds", 0.0)
            ),
            "codecs": rows,
        }
    return {
        "n_vectors": n_vectors,
        "n_queries": n_queries,
        "sample_cap": _SAMPLE_CAP,
        "dims": list(dims),
        "corpus": (
            "density.bench.synth.sample_embeddings(n_vectors, dim, seed): "
            "200-cluster low-rank manifold mixture on the unit sphere, "
            "Zipf weights, 35 percent near-duplicate points; single source "
            "of truth with the synth corpus"
        ),
        "cold_rerank_reference": (
            "COLD recall is measured with rerank against SQ8 (WARM int8) "
            "reconstructions of the leave-out database, per DECISIONS item 6."
        ),
        "by_dim": by_dim,
    }


def _trace_corpus(cache_root: Path, gb: float, seed: int) -> Path:
    """Return a cached synth corpus directory, generating it if needed.

    The corpus is deterministic per (gb, dim, n_vectors, seed), so reuse
    is safe whenever the stamp parameters match and the trace shards on
    disk still sum to the byte count recorded at generation time. Any
    mismatch (crashed run, tampering, version drift in the generator's
    output size) wipes and regenerates.
    """
    params = {
        "gb": gb,
        "dim": _TRACE_STUB_DIM,
        "n_vectors": _TRACE_STUB_VECTORS,
        "seed": seed,
    }
    corpus = cache_root / f"synth-{gb:g}gb-dim{_TRACE_STUB_DIM}-seed{seed}"
    stamp = corpus / "stamp.json"
    if stamp.exists():
        try:
            meta = json.loads(stamp.read_text(encoding="utf-8"))
            on_disk = sum(
                f.stat().st_size for f in (corpus / "traces").glob("part-*.jsonl")
            )
            if meta.get("params") == params and on_disk == meta.get("bytes_traces"):
                return corpus
        except (OSError, ValueError):
            pass
        shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir(parents=True, exist_ok=True)
    stats = generate(
        gb, corpus, seed=seed, dim=_TRACE_STUB_DIM, n_vectors=_TRACE_STUB_VECTORS
    )
    stamp.write_text(
        json.dumps({"params": params, "bytes_traces": stats.bytes_traces}, indent=2),
        encoding="utf-8",
    )
    return corpus


def _naive_zstd_bytes(shards: list[Path], level: int) -> int:
    """Compressed size of each shard through plain zstd at level, summed.

    This is literally the `zstd -19 *.jsonl` baseline of the Phase 2
    gate: per-file, no dictionary, no structure. Streaming compressobj
    keeps peak memory at one read chunk regardless of shard size.
    """
    cctx = zstandard.ZstdCompressor(level=level)
    total = 0
    for path in shards:
        cobj = cctx.compressobj()
        with path.open("rb") as fh:
            while chunk := fh.read(_READ_CHUNK):
                total += len(cobj.compress(chunk))
        total += len(cobj.flush())
    return total


def _measure_traces(cache_root: Path, gb: float, seed: int) -> dict[str, Any]:
    """Raw vs COLD shred bundle vs naive zstd-19 on one synth corpus."""
    corpus = _trace_corpus(cache_root, gb, seed)
    traces_dir = corpus / "traces"
    shards = sorted(traces_dir.glob("part-*.jsonl"))
    raw_file_bytes = sum(p.stat().st_size for p in shards)

    # The bundle lives only long enough to be measured: its size is the
    # number we keep, and full-mode bundles are tens of megabytes.
    work = Path(tempfile.mkdtemp(prefix="density-bench-shred-"))
    try:
        t0 = time.perf_counter()
        shred = shred_events(
            iter_events(traces_dir), work / "bundle", level=LEVEL_COLD, seed=seed
        )
        shred_seconds = time.perf_counter() - t0
        bundle_bytes = sum(
            f.stat().st_size for f in (work / "bundle").rglob("*") if f.is_file()
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    t0 = time.perf_counter()
    naive_bytes = _naive_zstd_bytes(shards, LEVEL_COLD)
    naive_seconds = time.perf_counter() - t0

    return {
        "corpus": {
            "gb": gb,
            "dim": _TRACE_STUB_DIM,
            "n_vectors": _TRACE_STUB_VECTORS,
            "seed": seed,
            "files": len(shards),
            "events": int(shred.events),
            "malformed": int(shred.malformed),
            "residual_lines": int(shred.residual_lines),
        },
        # File bytes are what ls shows (newlines included); payload bytes
        # match IngestStats.bytes_read. Ratios use file bytes so the
        # claim can be re-checked with du and zstd alone.
        "raw_file_bytes": int(raw_file_bytes),
        "raw_payload_bytes": int(shred.raw_bytes),
        "cold_bundle_bytes": int(bundle_bytes),
        "naive_zstd19_bytes": int(naive_bytes),
        "zstd_level": LEVEL_COLD,
        "ratio_vs_raw": raw_file_bytes / bundle_bytes,
        "ratio_vs_naive_zstd19": naive_bytes / bundle_bytes,
        "shred_seconds": float(shred_seconds),
        "naive_seconds": float(naive_seconds),
    }


def _measure_accel(kernel_n: int, seed: int) -> dict[str, Any]:
    """ACCEL_ACTIVE plus compiled-vs-fallback kernel wall times."""
    from density.engine._accel import ACCEL_ACTIVE
    from density.engine._accel.bench import measure_kernels

    return {
        "accel_active": bool(ACCEL_ACTIVE),
        "measure_kernels": measure_kernels(n=kernel_n, seed=seed),
    }


def run_bench(
    out_dir: str | Path,
    quick: bool = False,
    seed: int = 1337,
    corpus_cache: str | Path | None = None,
) -> Path:
    """Run the benchmark matrix and write one results JSON into out_dir.

    out_dir: results directory, created if missing; the file is named
    bench_quick.json or bench_full.json after the mode. quick shrinks
    the matrix (dim 64, 5k vectors, 200 queries, 0.01 GB traces, 200k
    kernel rows) to finish in under about two minutes; full is the
    standard corpus (dims 384 and 768, 100k vectors, 1k queries, 0.2 GB
    traces, 1M kernel rows). seed drives every stochastic step.
    corpus_cache: directory for the reusable synth trace corpus; None
    means a stable path under the system temp directory, which is what
    lets repeated full runs skip regeneration.

    Returns the path of the written JSON. Every number in it is measured
    in this process; only *_seconds fields and the machine section vary
    between identically seeded runs.
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise DensityError(f"bench out_dir is not a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cache_root = (
        Path(corpus_cache)
        if corpus_cache is not None
        else Path(tempfile.gettempdir()) / "density-bench-corpus"
    )

    dims = _QUICK_DIMS if quick else _FULL_DIMS
    n_vectors = _QUICK_N_VECTORS if quick else _FULL_N_VECTORS
    n_queries = _QUICK_N_QUERIES if quick else _FULL_N_QUERIES
    trace_gb = _QUICK_TRACE_GB if quick else _FULL_TRACE_GB
    kernel_n = _QUICK_KERNEL_N if quick else _FULL_KERNEL_N

    elapsed: dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    vectors = _measure_vectors(dims, n_vectors, n_queries, seed)
    elapsed["vectors"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    traces = _measure_traces(cache_root, trace_gb, seed)
    elapsed["traces"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    accel = _measure_accel(kernel_n, seed)
    elapsed["accel"] = time.perf_counter() - t0
    elapsed["total"] = time.perf_counter() - t_total

    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "quick" if quick else "full",
        "seed": seed,
        "accel_active": accel["accel_active"],
        "machine": _machine_info(),
        "versions": _library_versions(),
        "elapsed_seconds": elapsed,
        "vectors": vectors,
        "traces": traces,
        "accel": accel,
    }
    path = out / ("bench_quick.json" if quick else "bench_full.json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
