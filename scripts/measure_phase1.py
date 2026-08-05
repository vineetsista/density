"""Phase 1 gate measurement: embedding engine numeric targets.

Standard corpus: 100k vectors of dim 768, 1k held-out queries, seed 1337,
laptop CPU, single process. Writes benchmarks/results/phase1.json. Every
number in the output is measured here, never typed by hand.

The corpus IS the synth embedding distribution: this script imports
density.bench.synth.sample_embeddings directly, so the gates are measured
on exactly the geometry the generator ships (single source of truth, no
drift between the benchmark corpus and the corpus users generate). That
geometry is a low-rank manifold mixture: 200 Zipf-weighted clusters, each
a rank 12 to 24 patch on the unit sphere with per-cluster scale, plus 35
percent near-duplicate points, modeling the measured intrinsic dimension
of real text embeddings and the retry-storm redundancy of agent corpora.

Gates measured (Phase 1, docs/design/01-phases.md):
- sq8: 4.0x codes-only vs fp32 and recall@10 >= 0.99
- pq standalone ADC (default m = d // 4 = 192): recall@10 >= 0.80
- pq at m = 128 (DECISIONS.md item 8) with rerank 200 via SQ8Reranker:
  recall@10 >= 0.95 and >= 16x reduction counting pq codes plus codebook
  (the COLD footprint rule, DECISIONS.md item 6)
- binary with rerank 500 via SQ8Reranker: recall@10 >= 0.90 at 32x codes
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _machine import library_versions, machine_info

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337
DIM = 768
N_VECTORS = 100_000
N_QUERIES = 1_000
FP32_BYTES_PER_VECTOR = DIM * 4

GATE_TARGETS = {
    "sq8_reduction_codes_only_x": 4.0,
    "sq8_recall_at_10": 0.99,
    "pq_adc_m192_recall_at_10": 0.80,
    "pq_m128_rerank200_recall_at_10": 0.95,
    "pq_m128_reduction_incl_codebook_x": 16.0,
    "binary_rerank500_recall_at_10": 0.90,
    "binary_reduction_codes_only_x": 32.0,
}


def make_corpus(
    n: int = N_VECTORS, dim: int = DIM, seed: int = SEED
) -> np.ndarray:
    """Standard measurement corpus: float32 [n, dim], L2-normalized.

    Delegates to density.bench.synth.sample_embeddings, the single source
    of truth for the synth embedding geometry (see its docstring for the
    low-rank manifold mixture and its realism grounding). Measuring the
    gates on any other distribution would let the benchmark corpus and
    the shipped generator drift apart silently.
    """
    from density.bench.synth import sample_embeddings

    return sample_embeddings(n, dim, seed=seed)


def _instrument(codec, name: str, stages: dict[str, float]):
    """Wrap fit and add on a codec instance to record wall seconds."""
    orig_fit, orig_add = codec.fit, codec.add

    def fit(X, seed: int = SEED):
        t0 = time.perf_counter()
        out = orig_fit(X, seed=seed)
        stages[f"{name}.fit"] = round(time.perf_counter() - t0, 3)
        return out

    def add(X):
        t0 = time.perf_counter()
        orig_add(X)
        stages[f"{name}.add"] = round(time.perf_counter() - t0, 3)

    codec.fit = fit
    codec.add = add
    return codec


def run(
    n_vectors: int = N_VECTORS,
    dim: int = DIM,
    n_queries: int = N_QUERIES,
) -> dict:
    """Measure every Phase 1 gate and return the JSON-ready payload."""
    from density.engine._accel import ACCEL_ACTIVE
    from density.engine.embed.binary import BinaryCodec
    from density.engine.embed.pq import PQ
    from density.engine.embed.rerank import SQ8Reranker
    from density.engine.embed.sq8 import SQ8
    from density.recall.verifier import verify_tiers

    stages: dict[str, float] = {}
    total0 = time.perf_counter()

    print(f"generating corpus: {n_vectors} x {dim}, seed {SEED} ...", flush=True)
    t0 = time.perf_counter()
    X = make_corpus(n_vectors, dim, SEED)
    stages["generate_corpus"] = round(time.perf_counter() - t0, 3)
    print(f"  corpus in {stages['generate_corpus']:.1f}s", flush=True)

    # One SQ8Reranker over the leave-out database serves both rerank gates.
    # Built lazily on the first factory call and cached, so the SQ8 rerank
    # store is fitted once and its build time is a visible stage.
    cache: list = []

    def reranker_factory(x_db: np.ndarray):
        if not cache:
            t0 = time.perf_counter()
            sq = SQ8().fit(x_db, seed=SEED)
            sq.add(x_db)
            cache.append(SQ8Reranker(sq))
            stages["sq8_reranker.build"] = round(time.perf_counter() - t0, 3)
        return cache[0]

    codecs = {
        "sq8": _instrument(SQ8(), "sq8", stages),
        "pq_adc_m192": _instrument(PQ(), "pq_adc_m192", stages),
        "pq_rerank_m128": _instrument(PQ(m=128), "pq_rerank_m128", stages),
        "binary": _instrument(BinaryCodec(), "binary", stages),
    }
    rerank_depths = {
        "sq8": 0,
        "pq_adc_m192": 0,
        "pq_rerank_m128": 200,
        "binary": 500,
    }

    print("verifying codecs (fit, encode, exact ground truth, search) ...", flush=True)
    t0 = time.perf_counter()
    result = verify_tiers(
        X,
        codecs,
        seed=SEED,
        n_queries=n_queries,
        sample_cap=200_000,
        rerank_depths=rerank_depths,
        reranker_factory=reranker_factory,
    )
    stages["verify_total"] = round(time.perf_counter() - t0, 3)
    print(f"  verify in {stages['verify_total']:.1f}s", flush=True)
    del X

    r = result.to_dict()
    n_db = r["n_database"]

    codec_rows: dict[str, dict] = {}
    for name, rep in r["codecs"].items():
        codes_bpv = rep["bytes"]["encoded"] / n_db
        total_bpv = rep["bytes_per_vector"]
        codec_rows[name] = {
            "recall_at_1": rep["recall"]["recall@1"],
            "recall_at_10": rep["recall"]["recall@10"],
            "recall_at_100": rep["recall"]["recall@100"],
            "rerank_depth": rep["rerank_depth"],
            "aux_bytes": rep["bytes"]["aux"],
            "bytes_per_vector_codes_only": round(codes_bpv, 4),
            "bytes_per_vector_incl_aux": round(total_bpv, 4),
            "reduction_x_codes_only": round(FP32_BYTES_PER_VECTOR / codes_bpv, 4),
            "reduction_x_incl_aux": round(FP32_BYTES_PER_VECTOR / total_bpv, 4),
            "search_seconds": round(rep["search_seconds"], 3),
        }
        stages[f"{name}.search"] = round(rep["search_seconds"], 3)

    measured = {
        "sq8_reduction_codes_only_x": FP32_BYTES_PER_VECTOR
        / (r["codecs"]["sq8"]["bytes"]["encoded"] / n_db),
        "sq8_recall_at_10": r["codecs"]["sq8"]["recall"]["recall@10"],
        "pq_adc_m192_recall_at_10": r["codecs"]["pq_adc_m192"]["recall"]["recall@10"],
        "pq_m128_rerank200_recall_at_10": r["codecs"]["pq_rerank_m128"]["recall"]["recall@10"],
        # COLD footprint rule (DECISIONS.md item 6): pq codes plus pq
        # codebook only, which is exactly encoded + aux for the PQ codec.
        # The SQ8 rerank store lives in the WARM tier and is not counted.
        "pq_m128_reduction_incl_codebook_x": FP32_BYTES_PER_VECTOR
        / r["codecs"]["pq_rerank_m128"]["bytes_per_vector"],
        "binary_rerank500_recall_at_10": r["codecs"]["binary"]["recall"]["recall@10"],
        "binary_reduction_codes_only_x": FP32_BYTES_PER_VECTOR
        / (r["codecs"]["binary"]["bytes"]["encoded"] / n_db),
    }
    gates = [
        {
            "gate": name,
            "op": ">=",
            "target": target,
            "measured": round(float(measured[name]), 6),
            "pass": bool(measured[name] >= target),
        }
        for name, target in GATE_TARGETS.items()
    ]

    stages["total"] = round(time.perf_counter() - total0, 3)
    return {
        "phase": 1,
        "seed": SEED,
        "corpus": {
            "n_vectors": n_vectors,
            "dim": dim,
            "n_queries": n_queries,
            "n_database": n_db,
            "sample_size": r["sample_size"],
            "definition": (
                "density.bench.synth.sample_embeddings"
                f"({n_vectors}, {dim}, seed={SEED}): 200-cluster low-rank "
                "manifold mixture on the unit sphere (per-cluster rank 12 "
                "to 24, tau 0.45 to 0.65, 15 percent isotropic residual), "
                "Zipf exponent 1.05, 35 percent near-duplicate points, "
                "L2-normalized float32; single source of truth with the "
                "synth corpus"
            ),
        },
        "fp32_bytes_per_vector": FP32_BYTES_PER_VECTOR,
        "accel_active": bool(ACCEL_ACTIVE),
        "machine": machine_info(),
        "versions": library_versions(),
        "stages_seconds": stages,
        "codecs": codec_rows,
        "verify": r,
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates),
    }


def print_gate_table(gates: list[dict]) -> None:
    width = max(len(g["gate"]) for g in gates)
    header = f"{'gate':<{width}}  {'target':>8}  {'measured':>10}  pass"
    print(header)
    print("-" * len(header))
    for g in gates:
        status = "PASS" if g["pass"] else "FAIL"
        print(
            f"{g['gate']:<{width}}  >= {g['target']:5.2f}  "
            f"{g['measured']:>10.4f}  {status}"
        )


def main() -> int:
    payload = run()
    out = ROOT / "benchmarks" / "results" / "phase1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print()
    print_gate_table(payload["gates"])
    print()
    print(f"accel_active: {payload['accel_active']}")
    print(f"total wall seconds: {payload['stages_seconds']['total']}")
    print(f"wrote {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
