"""Phase 1 gate measurement: embedding engine numeric targets.

Standard corpus: 100k synth vectors of dim 768, 1k held-out queries,
seed 1337, laptop CPU, single process. Writes
benchmarks/results/phase1.json. Never edit the numbers by hand.

Gates measured here:
- sq8: 4.0x vs fp32 and recall@10 >= 0.99
- pq standalone ADC (default m = d/4): recall@10 >= 0.80
- pq with rerank 200 via sq8 at m = 128: recall@10 >= 0.95 and
  >= 16x total reduction including codebook overhead
- binary with rerank 500 via sq8: recall@10 >= 0.90 at 32x codes
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _machine import machine_info

ROOT = Path(__file__).resolve().parents[1]
SEED = 1337
DIM = 768
N_VECTORS = 100_000


def main() -> int:
    from density.bench.synth import generate
    from density.engine._accel import ACCEL_ACTIVE
    from density.engine.embed.binary import BinaryCodec
    from density.engine.embed.pq import PQ
    from density.engine.embed.rerank import SQ8Reranker
    from density.engine.embed.sq8 import SQ8
    from density.ingest.embeddings import read_embeddings
    from density.recall.verifier import verify_tiers

    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratch" / "phase1_corpus"
    if not (corpus / "embeddings" / "vectors.npy").exists():
        print(f"generating standard corpus at {corpus} ...")
        t0 = time.perf_counter()
        generate(0.33, corpus, seed=SEED, dim=DIM, n_vectors=N_VECTORS)
        print(f"  synth took {time.perf_counter() - t0:.1f}s")

    emb = read_embeddings(corpus / "embeddings")
    X = emb.X
    assert X.shape == (N_VECTORS, DIM), X.shape

    def reranker_factory(x_db):
        sq = SQ8().fit(x_db, seed=SEED)
        sq.add(x_db)
        return SQ8Reranker(sq)

    codecs = {
        "sq8": SQ8(),
        "pq_adc_m192": PQ(),          # default m = d / 4 = 192, no rerank
        "pq_rerank_m128": PQ(m=128),  # the 16x-including-codebook config
        "binary": BinaryCodec(),
    }
    rerank_depths = {"sq8": 0, "pq_adc_m192": 0, "pq_rerank_m128": 200, "binary": 500}

    t0 = time.perf_counter()
    result = verify_tiers(
        X,
        codecs,
        seed=SEED,
        n_queries=1000,
        sample_cap=200_000,
        rerank_depths=rerank_depths,
        reranker_factory=reranker_factory,
    )
    elapsed = time.perf_counter() - t0
    r = result.to_dict()

    raw_bytes_per_vec = DIM * 4

    def reduction(name: str) -> float:
        return raw_bytes_per_vec / r["codecs"][name]["bytes_per_vector"]

    gates = {
        "sq8_reduction_x": reduction("sq8"),
        "sq8_recall10": r["codecs"]["sq8"]["recall"]["10"],
        "pq_adc_recall10": r["codecs"]["pq_adc_m192"]["recall"]["10"],
        "pq_rerank_recall10": r["codecs"]["pq_rerank_m128"]["recall"]["10"],
        "pq_rerank_reduction_x": reduction("pq_rerank_m128"),
        "binary_recall10": r["codecs"]["binary"]["recall"]["10"],
        "binary_codes_reduction_x": raw_bytes_per_vec
        / (r["codecs"]["binary"]["bytes_per_vector_codes_only"]
           if "bytes_per_vector_codes_only" in r["codecs"]["binary"]
           else r["codecs"]["binary"]["bytes_per_vector"]),
    }
    passes = {
        "sq8_4x": gates["sq8_reduction_x"] >= 3.99,
        "sq8_recall10_ge_0.99": gates["sq8_recall10"] >= 0.99,
        "pq_adc_recall10_ge_0.80": gates["pq_adc_recall10"] >= 0.80,
        "pq_rerank_recall10_ge_0.95": gates["pq_rerank_recall10"] >= 0.95,
        "pq_rerank_ge_16x_incl_codebook": gates["pq_rerank_reduction_x"] >= 16.0,
        "binary_recall10_ge_0.90": gates["binary_recall10"] >= 0.90,
        "binary_32x_codes": gates["binary_codes_reduction_x"] >= 31.9,
    }

    payload = {
        "phase": 1,
        "seed": SEED,
        "corpus": {"n_vectors": N_VECTORS, "dim": DIM, "n_queries": 1000},
        "accel_active": ACCEL_ACTIVE,
        "machine": machine_info(),
        "verify_seconds": round(elapsed, 2),
        "verify": r,
        "gates": gates,
        "passes": passes,
        "all_pass": all(passes.values()),
    }
    out = ROOT / "benchmarks" / "results" / "phase1.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"gates": gates, "passes": passes}, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
