"""Phase 0 gate measurement: the corpus generator's own guarantees.

Three gates from docs/design/01-phases.md:

- `density synth --gb 0.05` completes in under 30 seconds.
- Same seed, same bytes: two generations into different directories
  produce byte-identical outputs, compared by sha256 per file.
- The generated corpus lands within 5 percent of its byte target.

Two further properties the plan calls for are measured alongside the
gates because they are what makes the corpus worth benchmarking on: the
malformed line rate (about 2 percent by design) and the embedding cluster
structure (nearest-centroid accuracy far above chance).

Writes benchmarks/results/phase0.json. Every number is measured here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _machine import library_versions, machine_info

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"

SEED = 1337
DIM = 768
GB = 0.05
SYNTH_BUDGET_S = 30.0
BYTE_TOLERANCE_PCT = 5.0
# Nearest-centroid accuracy is compared against chance, which is 1 / the
# cluster count the generator uses.
CHANCE_FACTOR = 10.0


def _tree_digest(root: Path) -> str:
    """One sha256 over every file under root, keyed by sorted relpath."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "little"))
        h.update(rel)
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def _malformed_rate(corpus: Path) -> tuple[float, int, int]:
    """Measured malformed fraction over every generated line."""
    from density.ingest.traces import iter_events

    events = malformed = 0
    for parsed in iter_events(corpus / "traces"):
        if parsed.event is None:
            malformed += 1
        else:
            events += 1
    total = events + malformed
    return (malformed / total if total else 0.0), malformed, total


def _cluster_separation(corpus: Path) -> tuple[float, float]:
    """Nearest-centroid accuracy of the generated embeddings vs chance.

    The generator does not ship labels, so the clusters are recovered:
    seeded k-means on a subsample gives centroids, and accuracy is the
    fraction of held-out rows whose nearest centroid is also their nearest
    neighbor's. A structureless (isotropic) cloud scores at chance.
    """
    X = np.load(corpus / "embeddings" / "vectors.npy")
    rng = np.random.default_rng(SEED)
    n_clusters = 200
    sample = X[rng.choice(X.shape[0], size=min(4000, X.shape[0]), replace=False)]
    centroids = sample[rng.choice(sample.shape[0], size=min(n_clusters, sample.shape[0]), replace=False)]
    for _ in range(10):
        assign = np.argmax(sample @ centroids.T, axis=1)
        for c in range(centroids.shape[0]):
            members = sample[assign == c]
            if members.shape[0]:
                v = members.mean(axis=0)
                norm = np.linalg.norm(v)
                if norm > 0:
                    centroids[c] = v / norm
    probe = X[rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)]
    own = np.argmax(probe @ centroids.T, axis=1)
    # A structured corpus puts a probe and its nearest other probe in the
    # same recovered cluster; an isotropic one does so at chance.
    sims = probe @ probe.T
    np.fill_diagonal(sims, -np.inf)
    neighbor = np.argmax(sims, axis=1)
    accuracy = float(np.mean(own == own[neighbor]))
    chance = 1.0 / centroids.shape[0]
    return accuracy, chance


def main() -> None:
    from density.bench.synth import generate

    work = Path(tempfile.mkdtemp(prefix="density-phase0-"))
    try:
        first = work / "a"
        second = work / "b"
        t0 = time.perf_counter()
        stats = generate(GB, first, seed=SEED, dim=DIM)
        synth_seconds = time.perf_counter() - t0
        generate(GB, second, seed=SEED, dim=DIM)

        digest_a = _tree_digest(first)
        digest_b = _tree_digest(second)
        target_bytes = int(GB * 1_000_000_000)
        actual_bytes = int(stats.bytes_total)
        drift_pct = abs(actual_bytes - target_bytes) / target_bytes * 100.0
        malformed_rate, malformed, total_lines = _malformed_rate(first)
        accuracy, chance = _cluster_separation(first)

        doc = {
            "phase": 0,
            "seed": SEED,
            "dim": DIM,
            "gb": GB,
            "machine": machine_info(),
            "versions": library_versions(),
            "synth_seconds": round(synth_seconds, 3),
            "bytes_target": target_bytes,
            "bytes_total": actual_bytes,
            "bytes_traces": int(stats.bytes_traces),
            "bytes_embeddings": int(stats.bytes_embeddings),
            "byte_drift_pct": round(drift_pct, 4),
            "sha256_run_a": digest_a,
            "sha256_run_b": digest_b,
            "malformed_rate": round(malformed_rate, 5),
            "malformed_lines": malformed,
            "total_lines": total_lines,
            "cluster_nearest_centroid_accuracy": round(accuracy, 4),
            "cluster_chance_accuracy": round(chance, 6),
            "gates": [],
        }
        doc["gates"] = [
            {
                "gate": "synth_0_05gb_under_30s",
                "target": SYNTH_BUDGET_S,
                "measured": round(synth_seconds, 2),
                "op": "<",
                "pass": synth_seconds < SYNTH_BUDGET_S,
            },
            {
                "gate": "double_run_sha256_identical",
                "target": True,
                "measured": digest_a == digest_b,
                "op": "==",
                "pass": digest_a == digest_b,
            },
            {
                "gate": "byte_target_within_5pct",
                "target": BYTE_TOLERANCE_PCT,
                "measured": round(drift_pct, 4),
                "op": "<=",
                "pass": drift_pct <= BYTE_TOLERANCE_PCT,
            },
            {
                "gate": "embedding_clusters_far_above_chance",
                "target": round(chance * CHANCE_FACTOR, 6),
                "measured": round(accuracy, 4),
                "op": ">=",
                "pass": accuracy >= chance * CHANCE_FACTOR,
            },
        ]
        doc["all_pass"] = all(g["pass"] for g in doc["gates"])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "phase0.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(doc["gates"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
