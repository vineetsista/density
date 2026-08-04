"""Recall referee: exact ground truth and per-codec verification.

Every recall number DENSITY reports flows through this module. It is the
trust anchor, so it follows referee rules strictly: it talks to codecs only
through their public protocol surface (fit, add, search, byte accounting),
it computes ground truth itself with exact float64 arithmetic, and its
queries are real vectors held out of the database by a seeded split, never
synthetic noise. A codec bug can therefore lower its score but never
inflate it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from density.engine.embed import Reranker, VectorCodec
from density.engine.embed.rerank import ExactReranker
from density.recall.metrics import BytesAccount, recall_at_k

__all__ = [
    "DEFAULT_RERANK_DEPTHS",
    "CodecReport",
    "VerifyResult",
    "exact_ground_truth",
    "verify_tiers",
]

# Contract defaults: pq and binary codes are coarse enough to need rerank,
# sq8 is accurate enough on its own that rerank would only add latency.
DEFAULT_RERANK_DEPTHS: dict[str, int] = {"pq": 200, "binary": 500, "sq8": 0}

# The referee always measures recall at these cutoffs, and ground truth is
# computed once at the largest of them.
_RECALL_KS: tuple[int, ...] = (1, 10, 100)
_GT_K: int = 100


def exact_ground_truth(
    X: np.ndarray,
    Q: np.ndarray,
    k: int = 100,
    batch: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k of Q against X by batched float64 dot products.

    X: float32 [n, d] L2-normalized corpus. Q: float32 [d] or [nq, d]
    queries. Vectors are normalized at ingest, so cosine similarity is a
    plain dot product. batch is the number of corpus rows scored per step:
    each step widens only that chunk to float64, so peak memory is bounded
    at nq x (k + batch) float64 scores plus one batch x d float64 chunk,
    no matter how large the corpus is.

    Returns (ids int64 [nq, k], scores float32 [nq, k]) sorted best-first.
    Ties are broken by lower row id so results are reproducible across
    machines and batch sizes.
    """
    X = np.asarray(X)
    Q64 = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    if X.ndim != 2 or Q64.ndim != 2:
        raise ValueError(f"expected 2-D corpus and queries, got {X.shape} and {Q64.shape}")
    n = int(X.shape[0])
    nq = int(Q64.shape[0])
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n:
        raise ValueError(f"k={k} exceeds corpus size {n}")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")

    top_scores = np.empty((nq, 0), dtype=np.float64)
    top_ids = np.empty((nq, 0), dtype=np.int64)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        # Widen one chunk at a time rather than the whole corpus up front:
        # float32 to float64 is exact elementwise, so results are bit
        # identical either way, but this keeps peak memory bounded by the
        # batch size instead of doubling the corpus footprint.
        chunk_scores = Q64 @ X[start:stop].astype(np.float64, copy=False).T
        chunk_ids = np.broadcast_to(
            np.arange(start, stop, dtype=np.int64), (nq, stop - start)
        )
        cand_scores = np.concatenate([top_scores, chunk_scores], axis=1)
        cand_ids = np.concatenate([top_ids, chunk_ids], axis=1)
        # Stable sort on negated scores breaks ties by lower id: within the
        # running top-k, tied entries are already id-ordered from previous
        # merges, every retained id precedes every id in the new chunk, and
        # chunk columns are id-ascending by construction. The invariant is
        # therefore preserved across merges, which makes the result
        # independent of the batch size.
        order = np.argsort(-cand_scores, axis=1, kind="stable")[:, :k]
        top_scores = np.take_along_axis(cand_scores, order, axis=1)
        top_ids = np.take_along_axis(cand_ids, order, axis=1)
    return top_ids, top_scores.astype(np.float32)


def _split_indices(
    n: int,
    seed: int,
    n_queries: int,
    sample_cap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Seeded leave-out split of row indices 0..n-1.

    Returns (database_indices int64, query_indices int64, sample_size).
    When n exceeds sample_cap, a seeded sample of sample_cap rows forms the
    evaluation pool and sample_size records that shrinkage; otherwise the
    pool is every row. Queries are then left out of the pool, so they are
    always real corpus vectors and never appear in the database.
    """
    if n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")
    sample_size = int(min(n, sample_cap))
    if n_queries >= sample_size:
        raise ValueError(
            f"n_queries={n_queries} leaves no database rows out of a pool of {sample_size}"
        )
    # One permutation serves both decisions: the first sample_size entries
    # are the seeded corpus sample, and its first n_queries entries are the
    # held-out queries. Same seed, same machine, same split.
    pool = np.random.default_rng(seed).permutation(n)[:sample_size].astype(np.int64)
    query_idx = pool[:n_queries]
    db_idx = pool[n_queries:]
    return db_idx, query_idx, sample_size


@dataclass
class CodecReport:
    """Verification outcome for one codec.

    recall maps "recall@1" / "recall@10" / "recall@100" to fractions in
    [0, 1]. bytes_per_vector is (encoded + aux) / n_database, in bytes.
    search_seconds is wall time of the search call only, the single
    nondeterministic field in a report.
    """

    name: str
    recall: dict[str, float]
    rerank_depth: int
    bytes: BytesAccount
    bytes_per_vector: float
    search_seconds: float

    def to_dict(self) -> dict:
        """Plain-Python dict, safe for json.dumps."""
        return {
            "name": str(self.name),
            "recall": {key: float(value) for key, value in self.recall.items()},
            "rerank_depth": int(self.rerank_depth),
            "bytes": {
                "raw": int(self.bytes.raw),
                "encoded": int(self.bytes.encoded),
                "aux": int(self.bytes.aux),
                "total": int(self.bytes.total),
            },
            "bytes_per_vector": float(self.bytes_per_vector),
            "search_seconds": float(self.search_seconds),
        }


@dataclass
class VerifyResult:
    """Full verification outcome across codecs.

    n_vectors is the corpus size as given, sample_size the evaluated pool
    after the seeded cap, n_queries the held-out query count, n_database
    the rows actually indexed (sample_size minus n_queries), k the ground
    truth depth.
    """

    codecs: dict[str, CodecReport]
    n_vectors: int
    sample_size: int
    n_queries: int
    n_database: int
    k: int
    seed: int

    def to_dict(self) -> dict:
        """Plain-Python dict, safe for json.dumps: no numpy scalars."""
        return {
            "n_vectors": int(self.n_vectors),
            "sample_size": int(self.sample_size),
            "n_queries": int(self.n_queries),
            "n_database": int(self.n_database),
            "k": int(self.k),
            "seed": int(self.seed),
            "codecs": {name: report.to_dict() for name, report in self.codecs.items()},
        }


def verify_tiers(
    X: np.ndarray,
    codecs: Mapping[str, VectorCodec],
    seed: int = 1337,
    n_queries: int = 1000,
    sample_cap: int = 200_000,
    rerank_depths: Mapping[str, int] | None = None,
    reranker: Reranker | None = None,
    reranker_factory: Callable[[np.ndarray], Reranker] | None = None,
) -> VerifyResult:
    """Measures recall and footprint of each codec against exact ground truth.

    X: float32 [n, d] L2-normalized corpus. codecs maps codec name to an
    unfitted VectorCodec. The corpus is capped to sample_cap rows by a
    seeded sample when larger, n_queries real vectors are held out as
    queries, and each codec is fitted on the remaining database rows only,
    so queries never leak into training.

    rerank_depths overrides DEFAULT_RERANK_DEPTHS per codec name; depths
    are clamped to the database size. When a depth is positive the reranker
    resolves in order: the explicit reranker argument, reranker_factory
    called with the database vectors, and finally an ExactReranker over the
    database vectors (the honest fallback: rescoring with true float32).

    Returns a VerifyResult with recall@1/10/100, bytes per vector, and the
    rerank depth per codec. Only search_seconds varies between runs with
    the same seed.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [n, d], got shape {X.shape}")
    n_vectors = int(X.shape[0])
    db_idx, query_idx, sample_size = _split_indices(
        n_vectors, seed=seed, n_queries=n_queries, sample_cap=sample_cap
    )
    X_db = X[db_idx]
    Q = X[query_idx]
    n_database = int(db_idx.size)
    # Recall@100 is undefined on fewer than 100 database rows. Refuse here
    # with the caller's vocabulary (n_database, the query split) instead of
    # letting exact_ground_truth fail deeper in with "corpus size", which
    # would misname the leave-out database and hide the split arithmetic.
    if n_database < _GT_K:
        raise ValueError(
            f"verify_tiers measures recall@{_GT_K}, which needs at least "
            f"{_GT_K} database rows after the query split; got "
            f"n_database={n_database} (sample_size={sample_size} minus "
            f"n_queries={n_queries}). Provide more vectors or fewer queries."
        )

    gt_ids, _ = exact_ground_truth(X_db, Q, k=_GT_K)

    depths = dict(DEFAULT_RERANK_DEPTHS)
    if rerank_depths:
        depths.update(rerank_depths)

    reports: dict[str, CodecReport] = {}
    for name, codec in codecs.items():
        fitted = codec.fit(X_db, seed=seed)
        fitted.add(X_db)
        # Depth beyond the database is meaningless: clamp so the recorded
        # number states what was actually rescored.
        depth = max(0, min(int(depths.get(name, 0)), n_database))
        active_reranker: Reranker | None = None
        if depth > 0:
            if reranker is not None:
                active_reranker = reranker
            elif reranker_factory is not None:
                active_reranker = reranker_factory(X_db)
            else:
                active_reranker = ExactReranker(X_db)
        started = time.perf_counter()
        pred_ids, _scores = fitted.search(
            Q, k=_GT_K, rerank_depth=depth, reranker=active_reranker
        )
        elapsed = time.perf_counter() - started
        recall = {f"recall@{kk}": recall_at_k(gt_ids, pred_ids, kk) for kk in _RECALL_KS}
        encoded = int(fitted.encoded_nbytes())
        aux = int(fitted.aux_nbytes())
        reports[name] = CodecReport(
            name=str(name),
            recall=recall,
            rerank_depth=depth,
            bytes=BytesAccount(raw=int(X_db.nbytes), encoded=encoded, aux=aux),
            bytes_per_vector=float((encoded + aux) / n_database),
            search_seconds=float(elapsed),
        )
    return VerifyResult(
        codecs=reports,
        n_vectors=n_vectors,
        sample_size=int(sample_size),
        n_queries=int(query_idx.size),
        n_database=n_database,
        k=_GT_K,
        seed=int(seed),
    )
