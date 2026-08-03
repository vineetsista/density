"""Tests for the PQ codec: ADC quality, determinism, state round-trips.

All randomness is seeded. The toy structured corpus is exactly
representable by the codebooks (every subvector is drawn from a small
finite set), so ADC scores must track exact dot products very closely.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from density.engine.embed.pq import PQ

SEED = 1337


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without scipy. a, b: 1-D, equal length."""
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _structured(n: int = 400, d: int = 16, n_words: int = 24, seed: int = SEED) -> np.ndarray:
    """Toy corpus where each 4-dim subvector comes from a finite codeword set.

    Returns float32 [n, d]. Exactly representable by a 256-centroid PQ, so
    quantization error should be zero up to float arithmetic.
    """
    rng = np.random.default_rng(seed)
    m, dsub = d // 4, 4
    words = rng.normal(size=(m, n_words, dsub)).astype(np.float32)
    picks = rng.integers(0, n_words, size=(n, m))
    x = np.concatenate([words[j][picks[:, j]] for j in range(m)], axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


@functools.lru_cache(maxsize=1)
def _fitted_toy() -> tuple[PQ, np.ndarray]:
    """One shared fitted codec so the hypothesis test does not refit per example."""
    x = _structured()
    pq = PQ().fit(x, seed=SEED)
    pq.add(x)
    return pq, x


def test_m_must_divide_d(unit_vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        PQ(m=5).fit(unit_vectors)


def test_nbits_validated() -> None:
    with pytest.raises(ValueError):
        PQ(nbits=4)


def test_default_m_is_d_over_4(unit_vectors: np.ndarray) -> None:
    pq = PQ().fit(unit_vectors, seed=SEED)
    codes = pq.encode(unit_vectors[:10])
    assert codes.shape == (10, 32 // 4)
    assert codes.dtype == np.uint8


def test_adc_monotonic_with_exact_dot_pairwise() -> None:
    pq, x = _fitted_toy()
    rng = np.random.default_rng(SEED + 1)
    q = rng.normal(size=(12, x.shape[1])).astype(np.float32)
    n = x.shape[0]
    ids, scores = pq.search(q, k=n)
    approx = np.empty((12, n), dtype=np.float32)
    approx[np.arange(12)[:, None], ids] = scores
    exact = q @ x.T
    rho = _spearman(exact.ravel(), approx.ravel())
    assert rho > 0.95


@given(qseed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=25, deadline=None, derandomize=True)
def test_adc_monotonic_property_per_query(qseed: int) -> None:
    pq, x = _fitted_toy()
    q = np.random.default_rng(qseed).normal(size=x.shape[1]).astype(np.float32)
    n = x.shape[0]
    ids, scores = pq.search(q, k=n)
    approx = np.empty(n, dtype=np.float32)
    approx[ids[0]] = scores[0]
    assert _spearman(x @ q, approx) > 0.95


def test_recall_at_10_adc_only(unit_vectors: np.ndarray) -> None:
    x = unit_vectors
    pq = PQ(m=8).fit(x, seed=SEED)
    pq.add(x)
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(len(x), size=50, replace=False)
    q = x[qidx]
    gt = np.argsort(-(q @ x.T), axis=1, kind="stable")[:, :10]
    ids, _ = pq.search(q, k=10)
    hits = [len(np.intersect1d(gt[i], ids[i])) for i in range(len(q))]
    recall = float(np.mean(hits)) / 10.0
    assert recall >= 0.6


def test_codebook_determinism_across_fits(unit_vectors: np.ndarray) -> None:
    a = PQ(m=8).fit(unit_vectors, seed=SEED)
    b = PQ(m=8).fit(unit_vectors, seed=SEED)
    cb_a = a.to_state()["codebooks"]
    cb_b = b.to_state()["codebooks"]
    assert cb_a.dtype == np.float32
    # Bit-identical, not just close: same seed must give the same model.
    assert np.array_equal(cb_a, cb_b)
    assert np.array_equal(a.encode(unit_vectors), b.encode(unit_vectors))


def test_encode_determinism(unit_vectors: np.ndarray) -> None:
    pq = PQ(m=8).fit(unit_vectors, seed=SEED)
    c1 = pq.encode(unit_vectors)
    c2 = pq.encode(unit_vectors)
    assert c1.dtype == np.uint8
    assert c1.shape == (unit_vectors.shape[0], 8)
    assert np.array_equal(c1, c2)


def test_decode_shape_and_finite(unit_vectors: np.ndarray) -> None:
    pq = PQ(m=8).fit(unit_vectors, seed=SEED)
    dec = pq.decode(pq.encode(unit_vectors[:100]))
    assert dec.shape == (100, unit_vectors.shape[1])
    assert dec.dtype == np.float32
    assert np.isfinite(dec).all()


def test_empty_clusters_reseeded_and_encoding_still_works() -> None:
    # Only 10 distinct points but 256 centroids per subspace: most clusters
    # stay empty and must be reseeded without crashing the fit.
    rng = np.random.default_rng(SEED)
    base = rng.normal(size=(10, 8)).astype(np.float32)
    x = np.ascontiguousarray(base[rng.integers(0, 10, size=300)])
    pq = PQ(m=2).fit(x, seed=SEED)
    codes = pq.encode(x)
    assert codes.shape == (300, 2)
    dec = pq.decode(codes)
    assert np.isfinite(dec).all()
    # The corpus is exactly representable, so reconstruction is exact.
    assert np.allclose(dec, x, atol=1e-4)


def test_state_round_trip_preserves_search(unit_vectors: np.ndarray) -> None:
    pq = PQ(m=8).fit(unit_vectors, seed=SEED)
    pq.add(unit_vectors)
    q = unit_vectors[:5]
    ids1, s1 = pq.search(q, k=10)
    state = pq.to_state()
    assert all(isinstance(v, np.ndarray) for v in state.values())
    pq2 = PQ.from_state(state)
    ids2, s2 = pq2.search(q, k=10)
    assert np.array_equal(ids1, ids2)
    assert np.array_equal(s1, s2)


def test_bytes_accounting(unit_vectors: np.ndarray) -> None:
    d, m = unit_vectors.shape[1], 8
    pq = PQ(m=m).fit(unit_vectors, seed=SEED)
    pq.add(unit_vectors[:500])
    assert pq.encoded_nbytes() == 500 * m
    assert pq.aux_nbytes() == m * 256 * (d // m) * 4


def test_search_shapes_and_ordering(unit_vectors: np.ndarray) -> None:
    pq = PQ(m=8).fit(unit_vectors, seed=SEED)
    pq.add(unit_vectors)
    ids, scores = pq.search(unit_vectors[0], k=7)
    assert ids.shape == (1, 7) and scores.shape == (1, 7)
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    assert np.all(np.diff(scores[0]) <= 0)


def test_ids_index_most_recent_add(unit_vectors: np.ndarray) -> None:
    # Contract: ids are row indices into the X passed to the most recent add.
    pq = PQ(m=8).fit(unit_vectors, seed=SEED)
    pq.add(unit_vectors)
    pq.add(unit_vectors[:50])
    ids, _ = pq.search(unit_vectors[0], k=10)
    assert ids.max() < 50


def test_rerank_requires_reranker(unit_vectors: np.ndarray) -> None:
    pq = PQ(m=8).fit(unit_vectors, seed=SEED)
    pq.add(unit_vectors)
    with pytest.raises(ValueError):
        pq.search(unit_vectors[0], k=5, rerank_depth=20)


def test_rerank_path_uses_reranker_scores(unit_vectors: np.ndarray) -> None:
    pytest.importorskip("density.engine.embed.rerank")

    class _ExactStub:
        """Contract-shaped reranker over the raw float32 matrix."""

        def __init__(self, x: np.ndarray) -> None:
            self.x = x

        def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
            return (self.x[ids] @ q).astype(np.float32)

    x = unit_vectors
    pq = PQ(m=8).fit(x, seed=SEED)
    pq.add(x)
    q = x[3]
    ids, scores = pq.search(q, k=10, rerank_depth=200, reranker=_ExactStub(x))
    assert ids.shape == (1, 10) and scores.shape == (1, 10)
    assert np.all(np.diff(scores[0]) <= 0)
    # Returned scores must be the reranker's exact scores for the returned ids.
    assert np.allclose(scores[0], x[ids[0]] @ q, atol=1e-5)
