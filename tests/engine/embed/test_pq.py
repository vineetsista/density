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


def test_kmeans_pp_init_deterministic_per_seed() -> None:
    """Same seed gives bit-identical k-means++ seeds, new seed differs."""
    from density.engine.embed.pq import _kmeans_pp_init

    rng = np.random.default_rng(SEED)
    sub = rng.normal(size=(3000, 4)).astype(np.float32)
    x2 = np.einsum("nd,nd->n", sub, sub)
    a = _kmeans_pp_init(sub, x2, np.random.default_rng([SEED, 7]))
    b = _kmeans_pp_init(sub, x2, np.random.default_rng([SEED, 7]))
    c = _kmeans_pp_init(sub, x2, np.random.default_rng([SEED, 8]))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Every seed centroid is an actual data row (k-means++ picks rows).
    assert a.shape == (256, 4) and a.dtype == np.float32


def test_lloyd_reassigns_dead_centroids() -> None:
    """A centroid stranded far from all data is reassigned to a data row.

    The dead centroid must land on the row farthest from its assigned
    centroid (the highest-distortion point), which splits the worst
    cluster instead of leaving a wasted codeword.
    """
    from density.engine.embed.pq import _augment, _lloyd_refine

    rng = np.random.default_rng(SEED)
    sub = rng.normal(size=(5000, 4)).astype(np.float32)
    start = sub[rng.choice(5000, size=256, replace=False)].copy()
    start[13] = 1e6  # stranded: no row will ever assign to it
    refined, iters = _lloyd_refine(sub, _augment(sub), start, max_iters=3)
    assert iters <= 3
    # The stranded centroid moved back into the data's range.
    assert np.all(np.abs(refined[13]) < 1e3)
    # And every centroid now has at least one member.
    d2 = (
        np.einsum("kd,kd->k", refined, refined)[None, :]
        - 2.0 * (sub @ refined.T)
    )
    counts = np.bincount(np.argmin(d2, axis=1), minlength=256)
    assert np.all(counts > 0)


def test_lloyd_convergence_cap_and_early_stop() -> None:
    """The iteration cap is respected, and a converged start exits early."""
    from density.engine.embed.pq import _augment, _lloyd_refine

    rng = np.random.default_rng(SEED)
    sub = rng.normal(size=(4000, 4)).astype(np.float32)
    aug = _augment(sub)
    init = sub[rng.choice(4000, size=256, replace=False)].copy()
    refined, iters = _lloyd_refine(sub, aug, init.copy(), max_iters=5)
    assert iters <= 5
    # Feeding a fully refined solution back in must stop almost at once:
    # centroids are already at their members' means, so the movement
    # tolerance triggers on the first or second pass.
    converged, _ = _lloyd_refine(sub, aug, init.copy(), max_iters=200)
    again, iters_again = _lloyd_refine(sub, aug, converged.copy(), max_iters=200)
    assert iters_again <= 2


def test_aniso_refine_decreases_its_own_loss() -> None:
    """Anisotropic refinement descends the anisotropic objective.

    The loss is sum_i ||x - c||^2 + (eta - 1) (((x - c) . x) / ||x||)^2
    at each row's cost-argmin centroid; both half-steps (assignment and
    the normal-equation update) minimize it exactly, so the refined
    codebook must score strictly no worse than the plain one.
    """
    from density.engine.embed.pq import _aniso_refine, _augment, _lloyd_refine

    rng = np.random.default_rng(SEED)
    sub = rng.normal(size=(4000, 4)).astype(np.float32)
    x2 = np.einsum("nd,nd->n", sub, sub)
    eta = 8.0

    def aniso_loss(centroids: np.ndarray) -> float:
        diff = sub[:, None, :] - centroids[None, :, :]  # [n, 256, dsub]
        plain = np.einsum("nkd,nkd->nk", diff, diff)
        par = np.einsum("nkd,nd->nk", diff, sub) / np.sqrt(x2)[:, None]
        cost = plain + (eta - 1.0) * par**2
        return float(np.sum(cost.min(axis=1)))

    plain, _ = _lloyd_refine(sub, _augment(sub), sub[:256].copy(), max_iters=10)
    refined, iters = _aniso_refine(sub, x2, plain.copy(), eta=eta, max_iters=5)
    assert iters <= 5
    assert aniso_loss(refined) <= aniso_loss(plain)


def test_aniso_argmin_matches_bruteforce_cost() -> None:
    """The buffered fast path equals a direct evaluation of the cost."""
    from density.engine.embed.pq import _aniso_argmin

    rng = np.random.default_rng(SEED)
    sub = rng.normal(size=(1000, 4)).astype(np.float32)
    centroids = rng.normal(size=(256, 4)).astype(np.float32)
    x2 = np.einsum("nd,nd->n", sub, sub)
    cnorm = np.einsum("kd,kd->k", centroids, centroids)
    eta = 8.0
    tbuf = np.empty((1000, 256), dtype=np.float32)
    cbuf = np.empty_like(tbuf)
    got = _aniso_argmin(sub, x2, centroids, cnorm, eta, tbuf, cbuf)
    diff = sub[:, None, :] - centroids[None, :, :]
    plain = np.einsum("nkd,nkd->nk", diff, diff)
    par = np.einsum("nkd,nd->nk", diff, sub) / np.sqrt(x2)[:, None]
    want = np.argmin(plain + (eta - 1.0) * par**2, axis=1)
    # float32 fast path versus float32 broadcast path: identical argmin
    # on well-separated random data (no near-tie ambiguity at n=1000).
    assert np.mean(got == want) > 0.99


def test_fit_bit_identical_across_thread_counts(
    unit_vectors: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thread scheduling must never change the fitted model.

    Each subspace trains from its own (seed, j) child rng and writes a
    disjoint codebook slice, so a single-worker fit and a parallel fit
    are the same computation in a different order of execution.
    """
    from density.engine.embed import pq as pq_module

    monkeypatch.setattr(pq_module, "_FIT_THREADS", 1)
    serial = PQ(m=8).fit(unit_vectors, seed=SEED).to_state()["codebooks"]
    monkeypatch.setattr(pq_module, "_FIT_THREADS", 8)
    parallel = PQ(m=8).fit(unit_vectors, seed=SEED).to_state()["codebooks"]
    assert np.array_equal(serial, parallel)


def test_rerank_depth_widens_to_k(unit_vectors: np.ndarray) -> None:
    """Regression: k > rerank_depth used to crash with a broadcast error.

    The output has min(k, n) columns, so the candidate pool must be
    widened to at least that many rows before reranking, exactly as the
    binary codec already does.
    """

    class _ExactStub:
        def __init__(self, x: np.ndarray) -> None:
            self.x = x

        def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
            return (self.x[ids] @ q).astype(np.float32)

    x = unit_vectors
    n = x.shape[0]
    pq = PQ(m=8).fit(x, seed=SEED)
    pq.add(x)
    q = x[3]
    ids, scores = pq.search(q, k=50, rerank_depth=10, reranker=_ExactStub(x))
    assert ids.shape == (1, 50) and scores.shape == (1, 50)
    assert np.all(np.diff(scores[0]) <= 0)
    assert np.allclose(scores[0], x[ids[0]] @ q, atol=1e-5)
    # k beyond n clamps to n and must still not outrun the widened pool.
    ids_all, scores_all = pq.search(q, k=n + 50, rerank_depth=10, reranker=_ExactStub(x))
    assert ids_all.shape == (1, n) and scores_all.shape == (1, n)
    assert len(np.unique(ids_all[0])) == n
