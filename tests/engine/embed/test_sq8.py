"""Tests for the SQ8 codec and the shared topk helper."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from density.engine.embed import topk
from density.engine.embed.rerank import ExactReranker
from density.engine.embed.sq8 import SQ8

SEED = 1337


def _fitted(x: np.ndarray) -> SQ8:
    codec = SQ8().fit(x, seed=SEED)
    codec.add(x)
    return codec


@settings(deadline=None, max_examples=60)
@given(
    x=hnp.arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(2, 24), st.integers(1, 12)),
        elements=st.floats(
            -1e4, 1e4, allow_nan=False, allow_infinity=False, width=32
        ),
    )
)
def test_reconstruction_within_half_step(x: np.ndarray) -> None:
    codec = SQ8().fit(x, seed=SEED)
    rec = codec.decode(codec.encode(x)).astype(np.float64)
    mins = x.min(axis=0).astype(np.float64)
    maxs = x.max(axis=0).astype(np.float64)
    span = maxs - mins
    # Half a quantization step per dim, plus tiny float fuzz for the
    # float32 scale and the final float32 cast of the reconstruction.
    tol = span / 255.0 / 2.0 + span * 1e-5 + (np.abs(mins) + np.abs(maxs)) * 1e-6 + 1e-7
    err = np.abs(rec - x.astype(np.float64))
    assert np.all(err <= tol[None, :])


def test_zero_range_dims_decode_to_constant(rng: np.random.Generator) -> None:
    x = rng.normal(size=(50, 6)).astype(np.float32)
    x[:, 2] = 3.25
    x[:, 5] = -1.5
    codec = SQ8().fit(x, seed=SEED)
    rec = codec.decode(codec.encode(x))
    assert np.all(rec[:, 2] == np.float32(3.25))
    assert np.all(rec[:, 5] == np.float32(-1.5))


def test_codes_are_uint8_and_4x_smaller_at_dim_768(rng: np.random.Generator) -> None:
    n, d = 128, 768
    x = rng.normal(size=(n, d)).astype(np.float32)
    codec = _fitted(x)
    codes = codec.encode(x)
    assert codes.dtype == np.uint8 and codes.shape == (n, d)
    fp32_bytes = n * d * 4
    assert fp32_bytes / codec.encoded_nbytes() == 4.0
    assert codec.aux_nbytes() == 2 * d * 4


def test_recall_at_10_vs_exact_on_unit_vectors(unit_vectors: np.ndarray) -> None:
    x = unit_vectors
    codec = _fitted(x)
    n_queries, k = 100, 10
    queries = x[:n_queries]
    ids, _ = codec.search(queries, k=k)
    exact = x @ queries.T
    hits = 0
    for i in range(n_queries):
        gt = set(topk(exact[:, i], k).tolist())
        hits += len(gt & set(ids[i].tolist()))
    recall = hits / (n_queries * k)
    # 0.95 gate at this 2000-vector scale; the 0.99 tier guarantee is
    # measured by the verifier at 100k scale, not claimed here.
    assert recall >= 0.95


def test_search_shapes_and_single_query(unit_vectors: np.ndarray) -> None:
    codec = _fitted(unit_vectors)
    ids, scores = codec.search(unit_vectors[0], k=7)
    assert ids.shape == (1, 7) and ids.dtype == np.int64
    assert scores.shape == (1, 7) and scores.dtype == np.float32
    assert np.all(np.diff(scores[0]) <= 0)
    # A unit vector is its own nearest neighbor.
    assert ids[0, 0] == 0


def test_add_appends_and_ids_continue(rng: np.random.Generator) -> None:
    x = rng.normal(size=(40, 8)).astype(np.float32)
    codec = SQ8().fit(x, seed=SEED)
    codec.add(x[:30])
    codec.add(x[30:])
    assert codec.stored_codes().shape == (40, 8)
    assert codec.encoded_nbytes() == 40 * 8
    ids, _ = codec.search(x[35], k=1)
    assert ids[0, 0] == 35


def test_determinism_bit_identical(rng: np.random.Generator) -> None:
    x = rng.normal(size=(300, 16)).astype(np.float32)
    q = x[:5]
    runs = []
    for _ in range(2):
        codec = SQ8().fit(x, seed=SEED)
        codec.add(x)
        ids, scores = codec.search(q, k=10)
        runs.append((codec.encode(x), ids, scores))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])
    assert np.array_equal(runs[0][2], runs[1][2])


def test_to_state_round_trip(unit_vectors: np.ndarray) -> None:
    codec = _fitted(unit_vectors)
    state = codec.to_state()
    assert set(state) == {"mins", "scale", "codes"}
    clone = SQ8.from_state(state)
    assert np.array_equal(codec.stored_codes(), clone.stored_codes())
    assert codec.encoded_nbytes() == clone.encoded_nbytes()
    assert codec.aux_nbytes() == clone.aux_nbytes()
    ids_a, scores_a = codec.search(unit_vectors[:8], k=10)
    ids_b, scores_b = clone.search(unit_vectors[:8], k=10)
    assert np.array_equal(ids_a, ids_b)
    assert np.array_equal(scores_a, scores_b)


def test_from_state_rejects_missing_key(unit_vectors: np.ndarray) -> None:
    state = _fitted(unit_vectors).to_state()
    del state["scale"]
    with pytest.raises(ValueError):
        SQ8.from_state(state)


def test_rerank_depth_requires_reranker(unit_vectors: np.ndarray) -> None:
    codec = _fitted(unit_vectors)
    with pytest.raises(ValueError):
        codec.search(unit_vectors[0], k=5, rerank_depth=50)


def test_unfitted_and_empty_store_raise(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        SQ8().encode(rng.normal(size=(4, 4)).astype(np.float32))
    codec = SQ8().fit(rng.normal(size=(10, 4)).astype(np.float32), seed=SEED)
    with pytest.raises(ValueError):
        codec.search(np.zeros(4, dtype=np.float32), k=3)


def test_search_with_rerank_matches_exact_topk(unit_vectors: np.ndarray) -> None:
    x = unit_vectors[:200]
    codec = SQ8().fit(x, seed=SEED)
    codec.add(x)
    q = x[3]
    # Rerank depth covering the whole store makes the result exact.
    ids, scores = codec.search(q, k=5, rerank_depth=200, reranker=ExactReranker(x))
    exact = x @ q
    expected = topk(exact, 5)
    assert np.array_equal(ids[0], expected)
    assert np.allclose(scores[0], exact[expected], rtol=1e-6, atol=1e-6)


def test_topk_helper() -> None:
    scores = np.array([0.1, 5.0, -2.0, 3.0, 3.0], dtype=np.float32)
    assert topk(scores, 3).tolist() == [1, 3, 4]
    assert topk(scores, 99).tolist() == [1, 3, 4, 0, 2]
    assert topk(scores, 0).tolist() == []
