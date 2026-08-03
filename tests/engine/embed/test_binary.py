"""Tests for the binary sign codec (engine/embed/binary.py).

rerank.py is owned by another agent and may be absent while this suite
runs, so a stub module matching the contract signature of rerank_topk is
installed into sys.modules for the rerank tests. The codec imports rerank
lazily inside search, which is exactly what makes the stub injectable.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from density.engine._accel import hamming_scan
from density.engine.embed.binary import BinaryCodec

SEED = 1337


class ExactRerankerStub:
    """Exact dot-product reranker over a float32 matrix.

    Matches the Reranker protocol: score(q [d], ids int64 [c]) -> float32 [c].
    """

    def __init__(self, X: np.ndarray) -> None:
        self._X = np.asarray(X, dtype=np.float32)

    def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        return (self._X[ids] @ np.asarray(q, dtype=np.float32)).astype(np.float32)


def _stub_rerank_topk(
    q: np.ndarray, candidate_ids: np.ndarray, reranker: ExactRerankerStub, k: int
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(reranker.score(q, candidate_ids), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[: int(k)]
    return candidate_ids[order].astype(np.int64), scores[order]


@pytest.fixture()
def rerank_stub(monkeypatch):
    # Installed unconditionally so this suite never depends on the real
    # rerank module, which a concurrent agent may still be writing.
    mod = types.ModuleType("density.engine.embed.rerank")
    mod.rerank_topk = _stub_rerank_topk
    monkeypatch.setitem(sys.modules, "density.engine.embed.rerank", mod)
    return mod


_byte_rows = st.integers(1, 16).flatmap(
    lambda b: st.tuples(
        st.lists(st.integers(0, 255), min_size=b, max_size=b),
        st.lists(st.integers(0, 255), min_size=b, max_size=b),
        st.lists(st.integers(0, 255), min_size=b, max_size=b),
    )
)


@given(_byte_rows)
def test_hamming_symmetry(triple):
    a = np.asarray(triple[0], dtype=np.uint8)
    b = np.asarray(triple[1], dtype=np.uint8)
    assert hamming_scan(a[None, :], b)[0] == hamming_scan(b[None, :], a)[0]


@given(_byte_rows)
def test_hamming_triangle_sanity(triple):
    a = np.asarray(triple[0], dtype=np.uint8)
    b = np.asarray(triple[1], dtype=np.uint8)
    c = np.asarray(triple[2], dtype=np.uint8)
    d_ab = hamming_scan(a[None, :], b)[0]
    d_bc = hamming_scan(b[None, :], c)[0]
    d_ac = hamming_scan(a[None, :], c)[0]
    assert d_ac <= d_ab + d_bc


@given(st.lists(st.integers(0, 255), min_size=1, max_size=32))
def test_hamming_distance_to_self_is_zero(vals):
    a = np.asarray(vals, dtype=np.uint8)
    assert hamming_scan(a[None, :], a)[0] == 0


def _reference_pack(X: np.ndarray) -> np.ndarray:
    """Bit-by-bit packing oracle: bit j of row i set iff X[i, j] > 0.

    Big-endian bit order within each byte, matching numpy packbits.
    """
    n, d = X.shape
    out = np.zeros((n, d // 8), dtype=np.uint8)
    for i in range(n):
        for j in range(d):
            if X[i, j] > 0:
                out[i, j // 8] |= np.uint8(1 << (7 - (j % 8)))
    return out


def test_encode_matches_unpacked_reference(rng):
    x = rng.normal(size=(40, 64)).astype(np.float32)
    # Zeros must pack as 0 bits: the contract is sign(x) > 0, strictly.
    x[0, :8] = 0.0
    codec = BinaryCodec().fit(x, seed=SEED)
    codes = codec.encode(x)
    assert codes.dtype == np.uint8
    assert codes.shape == (40, 8)
    np.testing.assert_array_equal(codes, _reference_pack(x))


def test_decode_unit_norm_and_signs(rng):
    x = rng.normal(size=(20, 32)).astype(np.float32)
    codec = BinaryCodec().fit(x, seed=SEED)
    dec = codec.decode(codec.encode(x))
    assert dec.dtype == np.float32
    assert dec.shape == x.shape
    np.testing.assert_allclose(np.linalg.norm(dec, axis=1), 1.0, atol=1e-5)
    # Positive inputs decode positive, everything else decodes negative.
    np.testing.assert_array_equal(dec > 0, x > 0)


def test_dim_not_divisible_by_8_rejected(rng):
    x = rng.normal(size=(4, 30)).astype(np.float32)
    with pytest.raises(ValueError):
        BinaryCodec().fit(x)
    with pytest.raises(ValueError):
        BinaryCodec().encode(x)


def test_bytes_ratio_is_exactly_32x_at_dim_768(rng):
    x = rng.normal(size=(100, 768)).astype(np.float32)
    codec = BinaryCodec().fit(x, seed=SEED)
    codec.add(x)
    assert codec.encoded_nbytes() * 32 == x.nbytes
    assert codec.aux_nbytes() == 0


def test_search_shapes_order_and_self_match(unit_vectors):
    codec = BinaryCodec().fit(unit_vectors, seed=SEED)
    codec.add(unit_vectors)
    ids, scores = codec.search(unit_vectors[0], k=10)
    assert ids.shape == (1, 10)
    assert scores.shape == (1, 10)
    assert ids.dtype == np.int64
    assert scores.dtype == np.float32
    # Scores are negative hamming, sorted best-first.
    assert np.all(np.diff(scores[0]) <= 0)
    assert ids[0, 0] == 0
    assert scores[0, 0] == 0.0


def test_rerank_depth_requires_reranker(unit_vectors):
    codec = BinaryCodec().fit(unit_vectors, seed=SEED)
    codec.add(unit_vectors)
    with pytest.raises(ValueError):
        codec.search(unit_vectors[0], k=5, rerank_depth=50)


def test_search_is_bit_identical_across_runs(unit_vectors):
    q = unit_vectors[:5]
    runs = []
    for _ in range(2):
        codec = BinaryCodec().fit(unit_vectors, seed=SEED)
        codec.add(unit_vectors)
        runs.append(codec.search(q, k=10))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_recall_at_10_with_rerank_depth_100(unit_vectors, rerank_stub):
    x = unit_vectors
    codec = BinaryCodec().fit(x, seed=SEED)
    codec.add(x)
    nq, k = 100, 10
    q = x[:nq]
    gt = np.argsort(-(q @ x.T), axis=1, kind="stable")[:, :k]
    ids, scores = codec.search(q, k=k, rerank_depth=100, reranker=ExactRerankerStub(x))
    assert ids.shape == (nq, k)
    # Reranked scores are exact dot products, so they must be best-first too.
    assert np.all(np.diff(scores, axis=1) <= 0)
    hits = sum(len(set(gt[i]) & set(ids[i])) for i in range(nq))
    recall = hits / (nq * k)
    assert recall >= 0.8


def test_to_state_round_trip_preserves_results(unit_vectors):
    codec = BinaryCodec().fit(unit_vectors, seed=SEED)
    codec.add(unit_vectors)
    q = unit_vectors[:7]
    ids0, scores0 = codec.search(q, k=10)

    state = codec.to_state()
    assert all(isinstance(v, np.ndarray) for v in state.values())
    restored = BinaryCodec.from_state(state)
    assert restored.encoded_nbytes() == codec.encoded_nbytes()

    ids1, scores1 = restored.search(q, k=10)
    np.testing.assert_array_equal(ids0, ids1)
    np.testing.assert_array_equal(scores0, scores1)
