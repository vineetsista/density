"""Tests for rerankers and rerank_topk."""

from __future__ import annotations

import numpy as np

from density.engine.embed.rerank import ExactReranker, SQ8Reranker, rerank_topk
from density.engine.embed.sq8 import SQ8

SEED = 1337


def test_rerank_topk_exact_dot_ordering(rng: np.random.Generator) -> None:
    x = rng.normal(size=(60, 8)).astype(np.float32)
    q = rng.normal(size=8).astype(np.float32)
    candidates = rng.choice(60, size=25, replace=False).astype(np.int64)
    ids, scores = rerank_topk(q, candidates, ExactReranker(x), k=6)
    exact = (x[candidates] @ q).astype(np.float32)
    expected = candidates[np.argsort(-exact, kind="stable")][:6]
    assert ids.dtype == np.int64 and scores.dtype == np.float32
    assert np.array_equal(ids, expected)
    assert np.array_equal(scores, np.sort(exact)[::-1][:6])
    assert np.all(np.diff(scores) <= 0)


def test_rerank_topk_clamps_k_to_candidates(rng: np.random.Generator) -> None:
    x = rng.normal(size=(20, 4)).astype(np.float32)
    q = rng.normal(size=4).astype(np.float32)
    candidates = np.array([3, 7, 11], dtype=np.int64)
    ids, scores = rerank_topk(q, candidates, ExactReranker(x), k=10)
    assert ids.shape == (3,) and scores.shape == (3,)
    assert set(ids.tolist()) == {3, 7, 11}


def test_sq8_reranker_matches_decode_dot(rng: np.random.Generator) -> None:
    x = rng.normal(size=(80, 12)).astype(np.float32)
    codec = SQ8().fit(x, seed=SEED)
    codec.add(x)
    q = rng.normal(size=12).astype(np.float32)
    ids = np.array([0, 17, 42, 79], dtype=np.int64)
    got = SQ8Reranker(codec).score(q, ids)
    expected = codec.decode(codec.stored_codes()[ids]) @ q
    assert got.dtype == np.float32
    assert np.array_equal(got, expected.astype(np.float32))


def test_sq8_search_rerank_improves_over_raw(unit_vectors: np.ndarray) -> None:
    # The reranked result restricted to full depth must equal exact search,
    # so a rerank pass can never do worse than the raw sq8 ordering.
    x = unit_vectors[:300]
    codec = SQ8().fit(x, seed=SEED)
    codec.add(x)
    q = x[10]
    exact = x @ q
    gt = set(np.argsort(-exact, kind="stable")[:10].tolist())
    ids_rr, _ = codec.search(q, k=10, rerank_depth=300, reranker=ExactReranker(x))
    assert set(ids_rr[0].tolist()) == gt
    ids_raw, _ = codec.search(q, k=10)
    raw_hits = len(gt & set(ids_raw[0].tolist()))
    assert len(gt & set(ids_rr[0].tolist())) >= raw_hits


def test_sq8_search_with_sq8_reranker_scores_are_decoded(rng: np.random.Generator) -> None:
    x = rng.normal(size=(120, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    codec = SQ8().fit(x, seed=SEED)
    codec.add(x)
    q = x[5]
    ids, scores = codec.search(q, k=4, rerank_depth=30, reranker=SQ8Reranker(codec))
    decoded = codec.decode(codec.stored_codes()[ids[0]])
    assert np.allclose(scores[0], decoded @ q, rtol=1e-6, atol=1e-6)
