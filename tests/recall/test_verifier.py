"""Tests for the recall referee: exact ground truth and tier verification."""

from __future__ import annotations

import json

import numpy as np
import pytest

from density.recall.verifier import (
    DEFAULT_RERANK_DEPTHS,
    VerifyResult,
    _split_indices,
    exact_ground_truth,
    verify_tiers,
)

SEED = 1337


def _oracle(X: np.ndarray, Q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force reference: full float64 matmul, ties broken by lower id."""
    s = np.asarray(Q, dtype=np.float64) @ np.asarray(X, dtype=np.float64).T
    n = X.shape[0]
    row_ids = np.arange(n, dtype=np.int64)
    ids = np.empty((s.shape[0], k), dtype=np.int64)
    scores = np.empty((s.shape[0], k), dtype=np.float64)
    for i in range(s.shape[0]):
        order = np.lexsort((row_ids, -s[i]))[:k]
        ids[i] = order
        scores[i] = s[i, order]
    return ids, scores.astype(np.float32)


class _ExactStub:
    """Codec stub that stores float32 verbatim and searches exactly.

    Scoring matches the referee arithmetic (float64 dot, stable ordering)
    so recall against exact ground truth is 1.0 by construction.
    """

    name = "stub"

    def __init__(self) -> None:
        self._X: np.ndarray | None = None
        self.fit_seed: int | None = None

    def fit(self, X: np.ndarray, seed: int = 1337) -> "_ExactStub":
        self.fit_seed = seed
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.float32)

    def add(self, X: np.ndarray) -> None:
        self._X = np.asarray(X, dtype=np.float32)

    def _candidate_scores(self, Q: np.ndarray) -> np.ndarray:
        assert self._X is not None
        return np.asarray(Q, dtype=np.float64) @ self._X.astype(np.float64).T

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        rerank_depth: int = 0,
        reranker: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        Q = np.atleast_2d(q)
        s = self._candidate_scores(Q)
        if rerank_depth > 0 and reranker is not None:
            depth = min(rerank_depth, s.shape[1])
            cand = np.argsort(-s, axis=1, kind="stable")[:, :depth]
            ids = np.empty((Q.shape[0], k), dtype=np.int64)
            scores = np.empty((Q.shape[0], k), dtype=np.float32)
            for i in range(Q.shape[0]):
                rescored = np.asarray(
                    reranker.score(np.asarray(Q[i], dtype=np.float32), cand[i]),
                    dtype=np.float64,
                )
                order = np.argsort(-rescored, kind="stable")[:k]
                ids[i] = cand[i][order]
                scores[i] = rescored[order].astype(np.float32)
            return ids, scores
        order = np.argsort(-s, axis=1, kind="stable")[:, :k]
        ids = order.astype(np.int64)
        scores = np.take_along_axis(s, order, axis=1).astype(np.float32)
        return ids, scores

    def encoded_nbytes(self) -> int:
        assert self._X is not None
        return int(self._X.nbytes)

    def aux_nbytes(self) -> int:
        return 0

    def to_state(self) -> dict:
        return {}

    @classmethod
    def from_state(cls, state: dict) -> "_ExactStub":
        return cls()


class _LossyStub(_ExactStub):
    """Codec stub that perturbs stored vectors so retrieval is imperfect."""

    name = "lossy"

    def __init__(self, noise: float = 0.6, seed: int = 7) -> None:
        super().__init__()
        self._noise = noise
        self._seed = seed

    def add(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self._seed)
        noisy = np.asarray(X, dtype=np.float64) + self._noise * rng.normal(size=X.shape)
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        self._X = noisy.astype(np.float32)


class TestExactGroundTruth:
    def test_matches_bruteforce_oracle(self, unit_vectors: np.ndarray) -> None:
        X = unit_vectors
        Q = unit_vectors[:40]
        for k in (1, 10, 100):
            ids, scores = exact_ground_truth(X, Q, k=k)
            oracle_ids, oracle_scores = _oracle(X, Q, k)
            assert ids.shape == (40, k)
            assert scores.shape == (40, k)
            assert ids.dtype == np.int64
            assert scores.dtype == np.float32
            np.testing.assert_array_equal(ids, oracle_ids)
            np.testing.assert_array_equal(scores, oracle_scores)

    def test_batch_boundaries_do_not_change_result(self, unit_vectors: np.ndarray) -> None:
        X = unit_vectors[:500]
        Q = unit_vectors[500:553]
        for k in (10, 100):
            ids_a, scores_a = exact_ground_truth(X, Q, k=k, batch=7)
            ids_b, scores_b = exact_ground_truth(X, Q, k=k, batch=4096)
            np.testing.assert_array_equal(ids_a, ids_b)
            np.testing.assert_array_equal(scores_a, scores_b)

    def test_rows_sorted_best_first(self, unit_vectors: np.ndarray) -> None:
        ids, scores = exact_ground_truth(unit_vectors, unit_vectors[:5], k=50)
        assert np.all(np.diff(scores.astype(np.float64), axis=1) <= 0.0)
        # A query drawn from the corpus must recover itself at rank 0.
        np.testing.assert_array_equal(ids[:, 0], np.arange(5, dtype=np.int64))

    def test_ties_broken_by_lower_id(self) -> None:
        d = 8
        v = np.zeros(d, dtype=np.float32)
        v[0] = 1.0
        u = np.zeros(d, dtype=np.float32)
        u[1] = 1.0
        # Rows 0, 2, 3 are bit-identical, so scores tie exactly.
        X = np.stack([v, u, v, v]).astype(np.float32)
        ids, scores = exact_ground_truth(X, v[None, :], k=3, batch=2)
        np.testing.assert_array_equal(ids[0], np.array([0, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(scores[0], np.ones(3, dtype=np.float32))

    def test_accepts_1d_query(self, unit_vectors: np.ndarray) -> None:
        ids2, scores2 = exact_ground_truth(unit_vectors, unit_vectors[:1], k=5)
        ids1, scores1 = exact_ground_truth(unit_vectors, unit_vectors[0], k=5)
        assert ids1.shape == (1, 5)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_equal(scores1, scores2)

    def test_k_larger_than_corpus_raises(self) -> None:
        X = np.eye(4, dtype=np.float32)
        with pytest.raises(ValueError):
            exact_ground_truth(X, X[:1], k=5)


class TestSplitIndices:
    def test_same_seed_same_split(self) -> None:
        db_a, q_a, size_a = _split_indices(2000, seed=SEED, n_queries=100, sample_cap=200_000)
        db_b, q_b, size_b = _split_indices(2000, seed=SEED, n_queries=100, sample_cap=200_000)
        np.testing.assert_array_equal(db_a, db_b)
        np.testing.assert_array_equal(q_a, q_b)
        assert size_a == size_b == 2000

    def test_different_seed_different_split(self) -> None:
        _, q_a, _ = _split_indices(2000, seed=SEED, n_queries=100, sample_cap=200_000)
        _, q_b, _ = _split_indices(2000, seed=SEED + 1, n_queries=100, sample_cap=200_000)
        assert not np.array_equal(q_a, q_b)

    def test_queries_removed_from_database(self) -> None:
        db, q, size = _split_indices(2000, seed=SEED, n_queries=100, sample_cap=200_000)
        assert q.size == 100
        assert db.size == 1900
        assert np.intersect1d(db, q).size == 0
        np.testing.assert_array_equal(np.sort(np.concatenate([db, q])), np.arange(2000))

    def test_sample_cap_shrinks_pool(self) -> None:
        db, q, size = _split_indices(2000, seed=SEED, n_queries=50, sample_cap=300)
        assert size == 300
        assert q.size == 50
        assert db.size == 250
        combined = np.concatenate([db, q])
        assert np.unique(combined).size == 300
        assert combined.min() >= 0 and combined.max() < 2000


class TestVerifyTiers:
    def test_exact_stub_has_recall_one_everywhere(self, unit_vectors: np.ndarray) -> None:
        result = verify_tiers(unit_vectors, {"stub": _ExactStub()}, seed=SEED, n_queries=50)
        report = result.codecs["stub"]
        assert report.recall["recall@1"] == 1.0
        assert report.recall["recall@10"] == 1.0
        assert report.recall["recall@100"] == 1.0
        assert report.rerank_depth == 0
        # fp32 storage: 32 dims times 4 bytes, no aux.
        assert report.bytes_per_vector == pytest.approx(128.0)
        assert report.bytes.raw == report.bytes.encoded
        assert report.bytes.aux == 0
        assert report.search_seconds >= 0.0
        assert result.n_vectors == 2000
        assert result.sample_size == 2000
        assert result.n_queries == 50
        assert result.n_database == 1950
        assert result.k == 100

    def test_fit_receives_seed(self, unit_vectors: np.ndarray) -> None:
        codec = _ExactStub()
        verify_tiers(unit_vectors, {"stub": codec}, seed=99, n_queries=20)
        assert codec.fit_seed == 99

    def test_lossy_stub_has_recall_below_one(self, unit_vectors: np.ndarray) -> None:
        result = verify_tiers(unit_vectors, {"lossy": _LossyStub()}, seed=SEED, n_queries=50)
        report = result.codecs["lossy"]
        assert report.recall["recall@1"] < 1.0
        assert 0.0 <= report.recall["recall@100"] <= 1.0

    def test_rerank_with_default_exact_reranker_recovers(self, unit_vectors: np.ndarray) -> None:
        # Full-depth rerank against the exact fallback reranker must undo
        # the noise entirely: every candidate is rescored with true float32.
        plain = verify_tiers(unit_vectors, {"lossy": _LossyStub()}, seed=SEED, n_queries=50)
        reranked = verify_tiers(
            unit_vectors,
            {"lossy": _LossyStub()},
            seed=SEED,
            n_queries=50,
            rerank_depths={"lossy": 10**9},
        )
        report = reranked.codecs["lossy"]
        # Depth is clamped to the database size.
        assert report.rerank_depth == reranked.n_database
        assert report.recall["recall@1"] == 1.0
        assert report.recall["recall@10"] == 1.0
        assert report.recall["recall@100"] == 1.0
        assert report.recall["recall@1"] > plain.codecs["lossy"].recall["recall@1"]

    def test_reranker_factory_is_used(self, unit_vectors: np.ndarray) -> None:
        seen: dict[str, object] = {}

        class _SpyReranker:
            def __init__(self, X: np.ndarray) -> None:
                self._X = np.asarray(X, dtype=np.float32)

            def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
                seen["called"] = True
                return self._X[np.asarray(ids, dtype=np.int64)] @ np.asarray(q, dtype=np.float32)

        def factory(X_db: np.ndarray) -> _SpyReranker:
            seen["n_db"] = X_db.shape[0]
            return _SpyReranker(X_db)

        result = verify_tiers(
            unit_vectors,
            {"lossy": _LossyStub()},
            seed=SEED,
            n_queries=50,
            rerank_depths={"lossy": 10**9},
            reranker_factory=factory,
        )
        assert seen["n_db"] == result.n_database
        assert seen["called"] is True
        assert result.codecs["lossy"].recall["recall@1"] == 1.0

    def test_query_split_and_result_deterministic(self, unit_vectors: np.ndarray) -> None:
        a = verify_tiers(unit_vectors, {"stub": _ExactStub()}, seed=SEED, n_queries=64)
        b = verify_tiers(unit_vectors, {"stub": _ExactStub()}, seed=SEED, n_queries=64)
        da, db = a.to_dict(), b.to_dict()
        # Wall time is the only field allowed to vary between runs.
        da["codecs"]["stub"].pop("search_seconds")
        db["codecs"]["stub"].pop("search_seconds")
        assert da == db

    def test_sample_cap_path_samples_and_records(self, unit_vectors: np.ndarray) -> None:
        result = verify_tiers(
            unit_vectors, {"stub": _ExactStub()}, seed=SEED, n_queries=50, sample_cap=300
        )
        assert result.n_vectors == 2000
        assert result.sample_size == 300
        assert result.n_queries == 50
        assert result.n_database == 250
        assert result.codecs["stub"].recall["recall@1"] == 1.0

    def test_to_dict_json_serializable(self, unit_vectors: np.ndarray) -> None:
        result = verify_tiers(
            unit_vectors,
            {"stub": _ExactStub(), "lossy": _LossyStub()},
            seed=SEED,
            n_queries=32,
        )
        d = result.to_dict()
        text = json.dumps(d)
        assert json.loads(text) == d

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    assert type(key) is str
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                assert type(value) in (int, float, str, bool, type(None))

        walk(d)

    def test_default_rerank_depths_match_contract(self) -> None:
        assert DEFAULT_RERANK_DEPTHS == {"pq": 200, "binary": 500, "sq8": 0}

    def test_too_small_database_raises_with_split_context(self, rng) -> None:
        # 80 vectors minus 10 queries leaves 70 database rows, below the
        # fixed recall@100 requirement. The guard must speak in verify_tiers
        # vocabulary (n_database, the query split), not "corpus size".
        x = rng.normal(size=(80, 32)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        with pytest.raises(ValueError, match=r"n_database=70") as excinfo:
            verify_tiers(x, {"stub": _ExactStub()}, seed=SEED, n_queries=10)
        message = str(excinfo.value)
        assert "recall@100" in message
        assert "n_queries=10" in message

    def test_result_type(self, unit_vectors: np.ndarray) -> None:
        result = verify_tiers(unit_vectors, {"stub": _ExactStub()}, seed=SEED, n_queries=16)
        assert isinstance(result, VerifyResult)
