"""Unit tests for recall and compression metrics."""

from __future__ import annotations

import numpy as np
import pytest

from density.recall.metrics import BytesAccount, compression_ratio, recall_at_k


class TestRecallAtK:
    def test_perfect_match_is_one(self) -> None:
        gt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        pred = gt.copy()
        assert recall_at_k(gt, pred, 3) == 1.0

    def test_partial_overlap(self) -> None:
        # Query 0 recovers 2 of 4, query 1 recovers 4 of 4: mean 0.75.
        gt = np.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=np.int64)
        pred = np.array([[0, 1, 8, 9], [13, 12, 11, 10]], dtype=np.int64)
        assert recall_at_k(gt, pred, 4) == pytest.approx(0.75)

    def test_order_within_row_does_not_matter(self) -> None:
        gt = np.array([[7, 3, 9]], dtype=np.int64)
        pred = np.array([[9, 7, 3]], dtype=np.int64)
        assert recall_at_k(gt, pred, 3) == 1.0

    def test_disjoint_is_zero(self) -> None:
        gt = np.array([[0, 1], [2, 3]], dtype=np.int64)
        pred = np.array([[8, 9], [6, 7]], dtype=np.int64)
        assert recall_at_k(gt, pred, 2) == 0.0

    def test_only_first_k_prediction_columns_count(self) -> None:
        # The true neighbor sits at prediction rank 2, outside k=1.
        gt = np.array([[5, 6]], dtype=np.int64)
        pred = np.array([[9, 5, 6]], dtype=np.int64)
        assert recall_at_k(gt, pred, 1) == 0.0
        assert recall_at_k(gt, pred, 2) == pytest.approx(0.5)

    def test_wider_gt_uses_first_k_columns(self) -> None:
        gt = np.array([[1, 2, 3, 4]], dtype=np.int64)
        pred = np.array([[1, 2, 99, 98]], dtype=np.int64)
        assert recall_at_k(gt, pred, 2) == 1.0

    def test_single_neighbor(self) -> None:
        gt = np.array([[42]], dtype=np.int64)
        assert recall_at_k(gt, np.array([[42]], dtype=np.int64), 1) == 1.0
        assert recall_at_k(gt, np.array([[41]], dtype=np.int64), 1) == 0.0

    def test_k_larger_than_gt_raises(self) -> None:
        gt = np.array([[1, 2]], dtype=np.int64)
        pred = np.array([[1, 2, 3]], dtype=np.int64)
        with pytest.raises(ValueError):
            recall_at_k(gt, pred, 3)

    def test_k_larger_than_pred_raises(self) -> None:
        gt = np.array([[1, 2, 3]], dtype=np.int64)
        pred = np.array([[1, 2]], dtype=np.int64)
        with pytest.raises(ValueError):
            recall_at_k(gt, pred, 3)

    def test_nonpositive_k_raises(self) -> None:
        gt = np.array([[1]], dtype=np.int64)
        with pytest.raises(ValueError):
            recall_at_k(gt, gt, 0)

    def test_row_count_mismatch_raises(self) -> None:
        gt = np.array([[1], [2]], dtype=np.int64)
        pred = np.array([[1]], dtype=np.int64)
        with pytest.raises(ValueError):
            recall_at_k(gt, pred, 1)

    def test_returns_plain_float(self) -> None:
        gt = np.array([[1, 2]], dtype=np.int64)
        out = recall_at_k(gt, gt, 2)
        assert type(out) is float


class TestCompressionRatio:
    def test_four_x(self) -> None:
        assert compression_ratio(1000, 250) == pytest.approx(4.0)

    def test_no_compression(self) -> None:
        assert compression_ratio(100, 100) == pytest.approx(1.0)

    def test_expansion_below_one(self) -> None:
        assert compression_ratio(100, 200) == pytest.approx(0.5)

    def test_zero_compressed_raises(self) -> None:
        with pytest.raises(ValueError):
            compression_ratio(100, 0)

    def test_negative_compressed_raises(self) -> None:
        with pytest.raises(ValueError):
            compression_ratio(100, -1)

    def test_returns_plain_float(self) -> None:
        assert type(compression_ratio(10, 5)) is float


class TestBytesAccount:
    def test_fields_and_total(self) -> None:
        acct = BytesAccount(raw=4000, encoded=1000, aux=24)
        assert acct.raw == 4000
        assert acct.encoded == 1000
        assert acct.aux == 24
        assert acct.total == 1024

    def test_equality(self) -> None:
        assert BytesAccount(1, 2, 3) == BytesAccount(1, 2, 3)
        assert BytesAccount(1, 2, 3) != BytesAccount(1, 2, 4)


def test_recall_at_k_empty_queries_raises() -> None:
    empty = np.empty((0, 10), dtype=np.int64)
    with pytest.raises(ValueError, match="at least one query"):
        recall_at_k(empty, empty, k=10)
