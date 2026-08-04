"""Recall and compression metrics.

These are the arithmetic primitives the verifier and audit report build on.
They stay deliberately dumb: no sampling, no codec knowledge, just counting.
All byte quantities are plain bytes, all recalls are fractions in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BytesAccount", "compression_ratio", "recall_at_k"]


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Mean fraction of true top-k neighbors recovered in the predicted top-k.

    gt: int64 [nq, >=k] exact neighbor ids, best-first.
    pred: int64 [nq, >=k] predicted neighbor ids, best-first.
    Only the first k columns of each matter, so a wide ground truth matrix
    can serve every k in one pass. Order within a row does not matter:
    recall is a set overlap, not a rank correlation.
    Returns a plain Python float in [0, 1].
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if gt.ndim != 2 or pred.ndim != 2:
        raise ValueError(
            f"gt and pred must be 2-D [nq, k], got shapes {gt.shape} and {pred.shape}"
        )
    if gt.shape[0] != pred.shape[0]:
        raise ValueError(
            f"row count mismatch: gt has {gt.shape[0]} queries, pred has {pred.shape[0]}"
        )
    if gt.shape[1] < k:
        raise ValueError(f"ground truth has only {gt.shape[1]} columns, need k={k}")
    if pred.shape[1] < k:
        raise ValueError(f"prediction has only {pred.shape[1]} columns, need k={k}")
    nq = gt.shape[0]
    if nq == 0:
        raise ValueError("gt and pred are empty: at least one query row is required")
    # intersect1d uniques both sides, so accidental duplicate ids in a row
    # can never push a query's recall above 1.0.
    hits = 0
    for i in range(nq):
        hits += int(np.intersect1d(gt[i, :k], pred[i, :k]).size)
    return float(hits / (nq * k))


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """How many times smaller the compressed form is: original / compressed.

    Both arguments are byte counts. Values below 1.0 mean the "compressed"
    form actually grew. Returns a plain Python float.
    """
    if compressed_bytes <= 0:
        raise ValueError(f"compressed_bytes must be positive, got {compressed_bytes}")
    return float(original_bytes) / float(compressed_bytes)


@dataclass(frozen=True)
class BytesAccount:
    """Byte accounting for one encoded representation.

    raw: bytes of the original float32 vectors.
    encoded: bytes of the stored codes.
    aux: bytes of side data required to score (codebooks, scales, mins).
    Aux counts toward the footprint because search cannot run without it,
    which is why total is encoded + aux, never encoded alone.
    """

    raw: int
    encoded: int
    aux: int

    @property
    def total(self) -> int:
        """Bytes actually occupied by the encoded form: encoded + aux."""
        return self.encoded + self.aux
