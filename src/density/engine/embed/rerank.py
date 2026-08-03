"""Reranking: rescore a candidate set with higher fidelity.

Codec search kernels trade accuracy for speed; rerank_topk recovers
accuracy by rescoring only the top candidates. SQ8Reranker decodes just
the candidate rows to their exact affine reconstruction, ExactReranker
keeps the original fp32 matrix (the HOT tier).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Reranker is defined next to VectorCodec so the protocols live together;
# re-exported here because the contract names this module for it.
from density.engine.embed import Reranker, topk

if TYPE_CHECKING:
    from density.engine.embed.sq8 import SQ8

__all__ = ["ExactReranker", "Reranker", "SQ8Reranker", "rerank_topk"]


class SQ8Reranker:
    """Rescores via exact affine reconstruction of SQ8 codes.

    Decodes only the candidate rows, not the whole store, because rerank
    depth is tiny compared to corpus size. This is the true decode, not
    the kernel's fused approximation, so scores match decode() @ q.
    """

    def __init__(self, codec: "SQ8") -> None:
        self._codec = codec

    def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """q: float32 [d], ids: int64 [c]. Returns float32 [c] dot products."""
        ids = np.asarray(ids, dtype=np.int64)
        rows = self._codec.stored_codes()[ids]
        decoded = self._codec.decode(rows)
        return (decoded @ np.asarray(q, dtype=np.float32)).astype(np.float32)


class ExactReranker:
    """Rescores against the original float32 vectors."""

    def __init__(self, X: np.ndarray) -> None:
        """X: float32 [n, d], rows indexed by candidate ids."""
        self._X = np.ascontiguousarray(X, dtype=np.float32)

    def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """q: float32 [d], ids: int64 [c]. Returns float32 [c] dot products."""
        ids = np.asarray(ids, dtype=np.int64)
        return (self._X[ids] @ np.asarray(q, dtype=np.float32)).astype(np.float32)


def rerank_topk(
    q: np.ndarray,
    candidate_ids: np.ndarray,
    reranker: Reranker,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescores candidates and keeps the best k.

    q: float32 [d], candidate_ids: int [c]. Returns (ids int64
    [min(k, c)], scores float32 [min(k, c)]) best-first, ids drawn from
    candidate_ids and scores from the reranker.
    """
    ids = np.asarray(candidate_ids, dtype=np.int64)
    scores = np.asarray(reranker.score(np.asarray(q, dtype=np.float32), ids), dtype=np.float32)
    order = topk(scores, k)
    return ids[order], scores[order]
