"""Shared vector codec interfaces for the embedding engine.

Defines the VectorCodec and Reranker protocols that sq8, pq, and binary
implement, plus the topk helper they all share. This module must not
import the concrete codec submodules: they are developed independently
and importing them here would pin a package import order.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["Reranker", "VectorCodec", "topk"]


class Reranker(Protocol):
    """Rescores candidate rows against a query with higher fidelity."""

    def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """q: float32 [d], ids: int64 [c] row indices. Returns float32 [c]."""
        ...


class VectorCodec(Protocol):
    """Lossy vector codec: fit, encode, retain codes, search.

    All vectors are float32 and L2-normalized at ingest, so every score
    in this interface is a cosine similarity expressed as a dot product.
    """

    name: str

    def fit(self, X: np.ndarray, seed: int = 1337) -> VectorCodec:
        """X: float32 [n, d] training vectors. Returns self, fitted."""
        ...

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X: float32 [n, d]. Returns codes, dtype and shape per codec."""
        ...

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """codes per codec. Returns float32 [n, d] approximation."""
        ...

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        rerank_depth: int = 0,
        reranker: Reranker | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """q: float32 [d] or [nq, d]. Returns (ids int64 [nq, k], scores
        float32 [nq, k]) sorted best-first. Ids are row indices into the
        vectors passed to add(). rerank_depth == 0 means no rerank,
        rerank_depth > 0 requires a reranker.
        """
        ...

    def add(self, X: np.ndarray) -> None:
        """X: float32 [n, d]. Encodes and retains codes for search."""
        ...

    def encoded_nbytes(self) -> int:
        """Bytes of stored codes."""
        ...

    def aux_nbytes(self) -> int:
        """Bytes of codebooks, scales, and other side data."""
        ...

    def to_state(self) -> dict[str, np.ndarray]:
        """Everything needed to rebuild the codec, for manifest persistence."""
        ...

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> VectorCodec:
        """Inverse of to_state."""
        ...


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Best-first indices of the k highest scores.

    scores: float [n]. Returns int64 [min(k, n)], highest score first.
    argpartition selects the winners in O(n), the final sort only touches
    the k survivors, which keeps large scans cheap.
    """
    scores = np.asarray(scores)
    n = int(scores.shape[0])
    k = min(int(k), n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(scores, n - k)[n - k :]
    # Stable sort on the negated scores keeps tie order deterministic.
    order = np.argsort(-scores[part], kind="stable")
    return part[order].astype(np.int64)
