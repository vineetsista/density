"""SQ8: per-dimension min/max affine quantization to uint8.

Codes are uint8 [n, d], a 4.0x reduction versus fp32 (the per-dimension
scale and min arrays are counted as aux bytes). Scoring never decodes:
with the affine model x ~= scale * code + mins, the dot product becomes
dot(q, x) = (q * scale) @ code + q @ mins, so a query reduces to one
uint8-domain matmul via the accel kernel plus a scalar constant.
"""

from __future__ import annotations

import numpy as np

from density.engine._accel import sq8_scores
from density.engine.embed import Reranker, topk
from density.engine.embed.rerank import rerank_topk


class SQ8:
    """Scalar 8-bit codec. See VectorCodec in engine/embed/__init__.py."""

    name: str = "sq8"

    def __init__(self) -> None:
        self._mins: np.ndarray | None = None  # float32 [d]
        self._scale: np.ndarray | None = None  # float32 [d], 0 on flat dims
        # Codes arrive in append-order blocks; concatenation is deferred and
        # cached because add() may be called many times before any search.
        self._blocks: list[np.ndarray] = []
        self._cache: np.ndarray | None = None

    def fit(self, X: np.ndarray, seed: int = 1337) -> "SQ8":
        """X: float32 [n, d]. Returns self, fitted.

        Deterministic: the min/max sweep has no stochastic step, seed is
        accepted only to honor the shared codec signature. Refitting drops
        stored codes because they were encoded under the old grid.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("fit expects a non-empty float32 [n, d] matrix")
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        scale = (maxs - mins) / np.float32(255.0)
        # Flat dimensions carry no information: scale 0 makes encode map
        # them to code 0 and decode reproduce the constant exactly.
        self._mins = mins.astype(np.float32)
        self._scale = np.where(maxs > mins, scale, np.float32(0.0)).astype(np.float32)
        self._blocks = []
        self._cache = None
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X: float32 [n, d]. Returns uint8 [n, d] codes.

        Round-trip guarantee: decode(encode(X)) is within half a
        quantization step, (max - min) / 255 / 2, per dimension, modulo
        float rounding. Arithmetic runs in float64 so the guarantee is
        not eroded by intermediate float32 rounding.
        """
        mins, scale = self._require_fit()
        X64 = np.asarray(X, dtype=np.float32).astype(np.float64)
        scale64 = scale.astype(np.float64)
        # Masked divide keeps flat dims (scale 0) at inv 0 without warning.
        inv = np.divide(1.0, scale64, out=np.zeros_like(scale64), where=scale64 > 0)
        t = (X64 - mins.astype(np.float64)) * inv
        return np.clip(np.rint(t), 0.0, 255.0).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """codes: uint8 [n, d]. Returns float32 [n, d] affine reconstruction."""
        mins, scale = self._require_fit()
        codes64 = np.asarray(codes, dtype=np.uint8).astype(np.float64)
        out = codes64 * scale.astype(np.float64) + mins.astype(np.float64)
        return out.astype(np.float32)

    def add(self, X: np.ndarray) -> None:
        """X: float32 [n, d]. Encodes and appends; ids continue from the
        rows added so far."""
        self._blocks.append(self.encode(X))
        self._cache = None

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        rerank_depth: int = 0,
        reranker: Reranker | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """q: float32 [d] or [nq, d]. Returns (ids int64 [nq, k], scores
        float32 [nq, k]) best-first; k is clamped to the stored row count.
        Scores are approximate dot products unless reranked."""
        mins, scale = self._require_fit()
        if rerank_depth > 0 and reranker is None:
            raise ValueError("rerank_depth > 0 requires a reranker")
        codes = self.stored_codes()
        if codes.shape[0] == 0:
            raise ValueError("search requires vectors: call add() first")
        Q = np.asarray(q, dtype=np.float32)
        Q = Q[None, :] if Q.ndim == 1 else Q
        ids_rows: list[np.ndarray] = []
        score_rows: list[np.ndarray] = []
        for row in Q:
            # const in float64: it is a single accumulation over d terms and
            # costs nothing, while keeping the additive constant exact.
            const = float(row.astype(np.float64) @ mins.astype(np.float64))
            approx = sq8_scores(codes, row * scale, const)
            if rerank_depth > 0:
                cand = topk(approx, rerank_depth)
                assert reranker is not None
                ids, scores = rerank_topk(row, cand, reranker, k)
            else:
                ids = topk(approx, k)
                scores = approx[ids].astype(np.float32)
            ids_rows.append(ids)
            score_rows.append(scores)
        return (
            np.stack(ids_rows).astype(np.int64),
            np.stack(score_rows).astype(np.float32),
        )

    def stored_codes(self) -> np.ndarray:
        """All retained codes as one uint8 [n, d] matrix (cached view)."""
        if self._cache is None:
            mins, _ = self._require_fit()
            if not self._blocks:
                self._cache = np.empty((0, mins.shape[0]), dtype=np.uint8)
            elif len(self._blocks) == 1:
                self._cache = np.ascontiguousarray(self._blocks[0])
            else:
                self._cache = np.concatenate(self._blocks, axis=0)
        return self._cache

    def encoded_nbytes(self) -> int:
        """Bytes of stored codes: 1 byte per stored dimension."""
        return sum(int(b.nbytes) for b in self._blocks)

    def aux_nbytes(self) -> int:
        """Bytes of the scale and min arrays."""
        mins, scale = self._require_fit()
        return int(mins.nbytes) + int(scale.nbytes)

    def to_state(self) -> dict[str, np.ndarray]:
        """Returns {mins float32 [d], scale float32 [d], codes uint8 [n, d]}."""
        mins, scale = self._require_fit()
        return {
            "mins": mins.copy(),
            "scale": scale.copy(),
            "codes": self.stored_codes().copy(),
        }

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "SQ8":
        """Inverse of to_state, including stored codes."""
        for key in ("mins", "scale", "codes"):
            if key not in state:
                raise ValueError(f"SQ8 state missing key: {key}")
        codec = cls()
        codec._mins = np.asarray(state["mins"], dtype=np.float32).copy()
        codec._scale = np.asarray(state["scale"], dtype=np.float32).copy()
        codes = np.asarray(state["codes"], dtype=np.uint8)
        codec._blocks = [] if codes.shape[0] == 0 else [codes.copy()]
        codec._cache = None
        return codec

    def _require_fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mins is None or self._scale is None:
            raise ValueError("SQ8 is not fitted: call fit() first")
        return self._mins, self._scale
