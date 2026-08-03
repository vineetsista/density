"""Binary sign codec: 1 bit per dimension, hamming distance search.

Codes are uint8 [n, d // 8], bit j of row i set iff X[i, j] > 0, packed
with numpy packbits (big-endian bit order within each byte). Storage is
exactly 1/32 of the fp32 bytes with zero aux bytes. Search scores are
negative hamming distance so that higher is better, matching the shared
VectorCodec protocol. Reranking routes through rerank_topk, imported
lazily so the codec loads even while rerank.py is still being written.
"""

from __future__ import annotations

import numpy as np

from density.engine import _accel


def _validated_2d(X: np.ndarray) -> np.ndarray:
    """Return X as a 2-D array with a bit-packable width.

    X: [n, d] numeric. Raises ValueError unless d is divisible by 8,
    because codes are whole bytes and sub-byte padding would silently
    corrupt hamming distances.
    """
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array [n, d], got shape {arr.shape}")
    if arr.shape[1] % 8 != 0:
        raise ValueError(f"dim {arr.shape[1]} is not divisible by 8")
    return arr


class BinaryCodec:
    """1-bit sign quantization over float vectors.

    fit records the dimensionality (no training is needed for signs),
    add encodes and retains codes, search scans retained codes with the
    popcount kernel. Ids returned by search are row indices into the X
    passed to the most recent add, so add replaces retained codes rather
    than appending.
    """

    name: str = "binary"

    def __init__(self) -> None:
        self._dim: int | None = None
        self._codes: np.ndarray | None = None

    def fit(self, X: np.ndarray, seed: int = 1337) -> "BinaryCodec":
        """Validate X: float [n, d], d divisible by 8, and record d.

        seed is accepted for protocol parity only: sign quantization has
        no trained state, so fitting is deterministic regardless of seed.
        Returns self.
        """
        arr = _validated_2d(X)
        self._dim = int(arr.shape[1])
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Pack sign(X) > 0 into bits. X: [n, d] float. Returns uint8 [n, d // 8]."""
        arr = _validated_2d(X)
        if self._dim is not None and arr.shape[1] != self._dim:
            raise ValueError(f"dim {arr.shape[1]} does not match fitted dim {self._dim}")
        return np.packbits(arr > 0, axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Approximate reconstruction from packed bits.

        codes: uint8 [n, d // 8]. Returns float32 [n, d]: plus or minus 1
        per dimension, scaled by 1 / sqrt(d) so every row has unit length,
        keeping decoded vectors on the same sphere as the fp32 originals.
        """
        arr = np.asarray(codes, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"expected codes [n, d // 8], got shape {arr.shape}")
        bits = np.unpackbits(arr, axis=1).astype(np.float32)
        d = bits.shape[1]
        signs = bits * np.float32(2.0) - np.float32(1.0)
        signs *= np.float32(1.0 / np.sqrt(np.float32(d)))
        return signs

    def add(self, X: np.ndarray) -> None:
        """Encode X: [n, d] float and retain the codes.

        Replaces any previously retained codes: search ids are defined as
        row indices into the X of the most recent add.
        """
        codes = self.encode(X)
        self._dim = int(np.asarray(X).shape[1])
        self._codes = np.ascontiguousarray(codes)

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        rerank_depth: int = 0,
        reranker: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k scan of retained codes by hamming distance.

        q: float32 [d] or [nq, d]. Returns (ids int64 [nq, k], scores
        float32 [nq, k]) sorted best-first; a 1-D query yields nq = 1.
        Without rerank, scores are negative hamming distances (ties broken
        by ascending row index for determinism). With rerank_depth > 0 the
        hamming top rerank_depth candidates (widened to at least k) are
        rescored through rerank_topk with the given reranker, and scores
        are whatever the reranker produced.
        """
        if self._codes is None or self._codes.shape[0] == 0:
            raise ValueError("no codes retained: call add() before search()")
        q_arr = np.asarray(q, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = q_arr[None, :]
        if q_arr.ndim != 2 or q_arr.shape[1] != self._codes.shape[1] * 8:
            raise ValueError(
                f"query shape {np.asarray(q).shape} does not match "
                f"stored dim {self._codes.shape[1] * 8}"
            )

        n = self._codes.shape[0]
        nq = q_arr.shape[0]
        kk = min(int(k), n)

        rerank_topk = None
        if rerank_depth > 0:
            if reranker is None:
                raise ValueError("rerank_depth > 0 requires a reranker")
            # Lazy import: rerank.py is a sibling module owned elsewhere,
            # deferring lets this codec import and run without it.
            from density.engine.embed.rerank import rerank_topk
            depth = min(max(int(rerank_depth), kk), n)

        ids = np.empty((nq, kk), dtype=np.int64)
        scores = np.empty((nq, kk), dtype=np.float32)
        for i in range(nq):
            qb = np.packbits(q_arr[i] > 0)
            dist = _accel.hamming_scan(self._codes, qb)
            # Stable sort so equal distances resolve by row index: hamming
            # ties are common at low bit widths and must not break
            # bit-identical determinism across runs.
            order = np.argsort(dist, kind="stable")
            if rerank_topk is not None:
                cand = order[:depth].astype(np.int64)
                r_ids, r_scores = rerank_topk(q_arr[i], cand, reranker, kk)
                ids[i] = r_ids[:kk]
                scores[i] = np.asarray(r_scores[:kk], dtype=np.float32)
            else:
                top = order[:kk]
                ids[i] = top
                scores[i] = -dist[top].astype(np.float32)
        return ids, scores

    def encoded_nbytes(self) -> int:
        """Bytes of retained codes: n * d / 8, exactly 1/32 of fp32 bytes."""
        return 0 if self._codes is None else int(self._codes.nbytes)

    def aux_nbytes(self) -> int:
        """Always 0: sign quantization carries no codebooks or scales."""
        return 0

    def to_state(self) -> dict[str, np.ndarray]:
        """Serializable state for manifest persistence.

        Returns {"codes": uint8 [n, d // 8], "dim": int64 [1]}. Arrays are
        views of live state, callers persist rather than mutate them.
        """
        codes = self._codes if self._codes is not None else np.zeros((0, 0), dtype=np.uint8)
        dim = self._dim if self._dim is not None else 0
        return {"codes": codes, "dim": np.asarray([dim], dtype=np.int64)}

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "BinaryCodec":
        """Rebuild a codec from to_state output. Inverse of to_state."""
        codec = cls()
        dim = int(np.asarray(state["dim"]).reshape(-1)[0])
        if dim > 0:
            codec._dim = dim
        codes = np.ascontiguousarray(np.asarray(state["codes"], dtype=np.uint8))
        if codes.shape[0] > 0:
            codec._codes = codes
        return codec
