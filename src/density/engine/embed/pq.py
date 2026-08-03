"""Product quantization codec.

Splits each d-dimensional float32 vector into m contiguous subvectors of
d // m dimensions, learns 256 centroids per subspace with seeded
mini-batch k-means, and stores one uint8 centroid index per subspace.
Codes are uint8 [n, m], codebooks are float32 [m, 256, d // m]. Search
builds a per-query ADC lookup table, lut float32 [m, 256], and scans
stored codes with the accel kernel. Vectors are L2-normalized upstream,
so the dot-product lut is a cosine lut.
"""

from __future__ import annotations

import numpy as np

from density.engine import _accel

# Training reads at most this many rows so fit memory stays bounded even
# on multi-million row corpora.
_FIT_SAMPLE_CAP = 120_000
_KMEANS_BATCH = 8_192
_KMEANS_EPOCHS = 8
# Encoding materializes one float32 [chunk, 256] distance block at a time.
_ENCODE_CHUNK = 65_536
_N_CENTROIDS = 256


def _topk_desc(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k largest scores, best-first.

    scores: float32 [n], k <= n. Returns (int64 [k], float32 [k]).
    Partition first so the full sort only touches k candidates; tie
    handling is deterministic, so repeated runs are bit-identical.
    """
    n = scores.shape[0]
    if k >= n:
        idx = np.argsort(-scores, kind="stable")[:k]
    else:
        part = np.argpartition(-scores, k - 1)[:k]
        idx = part[np.argsort(-scores[part], kind="stable")]
    return idx.astype(np.int64), np.ascontiguousarray(scores[idx], dtype=np.float32)


class PQ:
    """Product quantizer: m subquantizers, 256 centroids each.

    m defaults to d // 4 (resolved at fit time) and must divide d.
    Codes are uint8 [n, m], one byte per subspace, a d * 4 / m fold
    reduction versus float32. Aux state is the float32 codebook,
    m * 256 * (d // m) * 4 bytes.
    """

    name: str = "pq"

    def __init__(self, m: int | None = None, nbits: int = 8) -> None:
        if nbits != 8:
            raise ValueError(
                "PQ supports nbits=8 only: codes are uint8 and the ADC "
                "kernel scans a 256-entry lut per subquantizer"
            )
        if m is not None and m < 1:
            raise ValueError(f"m must be a positive integer, got {m}")
        self.m: int | None = m
        self.nbits: int = nbits
        self._d: int | None = None
        self._codebooks: np.ndarray | None = None  # float32 [m, 256, d // m]
        self._codes: np.ndarray | None = None  # uint8 [n, m]

    def fit(self, X: np.ndarray, seed: int = 1337) -> "PQ":
        """Learn per-subspace codebooks with seeded mini-batch k-means.

        X: float32 [n, d]. Trains on a seeded sample of at most 120000
        rows, batches of 8192, 8 epochs per subspace, empty centroids
        reseeded from random sample rows after every epoch. Same seed
        gives bit-identical codebooks. Stored codes are cleared because
        codes from a previous codebook are meaningless. Returns self.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D [n, d] array, got shape {X.shape}")
        n, d = X.shape
        if n == 0:
            raise ValueError("cannot fit PQ on an empty array")
        m = self.m if self.m is not None else d // 4
        if m < 1 or d % m != 0:
            raise ValueError(f"m={m} must be a positive divisor of d={d}")
        dsub = d // m
        rng = np.random.default_rng(seed)
        if n > _FIT_SAMPLE_CAP:
            # Sorted sample indices keep the gather cache-friendly without
            # changing which rows are seen.
            keep = np.sort(rng.choice(n, size=_FIT_SAMPLE_CAP, replace=False))
            sample = X[keep]
        else:
            sample = X
        codebooks = np.empty((m, _N_CENTROIDS, dsub), dtype=np.float32)
        for j in range(m):
            sub = np.ascontiguousarray(sample[:, j * dsub : (j + 1) * dsub])
            codebooks[j] = _fit_subspace(sub, rng)
        self.m = m
        self._d = d
        self._codebooks = codebooks
        self._codes = None
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Nearest centroid per subspace. X: float32 [n, d] -> uint8 [n, m].

        Works in chunks of at most 65536 rows so the distance block never
        exceeds chunk x 256 floats per subspace.
        """
        cb = self._require_fitted()
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(f"expected [n, {self._d}] input, got shape {X.shape}")
        m, _, dsub = cb.shape
        n = X.shape[0]
        codes = np.empty((n, m), dtype=np.uint8)
        cnorm = np.einsum("mkd,mkd->mk", cb, cb)
        for start in range(0, n, _ENCODE_CHUNK):
            stop = min(start + _ENCODE_CHUNK, n)
            chunk = X[start:stop]
            for j in range(m):
                sub = chunk[:, j * dsub : (j + 1) * dsub]
                # The row norm term is constant per row, so dropping it
                # leaves the squared-distance argmin unchanged.
                d2 = cnorm[j][None, :] - 2.0 * (sub @ cb[j].T)
                codes[start:stop, j] = np.argmin(d2, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Centroid lookup. codes: uint8 [n, m] -> float32 [n, d]."""
        cb = self._require_fitted()
        codes = np.asarray(codes)
        m, _, dsub = cb.shape
        if codes.ndim != 2 or codes.shape[1] != m:
            raise ValueError(f"expected [n, {m}] codes, got shape {codes.shape}")
        out = np.empty((codes.shape[0], m * dsub), dtype=np.float32)
        for j in range(m):
            out[:, j * dsub : (j + 1) * dsub] = cb[j][codes[:, j]]
        return out

    def add(self, X: np.ndarray) -> None:
        """Encode X (float32 [n, d]) and retain the codes for search.

        Replaces previously stored codes: per the contract, search ids are
        row indices into the X passed to the most recent add call.
        """
        self._codes = self.encode(X)

    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        rerank_depth: int = 0,
        reranker: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ADC search over stored codes.

        q: float32 [d] or [nq, d]. Returns (ids int64 [nq, k], scores
        float32 [nq, k]) sorted best-first; k is clamped to the number of
        stored codes. Scores are dot-product similarities reconstructed
        through the codebooks (cosine, since inputs are normalized
        upstream). rerank_depth > 0 rescores the ADC top rerank_depth
        candidates with the given reranker and returns its top k.
        """
        cb = self._require_fitted()
        codes = self._codes
        if codes is None or codes.shape[0] == 0:
            raise ValueError("no stored codes: call add() before search()")
        if rerank_depth > 0 and reranker is None:
            raise ValueError("rerank_depth > 0 requires a reranker")
        Q = np.ascontiguousarray(q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q[None, :]
        if Q.ndim != 2 or Q.shape[1] != self._d:
            raise ValueError(f"expected query dim {self._d}, got shape {np.shape(q)}")
        m, _, dsub = cb.shape
        n = codes.shape[0]
        nq = Q.shape[0]
        k_eff = min(k, n)
        ids = np.empty((nq, k_eff), dtype=np.int64)
        scores = np.empty((nq, k_eff), dtype=np.float32)
        for i in range(nq):
            qsub = Q[i].reshape(m, dsub)
            # lut[j, c] = centroids[j, c] . q_sub[j]; the scan then sums one
            # lut entry per subspace instead of decoding any vector.
            lut = np.ascontiguousarray((cb @ qsub[:, :, None])[:, :, 0], dtype=np.float32)
            adc = _accel.pq_adc_scan(codes, lut)
            if rerank_depth > 0:
                # Lazy import: rerank is an optional collaborator module and
                # must not be a hard dependency of ADC-only search.
                from density.engine.embed.rerank import rerank_topk

                cand, _ = _topk_desc(adc, min(rerank_depth, n))
                r_ids, r_scores = rerank_topk(Q[i], cand, reranker, k_eff)
                ids[i] = r_ids
                scores[i] = r_scores
            else:
                ids[i], scores[i] = _topk_desc(adc, k_eff)
        return ids, scores

    def encoded_nbytes(self) -> int:
        """Bytes of stored codes: n * m."""
        return 0 if self._codes is None else int(self._codes.nbytes)

    def aux_nbytes(self) -> int:
        """Codebook bytes: m * 256 * (d // m) * 4."""
        return 0 if self._codebooks is None else int(self._codebooks.nbytes)

    def to_state(self) -> dict[str, np.ndarray]:
        """Persistable state: codebooks float32 [m, 256, d // m], codes uint8 [n, m]."""
        cb = self._require_fitted()
        codes = (
            self._codes
            if self._codes is not None
            else np.empty((0, cb.shape[0]), dtype=np.uint8)
        )
        return {"codebooks": cb.copy(), "codes": codes.copy()}

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "PQ":
        """Rebuild a fitted codec from to_state output."""
        cb = np.ascontiguousarray(state["codebooks"], dtype=np.float32)
        if cb.ndim != 3 or cb.shape[1] != _N_CENTROIDS:
            raise ValueError(f"expected codebooks [m, 256, dsub], got shape {cb.shape}")
        m, _, dsub = cb.shape
        obj = cls(m=m)
        obj._d = m * dsub
        obj._codebooks = cb
        codes = np.ascontiguousarray(state["codes"], dtype=np.uint8)
        obj._codes = codes if codes.size else None
        return obj

    def _require_fitted(self) -> np.ndarray:
        if self._codebooks is None:
            raise ValueError("PQ is not fitted: call fit() first")
        return self._codebooks


def _fit_subspace(sub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mini-batch k-means for one subspace.

    sub: float32 [n_sample, dsub]. Returns float32 [256, dsub]. Consumes
    the shared rng in a fixed order, which is what makes whole-model fits
    bit-identical for a given seed.
    """
    n_s, dsub = sub.shape
    init = rng.choice(n_s, size=_N_CENTROIDS, replace=n_s < _N_CENTROIDS)
    centroids = sub[init].copy()
    counts = np.zeros(_N_CENTROIDS, dtype=np.int64)
    for _ in range(_KMEANS_EPOCHS):
        order = rng.permutation(n_s)
        for start in range(0, n_s, _KMEANS_BATCH):
            xb = sub[order[start : start + _KMEANS_BATCH]]
            cnorm = np.einsum("kd,kd->k", centroids, centroids)
            assign = np.argmin(cnorm[None, :] - 2.0 * (xb @ centroids.T), axis=1)
            bcount = np.bincount(assign, minlength=_N_CENTROIDS)
            sums = np.zeros((_N_CENTROIDS, dsub), dtype=np.float32)
            np.add.at(sums, assign, xb)
            counts += bcount
            upd = bcount > 0
            # Classic mini-batch learning rate batch_count / lifetime_count:
            # early batches move centroids a lot, later batches refine, and
            # the update is exact running-mean bookkeeping.
            cf = counts[upd].astype(np.float32)[:, None]
            bf = bcount[upd].astype(np.float32)[:, None]
            centroids[upd] += (sums[upd] - bf * centroids[upd]) / cf
        dead = counts == 0
        if dead.any():
            # Reseed from real rows so degenerate inputs (fewer distinct
            # points than centroids) still produce usable codebooks.
            centroids[dead] = sub[rng.integers(0, n_s, size=int(dead.sum()))]
    return centroids
