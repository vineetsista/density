"""Product quantization codec.

Splits each d-dimensional float32 vector into m contiguous subvectors of
d // m dimensions, learns 256 centroids per subspace, and stores one
uint8 centroid index per subspace. Codes are uint8 [n, m], codebooks are
float32 [m, 256, d // m]. Search builds a per-query ADC lookup table,
lut float32 [m, 256], and scans stored codes with the accel kernel.
Vectors are L2-normalized upstream, so the dot-product lut is a cosine
lut.

Training per subspace layers standard k-means engineering on the
contract floor (seeded mini-batch k-means, 256 centroids, at least 8
epochs over a capped sample):

1. greedy k-means++ initialization on a seeded subsample,
2. the 8 mini-batch epochs over the full capped sample (the floor),
3. full-batch Lloyd refinement to a movement tolerance, dead centroids
   reassigned to the highest-distortion points,
4. anisotropic (score-aware) Lloyd refinement: a weighted k-means
   variant that penalizes the residual component parallel to each data
   point by eta. ADC search ranks by dot products, and the candidates
   that decide recall are near-parallel to the query, so parallel
   residual error is what perturbs their scores; orthogonal residual
   error mostly perturbs scores of far-away points that were never
   competitive. Trading orthogonal for parallel accuracy raises recall
   at the same code size (Guo et al., "Accelerating Large-Scale
   Inference with Anisotropic Vector Quantization", ICML 2020).

encode() assigns codes under the same anisotropic cost, because
codebooks trained for a loss only pay off when assignment minimizes
that same loss. Codes, codebooks, decode, the ADC lut and scan, and the
search API are all unchanged by this: only which of the 256 centroids a
row maps to changes.

Subspaces train in parallel on threads with independent per-subspace
child seeds, so fits are bit-identical for a given seed regardless of
thread scheduling.
"""

from __future__ import annotations

import ctypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from density.engine import _accel

# Training reads at most this many rows so fit memory stays bounded even
# on multi-million row corpora.
_FIT_SAMPLE_CAP = 120_000
_KMEANS_BATCH = 8_192
_KMEANS_EPOCHS = 8
# Encoding materializes one float32 [chunk, 256] cost block at a time.
_ENCODE_CHUNK = 65_536
_N_CENTROIDS = 256
# Greedy k-means++ seeding: 2 + int(log(256)) candidate draws per
# centroid, the standard greedy candidate count (scikit-learn's rule),
# on a seeded subsample. Initialization only has to spread centroids
# sensibly before Lloyd polishes them, and its cost is quadratic-ish in
# sample size, so a subsample buys a large speedup for no measured
# quality change.
_KPP_CANDIDATES = 7
_KPP_SAMPLE_CAP = 20_000
# Full-batch Lloyd refinement: hard iteration caps for time, and a
# scale-free movement tolerance in the scikit-learn convention
# (converged when the squared Frobenius norm of the centroid shift
# drops below tol times the mean per-dimension variance of the
# training subvectors). Measured on the synth corpus, inertia at these
# caps sits within 0.2 percent of full convergence.
_LLOYD_MAX_ITERS = 25
_ANISO_MAX_ITERS = 10
_LLOYD_TOL = 1e-4
# Anisotropic weight: the residual component parallel to the data point
# costs eta times the orthogonal component, in training and in encode.
# Swept on held-out synth data at two scales (eta 2 to 16); recall@10
# peaks broadly at 8 and is flat within about 0.01 across 4 to 12.
_ANISO_ETA = 8.0
# Assignment materializes float32 [chunk, 256] blocks; 16384 keeps two
# such buffers per worker thread near L2/L3 size instead of DRAM.
_ASSIGN_CHUNK = 16_384
# Worker threads for per-subspace training. Assignment is memory
# bandwidth bound, and on a 12-core laptop measured throughput peaks
# near 8 workers and degrades beyond it, so more threads only thrash
# the shared bandwidth. None means min(8, cpu count); tests pin 1 to
# prove scheduling never changes the fitted codebooks.
_FIT_THREADS: int | None = None
_MAX_DEFAULT_FIT_THREADS = 8
# Per-worker ceiling on the anisotropic pair-column buffer. Every default
# and contracted configuration lands far under it (m = d / 4 gives dsub 4
# and about 5 MB at the sample cap), so the cap only bites for coarse m,
# where it trades recomputation for not exhausting RAM.
_ANISO_PAIR_BUDGET_BYTES = 64 << 20

# Guards against zero-norm subvectors in the anisotropic weight w / x2:
# for such rows the parallel direction is undefined and the cost falls
# back to the plain squared distance (t is 0 there anyway).
_NORM_FLOOR = 1e-30


# ---------------------------------------------------------------------------
# BLAS thread control. The per-subspace gemms here are tiny (inner
# dimension d // m, often 4), and OpenBLAS fanning each one out to its
# own thread pool fights the subspace-level thread pool: measured on
# the 12-core laptop, pinning BLAS to one thread during fit makes the
# whole fit about 2x faster. This is the same runtime API threadpoolctl
# uses; everything is probed defensively so an environment with a
# different BLAS simply skips the pin and stays correct, just slower.
# The pin is process-global while a fit runs and is always restored.
# ---------------------------------------------------------------------------

_BLAS_PROBE_LOCK = threading.Lock()
_BLAS_CTL: tuple | None = None
_BLAS_PROBED = False


def _openblas_thread_control() -> tuple | None:
    """(set_num_threads, get_num_threads) of the loaded OpenBLAS, or None.

    Probes the shared library numpy bundles (site-packages/numpy.libs)
    for the thread-control symbols of scipy-openblas wheels and of plain
    OpenBLAS builds. Cached after the first probe.
    """
    global _BLAS_CTL, _BLAS_PROBED
    with _BLAS_PROBE_LOCK:
        if _BLAS_PROBED:
            return _BLAS_CTL
        _BLAS_PROBED = True
        try:
            libs_dir = Path(np.__file__).resolve().parent.parent / "numpy.libs"
            candidates = sorted(libs_dir.glob("*openblas*")) if libs_dir.is_dir() else []
            for lib_path in candidates:
                handle = ctypes.CDLL(str(lib_path))
                for set_name, get_name in (
                    ("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
                    ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
                    ("openblas_set_num_threads", "openblas_get_num_threads"),
                ):
                    set_fn = getattr(handle, set_name, None)
                    get_fn = getattr(handle, get_name, None)
                    if set_fn is not None and get_fn is not None:
                        get_fn.restype = ctypes.c_int
                        set_fn.argtypes = [ctypes.c_int]
                        _BLAS_CTL = (set_fn, get_fn)
                        return _BLAS_CTL
        except Exception:
            _BLAS_CTL = None
        return _BLAS_CTL


# The BLAS pin is process-global state, so nested and concurrent fits share
# one saved value behind this lock and a depth counter. Without them, two
# fits in two threads would each save the other's pinned value of 1 and
# BLAS would stay pinned to a single thread for the rest of the process.
_BLAS_PIN_LOCK = threading.Lock()
_BLAS_PIN_DEPTH = 0
_BLAS_PIN_SAVED = 0


@contextmanager
def _blas_single_thread():
    """Pin BLAS to one thread for the duration, restoring on exit.

    Every gemm inside a fit always runs under the same pinned thread
    count, so fit results are bit-identical run to run whether or not
    the pin is available (without it they are just slower, computed
    under the process default which is likewise fixed).

    Reentrant and thread safe: only the outermost active pin records the
    original thread count, and only its exit restores it.
    """
    global _BLAS_PIN_DEPTH, _BLAS_PIN_SAVED
    ctl = _openblas_thread_control()
    if ctl is None:
        yield
        return
    set_fn, get_fn = ctl
    with _BLAS_PIN_LOCK:
        if _BLAS_PIN_DEPTH == 0:
            _BLAS_PIN_SAVED = int(get_fn())
            set_fn(1)
        _BLAS_PIN_DEPTH += 1
    try:
        yield
    finally:
        with _BLAS_PIN_LOCK:
            _BLAS_PIN_DEPTH -= 1
            if _BLAS_PIN_DEPTH == 0:
                set_fn(_BLAS_PIN_SAVED if _BLAS_PIN_SAVED > 0 else 1)


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

    def fit(self, X: np.ndarray, seed: int = 1337) -> PQ:
        """Learn per-subspace codebooks with seeded k-means.

        X: float32 [n, d]. Trains on a seeded sample of at most 120000
        rows. Each subspace runs greedy k-means++ initialization, 8
        mini-batch epochs of 8192 rows, full-batch Lloyd refinement to
        a movement tolerance, then anisotropic Lloyd refinement (see
        the module docstring), with dead centroids reassigned to the
        highest-distortion points throughout. Subspaces train in
        parallel on threads, each with an independent child rng derived
        from (seed, subspace index), so the same seed gives
        bit-identical codebooks no matter how threads are scheduled.
        Stored codes are cleared because codes from a previous codebook
        are meaningless. Returns self.
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

        def _train_one(j: int) -> None:
            sub = np.ascontiguousarray(sample[:, j * dsub : (j + 1) * dsub])
            # Child seed (seed, j): each subspace consumes its own rng
            # stream, so results do not depend on the order in which the
            # thread pool happens to run subspaces. Writes land in
            # disjoint codebook slices, so no synchronization is needed.
            codebooks[j] = _fit_subspace(sub, np.random.default_rng([seed, j]))

        workers = (
            _FIT_THREADS
            if _FIT_THREADS is not None
            else min(_MAX_DEFAULT_FIT_THREADS, os.cpu_count() or 1)
        )
        workers = max(1, min(workers, m))
        with _blas_single_thread():
            if workers == 1:
                for j in range(m):
                    _train_one(j)
            else:
                # Threads, not processes: the heavy work is gemms and numpy
                # reductions that release the GIL, and threads share the
                # sample array with zero copies.
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    # list() propagates the first worker exception, if any.
                    list(pool.map(_train_one, range(m)))
        self.m = m
        self._d = d
        self._codebooks = codebooks
        self._codes = None
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Anisotropic-cost assignment. X: float32 [n, d] -> uint8 [n, m].

        Each subvector maps to the centroid minimizing the same
        anisotropic quantization cost the codebooks were trained for
        (parallel residual weighted by eta), not the plain nearest
        centroid: assignment under the training loss is what realizes
        the loss's dot-product accuracy. On corpora a codebook
        represents exactly, both costs are zero at the true codeword,
        so exact reconstruction behavior is unchanged. Works in chunks
        of at most 65536 rows so cost blocks stay bounded.
        """
        cb = self._require_fitted()
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(f"expected [n, {self._d}] input, got shape {X.shape}")
        m, _, dsub = cb.shape
        n = X.shape[0]
        codes = np.empty((n, m), dtype=np.uint8)
        cnorm = np.einsum("mkd,mkd->mk", cb, cb)
        # Two reusable [chunk, 256] float32 blocks: dot products and costs.
        # _aniso_argmin chunks rows internally, so the buffers only need
        # to cover one assignment chunk regardless of the encode chunk.
        tbuf = np.empty((min(_ASSIGN_CHUNK, n), _N_CENTROIDS), dtype=np.float32)
        cbuf = np.empty_like(tbuf)
        for start in range(0, n, _ENCODE_CHUNK):
            stop = min(start + _ENCODE_CHUNK, n)
            chunk = X[start:stop]
            for j in range(m):
                sub = np.ascontiguousarray(chunk[:, j * dsub : (j + 1) * dsub])
                x2 = np.einsum("nd,nd->n", sub, sub)
                codes[start:stop, j] = _aniso_argmin(
                    sub, x2, cb[j], cnorm[j], _ANISO_ETA, tbuf, cbuf
                )
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
        candidates (widened to at least k, capped at n) with the given
        reranker and returns its top k.
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
        # Clamped at 0 in all three codecs: a negative k used to reach
        # np.empty((nq, -1)) and raise about negative dimensions instead of
        # returning an empty result.
        k_eff = max(0, min(int(k), n))
        # Widen the candidate pool to at least k_eff (and at most n): the
        # output has k_eff columns, so a depth below k_eff would leave the
        # rerank returning fewer rows than the slots it must fill, which
        # used to crash with a broadcast error for any k > rerank_depth.
        # Mirrors binary.py's widening so both cold codecs agree.
        depth = min(max(int(rerank_depth), k_eff), n)
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

                cand, _ = _topk_desc(adc, depth)
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
    def from_state(cls, state: dict[str, np.ndarray]) -> PQ:
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


# ---------------------------------------------------------------------------
# Training internals. All helpers are deterministic given their inputs;
# the only randomness enters through the per-subspace child rng.
# ---------------------------------------------------------------------------


def _partial_sums(
    assign: np.ndarray, x: np.ndarray, dsub: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-centroid member counts and coordinate sums.

    assign: int [n] centroid ids, x: float32 [n, >= dsub]. Returns
    (counts int64 [256], sums float64 [256, dsub]). bincount with
    weights is the fast deterministic scatter-add here (np.add.at is an
    order of magnitude slower), and its float64 accumulation keeps
    centroid means accurate on large clusters.
    """
    counts = np.bincount(assign, minlength=_N_CENTROIDS)
    sums = np.empty((_N_CENTROIDS, dsub), dtype=np.float64)
    for dd in range(dsub):
        sums[:, dd] = np.bincount(assign, weights=x[:, dd], minlength=_N_CENTROIDS)
    return counts, sums


def _augment(sub: np.ndarray) -> np.ndarray:
    """[n, dsub] -> [n, dsub + 1] with a trailing ones column.

    Lets one gemm produce the full assignment block: with G stacking
    -2 * centroids.T over the centroid squared norms, aug @ G equals
    ||c||^2 - 2 x.c per (row, centroid), so the add pass that would
    otherwise stream the whole block through memory again disappears.
    """
    n_s, dsub = sub.shape
    aug = np.empty((n_s, dsub + 1), dtype=np.float32)
    aug[:, :dsub] = sub
    aug[:, dsub] = 1.0
    return aug


def _assign_matrix(centroids: np.ndarray) -> np.ndarray:
    """The [dsub + 1, 256] gemm operand paired with _augment."""
    dsub = centroids.shape[1]
    g = np.empty((dsub + 1, _N_CENTROIDS), dtype=np.float32)
    g[:dsub] = -2.0 * centroids.T
    g[dsub] = np.einsum("kd,kd->k", centroids, centroids)
    return g


def _assign_plain(
    aug: np.ndarray,
    centroids: np.ndarray,
    buf: np.ndarray,
    assign: np.ndarray,
    dmin: np.ndarray,
) -> None:
    """Nearest-centroid assignment for every row of aug, chunked.

    aug: float32 [n, dsub + 1] (see _augment), buf: float32
    [>= chunk, 256] scratch, assign: int64 [n] and dmin: float32 [n]
    outputs. dmin holds ||c||^2 - 2 x.c at the argmin; the row term
    ||x||^2 is constant per row so the argmin does not need it, and
    callers add it back only where a true distance is required.
    """
    n_s = aug.shape[0]
    g = _assign_matrix(centroids)
    for start in range(0, n_s, _ASSIGN_CHUNK):
        stop = min(start + _ASSIGN_CHUNK, n_s)
        b = buf[: stop - start]
        np.dot(aug[start:stop], g, out=b)
        a = np.argmin(b, axis=1)
        assign[start:stop] = a
        dmin[start:stop] = b[np.arange(stop - start), a]


def _aniso_argmin(
    sub: np.ndarray,
    x2: np.ndarray,
    centroids: np.ndarray,
    cnorm: np.ndarray,
    eta: float,
    tbuf: np.ndarray,
    cbuf: np.ndarray,
    assign: np.ndarray | None = None,
    dmin: np.ndarray | None = None,
) -> np.ndarray:
    """Argmin of the anisotropic cost over centroids, chunked.

    The cost of assigning row x to centroid c is
        ||x - c||^2 + (eta - 1) * (((x - c) . x) / ||x||)^2,
    the plain squared distance plus an extra eta - 1 charge on the
    residual component parallel to x. With t = x . c it reduces (up to
    row-constant terms) to
        ||c||^2 - 2 * eta * t + ((eta - 1) / ||x||^2) * t^2,
    so one gemm for t plus elementwise passes suffice. sub: float32
    [n, dsub], x2: float32 [n] squared row norms, tbuf and cbuf:
    float32 [>= chunk, 256] scratch. Writes assign and dmin when given
    (dmin excludes row-constant terms; add eta * x2 for the true cost)
    and returns the argmin ids.
    """
    n_s = sub.shape[0]
    ct = np.ascontiguousarray(centroids.T)
    # w / ||x||^2 with a floor: zero-norm rows have t = 0, so their cost
    # degenerates to the plain distance to the origin-nearest centroid.
    winv = (float(eta) - 1.0) / np.maximum(x2, _NORM_FLOOR)
    out = np.empty(n_s, dtype=np.int64) if assign is None else assign
    for start in range(0, n_s, _ASSIGN_CHUNK):
        stop = min(start + _ASSIGN_CHUNK, n_s)
        rows = stop - start
        t = tbuf[:rows]
        cost = cbuf[:rows]
        np.dot(sub[start:stop], ct, out=t)
        np.multiply(t, t, out=cost)
        cost *= winv[start:stop, None]
        t *= -2.0 * float(eta)
        cost += t
        cost += cnorm[None, :]
        a = np.argmin(cost, axis=1)
        out[start:stop] = a
        if dmin is not None:
            dmin[start:stop] = cost[np.arange(rows), a]
    return out


def _split_dead_centroids(
    new_centroids: np.ndarray, sub: np.ndarray, counts: np.ndarray, dcost: np.ndarray
) -> None:
    """Reassign empty centroids to the worst-quantized rows, in place.

    counts: int64 [256] member counts, dcost: float32 [n] per-row
    assignment cost (any row-comparable scale). A dead centroid jumps
    to the row with the largest current cost, which splits the
    highest-distortion cluster: that row and its neighborhood peel off
    from their old centroid on the next assignment, cutting distortion
    where it is worst. argsort with a stable kind keeps ties, and hence
    fits, deterministic.
    """
    empty = np.flatnonzero(counts == 0)
    if empty.size == 0:
        return
    far = np.argsort(-dcost, kind="stable")[: empty.size]
    if far.size < empty.size:
        # Fewer rows than dead centroids (degenerate tiny inputs): recycle
        # rows deterministically so every centroid stays finite.
        far = np.resize(far, empty.size)
    new_centroids[empty] = sub[far]


def _kmeans_pp_init(
    sub: np.ndarray, x2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Greedy k-means++ initialization for one subspace.

    sub: float32 [n_s, dsub], x2: float32 [n_s] squared row norms.
    Returns float32 [256, dsub]. Each new centroid is sampled with
    probability proportional to the squared distance to the nearest
    already-chosen centroid (the classic k-means++ bias toward
    uncovered regions), and greedily: of 7 such candidates, keep the
    one that lowers the total potential most. This start lets Lloyd
    converge to a much lower distortion than random-row initialization,
    which wastes many of the 256 centroids on dense regions.
    """
    n_s, dsub = sub.shape
    centroids = np.empty((_N_CENTROIDS, dsub), dtype=np.float32)
    first = int(rng.integers(n_s))
    centroids[0] = sub[first]
    # d2[i] is the squared distance from row i to its nearest chosen
    # centroid so far; expanded form x2 - 2 x.c + c2 with a clamp at 0
    # because float cancellation can dip a true zero slightly negative.
    d2 = x2 - 2.0 * (sub @ centroids[0]) + np.float32(centroids[0] @ centroids[0])
    np.maximum(d2, 0.0, out=d2)
    cd2 = np.empty((_KPP_CANDIDATES, n_s), dtype=np.float32)
    for c in range(1, _N_CENTROIDS):
        # float64 cumsum: the sampling inverse-CDF must be exact enough
        # that searchsorted stays deterministic and in range.
        cum = np.cumsum(d2, dtype=np.float64)
        pot = float(cum[-1])
        if pot <= 0.0:
            # Every row already coincides with a centroid (fewer distinct
            # points than centroids): fill the rest from random rows,
            # matching the old degenerate-input behavior.
            centroids[c:] = sub[rng.integers(0, n_s, size=_N_CENTROIDS - c)]
            break
        draws = rng.random(_KPP_CANDIDATES) * pot
        cand = np.searchsorted(cum, draws, side="right")
        np.clip(cand, 0, n_s - 1, out=cand)
        # cd2[t, i]: squared distance from row i to candidate t, capped by
        # the current d2 so each row keeps its nearest-centroid distance.
        np.dot(sub[cand], sub.T, out=cd2)
        cd2 *= -2.0
        cd2 += x2[None, :]
        cd2 += x2[cand][:, None]
        np.maximum(cd2, 0.0, out=cd2)
        np.minimum(cd2, d2[None, :], out=cd2)
        pots = cd2.sum(axis=1, dtype=np.float64)
        best = int(np.argmin(pots))
        centroids[c] = sub[cand[best]]
        d2 = cd2[best].copy()
    return centroids


def _minibatch_epochs(
    sub: np.ndarray, aug: np.ndarray, centroids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Mini-batch k-means warm start: the contract floor of 8 epochs.

    sub: float32 [n_s, dsub], aug its augmented form, centroids float32
    [256, dsub] updated in place and returned. Batches of 8192 with the
    classic running-mean learning rate batch_count / lifetime_count:
    early batches move centroids a lot, later batches refine. Cheap
    relative to full-batch Lloyd, and it hands Lloyd a near-converged
    start so few full-batch iterations remain.
    """
    n_s, dsub = sub.shape
    counts = np.zeros(_N_CENTROIDS, dtype=np.int64)
    buf = np.empty((min(_KMEANS_BATCH, n_s), _N_CENTROIDS), dtype=np.float32)
    for _ in range(_KMEANS_EPOCHS):
        order = rng.permutation(n_s)
        for start in range(0, n_s, _KMEANS_BATCH):
            rows = order[start : start + _KMEANS_BATCH]
            xb = aug[rows]
            b = buf[: xb.shape[0]]
            np.dot(xb, _assign_matrix(centroids), out=b)
            assign = np.argmin(b, axis=1)
            bcount, sums = _partial_sums(assign, xb, dsub)
            counts += bcount
            upd = bcount > 0
            cf = counts[upd].astype(np.float32)[:, None]
            bf = bcount[upd].astype(np.float32)[:, None]
            centroids[upd] += (sums[upd].astype(np.float32) - bf * centroids[upd]) / cf
        dead = counts == 0
        if dead.any():
            # Reseed from real rows so degenerate inputs (fewer distinct
            # points than centroids) still produce usable codebooks.
            centroids[dead] = sub[rng.integers(0, n_s, size=int(dead.sum()))]
    return centroids


def _lloyd_refine(
    sub: np.ndarray,
    aug: np.ndarray,
    centroids: np.ndarray,
    max_iters: int = _LLOYD_MAX_ITERS,
    tol: float = _LLOYD_TOL,
) -> tuple[np.ndarray, int]:
    """Full-batch Lloyd refinement with empty-cluster splitting.

    sub: float32 [n_s, dsub], aug its augmented form, centroids float32
    [256, dsub] consumed as the starting point. Returns (centroids,
    iterations_run). Each iteration assigns every training row to its
    nearest centroid and moves each centroid to its members' mean, the
    exact descent step on quantization MSE. Stops when the squared
    Frobenius norm of the centroid shift drops below tol times the mean
    per-dimension variance of sub (the scikit-learn tolerance
    convention, so tol is scale-free), or at max_iters. No rng: given a
    starting point the refinement is fully deterministic.
    """
    n_s, dsub = sub.shape
    x2 = np.einsum("nd,nd->n", sub, sub)
    threshold = float(tol) * float(np.mean(np.var(sub, axis=0)))
    buf = np.empty((min(_ASSIGN_CHUNK, n_s), _N_CENTROIDS), dtype=np.float32)
    assign = np.empty(n_s, dtype=np.int64)
    dmin = np.empty(n_s, dtype=np.float32)
    iters_run = 0
    for _ in range(max_iters):
        iters_run += 1
        _assign_plain(aug, centroids, buf, assign, dmin)
        counts, sums = _partial_sums(assign, sub, dsub)
        new_centroids = centroids.astype(np.float64)
        occupied = counts > 0
        new_centroids[occupied] = sums[occupied] / counts[occupied][:, None]
        new_centroids = new_centroids.astype(np.float32)
        # dmin + x2 is the true squared distance to the assigned centroid.
        _split_dead_centroids(new_centroids, sub, counts, dmin + x2)
        delta = new_centroids - centroids
        shift = float(np.einsum("kd,kd->", delta, delta))
        centroids = new_centroids
        if shift <= threshold:
            break
    return centroids, iters_run


def _fill_pair_cols(
    sub: np.ndarray,
    winv: np.ndarray,
    pairs: list[tuple[int, int]],
    start: int,
    out: np.ndarray,
) -> None:
    """Write the scaled outer-product columns for pairs[start:start+width].

    out: float32 [n_s, width]. Column p holds x_a * x_b * w / ||x||^2 for
    pairs[start + p], the per-row term S_k accumulates over its members.
    """
    for p in range(out.shape[1]):
        a, b = pairs[start + p]
        np.multiply(sub[:, a], sub[:, b], out=out[:, p])
        out[:, p] *= winv


def _aniso_refine(
    sub: np.ndarray,
    x2: np.ndarray,
    centroids: np.ndarray,
    eta: float = _ANISO_ETA,
    max_iters: int = _ANISO_MAX_ITERS,
    tol: float = _LLOYD_TOL,
) -> tuple[np.ndarray, int]:
    """Anisotropic Lloyd refinement: score-aware weighted k-means.

    Alternating minimization of the anisotropic quantization loss
        sum_i ||x_i - c_a(i)||^2 + (eta - 1) * (((x_i - c_a(i)) . x_i) / ||x_i||)^2:
    assignment takes the exact cost argmin (_aniso_argmin) and the
    update solves each centroid's normal equations in closed form. For
    centroid k with members M, the stationarity condition is
        [|M| I + w S_k] c = eta * sum_{i in M} x_i,
    with w = eta - 1 and S_k = sum_{i in M} x_i x_i^T / ||x_i||^2 (the
    parallel-direction outer products), a dsub x dsub symmetric
    positive definite solve per centroid. Both half-steps decrease the
    same loss, so the refinement descends monotonically, and the same
    scale-free movement tolerance as plain Lloyd stops it early.

    sub: float32 [n_s, dsub], x2 its squared row norms, centroids
    consumed as the starting point (train plain k-means first: the
    anisotropic loss equals plain MSE at eta 1 and stays close for
    small residuals, so plain Lloyd's solution is a strong warm start).
    Returns (centroids float32 [256, dsub], iterations_run).
    """
    n_s, dsub = sub.shape
    threshold = float(tol) * float(np.mean(np.var(sub, axis=0)))
    w = float(eta) - 1.0
    winv = w / np.maximum(x2, _NORM_FLOOR)
    # Precompute the scaled outer-product columns once: pair (a, b) of
    # S_k needs sum over members of x_a * x_b / ||x||^2, so the per-row
    # products are iteration-invariant.
    pairs = [(a, b) for a in range(dsub) for b in range(a, dsub)]
    # Holding every pair column is iteration-invariant work, but the array
    # is n_s x dsub(dsub+1)/2 float32: at the default m = d / 4 that is a
    # few megabytes, while a legal but coarse m (m=8 on dim 768 gives
    # dsub=96) would want 2.2 GB per subspace, times the concurrent fit
    # workers. Past the budget the columns are rebuilt in pair-sized
    # chunks each iteration. Chunking is over pairs, never over rows, so
    # every bincount still reduces the same full column in the same order
    # and both paths produce bit-identical codebooks.
    chunk_pairs = max(1, _ANISO_PAIR_BUDGET_BYTES // max(1, n_s * 4))
    precomputed = chunk_pairs >= len(pairs)
    pair_cols = np.empty((n_s, min(chunk_pairs, len(pairs))), dtype=np.float32)
    if precomputed:
        _fill_pair_cols(sub, winv, pairs, 0, pair_cols)
    tbuf = np.empty((min(_ASSIGN_CHUNK, n_s), _N_CENTROIDS), dtype=np.float32)
    cbuf = np.empty_like(tbuf)
    assign = np.empty(n_s, dtype=np.int64)
    dmin = np.empty(n_s, dtype=np.float32)
    iters_run = 0
    for _ in range(max_iters):
        iters_run += 1
        cnorm = np.einsum("kd,kd->k", centroids, centroids)
        _aniso_argmin(
            sub, x2, centroids, cnorm, eta, tbuf, cbuf, assign=assign, dmin=dmin
        )
        counts, sums = _partial_sums(assign, sub, dsub)
        # Normal-equation systems: A_k = count_k I + w S_k is SPD for
        # occupied centroids, so the batched solve is well posed.
        a_mats = np.zeros((_N_CENTROIDS, dsub, dsub), dtype=np.float64)
        width = pair_cols.shape[1]
        for start in range(0, len(pairs), width):
            block = pairs[start : start + width]
            cols = pair_cols[:, : len(block)]
            if not precomputed:
                _fill_pair_cols(sub, winv, pairs, start, cols)
            for p, (a, b) in enumerate(block):
                s = np.bincount(assign, weights=cols[:, p], minlength=_N_CENTROIDS)
                a_mats[:, a, b] += s
                if a != b:
                    a_mats[:, b, a] += s
        diag = np.arange(dsub)
        a_mats[:, diag, diag] += counts[:, None]
        occupied = counts > 0
        new_centroids = centroids.astype(np.float64)
        rhs = float(eta) * sums[occupied]
        new_centroids[occupied] = np.linalg.solve(
            a_mats[occupied], rhs[:, :, None]
        )[:, :, 0]
        new_centroids = new_centroids.astype(np.float32)
        # dmin + eta * x2 restores the row-constant terms of the true
        # anisotropic cost, giving the worst-quantized rows for splits.
        _split_dead_centroids(new_centroids, sub, counts, dmin + np.float32(eta) * x2)
        delta = new_centroids - centroids
        shift = float(np.einsum("kd,kd->", delta, delta))
        centroids = new_centroids
        if shift <= threshold:
            break
    return centroids, iters_run


def _fit_subspace(sub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train one subspace codebook: k-means++, mini-batch, Lloyd, aniso.

    sub: float32 [n_sample, dsub]. Returns float32 [256, dsub]. The rng
    is private to this subspace (a child seed derived in fit), so the
    draw order here never interacts with other subspaces and fits stay
    bit-identical for a given seed.
    """
    n_s = sub.shape[0]
    x2 = np.einsum("nd,nd->n", sub, sub)
    if n_s > _KPP_SAMPLE_CAP:
        # Initialization quality saturates long before the full sample:
        # seed on a subsample, then every later phase sees all rows.
        rows = np.sort(rng.choice(n_s, size=_KPP_SAMPLE_CAP, replace=False))
        centroids = _kmeans_pp_init(sub[rows], x2[rows], rng)
    else:
        centroids = _kmeans_pp_init(sub, x2, rng)
    aug = _augment(sub)
    centroids = _minibatch_epochs(sub, aug, centroids, rng)
    centroids, _ = _lloyd_refine(sub, aug, centroids)
    centroids, _ = _aniso_refine(sub, x2, centroids)
    return centroids
