"""Pure numpy implementations of the acceleration kernels.

These are the reference semantics. The compiled C++ kernels in cpp/ must
match these within 1e-5 relative error. All three functions are careful to
bound peak memory: they process rows in chunks instead of materializing
large float temporaries, because corpora reach millions of rows.

Shared conventions:
- codes matrices are C-contiguous uint8, one row per stored vector
- outputs are 1-D over rows, higher score = better match (hamming returns
  raw distances, the caller negates when it needs a score)

Input validation matches cpp/bindings.cpp message for message. Without it
the two backends disagree on invalid input: numpy would broadcast a
mis-shaped query into a plausible-looking answer where the compiled kernel
raises, so a bug would surface only on machines without a compiler.
"""

from __future__ import annotations

import numpy as np

# Rows processed per chunk. 65536 rows x 768 dims x 4 bytes = 192 MB peak
# for the widest supported dim, which stays comfortable on a laptop.
_CHUNK_ROWS = 65_536

# Popcount lookup for one byte. 256 entries, built once at import time.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def sq8_scores(codes: np.ndarray, qa: np.ndarray, const: float) -> np.ndarray:
    """Approximate dot products against SQ8-encoded vectors.

    codes: uint8 [n, d], qa: float32 [d] (query pre-multiplied by the
    per-dimension scale), const: scalar (query dotted with per-dimension
    minimums). Returns float32 [n].

    The affine decode is x ~= scale * code + mins, so
    dot(q, x) = (q * scale) @ code + q @ mins = qa @ code + const,
    which lets us score without ever materializing decoded float vectors.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    qa = np.ascontiguousarray(qa, dtype=np.float32)
    if codes.ndim != 2:
        raise ValueError("sq8_scores: codes must be 2-D [n, d]")
    if qa.ndim != 1:
        raise ValueError("sq8_scores: qa must be 1-D [d]")
    if qa.shape[0] != codes.shape[1]:
        raise ValueError("sq8_scores: qa length must equal codes.shape[1]")
    n = codes.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        # uint8 @ float32 upcasts chunk-by-chunk, bounding the temporary.
        out[start:stop] = codes[start:stop].astype(np.float32) @ qa
    out += np.float32(const)
    return out


def pq_adc_scan(codes: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Asymmetric distance computation scan for product quantization.

    codes: uint8 [n, m] (one centroid index per subquantizer), lut: float32
    [m, 256] (per-subquantizer score of the query against each centroid).
    Returns float32 [n]: sum over subquantizers of lut[j, codes[i, j]].

    The scan walks subquantizer-by-subquantizer so each step is a flat
    256-entry gather, which numpy executes as a single vectorized take.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    lut = np.ascontiguousarray(lut, dtype=np.float32)
    if codes.ndim != 2:
        raise ValueError("pq_adc_scan: codes must be 2-D [n, m]")
    if lut.ndim != 2:
        raise ValueError("pq_adc_scan: lut must be 2-D [m, 256]")
    if lut.shape[0] != codes.shape[1]:
        raise ValueError("pq_adc_scan: lut.shape[0] must equal codes.shape[1]")
    if lut.shape[1] != 256:
        raise ValueError("pq_adc_scan: lut.shape[1] must be 256")
    n, m = codes.shape
    out = np.zeros(n, dtype=np.float32)
    for j in range(m):
        out += lut[j][codes[:, j]]
    return out


def hamming_scan(codes: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamming distances between a packed binary query and packed codes.

    codes: uint8 [n, b] (d bits packed into b = d / 8 bytes), q: uint8 [b].
    Returns int32 [n]: number of differing bits per row. Lower is closer.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    q = np.ascontiguousarray(q, dtype=np.uint8)
    if codes.ndim != 2:
        raise ValueError("hamming_scan: codes must be 2-D [n, b]")
    if q.ndim != 1:
        raise ValueError("hamming_scan: q must be 1-D [b]")
    if q.shape[0] != codes.shape[1]:
        raise ValueError("hamming_scan: q length must equal codes.shape[1]")
    n = codes.shape[0]
    out = np.empty(n, dtype=np.int32)
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        xored = np.bitwise_xor(codes[start:stop], q)
        out[start:stop] = _POPCOUNT[xored].sum(axis=1, dtype=np.int32)
    return out
