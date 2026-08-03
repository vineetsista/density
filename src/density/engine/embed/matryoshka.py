"""Matryoshka truncation: keep leading dimensions, re-normalize.

Matryoshka-trained embedders front-load information into the leading
dimensions, so truncating and re-normalizing is a cheap extra compression
lever. Off by default everywhere; exposed as an optional flag on audit
and store APIs.
"""

from __future__ import annotations

import numpy as np


def truncate(X: np.ndarray, dims: int) -> np.ndarray:
    """Slices the first dims dimensions and re-normalizes each row to
    unit L2 norm.

    X: float32 [n, d]. dims: 1 <= dims <= d, else ValueError. Returns a
    new float32 [n, dims] matrix; the input is never modified. Rows that
    are all zero stay all zero, matching the ingest convention that zero
    vectors are left as zeros.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("truncate expects a 2-D [n, d] matrix")
    d = X.shape[1]
    if not 1 <= int(dims) <= d:
        raise ValueError(f"dims must be in [1, {d}], got {dims}")
    out = np.array(X[:, : int(dims)], dtype=np.float32, copy=True)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    out[nonzero] /= norms[nonzero]
    return out
