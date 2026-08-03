"""Query embedders for text search.

DEMO ONLY. The hashing embedder below exists so that the demo session can
run text queries against a store without any model, network, or GPU. It is
a deterministic bag-of-words feature hasher: it preserves coarse token
overlap, nothing more. It is NOT a semantic embedding model and search
quality with it is limited to keyword-ish matching. Production users supply
their own callable via Store.set_embedder(fn), where fn(text) returns a
float32 vector of the store's dimensionality, produced by the same model
that produced the stored vectors.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic feature-hashing text embedder. Demo only.

    Tokenizes to lowercase alphanumeric runs, hashes each token (and each
    adjacent token bigram, which adds a little phrase sensitivity) into one
    of dim buckets with a sign, and L2-normalizes the result. Stable across
    processes and machines because it uses blake2b, not the salted builtin
    hash.
    """

    demo_only = True

    def __init__(self, dim: int = 768) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def __call__(self, text: str) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        for gram in tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]:
            idx, sign = self._bucket(gram)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec
