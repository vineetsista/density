"""Shared fixtures. Every test seeds explicitly, nothing reads the clock."""

from __future__ import annotations

import json

import numpy as np
import pytest

SEED = 1337


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture()
def unit_vectors(rng):
    """Clustered unit vectors so nearest-neighbor structure is real.

    2000 vectors, dim 32, 16 Gaussian clusters on the unit sphere.
    """
    n, d, k = 2000, 32, 16
    centers = rng.normal(size=(k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.integers(0, k, size=n)
    x = centers[assign] + 0.15 * rng.normal(size=(n, d)).astype(np.float32)
    x = x.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


@pytest.fixture()
def tiny_corpus(tmp_path, rng):
    """A small on-disk corpus: one JSONL trace file + matching embeddings.

    200 well-formed events across 20 traces, plus 4 malformed lines.
    Returns (corpus_dir, n_events, n_malformed).
    """
    corpus = tmp_path / "corpus"
    traces = corpus / "traces"
    traces.mkdir(parents=True)
    n_events, n_traces = 200, 20
    lines: list[bytes] = []
    system_prompt = "You are a helpful support agent. " * 40
    for i in range(n_events):
        trace = f"tr-{i % n_traces:04d}"
        role = ["system", "user", "assistant", "tool"][i % 4]
        event = {
            "trace_id": trace,
            "ts": 1_700_000_000_000_000 + i * 250_000,
            "role": role,
            "type": "message" if role != "tool" else "tool_result",
            "content": system_prompt if role == "system" else f"event {i} body é\U0001f680",
            "model": "gpt-x" if role == "assistant" else None,
            "custom_field": {"nested": i},
        }
        lines.append(json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode())
    malformed = [b"{truncated", b"\x00\x01\x02binary", b"just text", b"[1,2,3]"]
    all_lines = lines + malformed
    (traces / "part-0000.jsonl").write_bytes(b"\n".join(all_lines) + b"\n")

    emb = corpus / "embeddings"
    emb.mkdir()
    x = rng.normal(size=(256, 32)).astype(np.float32)
    np.save(emb / "vectors.npy", x)
    return corpus, len(lines), len(malformed)
