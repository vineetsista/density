"""load_corpus tests: synth output and hand-built user layouts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from density.bench.datasets import load_corpus
from density.bench.synth import generate
from density.errors import IngestError
from density.ingest.embeddings import EmbeddingSet

SEED = 1337


def _write_jsonl(path, n=3):
    lines = [
        json.dumps(
            {"trace_id": f"t-{i}", "ts": i, "role": "user", "type": "message", "content": f"c{i}"},
            separators=(",", ":"),
        ).encode("utf-8")
        for i in range(n)
    ]
    path.write_bytes(b"\n".join(lines) + b"\n")


def test_load_corpus_on_synth_output(tmp_path):
    # 0.01 GB at dim 64 keeps the embedding floor small enough for the
    # byte target while exercising the real generator layout.
    stats = generate(0.01, tmp_path / "raw", seed=SEED, dim=64)
    trace_paths, emb = load_corpus(tmp_path / "raw")

    assert len(trace_paths) == stats.files
    assert all(p.is_file() and p.suffix == ".jsonl" for p in trace_paths)
    assert trace_paths == sorted(trace_paths)
    assert isinstance(emb, EmbeddingSet)
    assert emb.X.shape == (stats.n_vectors, 64)
    assert emb.X.dtype == np.float32
    assert emb.ids.shape == (stats.n_vectors,)
    # read_embeddings normalizes; unit rows are the downstream contract.
    norms = np.linalg.norm(emb.X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_load_corpus_flat_traces_only(tmp_path):
    root = tmp_path / "user"
    root.mkdir()
    _write_jsonl(root / "a.jsonl")
    _write_jsonl(root / "b.jsonl.1")  # numeric shard suffix counts too
    (root / "notes.txt").write_text("not a trace")

    trace_paths, emb = load_corpus(root)
    assert [p.name for p in trace_paths] == ["a.jsonl", "b.jsonl.1"]
    assert emb is None  # missing embeddings is None, never an error


def test_load_corpus_embeddings_only(tmp_path):
    root = tmp_path / "vecs"
    (root / "embeddings").mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    np.save(root / "embeddings" / "vectors.npy", rng.normal(size=(10, 8)).astype(np.float32))

    trace_paths, emb = load_corpus(root)
    assert trace_paths == []
    assert emb is not None and emb.X.shape == (10, 8)


def test_load_corpus_excludes_embeddings_jsonl_from_traces(tmp_path):
    # A flat layout where embeddings arrive as JSONL: those lines are
    # vectors and must not be double-counted as trace files.
    root = tmp_path / "mixed"
    (root / "embeddings").mkdir(parents=True)
    _write_jsonl(root / "log.jsonl")
    vec_lines = [
        json.dumps({"id": i, "vector": [float(i), 1.0, 0.0, 0.5]}).encode("utf-8")
        for i in range(6)
    ]
    (root / "embeddings" / "emb.jsonl").write_bytes(b"\n".join(vec_lines) + b"\n")

    trace_paths, emb = load_corpus(root)
    assert [p.name for p in trace_paths] == ["log.jsonl"]
    assert emb is not None and emb.X.shape == (6, 4)


def test_load_corpus_single_jsonl_file(tmp_path):
    path = tmp_path / "solo.jsonl"
    _write_jsonl(path)
    trace_paths, emb = load_corpus(path)
    assert trace_paths == [path.resolve()]
    assert emb is None


def test_load_corpus_single_npy_file(tmp_path):
    path = tmp_path / "solo.npy"
    rng = np.random.default_rng(SEED)
    np.save(path, rng.normal(size=(5, 4)).astype(np.float32))
    trace_paths, emb = load_corpus(path)
    assert trace_paths == []
    assert emb is not None and emb.X.shape == (5, 4)


def test_load_corpus_empty_dir_raises(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(IngestError, match="nothing to load"):
        load_corpus(root)


def test_load_corpus_empty_embeddings_dir_is_missing(tmp_path):
    # An empty embeddings/ folder means "no embeddings", not a read error,
    # but the corpus as a whole still needs traces or vectors.
    root = tmp_path / "hollow"
    (root / "embeddings").mkdir(parents=True)
    with pytest.raises(IngestError, match="nothing to load"):
        load_corpus(root)
    _write_jsonl(root / "a.jsonl")
    trace_paths, emb = load_corpus(root)
    assert len(trace_paths) == 1 and emb is None


def test_load_corpus_missing_path_raises(tmp_path):
    with pytest.raises(IngestError, match="does not exist"):
        load_corpus(tmp_path / "nope")


def test_load_corpus_broken_embeddings_raise(tmp_path):
    # Present-but-unreadable embeddings are a real problem, unlike
    # absent ones: the error must surface, not silently become None.
    root = tmp_path / "broken"
    (root / "embeddings").mkdir(parents=True)
    _write_jsonl(root / "a.jsonl")
    (root / "embeddings" / "vectors.npy").write_bytes(b"not an npy file")
    with pytest.raises(IngestError):
        load_corpus(root)
