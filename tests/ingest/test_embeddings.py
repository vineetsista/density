"""read_embeddings: format readers, dim validation, L2 normalization."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from density.errors import IngestError
from density.ingest.embeddings import EmbeddingSet, l2_normalize, read_embeddings


def _assert_unit_rows(X: np.ndarray) -> None:
    norms = np.linalg.norm(X, axis=1)
    nonzero = norms > 0
    assert np.allclose(norms[nonzero], 1.0, atol=1e-5)


def test_npy_reader(tmp_path, rng) -> None:
    x = rng.normal(size=(20, 8)).astype(np.float64)
    x[3] = 0.0  # zero row must survive as exact zeros
    f = tmp_path / "vecs.npy"
    np.save(f, x)
    es = read_embeddings(f)
    assert isinstance(es, EmbeddingSet)
    assert es.X.dtype == np.float32
    assert es.X.shape == (20, 8)
    assert es.normalized is True
    assert es.ids.dtype == np.int64
    assert np.array_equal(es.ids, np.arange(20, dtype=np.int64))
    _assert_unit_rows(es.X)
    assert np.all(es.X[3] == 0.0)


def test_npy_rejects_1d(tmp_path, rng) -> None:
    f = tmp_path / "bad.npy"
    np.save(f, rng.normal(size=17))
    with pytest.raises(IngestError):
        read_embeddings(f)


def test_parquet_fixed_size_list(tmp_path, rng) -> None:
    n, d = 12, 6
    x = rng.normal(size=(n, d)).astype(np.float32)
    flat = pa.array(x.reshape(-1), type=pa.float32())
    vec = pa.FixedSizeListArray.from_arrays(flat, d)
    ids = pa.array([f"doc-{i:03d}" for i in range(n)])
    pq.write_table(pa.table({"id": ids, "vector": vec}), tmp_path / "e.parquet")
    es = read_embeddings(tmp_path / "e.parquet")
    assert es.X.shape == (n, d)
    assert es.X.dtype == np.float32
    assert list(es.ids) == [f"doc-{i:03d}" for i in range(n)]
    _assert_unit_rows(es.X)
    assert np.allclose(es.X, l2_normalize(x), atol=1e-6)


def test_parquet_list_column_named_embedding_no_ids(tmp_path, rng) -> None:
    n, d = 9, 5
    x = rng.normal(size=(n, d)).astype(np.float64)
    vec = pa.array([row.tolist() for row in x], type=pa.list_(pa.float64()))
    pq.write_table(pa.table({"embedding": vec}), tmp_path / "e.parquet")
    es = read_embeddings(tmp_path / "e.parquet")
    assert es.X.shape == (n, d)
    assert np.array_equal(es.ids, np.arange(n, dtype=np.int64))
    _assert_unit_rows(es.X)


def test_parquet_ragged_list_raises(tmp_path) -> None:
    vec = pa.array([[1.0, 2.0], [1.0, 2.0, 3.0]], type=pa.list_(pa.float32()))
    pq.write_table(pa.table({"vector": vec}), tmp_path / "e.parquet")
    with pytest.raises(IngestError):
        read_embeddings(tmp_path / "e.parquet")


def test_jsonl_reader_int_ids(tmp_path, rng) -> None:
    n, d = 7, 4
    x = rng.normal(size=(n, d))
    f = tmp_path / "e.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"id": i * 10, "vector": x[i].tolist()}) + "\n")
    es = read_embeddings(f)
    assert es.X.shape == (n, d)
    assert es.ids.dtype == np.int64
    assert list(es.ids) == [i * 10 for i in range(n)]
    _assert_unit_rows(es.X)


def test_jsonl_reader_embedding_key_string_ids(tmp_path, rng) -> None:
    n, d = 5, 3
    x = rng.normal(size=(n, d))
    f = tmp_path / "e.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"id": f"v{i}", "embedding": x[i].tolist()}) + "\n")
    es = read_embeddings(f)
    assert list(es.ids) == [f"v{i}" for i in range(n)]
    assert es.X.shape == (n, d)


def test_jsonl_missing_ids_default_to_arange(tmp_path, rng) -> None:
    f = tmp_path / "e.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({"vector": rng.normal(size=3).tolist()}) + "\n")
    es = read_embeddings(f)
    assert np.array_equal(es.ids, np.arange(4, dtype=np.int64))


def test_jsonl_dim_mismatch_raises(tmp_path) -> None:
    f = tmp_path / "e.jsonl"
    f.write_text('{"vector":[1.0,2.0]}\n{"vector":[1.0,2.0,3.0]}\n')
    with pytest.raises(IngestError):
        read_embeddings(f)


def test_expected_dim_mismatch_raises(tmp_path, rng) -> None:
    f = tmp_path / "v.npy"
    np.save(f, rng.normal(size=(4, 8)).astype(np.float32))
    with pytest.raises(IngestError):
        read_embeddings(f, expected_dim=16)
    es = read_embeddings(f, expected_dim=8)
    assert es.X.shape == (4, 8)


def test_directory_vectors_npy_with_ids(tmp_path, rng) -> None:
    d = tmp_path / "emb"
    d.mkdir()
    x = rng.normal(size=(6, 4)).astype(np.float32)
    np.save(d / "vectors.npy", x)
    np.save(d / "ids.npy", np.arange(100, 106, dtype=np.int64))
    es = read_embeddings(d)
    assert es.X.shape == (6, 4)
    assert list(es.ids) == list(range(100, 106))
    _assert_unit_rows(es.X)


def test_directory_vectors_npy_without_ids(tiny_corpus) -> None:
    corpus, _, _ = tiny_corpus
    es = read_embeddings(corpus / "embeddings")
    assert es.X.shape == (256, 32)
    assert np.array_equal(es.ids, np.arange(256, dtype=np.int64))
    _assert_unit_rows(es.X)


def test_directory_single_supported_file(tmp_path, rng) -> None:
    d = tmp_path / "emb"
    d.mkdir()
    n, dim = 8, 4
    x = rng.normal(size=(n, dim)).astype(np.float32)
    flat = pa.array(x.reshape(-1), type=pa.float32())
    vec = pa.FixedSizeListArray.from_arrays(flat, dim)
    pq.write_table(pa.table({"vector": vec}), d / "e.parquet")
    (d / "readme.txt").write_text("not embeddings")
    es = read_embeddings(d)
    assert es.X.shape == (n, dim)


def test_directory_with_nothing_supported_raises(tmp_path) -> None:
    d = tmp_path / "emb"
    d.mkdir()
    (d / "readme.txt").write_text("nope")
    with pytest.raises(IngestError):
        read_embeddings(d)


def test_missing_path_raises() -> None:
    with pytest.raises(IngestError):
        read_embeddings("/nonexistent/definitely/not/here.npy")


def test_unsupported_extension_raises(tmp_path) -> None:
    f = tmp_path / "e.csv"
    f.write_text("1,2,3\n")
    with pytest.raises(IngestError):
        read_embeddings(f)


def test_normalization_idempotent(rng) -> None:
    x = rng.normal(size=(50, 16)).astype(np.float32)
    x[7] = 0.0
    once = l2_normalize(x)
    twice = l2_normalize(once)
    assert np.allclose(once, twice, atol=1e-6)
    assert np.all(once[7] == 0.0)
    assert np.all(twice[7] == 0.0)
    assert not np.any(np.isnan(twice))


def test_reading_already_normalized_data_is_stable(tmp_path, rng) -> None:
    x = rng.normal(size=(10, 8)).astype(np.float32)
    x = l2_normalize(x)
    f = tmp_path / "v.npy"
    np.save(f, x)
    es = read_embeddings(f)
    assert np.allclose(es.X, x, atol=1e-6)
    assert es.normalized is True
