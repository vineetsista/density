"""Embedding readers: .npy, .parquet, .jsonl files, or a directory mix.

Every reader converges on the same output: float32 vectors of one agreed
dimension, L2-normalized row-wise so cosine similarity is a plain dot
product downstream. Zero rows are left as exact zeros rather than NaN.
Ids default to 0..n-1 int64 when the source format carries none.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from density.errors import IngestError

_SUPPORTED_SUFFIXES = (".npy", ".parquet", ".jsonl")
# Column name candidates, checked lowercase, in preference order.
_VECTOR_COLUMNS = ("vector", "embedding", "embeddings", "vec")
_ID_COLUMNS = ("id", "ids")


@dataclass
class EmbeddingSet:
    """Vectors ready for indexing.

    ids: [n] array, int64 or strings, one id per row of X.
    X: float32 [n, d], L2-normalized rows (zero rows stay exact zeros).
    source: the path the set was read from.
    normalized: True once rows are unit length.
    """

    ids: np.ndarray
    X: np.ndarray
    source: str
    normalized: bool


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Return X as float32 with unit L2 rows; zero rows stay exact zeros.

    X: [n, d] any float or int dtype. Output: new float32 [n, d] array.
    Idempotent up to float32 rounding.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # where= skips zero-norm rows entirely, so they come out as the zeros
    # from `out` instead of 0/0 NaN.
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)


def _ids_array(values: list[Any]) -> np.ndarray:
    """Build an ids array: int64 when every id is an int, strings otherwise."""
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return np.asarray(values, dtype=np.int64)
    return np.asarray([str(v) for v in values])


def _read_npy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-D numeric .npy file. Returns (ids 0..n-1, X float32)."""
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise IngestError(f"cannot read npy file {path}: {exc}") from exc
    if arr.ndim != 2:
        raise IngestError(f"{path}: expected a 2-D array, got shape {arr.shape}")
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise IngestError(f"{path}: expected a numeric array, got dtype {arr.dtype}")
    X = arr.astype(np.float32, copy=False)
    return np.arange(X.shape[0], dtype=np.int64), X


def _load_ids_npy(path: Path, n: int) -> np.ndarray:
    """Read a 1-D ids.npy companion file and validate its length."""
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise IngestError(f"cannot read ids file {path}: {exc}") from exc
    if arr.ndim != 1:
        raise IngestError(f"{path}: ids must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise IngestError(f"{path}: {arr.shape[0]} ids for {n} vectors")
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    return arr


def _matrix_from_list_column(col: pa.Array, path: Path, name: str) -> np.ndarray:
    """Turn a (fixed size) list or binary arrow column into float32 [n, d]."""
    col_type = col.type
    if col.null_count:
        raise IngestError(f"{path}: column {name!r} contains null rows")
    if pa.types.is_fixed_size_list(col_type):
        d = col_type.list_size
        flat = np.asarray(col.flatten(), dtype=np.float32)
        return flat.reshape(len(col), d)
    if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
        offsets = np.asarray(col.offsets)
        lengths = np.diff(offsets)
        if lengths.size and not np.all(lengths == lengths[0]):
            raise IngestError(f"{path}: column {name!r} has ragged row lengths")
        d = int(lengths[0]) if lengths.size else 0
        if d == 0:
            raise IngestError(f"{path}: column {name!r} has empty vectors")
        flat = np.asarray(col.flatten(), dtype=np.float32)
        return flat.reshape(len(col), d)
    if (
        pa.types.is_binary(col_type)
        or pa.types.is_large_binary(col_type)
        or pa.types.is_fixed_size_binary(col_type)
    ):
        blobs = col.to_pylist()
        lengths = {len(b) for b in blobs}
        if len(lengths) != 1:
            raise IngestError(f"{path}: column {name!r} has ragged blob lengths")
        (nbytes,) = lengths
        if nbytes == 0 or nbytes % 4:
            raise IngestError(f"{path}: column {name!r} blobs are not float32 vectors")
        # Blobs are little-endian float32 by on-disk convention.
        flat = np.frombuffer(b"".join(blobs), dtype="<f4")
        return flat.reshape(len(blobs), nbytes // 4).astype(np.float32)
    raise IngestError(f"{path}: column {name!r} has unsupported type {col_type}")


def _vectors_from_table(table: pa.Table, path: Path) -> np.ndarray:
    """Locate and decode the vector column of a parquet table."""
    names_lower = {n.lower(): n for n in table.column_names}
    for candidate in _VECTOR_COLUMNS:
        if candidate in names_lower:
            name = names_lower[candidate]
            return _matrix_from_list_column(table.column(name).combine_chunks(), path, name)
    # No conventional name: fall back to the first list-like or binary
    # column, then to per-dimension numeric columns.
    for name in table.column_names:
        col_type = table.schema.field(name).type
        if (
            pa.types.is_list(col_type)
            or pa.types.is_large_list(col_type)
            or pa.types.is_fixed_size_list(col_type)
            or pa.types.is_binary(col_type)
            or pa.types.is_large_binary(col_type)
            or pa.types.is_fixed_size_binary(col_type)
        ):
            return _matrix_from_list_column(table.column(name).combine_chunks(), path, name)
    numeric = [
        name
        for name in table.column_names
        if name.lower() not in _ID_COLUMNS
        and (
            pa.types.is_floating(table.schema.field(name).type)
            or pa.types.is_integer(table.schema.field(name).type)
        )
    ]
    if len(numeric) >= 2:
        cols = [
            table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
            for name in numeric
        ]
        return np.stack(cols, axis=1).astype(np.float32)
    raise IngestError(f"{path}: no vector column found (columns: {table.column_names})")


def _ids_from_table(table: pa.Table, n: int, path: Path) -> np.ndarray:
    """Read the id column if present, else default to 0..n-1."""
    names_lower = {name.lower(): name for name in table.column_names}
    for candidate in _ID_COLUMNS:
        if candidate not in names_lower:
            continue
        col = table.column(names_lower[candidate]).combine_chunks()
        if len(col) != n:
            raise IngestError(f"{path}: {len(col)} ids for {n} vectors")
        if pa.types.is_integer(col.type):
            return col.to_numpy(zero_copy_only=False).astype(np.int64)
        return np.asarray([str(v) for v in col.to_pylist()])
    return np.arange(n, dtype=np.int64)


def _read_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a parquet embeddings file. Returns (ids, X float32)."""
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise IngestError(f"cannot read parquet file {path}: {exc}") from exc
    X = _vectors_from_table(table, path)
    ids = _ids_from_table(table, X.shape[0], path)
    return ids, X


def _read_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a JSONL embeddings file of {"id": ..., "vector"|"embedding": [...]}."""
    rows: list[list[float]] = []
    ids: list[Any] = []
    dim: int | None = None
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise IngestError(f"cannot read jsonl file {path}: {exc}") from exc
    with handle:
        for line_index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path}:{line_index}: invalid json: {exc}") from exc
            if not isinstance(obj, dict):
                raise IngestError(f"{path}:{line_index}: expected a json object")
            vec = obj.get("vector", obj.get("embedding"))
            if not isinstance(vec, list):
                raise IngestError(f"{path}:{line_index}: no vector or embedding list")
            if dim is None:
                dim = len(vec)
            elif len(vec) != dim:
                raise IngestError(
                    f"{path}:{line_index}: vector dim {len(vec)} disagrees with {dim}"
                )
            rows.append(vec)
            if "id" in obj:
                ids.append(obj["id"])
    if not rows:
        raise IngestError(f"{path}: no vectors found")
    try:
        X = np.asarray(rows, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise IngestError(f"{path}: non-numeric vector values: {exc}") from exc
    if len(ids) == len(rows):
        return _ids_array(ids), X
    if not ids:
        return np.arange(len(rows), dtype=np.int64), X
    raise IngestError(f"{path}: only {len(ids)} of {len(rows)} lines carry an id")


def _read_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch a single embeddings file to its format reader."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _read_npy(path)
    if suffix == ".parquet":
        return _read_parquet(path)
    if suffix == ".jsonl":
        return _read_jsonl(path)
    raise IngestError(f"unsupported embeddings format: {path} (want .npy, .parquet, .jsonl)")


def _read_directory(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read every supported embeddings file directly under root.

    The vectors.npy + ids.npy pair is a first-class convention (it is what
    the synth generator writes): ids.npy supplies ids for vectors.npy and
    is never treated as a vector file of its own. Remaining supported
    files are read in sorted name order and concatenated, which keeps the
    result deterministic across filesystems.
    """
    files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    vectors_npy = root / "vectors.npy"
    ids_npy = root / "ids.npy"
    if vectors_npy in files and ids_npy in files:
        _, X = _read_npy(vectors_npy)
        parts.append((_load_ids_npy(ids_npy, X.shape[0]), X))
        files = [f for f in files if f not in (vectors_npy, ids_npy)]
    for f in files:
        parts.append(_read_file(f))
    if not parts:
        raise IngestError(
            f"{root}: no supported embedding files found (want .npy, .parquet, .jsonl)"
        )
    dims = {X.shape[1] for _, X in parts}
    if len(dims) != 1:
        raise IngestError(f"{root}: files disagree on vector dim: {sorted(dims)}")
    if len(parts) == 1:
        return parts[0]
    return (
        np.concatenate([ids for ids, _ in parts]),
        np.vstack([X for _, X in parts]),
    )


def read_embeddings(path: str | Path, expected_dim: int | None = None) -> EmbeddingSet:
    """Read embeddings from a file or directory into an EmbeddingSet.

    Accepts .npy (2-D float array), .parquet (id column plus a fixed-size
    list, list, binary blob, or per-dimension float columns), .jsonl
    ({"id": ..., "vector" or "embedding": [...]}), or a directory holding
    any mix. The dimension is sniffed from the data and every row must
    agree; expected_dim, when given, is validated against the sniffed
    dimension. Rows are L2-normalized (zero rows stay zeros). Ids default
    to 0..n-1 int64 when the format carries none.

    Raises IngestError for missing paths, unsupported formats, ragged or
    mismatched dimensions, and id/vector count mismatches.
    """
    root = Path(path)
    if not root.exists():
        raise IngestError(f"embeddings path does not exist: {root}")
    if root.is_dir():
        ids, X = _read_directory(root)
    else:
        ids, X = _read_file(root)
    if X.shape[0] == 0:
        raise IngestError(f"{root}: no vectors found")
    if expected_dim is not None and X.shape[1] != expected_dim:
        raise IngestError(
            f"{root}: vector dim {X.shape[1]} does not match expected_dim {expected_dim}"
        )
    return EmbeddingSet(ids=ids, X=l2_normalize(X), source=str(root), normalized=True)
