"""Tests for matryoshka truncation."""

from __future__ import annotations

import numpy as np
import pytest

from density.engine.embed.matryoshka import truncate


def test_truncate_renormalizes(unit_vectors: np.ndarray) -> None:
    out = truncate(unit_vectors, 16)
    assert out.shape == (unit_vectors.shape[0], 16)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # Direction must match the raw slice, only the length changes.
    raw = unit_vectors[:, :16]
    expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert np.allclose(out, expected, atol=1e-6)


def test_zero_rows_stay_zero(rng: np.random.Generator) -> None:
    x = rng.normal(size=(10, 8)).astype(np.float32)
    x[3] = 0.0
    x[7, :4] = 0.0  # nonzero row whose leading slice is all zero
    out = truncate(x, 4)
    assert np.all(out[3] == 0.0)
    assert np.all(out[7] == 0.0)
    other = np.delete(np.arange(10), [3, 7])
    assert np.allclose(np.linalg.norm(out[other], axis=1), 1.0, atol=1e-5)


def test_full_width_is_plain_renormalize(unit_vectors: np.ndarray) -> None:
    out = truncate(unit_vectors, unit_vectors.shape[1])
    assert np.allclose(out, unit_vectors, atol=1e-6)


def test_input_is_not_modified(rng: np.random.Generator) -> None:
    x = rng.normal(size=(5, 6)).astype(np.float32)
    before = x.copy()
    truncate(x, 3)
    assert np.array_equal(x, before)


def test_dims_validation(rng: np.random.Generator) -> None:
    x = rng.normal(size=(4, 8)).astype(np.float32)
    with pytest.raises(ValueError):
        truncate(x, 9)
    with pytest.raises(ValueError):
        truncate(x, 0)
    with pytest.raises(ValueError):
        truncate(x[0], 2)
