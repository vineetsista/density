"""Parity and fallback tests for the compiled acceleration kernels.

The compiled path is optional by design: on machines without a C++
toolchain or pybind11 the parity tests skip with a clear reason and the
fallback tests still guarantee the dispatch answers correctly.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from density.engine._accel import fallback

RTOL = 1e-5
NS = [1, 7, 1000]
DIMS = [384, 768, 1536, 3072]
MS = [96, 192]


def _load_compiled():
    try:
        from density.engine._accel.build import load_compiled

        return load_compiled()
    except Exception:
        return None


COMPILED = _load_compiled()

needs_compiled = pytest.mark.skipif(
    COMPILED is None,
    reason=(
        "compiled kernels unavailable (no C++ toolchain or pybind11); "
        "numpy fallback is the active path on this machine"
    ),
)


def _assert_parity(compiled_out: np.ndarray, fallback_out: np.ndarray) -> None:
    assert compiled_out.shape == fallback_out.shape
    a = np.asarray(compiled_out, dtype=np.float64)
    b = np.asarray(fallback_out, dtype=np.float64)
    if b.size == 0:
        return
    # Relative to the result scale: per-element ratios blow up on scores
    # that legitimately cancel toward zero.
    scale = max(1.0, float(np.max(np.abs(b))))
    assert float(np.max(np.abs(a - b))) / scale <= RTOL


@needs_compiled
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("n", NS)
def test_sq8_parity(n: int, d: int) -> None:
    rng = np.random.default_rng(1337 + 7 * n + d)
    codes = rng.integers(0, 256, size=(n, d), dtype=np.uint8)
    qa = rng.standard_normal(d).astype(np.float32)
    const = float(rng.standard_normal())
    got = COMPILED.sq8_scores(codes, qa, const)
    want = fallback.sq8_scores(codes, qa, const)
    assert got.dtype == np.float32
    _assert_parity(got, want)


@needs_compiled
@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("n", NS)
def test_pq_parity(n: int, m: int) -> None:
    rng = np.random.default_rng(1337 + 7 * n + m)
    codes = rng.integers(0, 256, size=(n, m), dtype=np.uint8)
    lut = rng.standard_normal((m, 256)).astype(np.float32)
    got = COMPILED.pq_adc_scan(codes, lut)
    want = fallback.pq_adc_scan(codes, lut)
    assert got.dtype == np.float32
    _assert_parity(got, want)


@needs_compiled
@pytest.mark.parametrize("d", DIMS)
@pytest.mark.parametrize("n", NS)
def test_hamming_parity(n: int, d: int) -> None:
    b = d // 8
    rng = np.random.default_rng(1337 + 7 * n + b)
    codes = rng.integers(0, 256, size=(n, b), dtype=np.uint8)
    q = rng.integers(0, 256, size=b, dtype=np.uint8)
    got = COMPILED.hamming_scan(codes, q)
    want = fallback.hamming_scan(codes, q)
    assert got.dtype == np.int32
    # Hamming is integer arithmetic: both paths must agree exactly.
    np.testing.assert_array_equal(got, want)


@needs_compiled
def test_hamming_tail_bytes_parity() -> None:
    # b = 11 is not a multiple of 8, exercising the byte-wise tail loop
    # after the 64-bit word loop in the compiled kernel.
    rng = np.random.default_rng(1337)
    codes = rng.integers(0, 256, size=(50, 11), dtype=np.uint8)
    q = rng.integers(0, 256, size=11, dtype=np.uint8)
    np.testing.assert_array_equal(
        COMPILED.hamming_scan(codes, q), fallback.hamming_scan(codes, q)
    )


def test_fallback_empty_inputs() -> None:
    d, m, b = 64, 16, 8
    out = fallback.sq8_scores(
        np.empty((0, d), dtype=np.uint8), np.zeros(d, dtype=np.float32), 0.0
    )
    assert out.shape == (0,) and out.dtype == np.float32
    out = fallback.pq_adc_scan(
        np.empty((0, m), dtype=np.uint8), np.zeros((m, 256), dtype=np.float32)
    )
    assert out.shape == (0,) and out.dtype == np.float32
    out = fallback.hamming_scan(
        np.empty((0, b), dtype=np.uint8), np.zeros(b, dtype=np.uint8)
    )
    assert out.shape == (0,) and out.dtype == np.int32


@needs_compiled
def test_compiled_empty_inputs() -> None:
    d, m, b = 64, 16, 8
    out = COMPILED.sq8_scores(
        np.empty((0, d), dtype=np.uint8), np.zeros(d, dtype=np.float32), 0.0
    )
    assert out.shape == (0,) and out.dtype == np.float32
    out = COMPILED.pq_adc_scan(
        np.empty((0, m), dtype=np.uint8), np.zeros((m, 256), dtype=np.float32)
    )
    assert out.shape == (0,) and out.dtype == np.float32
    out = COMPILED.hamming_scan(
        np.empty((0, b), dtype=np.uint8), np.zeros(b, dtype=np.uint8)
    )
    assert out.shape == (0,) and out.dtype == np.int32


def test_build_failure_falls_back_silently(monkeypatch) -> None:
    import density.engine
    import density.engine._accel.build as build

    def boom():
        raise RuntimeError("simulated toolchain failure")

    saved = sys.modules["density.engine._accel"]
    monkeypatch.setattr(build, "load_compiled", boom)
    sys.modules.pop("density.engine._accel")
    try:
        fresh = importlib.import_module("density.engine._accel")
        assert fresh.ACCEL_ACTIVE is False
        # The dispatch must still answer, through the numpy implementations.
        assert fresh.sq8_scores is fallback.sq8_scores
        assert fresh.pq_adc_scan is fallback.pq_adc_scan
        assert fresh.hamming_scan is fallback.hamming_scan
        rng = np.random.default_rng(1337)
        codes = rng.integers(0, 256, size=(5, 32), dtype=np.uint8)
        qa = rng.standard_normal(32).astype(np.float32)
        out = fresh.sq8_scores(codes, qa, 0.25)
        assert out.shape == (5,) and out.dtype == np.float32
    finally:
        # Restore the original module object so every other test and any
        # already-imported consumer keeps seeing the real dispatch.
        sys.modules["density.engine._accel"] = saved
        density.engine._accel = saved
