"""Wall-clock comparison of the compiled kernels against the numpy fallback.

Bench-only module: wall-clock timing with time.perf_counter is allowed here
and nowhere else in the library. The Phase 5 harness calls measure_kernels
and records the result next to ACCEL_ACTIVE, so the return value is plain
python (ints, floats, bools, None) and JSON-serializable as-is.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from density.engine._accel import fallback
from density.engine._accel.build import load_compiled

# Rows used to warm each path before the timed run, so one-time costs
# (lazy allocations, code paging) do not pollute the measurement.
_WARMUP_ROWS = 256


def _timed(fn: Callable[..., np.ndarray], *args: Any) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    out = fn(*args)
    return time.perf_counter() - start, out


def _max_rel_diff(compiled_out: np.ndarray, fallback_out: np.ndarray) -> float:
    a = np.asarray(compiled_out, dtype=np.float64)
    b = np.asarray(fallback_out, dtype=np.float64)
    if b.size == 0:
        return 0.0
    # Relative to the problem scale: per-element ratios explode when a score
    # happens to land near zero, which says nothing about kernel accuracy.
    scale = max(1.0, float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)


def _measure_one(
    name: str,
    compiled: Any,
    args: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    fallback_fn = getattr(fallback, name)
    warm_args = (args[0][:_WARMUP_ROWS],) + args[1:]
    fallback_fn(*warm_args)
    fallback_s, fallback_out = _timed(fallback_fn, *args)
    entry: dict[str, Any] = {
        "fallback_s": fallback_s,
        "compiled_s": None,
        "speedup": None,
        "max_rel_diff": None,
    }
    if compiled is None:
        return entry
    compiled_fn = getattr(compiled, name)
    compiled_fn(*warm_args)
    compiled_s, compiled_out = _timed(compiled_fn, *args)
    entry["compiled_s"] = compiled_s
    entry["speedup"] = fallback_s / compiled_s if compiled_s > 0 else None
    entry["max_rel_diff"] = _max_rel_diff(compiled_out, fallback_out)
    return entry


def measure_kernels(
    n: int = 1_000_000, d: int = 768, m: int = 192, seed: int = 1337
) -> dict[str, Any]:
    """Time all three kernels on seeded random inputs, compiled vs fallback.

    Shapes: sq8 codes uint8 [n, d], pq codes uint8 [n, m] with lut float32
    [m, 256], hamming codes uint8 [n, d // 8]. When the compiled path is
    unavailable the per-kernel compiled_s, speedup, and max_rel_diff are
    None and the fallback timings still fill in.
    """
    compiled = load_compiled()
    b = d // 8
    rng = np.random.default_rng(seed)
    kernels: dict[str, Any] = {}

    # Inputs are generated and released one kernel at a time: at the default
    # n the sq8 codes alone are n * d bytes and machines running the bench
    # may not fit all three matrices at once.
    codes = rng.integers(0, 256, size=(n, d), dtype=np.uint8)
    qa = rng.standard_normal(d).astype(np.float32)
    const = float(rng.standard_normal())
    kernels["sq8_scores"] = _measure_one("sq8_scores", compiled, (codes, qa, const))
    del codes

    codes = rng.integers(0, 256, size=(n, m), dtype=np.uint8)
    lut = rng.standard_normal((m, 256)).astype(np.float32)
    kernels["pq_adc_scan"] = _measure_one("pq_adc_scan", compiled, (codes, lut))
    del codes

    codes = rng.integers(0, 256, size=(n, b), dtype=np.uint8)
    q = rng.integers(0, 256, size=b, dtype=np.uint8)
    kernels["hamming_scan"] = _measure_one("hamming_scan", compiled, (codes, q))
    del codes

    from density.engine import _accel

    return {
        "n": n,
        "d": d,
        "m": m,
        "b": b,
        "seed": seed,
        "accel_active": bool(_accel.ACCEL_ACTIVE),
        "accel_available": compiled is not None,
        "kernels": kernels,
    }
