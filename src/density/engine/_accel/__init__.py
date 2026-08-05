"""Kernel dispatch: compiled C++ when available, numpy fallback otherwise.

The compiled path is built lazily from cpp/kernels.cpp + cpp/bindings.cpp
on first import after `pip install -e ".[accel]"` (which provides pybind11).
Any failure at any stage, missing pybind11, missing compiler, build error,
import error, falls back silently to the numpy implementations, and
ACCEL_ACTIVE records which path is live. Callers import the three kernel
functions from this module and never care which backend answered.
"""

from __future__ import annotations

ACCEL_ACTIVE: bool = False

# Declared above the import on purpose: the flag is this module's contract
# and reads first, and the fallback import below is what defines the three
# kernel names it describes.
from density.engine._accel import fallback as _fb  # noqa: E402

sq8_scores = _fb.sq8_scores
pq_adc_scan = _fb.pq_adc_scan
hamming_scan = _fb.hamming_scan


def _try_load_compiled() -> None:
    """Attempt to load (building if needed) the compiled kernels."""
    global ACCEL_ACTIVE, sq8_scores, pq_adc_scan, hamming_scan
    try:
        from density.engine._accel.build import load_compiled

        mod = load_compiled()
    except Exception:
        return
    if mod is None:
        return
    sq8_scores = mod.sq8_scores
    pq_adc_scan = mod.pq_adc_scan
    hamming_scan = mod.hamming_scan
    ACCEL_ACTIVE = True


_try_load_compiled()
