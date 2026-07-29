"""DENSITY: the flight recorder for AI agents.

Public API:
    density.open(path)   -> Store        (local tiered store)
    density.audit(path)  -> AuditResult  (compress, verify recall, price, report)
    density.synth(gb)    -> SynthStats   (generate a realistic agent corpus)
    density.ACCEL_ACTIVE -> bool         (True when compiled kernels loaded)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["open", "audit", "synth", "ACCEL_ACTIVE", "__version__"]


def open(path):  # noqa: A001  (deliberate shadow, this is the SDK entry point)
    """Open (or create) a DENSITY store directory and return a Store."""
    from density.tiers.store import Store

    return Store.open(path)


def audit(path, out: str = "report.html", **kwargs):
    """Run a full audit: ingest, compress all tiers, verify recall, price, report."""
    from density.audit.runner import run_audit

    return run_audit(path, out=out, **kwargs)


def synth(gb: float, out: str = "./raw", seed: int = 1337, **kwargs):
    """Generate a realistic synthetic agent corpus (traces + embeddings)."""
    from density.bench.synth import generate

    return generate(gb, out, seed=seed, **kwargs)


def _accel_active() -> bool:
    from density.engine._accel import ACCEL_ACTIVE

    return ACCEL_ACTIVE


def __getattr__(name: str):
    # ACCEL_ACTIVE is resolved lazily so that importing density never pays
    # the (one-time) kernel compilation cost unless acceleration is used.
    if name == "ACCEL_ACTIVE":
        return _accel_active()
    raise AttributeError(f"module 'density' has no attribute {name!r}")
