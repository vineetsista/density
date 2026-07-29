"""Exception hierarchy for DENSITY.

User input problems never crash a pipeline: they are counted, quarantined,
and reported. These exceptions are for programming errors and unrecoverable
conditions (missing store, corrupt manifest), not for malformed data lines.
"""

from __future__ import annotations


class DensityError(Exception):
    """Base class for all DENSITY errors."""


class IngestError(DensityError):
    """Unrecoverable ingest problem, for example an unreadable path."""


class StoreError(DensityError):
    """Store-level problem, for example a missing or corrupt manifest."""


class AuditError(DensityError):
    """Audit orchestration problem."""
