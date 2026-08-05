"""Pricing assumptions for the savings model.

These are deliberately editable constants, stated loudly in the report.
The model is simple on purpose: storage_cost = vectors_gb * vectordb_gb_month
+ traces_gb * s3_gb_month, computed before and after compression. A user who
disagrees with the constants drops a pricing.toml next to their data and the
audit uses that instead.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from density.errors import AuditError

# Defaults: S3 standard storage, and a mid-market managed vector database
# price per GB-month. Both appear verbatim in the report methodology.
S3_GB_MONTH = 0.023
VECTORDB_GB_MONTH = 0.33


@dataclass(frozen=True)
class Pricing:
    s3_gb_month: float = S3_GB_MONTH
    vectordb_gb_month: float = VECTORDB_GB_MONTH
    source: str = "defaults"

    def monthly_cost(self, traces_gb: float, vectors_gb: float) -> float:
        """Dollars per month to hold this much data at these prices."""
        return traces_gb * self.s3_gb_month + vectors_gb * self.vectordb_gb_month

    def to_dict(self) -> dict:
        return {
            "s3_gb_month": self.s3_gb_month,
            "vectordb_gb_month": self.vectordb_gb_month,
            "source": self.source,
        }


def load_pricing(path: str | Path | None) -> Pricing:
    """Load pricing.toml when given (or found), else return defaults.

    Expected shape:

        [pricing]
        s3_gb_month = 0.023
        vectordb_gb_month = 0.33
    """
    if path is None:
        return Pricing()
    p = Path(path)
    if p.is_dir():
        p = p / "pricing.toml"
    if not p.exists():
        return Pricing()
    # A hand-edited price file is user input, so its failures are reported
    # as AuditError with the offending file named, never as a raw
    # TOMLDecodeError or ValueError traceback out of the middle of a run.
    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
        table = data.get("pricing", data)
        if not isinstance(table, dict):
            raise TypeError("[pricing] must be a table")
        return Pricing(
            s3_gb_month=float(table.get("s3_gb_month", S3_GB_MONTH)),
            vectordb_gb_month=float(table.get("vectordb_gb_month", VECTORDB_GB_MONTH)),
            source=str(p),
        )
    except (tomllib.TOMLDecodeError, TypeError, ValueError, OSError) as exc:
        raise AuditError(f"bad pricing file {p}: {exc}") from exc
