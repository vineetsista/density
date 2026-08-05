"""Tier definitions: the coined primitive of DENSITY.

A tier is not a compression setting. It is a measured contract: a recall
floor at a footprint target. The audit measures recall on the user's own
data and prints the measured number next to the guarantee. Policy holds the
guarantees; the recall verifier is the referee that checks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from density.engine.trace.zdict import LEVEL_COLD, LEVEL_WARM


class Tier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True)
class TierSpec:
    """One tier's contract.

    recall10_floor is the measured guarantee for recall@10 against exact
    search over the HOT originals, None for HOT itself (exact by
    definition). footprint_target is the promised fraction of original
    bytes, and the audit reports the measured fraction beside it.
    """

    tier: Tier
    vector_codec: str | None      # None means fp32 originals
    trace_zstd_level: int | None  # None means raw, uncompressed traces
    recall10_floor: float | None
    footprint_target: float
    rerank_depth: int             # candidates rescored at search time


TIER_SPECS: dict[Tier, TierSpec] = {
    Tier.HOT: TierSpec(
        tier=Tier.HOT,
        vector_codec=None,
        trace_zstd_level=None,
        recall10_floor=None,
        footprint_target=1.0,
        rerank_depth=0,
    ),
    Tier.WARM: TierSpec(
        tier=Tier.WARM,
        vector_codec="sq8",
        trace_zstd_level=LEVEL_WARM,
        recall10_floor=0.99,
        footprint_target=0.25,
        rerank_depth=0,
    ),
    Tier.COLD: TierSpec(
        tier=Tier.COLD,
        vector_codec="pq",
        trace_zstd_level=LEVEL_COLD,
        recall10_floor=0.95,
        footprint_target=0.08,
        rerank_depth=200,
    ),
}

# COLD with the binary codec instead of PQ trades recall floor for size:
# 32x vectors at a 0.90 floor with rerank depth 500. Kept as a variant
# spec so the audit can report it without redefining the tier.
COLD_BINARY_SPEC = TierSpec(
    tier=Tier.COLD,
    vector_codec="binary",
    trace_zstd_level=LEVEL_COLD,
    recall10_floor=0.90,
    footprint_target=0.08,
    rerank_depth=500,
)


def spec_for(tier: Tier | str, vector_codec: str | None = None) -> TierSpec:
    """Resolve a tier name (and optional codec override) to its spec."""
    t = Tier(tier)
    if t is Tier.COLD and vector_codec == "binary":
        return COLD_BINARY_SPEC
    return TIER_SPECS[t]
