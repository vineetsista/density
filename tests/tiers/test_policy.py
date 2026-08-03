from density.tiers.policy import COLD_BINARY_SPEC, TIER_SPECS, Tier, spec_for


def test_all_tiers_defined():
    assert set(TIER_SPECS) == {Tier.HOT, Tier.WARM, Tier.COLD}


def test_hot_is_exact_and_full_size():
    hot = TIER_SPECS[Tier.HOT]
    assert hot.vector_codec is None
    assert hot.trace_zstd_level is None
    assert hot.recall10_floor is None
    assert hot.footprint_target == 1.0


def test_guarantees_match_spec_table():
    assert TIER_SPECS[Tier.WARM].recall10_floor == 0.99
    assert TIER_SPECS[Tier.WARM].footprint_target == 0.25
    assert TIER_SPECS[Tier.COLD].recall10_floor == 0.95
    assert TIER_SPECS[Tier.COLD].footprint_target == 0.08
    assert COLD_BINARY_SPEC.recall10_floor == 0.90


def test_spec_for_resolves_strings_and_variants():
    assert spec_for("warm") is TIER_SPECS[Tier.WARM]
    assert spec_for("cold") is TIER_SPECS[Tier.COLD]
    assert spec_for("cold", vector_codec="binary") is COLD_BINARY_SPEC
    assert spec_for(Tier.HOT) is TIER_SPECS[Tier.HOT]


def test_cold_levels_harder_than_warm():
    assert TIER_SPECS[Tier.COLD].trace_zstd_level > TIER_SPECS[Tier.WARM].trace_zstd_level
