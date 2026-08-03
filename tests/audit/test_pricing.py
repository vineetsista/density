import pytest

from density.audit.pricing import Pricing, load_pricing


def test_defaults_match_spec_constants():
    p = Pricing()
    assert p.s3_gb_month == 0.023
    assert p.vectordb_gb_month == 0.33
    assert p.source == "defaults"


def test_monthly_cost_model():
    p = Pricing()
    # 100 GB traces on object storage plus 10 GB vectors in a vector DB.
    assert p.monthly_cost(traces_gb=100, vectors_gb=10) == pytest.approx(
        100 * 0.023 + 10 * 0.33
    )
    assert p.monthly_cost(0, 0) == 0.0


def test_load_pricing_none_gives_defaults():
    assert load_pricing(None) == Pricing()


def test_load_pricing_missing_file_gives_defaults(tmp_path):
    assert load_pricing(tmp_path / "nope.toml").source == "defaults"


def test_load_pricing_from_toml(tmp_path):
    f = tmp_path / "pricing.toml"
    f.write_text("[pricing]\ns3_gb_month = 0.01\nvectordb_gb_month = 0.5\n")
    p = load_pricing(f)
    assert p.s3_gb_month == 0.01
    assert p.vectordb_gb_month == 0.5
    assert p.source == str(f)


def test_load_pricing_from_directory(tmp_path):
    (tmp_path / "pricing.toml").write_text("[pricing]\ns3_gb_month = 0.02\n")
    p = load_pricing(tmp_path)
    assert p.s3_gb_month == 0.02
    # Missing keys keep their defaults.
    assert p.vectordb_gb_month == 0.33
