import numpy as np
import pytest

from density.embedders import HashingEmbedder


def test_labeled_demo_only():
    assert HashingEmbedder.demo_only is True


def test_deterministic_across_instances():
    a = HashingEmbedder(dim=64)("user asked about refunds")
    b = HashingEmbedder(dim=64)("user asked about refunds")
    np.testing.assert_array_equal(a, b)


def test_unit_norm_and_dtype():
    v = HashingEmbedder(dim=128)("hello world")
    assert v.dtype == np.float32
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


def test_empty_text_is_zero_vector():
    v = HashingEmbedder(dim=32)("")
    assert np.all(v == 0)


def test_related_text_closer_than_unrelated():
    e = HashingEmbedder(dim=768)
    q = e("refund policy for enterprise customers")
    near = e("what is the refund policy")
    far = e("kubernetes pod scheduling latency")
    assert float(q @ near) > float(q @ far)


def test_dim_validation():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
