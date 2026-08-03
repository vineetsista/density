import json

import pytest

from density.errors import StoreError
from density.tiers.manifest import (
    FORMAT_VERSION,
    Manifest,
    TierBytes,
    TierEntry,
    collect_versions,
    sha256_file,
)


def make_manifest() -> Manifest:
    m = Manifest(created_at="2026-01-01T00:00:00+00:00", seed=1337)
    m.datasets.traces.events = 100
    m.datasets.embeddings.count = 256
    m.datasets.embeddings.dim = 32
    m.tiers["warm"] = TierEntry(
        tier="warm",
        vector_codec="sq8",
        vectors_file="vectors/warm.sq8",
        trace_zstd_level=10,
        bytes=TierBytes(traces=10, vectors=20, vectors_aux=1, total=31),
    )
    return m


def test_save_load_round_trip(tmp_path):
    m = make_manifest()
    m.save(tmp_path)
    loaded = Manifest.load(tmp_path)
    assert loaded == m


def test_save_is_atomic_and_stable(tmp_path):
    m = make_manifest()
    m.save(tmp_path)
    first = (tmp_path / "manifest.json").read_bytes()
    m.save(tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == first
    # No temp droppings left behind.
    assert list(tmp_path.glob(".manifest-*")) == []


def test_load_missing_raises(tmp_path):
    with pytest.raises(StoreError, match="no manifest"):
        Manifest.load(tmp_path)


def test_load_corrupt_raises(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(StoreError, match="corrupt"):
        Manifest.load(tmp_path)


def test_load_wrong_version_raises(tmp_path):
    payload = {"format_version": FORMAT_VERSION + 99}
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(StoreError, match="unsupported"):
        Manifest.load(tmp_path)


def test_checksum_record_and_verify(tmp_path):
    data = tmp_path / "blob.bin"
    data.write_bytes(b"hello density")
    m = make_manifest()
    digest = m.record_checksum(tmp_path, "blob.bin")
    assert digest == sha256_file(data)
    assert m.verify_checksums(tmp_path) == []
    data.write_bytes(b"tampered")
    assert m.verify_checksums(tmp_path) == ["blob.bin"]
    data.unlink()
    assert m.verify_checksums(tmp_path) == ["blob.bin"]


def test_collect_versions_names():
    v = collect_versions()
    assert set(v) == {"density", "numpy", "pyarrow", "zstandard"}
    assert all(isinstance(x, str) and x for x in v.values())
