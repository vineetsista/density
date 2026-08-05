"""Tests for zstd block storage: train_dict, BlockWriter, BlockReader.

Every round trip must be byte-exact and every write must be deterministic:
same items give bit-identical files regardless of compression pool size.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from density.engine.trace.zdict import (
    LEVEL_COLD,
    LEVEL_WARM,
    BlockReader,
    BlockWriter,
    SpilledItems,
    train_dict,
)

SEED = 1337


def _varied_items(rng: np.random.Generator) -> list[bytes]:
    """Binary, empty, unicode, and repetitive items, a few KB total."""
    items: list[bytes] = [
        b"",
        b"plain ascii line",
        "unicode: café 日本語 \U0001f680 ☃".encode("utf-8"),
        b"\x00\x01\x02\xff\xfe binary \x00 with NULs",
        b"x" * 10_000,
    ]
    for _ in range(60):
        n = int(rng.integers(0, 2000))
        items.append(rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())
    items.append(b"")
    return items


def _write_all(
    dir_path: Path,
    items: list[bytes],
    *,
    level: int = 3,
    dict_bytes: bytes = b"",
    workers: int = 1,
    block_bytes: int = 4 << 20,
) -> tuple[list[tuple[int, int, int]], int]:
    writer = BlockWriter(
        dir_path,
        "t",
        level,
        dict_bytes=dict_bytes,
        workers=workers,
        block_bytes=block_bytes,
    )
    refs = [writer.append(item) for item in items]
    total = writer.finish()
    return refs, total


def test_levels() -> None:
    assert LEVEL_COLD == 19
    assert LEVEL_WARM == 10


def test_round_trip_byte_exact(tmp_path: Path, rng: np.random.Generator) -> None:
    items = _varied_items(rng)
    # Small blocks force multiple frames so refs cross block boundaries.
    refs, total = _write_all(tmp_path, items, block_bytes=4096)
    assert total > 0
    assert len({r[0] for r in refs}) > 1
    reader = BlockReader(tmp_path, "t")
    for ref, item in zip(refs, items):
        assert reader.get(ref) == item
    assert list(reader.iter_all()) == items


def test_empty_store_round_trips(tmp_path: Path) -> None:
    refs, total = _write_all(tmp_path, [])
    assert refs == []
    reader = BlockReader(tmp_path, "t")
    assert list(reader.iter_all()) == []


def test_large_item_spans_own_block(tmp_path: Path, rng: np.random.Generator) -> None:
    big = rng.integers(0, 256, size=10 * 1024 * 1024, dtype=np.uint8).tobytes()
    items = [b"before-1", b"before-2", big, b"after-1", b""]
    refs, _ = _write_all(tmp_path, items, level=LEVEL_WARM)
    big_ref = refs[2]
    # The oversized item starts a fresh block and is sealed alone.
    assert big_ref[1] == 0
    index = json.loads((tmp_path / "t.index.json").read_text())
    assert index["blocks"][big_ref[0]]["item_lengths"] == [len(big)]
    reader = BlockReader(tmp_path, "t")
    assert reader.get(big_ref) == big
    assert list(reader.iter_all()) == items


def test_random_access_out_of_order(tmp_path: Path, rng: np.random.Generator) -> None:
    items = _varied_items(rng)
    refs, _ = _write_all(tmp_path, items, block_bytes=2048)
    reader = BlockReader(tmp_path, "t")
    order = np.random.default_rng(SEED).permutation(len(items))
    for i in order:
        assert reader.get(refs[i]) == items[i]


def _boilerplate_items(rng: np.random.Generator, n: int = 300) -> list[bytes]:
    """Items sharing a 1.2 KB boilerplate prefix with unique tails."""
    boiler = (b"SYSTEM PROMPT: you are a meticulous support agent. " * 24)[:1200]
    items = []
    for _ in range(n):
        tail = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes().hex().encode()
        items.append(boiler + b" case:" + tail)
    return items


def test_dictionary_improves_compression(tmp_path: Path, rng: np.random.Generator) -> None:
    items = _boilerplate_items(rng)
    d = train_dict(items, dict_size=8192)
    assert d != b""
    # Tiny blocks give one or two items per frame, the regime dictionaries
    # exist for: without one, every frame pays for the boilerplate again.
    plain_dir = tmp_path / "plain"
    dict_dir = tmp_path / "dict"
    _, plain_total = _write_all(plain_dir, items, level=LEVEL_COLD, block_bytes=2048)
    refs, dict_total = _write_all(
        dict_dir, items, level=LEVEL_COLD, dict_bytes=d, block_bytes=2048
    )
    # The dictionary must pay for its own storage and still win.
    assert dict_total + len(d) < plain_total
    reader = BlockReader(dict_dir, "t")
    for ref, item in zip(refs, items):
        assert reader.get(ref) == item
    assert list(reader.iter_all()) == items


def test_degenerate_training_falls_back(tmp_path: Path, rng: np.random.Generator) -> None:
    assert train_dict([]) == b""
    assert train_dict([b"a"]) == b""
    assert train_dict([b"aaaa" * 10] * 5) == b""
    # No dictionary is a normal mode, round trips stay byte-exact.
    items = _varied_items(rng)
    refs, _ = _write_all(tmp_path, items, dict_bytes=train_dict([b"a"]))
    reader = BlockReader(tmp_path, "t")
    for ref, item in zip(refs, items):
        assert reader.get(ref) == item
    assert not (tmp_path / "t.dict").exists()


def test_sample_cap_is_applied(rng: np.random.Generator) -> None:
    items = _boilerplate_items(rng, n=200)
    capped = train_dict(items, dict_size=8192, sample_cap_bytes=100 * 1300)
    assert isinstance(capped, bytes)
    # A cap below one sample leaves nothing to train on.
    assert train_dict(items, dict_size=8192, sample_cap_bytes=8) == b""


def test_determinism_across_runs_and_pool_sizes(
    tmp_path: Path, rng: np.random.Generator
) -> None:
    items = _varied_items(rng)
    d = train_dict(_boilerplate_items(rng), dict_size=8192)
    outputs = []
    for name, workers in (("a", 1), ("b", 1), ("c", 4)):
        run_dir = tmp_path / name
        _write_all(
            run_dir, items, level=LEVEL_WARM, dict_bytes=d, workers=workers,
            block_bytes=4096,
        )
        outputs.append(
            (
                (run_dir / "t.blocks.bin").read_bytes(),
                (run_dir / "t.index.json").read_bytes(),
                (run_dir / "t.dict").read_bytes(),
            )
        )
    assert outputs[0] == outputs[1] == outputs[2]


def test_index_json_structure(tmp_path: Path, rng: np.random.Generator) -> None:
    items = _varied_items(rng)
    refs, total = _write_all(tmp_path, items, level=LEVEL_WARM, block_bytes=4096)
    index = json.loads((tmp_path / "t.index.json").read_text())
    assert index["level"] == LEVEL_WARM
    assert index["dict_used"] is False
    assert index["item_count"] == len(items)
    assert index["uncompressed_bytes"] == sum(len(i) for i in items)
    blocks = index["blocks"]
    assert sum(len(b["item_lengths"]) for b in blocks) == len(items)
    # Frames are laid out back to back and account for every byte on disk.
    pos = 0
    for b in blocks:
        assert b["offset"] == pos
        assert b["compressed_length"] > 0
        assert b["uncompressed_length"] == sum(b["item_lengths"])
        pos += b["compressed_length"]
    assert pos == index["compressed_bytes"] == total
    assert (tmp_path / "t.blocks.bin").stat().st_size == total


def test_bad_ref_raises(tmp_path: Path, rng: np.random.Generator) -> None:
    from density.errors import StoreError

    items = _varied_items(rng)
    refs, _ = _write_all(tmp_path, items, block_bytes=4096)
    reader = BlockReader(tmp_path, "t")
    with pytest.raises(StoreError):
        reader.get((10_000, 0, 1))
    last_block = max(r[0] for r in refs)
    with pytest.raises(StoreError):
        reader.get((last_block, 0, 1 << 30))


def test_item_index_access(tmp_path: Path, rng: np.random.Generator) -> None:
    """item_ref and get_item resolve by append position alone.

    The index already lists every item length per block, so a ref must be
    derivable from the position: this is what lets the parquet ref
    columns store one int instead of a triple.
    """
    items = _varied_items(rng)
    refs, _ = _write_all(tmp_path, items, block_bytes=2048)
    reader = BlockReader(tmp_path, "t")
    assert reader.item_count == len(items)
    for i, (ref, item) in enumerate(zip(refs, items)):
        assert reader.item_ref(i) == ref
        assert reader.get_item(i) == item
    from density.errors import StoreError

    with pytest.raises(StoreError):
        reader.item_ref(len(items))
    with pytest.raises(StoreError):
        reader.item_ref(-1)


def test_spilled_items_round_trip(tmp_path: Path, rng: np.random.Generator) -> None:
    """SpilledItems serves the same bytes as the reader, in any order."""
    items = _varied_items(rng)
    _write_all(tmp_path, items, block_bytes=2048)
    with SpilledItems(tmp_path, "t") as spilled:
        order = np.random.default_rng(SEED).permutation(len(items))
        for i in order:
            assert spilled.get(int(i)) == items[int(i)]
    from density.errors import StoreError

    with SpilledItems(tmp_path, "t") as spilled:
        with pytest.raises(StoreError):
            spilled.get(len(items))


def test_spilled_items_empty_store(tmp_path: Path) -> None:
    _write_all(tmp_path, [])
    with SpilledItems(tmp_path, "t") as spilled:
        from density.errors import StoreError

        with pytest.raises(StoreError):
            spilled.get(0)


def test_window_log_and_ldm_frames(tmp_path: Path, rng: np.random.Generator) -> None:
    """Wide-window LDM frames round-trip, record their parameters, and
    stay deterministic across pool sizes; the portable window cap is
    enforced so a bundle always decompresses on stock zstd builds."""
    from density.errors import StoreError

    items = _boilerplate_items(rng, n=400)
    outputs = []
    for name, workers in (("a", 1), ("b", 4)):
        run_dir = tmp_path / name
        writer = BlockWriter(
            run_dir, "t", LEVEL_COLD, workers=workers, block_bytes=1 << 20,
            window_log=20, enable_ldm=True,
        )
        refs = [writer.append(item) for item in items]
        writer.finish()
        outputs.append((run_dir / "t.blocks.bin").read_bytes())
        reader = BlockReader(run_dir, "t")
        for ref, item in zip(refs, items):
            assert reader.get(ref) == item
    assert outputs[0] == outputs[1]
    index = json.loads((tmp_path / "a" / "t.index.json").read_text())
    assert index["window_log"] == 20
    assert index["enable_ldm"] is True
    with pytest.raises(StoreError):
        BlockWriter(tmp_path / "c", "t", LEVEL_COLD, window_log=28)
