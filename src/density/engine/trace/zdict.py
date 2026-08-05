"""Dictionary-trained zstd block storage for trace text stores.

Items (arbitrary byte strings, possibly empty) are appended into blocks of
about 4 MB uncompressed. Each full block is compressed as one independent
zstd frame, optionally with a shared trained dictionary, and addressed by a
ref tuple `(block_index, offset_in_decompressed_block, length)`, all in
items of bytes. Round trips are byte-exact and on-disk output is
deterministic for a given item sequence, level, and dictionary.

On disk, under `dir_path` with a caller-chosen `prefix`:
- `{prefix}.blocks.bin`: concatenated zstd frames.
- `{prefix}.index.json`: per-frame compressed offset and length (bytes),
  per-block item lengths (bytes), item counts, level, dictionary flag,
  and the frame-shaping parameters (window_log, enable_ldm) the writer
  used, recorded for provenance: the frames themselves stay readable by
  any stock zstd decompressor because window_log is capped at 27.
- `{prefix}.dict`: raw dictionary bytes, written only when non-empty.

Because the index lists every item length per block, an item's ref is
derivable from its append position alone: `BlockReader.item_ref` and
`get_item` resolve items by index, and `SpilledItems` serves index-based
access after one bounded-memory decompression pass to a temp file (the
right tool when access order is unrelated to placement order).

Module constants: LEVEL_COLD = 19, LEVEL_WARM = 10.
"""

from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard

from density.errors import StoreError

LEVEL_COLD = 19
LEVEL_WARM = 10

_FORMAT_VERSION = 1
_DEFAULT_BLOCK_BYTES = 4 << 20

# Frames written with window_log above 27 would refuse to decompress on
# stock zstd builds (ZSTD_WINDOWLOG_LIMIT_DEFAULT), so writers cap there:
# a bundle must decompress everywhere, not just on the machine that wrote
# it. The reader still honors larger values recorded in an index by
# raising its own limit, for forward compatibility.
_MAX_PORTABLE_WINDOW_LOG = 27

# Uncompressed block payloads waiting in the compression pool are the one
# unbounded buffer in a streaming write: zstd level 19 compresses slower
# than callers can append, so without backpressure the whole store queues
# up in RAM. Capping in-flight seals keeps peak memory at about
# (cap + 1) * block_bytes regardless of corpus size.
_MAX_INFLIGHT_SEALS = 4

Ref = tuple[int, int, int]


def train_dict(
    samples: list[bytes],
    dict_size: int = 110_000,
    sample_cap_bytes: int = 100_000_000,
) -> bytes:
    """Train a zstd dictionary of at most `dict_size` bytes over `samples`.

    Samples are taken in order until their cumulative size reaches
    `sample_cap_bytes` (bytes), so huge corpora train in bounded memory.
    Returns the dictionary bytes, or b"" when training is impossible (too
    few, too small, or degenerate samples): b"" means compression proceeds
    dictionary-free, which is a fully supported mode, not an error.
    """
    capped: list[bytes] = []
    budget = sample_cap_bytes
    for sample in samples:
        # Empty samples carry no signal and some zstd builds reject them.
        if not sample:
            continue
        if len(sample) > budget:
            break
        capped.append(sample)
        budget -= len(sample)
    if not capped:
        return b""
    try:
        trained = zstandard.train_dictionary(dict_size, capped)
    except zstandard.ZstdError:
        # zstd raises on tiny or degenerate sample sets; the caller simply
        # compresses without a dictionary in that case.
        return b""
    return trained.as_bytes()


def _compress_block(
    payload: bytes,
    level: int,
    dict_bytes: bytes,
    window_log: int | None = None,
    enable_ldm: bool = False,
) -> bytes:
    """Compress one block as a single zstd frame, single-threaded.

    A fresh compressor per frame keeps frames independent of each other and
    of the calling thread, so output bytes never depend on pool size.

    window_log widens the match window past the level default (level 19
    stops at 8 MiB), and enable_ldm turns on long-distance matching so
    repeats far apart inside a large block are actually found. Both are
    plain zstd frame features: any stock decompressor reads the result,
    no side channel needed beyond the frame itself.
    """
    if window_log is not None or enable_ldm:
        overrides: dict[str, int] = {}
        if window_log is not None:
            overrides["window_log"] = int(window_log)
        if enable_ldm:
            overrides["enable_ldm"] = 1
        params = zstandard.ZstdCompressionParameters.from_level(level, **overrides)
        if dict_bytes:
            cctx = zstandard.ZstdCompressor(
                compression_params=params,
                dict_data=zstandard.ZstdCompressionDict(dict_bytes),
            )
        else:
            cctx = zstandard.ZstdCompressor(compression_params=params)
    elif dict_bytes:
        cctx = zstandard.ZstdCompressor(
            level=level, dict_data=zstandard.ZstdCompressionDict(dict_bytes)
        )
    else:
        cctx = zstandard.ZstdCompressor(level=level)
    return cctx.compress(payload)


class BlockWriter:
    """Append byte items into ~`block_bytes` zstd frames under `dir_path`.

    `append` returns a ref `(block_index, offset, length)` (bytes within the
    decompressed block). `finish` flushes, writes the three output files,
    and returns total compressed frame bytes (the size of blocks.bin).
    Blocks queued for compression may run on a ThreadPoolExecutor of
    `workers` threads; frames are written in block order so the files are
    byte-identical for any worker count.
    """

    def __init__(
        self,
        dir_path: str | Path,
        prefix: str,
        level: int,
        dict_bytes: bytes = b"",
        *,
        workers: int = 1,
        block_bytes: int = _DEFAULT_BLOCK_BYTES,
        window_log: int | None = None,
        enable_ldm: bool = False,
    ) -> None:
        if window_log is not None and window_log > _MAX_PORTABLE_WINDOW_LOG:
            raise StoreError(
                f"window_log {window_log} exceeds the portable decompression "
                f"limit of {_MAX_PORTABLE_WINDOW_LOG}: such frames would fail "
                "on stock zstd decompressors"
            )
        self._dir = Path(dir_path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._level = int(level)
        self._dict_bytes = bytes(dict_bytes)
        self._block_bytes = int(block_bytes)
        self._window_log = int(window_log) if window_log is not None else None
        self._enable_ldm = bool(enable_ldm)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._futures: list[Future[bytes]] = []
        # Index of the oldest seal not yet proven finished; backpressure
        # waits on it so at most _MAX_INFLIGHT_SEALS payloads sit in RAM.
        self._drain_idx = 0
        self._item_lengths: list[list[int]] = []
        self._buf: list[bytes] = []
        self._buf_lens: list[int] = []
        self._buf_size = 0
        self._finished = False

    def append(self, item_bytes: bytes) -> Ref:
        """Buffer one item; returns its ref (block_index, offset, length)."""
        if self._finished:
            raise StoreError("BlockWriter.append after finish")
        # Seal the current block first when this item would overflow it, so
        # an oversized item always starts at offset 0 of its own block.
        if self._buf_size and self._buf_size + len(item_bytes) > self._block_bytes:
            self._seal()
        ref = (len(self._futures), self._buf_size, len(item_bytes))
        self._buf.append(item_bytes)
        self._buf_lens.append(len(item_bytes))
        self._buf_size += len(item_bytes)
        if self._buf_size >= self._block_bytes:
            self._seal()
        return ref

    def _seal(self) -> None:
        # Backpressure before queueing another payload: waiting on the
        # oldest unfinished seal blocks until the pool catches up, which
        # caps queued uncompressed bytes without changing output order.
        while len(self._futures) - self._drain_idx >= _MAX_INFLIGHT_SEALS:
            self._futures[self._drain_idx].result()
            self._drain_idx += 1
        payload = b"".join(self._buf)
        self._item_lengths.append(self._buf_lens)
        self._futures.append(
            self._pool.submit(
                _compress_block,
                payload,
                self._level,
                self._dict_bytes,
                self._window_log,
                self._enable_ldm,
            )
        )
        self._buf = []
        self._buf_lens = []
        self._buf_size = 0

    def abort(self) -> None:
        """Release the compression pool without finishing the store.

        finish() is the only other place the pool is shut down, so a failed
        shred that never reaches it would leak _COMPRESS_WORKERS non-daemon
        threads per attempt. Idempotent, and safe to call after finish().
        """
        if getattr(self, "_finished", False):
            return
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._finished = True

    def finish(self) -> int:
        """Flush, write blocks.bin, index.json, and the optional dict file.

        Returns total compressed bytes of all frames (size of blocks.bin,
        excluding the dictionary and index files).
        """
        if self._finished:
            raise StoreError("BlockWriter.finish called twice")
        self._finished = True
        if self._buf:
            self._seal()
        frames = [f.result() for f in self._futures]
        self._pool.shutdown(wait=True)

        blocks = []
        pos = 0
        for frame, lens in zip(frames, self._item_lengths):
            blocks.append(
                {
                    "offset": pos,
                    "compressed_length": len(frame),
                    "uncompressed_length": sum(lens),
                    "item_lengths": lens,
                }
            )
            pos += len(frame)
        index = {
            "format_version": _FORMAT_VERSION,
            "level": self._level,
            "dict_used": bool(self._dict_bytes),
            "dict_bytes": len(self._dict_bytes),
            # Compression parameters are recorded so any reader knows how
            # the frames were produced; decompression itself needs nothing
            # extra as long as window_log stays within the portable limit.
            "window_log": self._window_log,
            "enable_ldm": self._enable_ldm,
            "item_count": sum(len(lens) for lens in self._item_lengths),
            "uncompressed_bytes": sum(b["uncompressed_length"] for b in blocks),
            "compressed_bytes": pos,
            "blocks": blocks,
        }
        (self._dir / f"{self._prefix}.blocks.bin").write_bytes(b"".join(frames))
        # sort_keys pins the serialization so index bytes are deterministic.
        (self._dir / f"{self._prefix}.index.json").write_text(
            json.dumps(index, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        if self._dict_bytes:
            (self._dir / f"{self._prefix}.dict").write_bytes(self._dict_bytes)
        return pos


class BlockReader:
    """Random access and full iteration over a BlockWriter output.

    `get(ref)` returns the exact item bytes for a ref from the matching
    writer. `iter_all()` yields every item in append order. Decompressed
    blocks are held in a small LRU cache (`cache_blocks` blocks, each about
    4 MB uncompressed) so clustered access does not re-decompress.
    """

    def __init__(
        self, dir_path: str | Path, prefix: str, *, cache_blocks: int = 4
    ) -> None:
        self._dir = Path(dir_path)
        self._prefix = prefix
        index_path = self._dir / f"{prefix}.index.json"
        self._blocks_path = self._dir / f"{prefix}.blocks.bin"
        if not index_path.exists() or not self._blocks_path.exists():
            raise StoreError(f"missing block store files for prefix {prefix!r} in {dir_path}")
        self._index = json.loads(index_path.read_text(encoding="utf-8"))
        self._blocks = self._index["blocks"]
        dict_bytes = b""
        if self._index.get("dict_used"):
            dict_bytes = (self._dir / f"{prefix}.dict").read_bytes()
        # Writers cap window_log at the portable limit, but a future format
        # may not: honoring a larger recorded value here costs nothing and
        # keeps old readers working on such stores. Older indexes have no
        # window_log key at all, hence the .get default.
        window_log = self._index.get("window_log")
        max_window = 1 << window_log if window_log and window_log > 27 else 0
        if dict_bytes:
            self._dctx = zstandard.ZstdDecompressor(
                dict_data=zstandard.ZstdCompressionDict(dict_bytes),
                max_window_size=max_window,
            )
        else:
            self._dctx = zstandard.ZstdDecompressor(max_window_size=max_window)
        self._cache_blocks = max(1, int(cache_blocks))
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        # Lazy item-index lookup tables, built on first item_ref call.
        self._item_starts: np.ndarray | None = None
        self._item_offsets: np.ndarray | None = None
        self._item_lengths_flat: np.ndarray | None = None

    def _block(self, block_index: int) -> bytes:
        if not 0 <= block_index < len(self._blocks):
            raise StoreError(f"block index {block_index} out of range")
        cached = self._cache.get(block_index)
        if cached is not None:
            self._cache.move_to_end(block_index)
            return cached
        meta = self._blocks[block_index]
        with open(self._blocks_path, "rb") as fh:
            fh.seek(meta["offset"])
            frame = fh.read(meta["compressed_length"])
        data = self._dctx.decompress(frame)
        if len(data) != meta["uncompressed_length"]:
            raise StoreError(f"block {block_index} decompressed to unexpected size")
        self._cache[block_index] = data
        if len(self._cache) > self._cache_blocks:
            self._cache.popitem(last=False)
        return data

    def get(self, ref: Ref) -> bytes:
        """Return the exact item bytes for `ref = (block, offset, length)`."""
        block_index, offset, length = ref
        data = self._block(block_index)
        if offset < 0 or length < 0 or offset + length > len(data):
            raise StoreError(f"ref {ref} out of bounds for block of {len(data)} bytes")
        return data[offset : offset + length]

    def iter_all(self) -> Iterator[bytes]:
        """Yield every stored item, byte-exact, in append order."""
        for block_index, meta in enumerate(self._blocks):
            data = self._block(block_index)
            offset = 0
            for length in meta["item_lengths"]:
                yield data[offset : offset + length]
                offset += length

    @property
    def item_count(self) -> int:
        """Number of stored items across all blocks."""
        return int(self._index["item_count"])

    def _build_item_tables(self) -> None:
        """Derive per-item (block, offset, length) from the index lengths.

        The index already lists every item length per block, so a ref is a
        pure function of the item's append position: no per-item triple
        needs to be stored anywhere else. Costs O(items) int64 memory,
        built once on first use.
        """
        starts = [0]
        offsets: list[np.ndarray] = []
        lengths: list[np.ndarray] = []
        for meta in self._blocks:
            lens = np.asarray(meta["item_lengths"], dtype=np.int64)
            starts.append(starts[-1] + len(lens))
            offs = np.zeros(len(lens), dtype=np.int64)
            if len(lens) > 1:
                np.cumsum(lens[:-1], out=offs[1:])
            offsets.append(offs)
            lengths.append(lens)
        self._item_starts = np.asarray(starts, dtype=np.int64)
        self._item_offsets = (
            np.concatenate(offsets) if offsets else np.zeros(0, dtype=np.int64)
        )
        self._item_lengths_flat = (
            np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
        )

    def item_ref(self, index: int) -> Ref:
        """Ref (block, offset, length) of the item at append position `index`."""
        if self._item_starts is None:
            self._build_item_tables()
        assert self._item_offsets is not None and self._item_lengths_flat is not None
        if not 0 <= index < len(self._item_offsets):
            raise StoreError(f"item index {index} out of range")
        block = int(np.searchsorted(self._item_starts, index, side="right")) - 1
        return (block, int(self._item_offsets[index]), int(self._item_lengths_flat[index]))

    def get_item(self, index: int) -> bytes:
        """Exact bytes of the item at append position `index`."""
        return self.get(self.item_ref(index))


class SpilledItems:
    """Random access by item index after decompressing a store to disk once.

    Full-corpus replay touches items in row order while the writer placed
    them in similarity order, so per-item access through BlockReader would
    re-decompress large frames constantly. This class pays one sequential
    decompression pass into an anonymous temp file (bounded memory: one
    zstd output chunk at a time) and serves items with a seek and a read.
    Temp disk equals the store's uncompressed bytes, which interning keeps
    far below the raw corpus. The file is unlinked by the OS on close, so
    a crash leaks nothing.
    """

    def __init__(self, dir_path: str | Path, prefix: str) -> None:
        reader = BlockReader(dir_path, prefix, cache_blocks=1)
        reader._build_item_tables()
        assert reader._item_starts is not None
        assert reader._item_offsets is not None and reader._item_lengths_flat is not None
        # Global offsets: per-block offset plus that block's start in the file.
        block_bases = np.zeros(len(reader._blocks), dtype=np.int64)
        pos = 0
        for i, meta in enumerate(reader._blocks):
            block_bases[i] = pos
            pos += int(meta["uncompressed_length"])
        item_block = (
            np.searchsorted(reader._item_starts, np.arange(reader.item_count), side="right")
            - 1
        )
        self._offsets = reader._item_offsets + block_bases[item_block]
        self._lengths = reader._item_lengths_flat
        self._spill = tempfile.TemporaryFile(prefix=f"density-{prefix}-spill-")
        blocks_path = Path(dir_path) / f"{prefix}.blocks.bin"
        # A corrupt or truncated store raises out of the loop below. Without
        # this guard the exception would escape __init__, so close() and
        # __exit__ would never run and the spill handle (plus whatever it
        # already wrote to disk) would survive until the garbage collector
        # happened to reach it. A long-lived server replaying a damaged
        # bundle would leak one per request.
        try:
            with open(blocks_path, "rb") as fh:
                for meta in reader._blocks:
                    fh.seek(meta["offset"])
                    dobj = reader._dctx.decompressobj()
                    remaining = meta["compressed_length"]
                    written = 0
                    while remaining > 0:
                        chunk = fh.read(min(1 << 20, remaining))
                        if not chunk:
                            raise StoreError(f"truncated block store {blocks_path}")
                        remaining -= len(chunk)
                        out = dobj.decompress(chunk)
                        written += len(out)
                        self._spill.write(out)
                    if written != meta["uncompressed_length"]:
                        raise StoreError(
                            f"block in {blocks_path} decompressed to unexpected size"
                        )
        except BaseException:
            self._spill.close()
            raise

    def get(self, index: int) -> bytes:
        """Exact bytes of the item at append position `index`."""
        if not 0 <= index < len(self._offsets):
            raise StoreError(f"item index {index} out of range")
        self._spill.seek(int(self._offsets[index]))
        return self._spill.read(int(self._lengths[index]))

    def close(self) -> None:
        self._spill.close()

    def __enter__(self) -> SpilledItems:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
