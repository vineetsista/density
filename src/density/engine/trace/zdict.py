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
  per-block item lengths (bytes), item counts, level, dictionary flag.
- `{prefix}.dict`: raw dictionary bytes, written only when non-empty.

Module constants: LEVEL_COLD = 19, LEVEL_WARM = 10.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import zstandard

from density.errors import StoreError

LEVEL_COLD = 19
LEVEL_WARM = 10

_FORMAT_VERSION = 1
_DEFAULT_BLOCK_BYTES = 4 << 20

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


def _compress_block(payload: bytes, level: int, dict_bytes: bytes) -> bytes:
    """Compress one block as a single zstd frame, single-threaded.

    A fresh compressor per frame keeps frames independent of each other and
    of the calling thread, so output bytes never depend on pool size.
    """
    if dict_bytes:
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
    ) -> None:
        self._dir = Path(dir_path)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._level = int(level)
        self._dict_bytes = bytes(dict_bytes)
        self._block_bytes = int(block_bytes)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._futures: list[Future[bytes]] = []
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
        payload = b"".join(self._buf)
        self._item_lengths.append(self._buf_lens)
        self._futures.append(
            self._pool.submit(_compress_block, payload, self._level, self._dict_bytes)
        )
        self._buf = []
        self._buf_lens = []
        self._buf_size = 0

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
        if dict_bytes:
            self._dctx = zstandard.ZstdDecompressor(
                dict_data=zstandard.ZstdCompressionDict(dict_bytes)
            )
        else:
            self._dctx = zstandard.ZstdDecompressor()
        self._cache_blocks = max(1, int(cache_blocks))
        self._cache: OrderedDict[int, bytes] = OrderedDict()

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
