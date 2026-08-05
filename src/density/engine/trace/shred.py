"""Shred parsed trace lines into a byte-exact columnar bundle.

shred_events writes, under one output directory:

- ``structured.parquet``: one row per input line. Scalar columns are
  trace_id, ts, role, type, model, tool_name, tokens_in, tokens_out,
  file, line_index; ref columns are content_ref, extra_ref,
  residual_ref, each a nullable int64 item index into the matching
  block store. An index resolves to exact bytes through the store's own
  metadata (zdict records every item length per block, so the triple
  (block, offset, length) is a pure function of the index): one small
  int per row instead of three large ones, and repeated payloads repeat
  the same single value. ts is int64 microseconds since the Unix epoch,
  stored absolute and delta-encoded on disk (DELTA_BINARY_PACKED);
  low-cardinality strings are dictionary-encoded.
- three zstd block stores (see zdict): ``content`` holds the event text
  body, ``extra`` holds the compact JSON dump of unrecognized input
  fields, ``residual`` holds the exact original bytes of every line the
  canonical form cannot reproduce. A residual line writes only its raw
  bytes: its content and extra are not stored a second time.
  Byte-identical content and extra payloads are stored once and shared
  by ref (sha256-interned); residual items are never shared, and sit in
  row order. Content and extra items are packed in similarity-sorted
  order, not arrival order: replay always goes through refs, so the
  store owns placement, and putting near-duplicate payloads next to
  each other is what lets zstd see their shared bytes inside one
  window (see _PackedStore).

The one non-negotiable property: ``unshred`` yields every input line
byte-for-byte in original order. A row keeps a null residual_ref only
when re-serializing its canonical event, canonical fields in an order
derived deterministically from the field values (see _canonical_line)
and extra keys in original order, with
``json.dumps(obj, separators=(",", ":"), ensure_ascii=False)``
reproduces the raw line exactly. Every other line, malformed ones
included, keeps its raw bytes whole in the residual store, and those
bytes win at replay time. All sizes in this module are bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from density.engine.trace.zdict import (
    LEVEL_COLD,
    BlockReader,
    BlockWriter,
    SpilledItems,
)
from density.errors import StoreError
from density.ingest.traces import ParsedLine

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# Rows per parquet row group. Boundaries depend only on the row count, so
# the output file is bit-identical for a given input sequence.
_ROW_GROUP_ROWS = 65_536
_READ_BATCH_ROWS = 8_192

# Block size for the residual store. Measured on the seed-1337 synth
# corpus at level 19: 16 MB blocks compress 45 percent smaller than the
# zdict default of 4 MB, because repeated multi-KB payloads stop being
# re-learned at every block boundary. The cost is coarser random access:
# a reader decompresses one 16 MB block to serve a ref, which sequential
# replay (residual items sit in row order) amortizes fully.
#
# No shared dictionary is trained. Measured on the same corpus, a trained
# dictionary never repays its own stored bytes here: items are compressed
# inside large concatenated blocks, so the block itself already provides
# the context a dictionary would supply (110 KB dict saved only 60 KB of
# block bytes at 4 MB blocks, and made blocks larger at 16 MB). Skipping
# training also lets every line stream straight to the writers, so memory
# stays bounded regardless of corpus shape.
_BLOCK_BYTES = 16 << 20

# Frame shape for the packed (content, extra) stores. Once items are
# similarity-sorted, most redundancy is local, but template families
# larger than zstd level 19's default 8 MiB window still repeat across
# multi-MB spans: 64 MB frames with the window widened to 64 MiB and
# long-distance matching enabled catch those. Measured on the seed-1337
# synth corpus: sorted 16 MB frames to sorted 64 MB + LDM is -2.3
# percent at 0.2 GB; going wider at 1 GB (128 MB frames, or one frame
# with window_log 27) recovers under 0.8 percent more, so 64 MB is the
# knee and keeps random access and writer memory sane. window_log stays
# within the stock decompressor limit, so the frames read anywhere; the
# parameters are recorded in the store index for provenance. Peak
# writer memory stays bounded by zdict's in-flight seal cap regardless
# of corpus size.
_PACK_BLOCK_BYTES = 64 << 20
_PACK_WINDOW_LOG = 26
_PACK_ENABLE_LDM = True

# zdict guarantees byte-identical frames for any worker count, so a small
# compression pool is free speed, not a determinism risk.
_COMPRESS_WORKERS = 4

# Byte-identical content and extra payloads share one stored item, keyed
# by sha256 (the same interning rule the dedup contract states for
# bodies). Agent traces resend system prompts and pooled snippets
# constantly, so interning removes most items outright, which shrinks
# blocks, the per-item spill, and the ref column (repeated rows repeat
# the same index). The map is capped and evicts oldest-first so memory
# stays bounded on corpora of unique payloads; eviction only costs a
# duplicate store later, never correctness. The residual store is never
# interned: its append order carries one item per residual line, which
# replay accounting and tests rely on.
_INTERN_CAP = 1 << 20

_STORE_PREFIXES = ("content", "extra", "residual")

_REF_TYPE = pa.int64()
_SCHEMA = pa.schema(
    [
        ("trace_id", pa.string()),
        ("ts", pa.int64()),
        ("role", pa.string()),
        ("type", pa.string()),
        ("model", pa.string()),
        ("tool_name", pa.string()),
        ("tokens_in", pa.int64()),
        ("tokens_out", pa.int64()),
        ("file", pa.string()),
        ("line_index", pa.int64()),
        ("content_ref", _REF_TYPE),
        ("extra_ref", _REF_TYPE),
        ("residual_ref", _REF_TYPE),
    ]
)

_DICT_COLUMNS = ["trace_id", "role", "type", "model", "tool_name", "file"]
# Integer encodings, measured per column on the seed-1337 synth corpus
# at 1 GB. ts and line_index are near-monotone, so delta packing stores
# small gaps. residual_ref is exactly the running count of residual
# rows: deltas are almost all zero-width. content_ref and extra_ref are
# similarity-sorted item indices, effectively shuffled relative to row
# order: delta packing EXPANDS them (random deltas span twice the value
# range), while BYTE_STREAM_SPLIT groups the mostly-zero high bytes for
# zstd and lands closest to the column's empirical entropy (1.60 vs
# 1.72 MB delta at 1 GB). Token counts are small skewed ints: splitting
# byte planes beats delta there too, by a hair.
_COLUMN_ENCODING = {
    "ts": "DELTA_BINARY_PACKED",
    "line_index": "DELTA_BINARY_PACKED",
    "content_ref": "BYTE_STREAM_SPLIT",
    "extra_ref": "BYTE_STREAM_SPLIT",
    "residual_ref": "DELTA_BINARY_PACKED",
    "tokens_in": "BYTE_STREAM_SPLIT",
    "tokens_out": "BYTE_STREAM_SPLIT",
}

# Rewrite pass batch size: one row group at a time keeps the final file's
# row group boundaries identical to the streaming layout (they depend
# only on row count), which keeps output bit-identical across runs.
_TMP_SUFFIX = ".tmp.parquet"


@dataclass(frozen=True)
class ShredResult:
    """Counters and sizes from one shred run. All byte counts are bytes.

    raw_bytes counts input payload bytes (line bytes, newlines excluded),
    matching IngestStats.bytes_read. store_bytes maps store prefix to
    compressed bytes of its blocks.bin. residual_lines counts every line
    whose replay comes from the residual store, malformed lines included.
    """

    lines: int
    events: int
    malformed: int
    residual_lines: int
    raw_bytes: int
    structured_bytes: int
    store_bytes: dict[str, int]
    out_dir: Path


def _fits_int64(value: int) -> bool:
    return _INT64_MIN <= value <= _INT64_MAX


def _parquet_text(value: str | None) -> str | None:
    """Make a str safe for a parquet UTF-8 column.

    JSON escape sequences can decode to lone surrogates, which UTF-8
    refuses to encode. Such a line is always residual (its canonical
    re-serialization fails the same way), so the column value is only
    informational and a deterministic backslash-escaped form is fine.
    """
    if value is None:
        return None
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError:
        return value.encode("utf-8", "backslashreplace").decode("utf-8")


def _dump_extra(extra: dict) -> bytes:
    """Compact JSON dump of the extra dict, key order preserved.

    surrogatepass keeps the store writable even when a value contains a
    lone surrogate; rows needing it are residual, so these bytes are
    never used to rebuild a line.
    """
    text = json.dumps(extra, separators=(",", ":"), ensure_ascii=False)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "surrogatepass")


def _canonical_line(
    trace_id: str,
    ts: int,
    role: str,
    type_: str,
    content: str,
    model: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
    tool_name: str | None,
    extra: dict,
) -> bytes | None:
    """Re-serialize a canonical event to line bytes, or None if impossible.

    The original positions of recognized keys are not stored, so the field
    order must be a deterministic function of the stored field values
    alone: unshred recomputes the same order from the same row. Two
    layouts exist, both starting trace_id, ts, role, type and ending with
    extra keys in original input order:

    - default: content, then model, tokens_in, tokens_out, tool_name as
      present. This is the common message shape (token-counted assistant
      turns put tool_name last if they have one at all).
    - tool layout, chosen when tool_name is present and both token counts
      are absent: tool_name, content, then model if present. Tool call
      and tool result lines are usually emitted as "which tool, then its
      payload" and carry no token counts, so keying the layout on that
      combination keeps such lines out of the residual store.

    Lines whose key order matches neither layout still replay exactly,
    from the residual store. dict.update keeps the position of a colliding
    key, so an unconsumed alias in extra (for example an unparseable "ts")
    lands in the same slot at shred and unshred time. Encoding fails only
    for lone surrogates, and that failure is the signal to keep the raw
    bytes instead.
    """
    tool_layout = tool_name is not None and tokens_in is None and tokens_out is None
    obj: dict[str, Any] = {
        "trace_id": trace_id,
        "ts": ts,
        "role": role,
        "type": type_,
    }
    if tool_layout:
        obj["tool_name"] = tool_name
    obj["content"] = content
    if model is not None:
        obj["model"] = model
    if tokens_in is not None:
        obj["tokens_in"] = tokens_in
    if tokens_out is not None:
        obj["tokens_out"] = tokens_out
    if tool_name is not None and not tool_layout:
        obj["tool_name"] = tool_name
    obj.update(extra)
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        return None


@dataclass(slots=True)
class _Pending:
    """One line prepared for writing: parquet scalars plus store payloads."""

    file: str
    line_index: int
    trace_id: str | None = None
    ts: int | None = None
    role: str | None = None
    type: str | None = None
    model: str | None = None
    tool_name: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    content: bytes | None = None
    extra: bytes | None = None
    residual: bytes | None = None


def _build_pending(parsed: ParsedLine) -> _Pending:
    """Classify one parsed line and prepare its row and store payloads."""
    if parsed.event is None:
        # Malformed: no structured fields at all, the raw bytes are the
        # whole story and they go to the residual store untouched.
        return _Pending(file=parsed.file, line_index=parsed.line_index, residual=parsed.raw)

    ev = parsed.event
    canon = _canonical_line(
        ev.trace_id,
        ev.ts,
        ev.role,
        ev.type,
        ev.content,
        ev.model,
        ev.tokens_in,
        ev.tokens_out,
        ev.tool_name,
        ev.extra,
    )
    residual = None if canon == parsed.raw else parsed.raw

    # Parsed ints can exceed int64 (a giant "ts" is still valid JSON).
    # The parquet column cannot hold them, so the value becomes null and
    # the line is forced residual: replay must never depend on a value
    # the row could not store.
    ts_value: int | None = ev.ts
    tokens_in = ev.tokens_in
    tokens_out = ev.tokens_out
    if ts_value is not None and not _fits_int64(ts_value):
        ts_value = None
        residual = parsed.raw
    if tokens_in is not None and not _fits_int64(tokens_in):
        tokens_in = None
        residual = parsed.raw
    if tokens_out is not None and not _fits_int64(tokens_out):
        tokens_out = None
        residual = parsed.raw

    # A residual line replays from its raw bytes alone, so writing its
    # content or extra to the block stores would store the largest
    # payloads twice for zero replay value. Empty content stores nothing
    # either: a null content_ref on a canonical row means empty body,
    # which saves three ref ints per heartbeat-style line.
    if residual is not None:
        content = None
        extra = None
    else:
        content = ev.content.encode("utf-8", "surrogatepass") if ev.content else None
        extra = _dump_extra(ev.extra) if ev.extra else None

    return _Pending(
        file=parsed.file,
        line_index=parsed.line_index,
        trace_id=_parquet_text(ev.trace_id),
        ts=ts_value,
        role=_parquet_text(ev.role),
        type=_parquet_text(ev.type),
        model=_parquet_text(ev.model),
        tool_name=_parquet_text(ev.tool_name),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        content=content,
        extra=extra,
        residual=residual,
    )


# Fixed-width sort-key prefixes kept per unique payload. 64 bytes of
# whitespace-collapsed text is enough to group payloads generated from
# the same template (log lines, tool call JSON, prompts) even when early
# slot values differ in whitespace; the raw 96-byte prefix then orders
# within a group, and the sha256 makes the order total and deterministic
# for payloads that agree on both prefixes.
_KEY_TEMPLATE_BYTES = 64
_KEY_RAW_BYTES = 96
_WS_RUN = re.compile(rb"[ \t\r\n\f\v]+")


class _PackedStore:
    """Collect unique payloads streaming, then pack them similarity-sorted.

    Streaming phase: ``add`` interns payloads by sha256 and appends first
    occurrences to a spill file beside the bundle, returning a dense
    first-occurrence uid that rows can hold immediately. Pack phase:
    ``pack`` sorts the unique payloads by (whitespace-collapsed prefix,
    raw prefix, sha256, uid) and appends them to a BlockWriter in that
    order, so near-duplicate payloads (same template, different slot
    values) become neighbors and compress against each other inside one
    zstd window. Placement is free because replay always resolves items
    through refs. Returns the uid -> packed-position permutation for the
    parquet rewrite.

    Memory stays bounded by construction: payload bytes live only in the
    spill file, and the sort operates on fixed-width key prefixes plus
    hashes, about 200 bytes per unique payload. Determinism: the sort key
    is a pure function of payload bytes with the uid as final tiebreaker,
    so identical input streams pack identically.
    """

    def __init__(self, out_dir: Path, prefix: str, level: int) -> None:
        self._dir = out_dir
        self._prefix = prefix
        self._level = int(level)
        self._spill_path = out_dir / f"{prefix}.spill.tmp"
        self._spill = open(self._spill_path, "w+b")
        self._spill_pos = 0
        self._offsets: list[int] = []
        self._lengths: list[int] = []
        self._keys: list[bytes] = []
        self._raws: list[bytes] = []
        self._shas: list[bytes] = []
        self._intern: dict[bytes, int] = {}

    def add(self, payload: bytes) -> int:
        """Intern one payload; returns its first-occurrence uid."""
        digest = hashlib.sha256(payload).digest()
        uid = self._intern.get(digest)
        if uid is not None:
            return uid
        uid = len(self._lengths)
        self._spill.write(payload)
        self._offsets.append(self._spill_pos)
        self._spill_pos += len(payload)
        self._lengths.append(len(payload))
        self._keys.append(_WS_RUN.sub(b" ", payload[: 4 * _KEY_TEMPLATE_BYTES])[:_KEY_TEMPLATE_BYTES])
        self._raws.append(payload[:_KEY_RAW_BYTES])
        self._shas.append(digest)
        if len(self._intern) >= _INTERN_CAP:
            # Oldest-first eviction: dict preserves insertion order. A
            # re-added payload gets a fresh uid and a duplicate stored
            # item; sorting puts the twins adjacent, so the duplicate
            # compresses to almost nothing.
            del self._intern[next(iter(self._intern))]
        self._intern[digest] = uid
        return uid

    def pack(self, workers: int) -> tuple[np.ndarray, int]:
        """Sort, write the block store, drop the spill.

        Returns (perm, compressed_bytes) where perm[uid] is the item's
        packed position in the store's append order.
        """
        n = len(self._lengths)
        order: np.ndarray
        if n:
            keys = np.zeros(
                n,
                dtype=[
                    ("key", f"S{_KEY_TEMPLATE_BYTES}"),
                    ("raw", f"S{_KEY_RAW_BYTES}"),
                    ("sha", "S32"),
                    ("uid", "<i8"),
                ],
            )
            keys["key"] = self._keys
            keys["raw"] = self._raws
            keys["sha"] = self._shas
            keys["uid"] = np.arange(n, dtype=np.int64)
            # numpy S-dtype comparison ignores trailing NUL bytes, which
            # only perturbs neighbor choice, never determinism: the same
            # payloads always produce the same stripped keys, and the uid
            # field makes the order total.
            order = np.argsort(keys, order=("key", "raw", "sha", "uid"), kind="stable")
            del keys
        else:
            order = np.zeros(0, dtype=np.int64)
        self._keys = []
        self._raws = []
        self._shas = []
        self._intern = {}

        writer = BlockWriter(
            self._dir,
            self._prefix,
            self._level,
            workers=workers,
            block_bytes=_PACK_BLOCK_BYTES,
            window_log=_PACK_WINDOW_LOG,
            enable_ldm=_PACK_ENABLE_LDM,
        )
        perm = np.zeros(n, dtype=np.int64)
        for position, uid in enumerate(order):
            uid_i = int(uid)
            self._spill.seek(self._offsets[uid_i])
            writer.append(self._spill.read(self._lengths[uid_i]))
            perm[uid_i] = position
        compressed = writer.finish()
        self._spill.close()
        self._spill_path.unlink(missing_ok=True)
        return perm, compressed


def shred_events(
    parsed_iter: Iterable[ParsedLine],
    out_dir: str | Path,
    level: int = LEVEL_COLD,
    seed: int = 1337,
) -> ShredResult:
    """Write a columnar bundle for a stream of parsed lines.

    parsed_iter is consumed once, in order (see ingest.traces.iter_events).
    level is the zstd level for the block stores and the parquet pages
    (zdict.LEVEL_COLD or LEVEL_WARM in normal use). seed is accepted for
    interface uniformity: shredding draws no random numbers, so output is
    bit-identical for the same input regardless of seed.

    Two phases, both with bounded memory. The streaming phase sends each
    line straight to a temporary parquet file (rows hold provisional
    first-occurrence uids) and the stores' spill files; nothing scales
    with corpus size beyond one row group, a few in-flight blocks, and
    per-unique-payload sort keys. The pack phase sorts unique payloads,
    writes the final block stores, and rewrites the parquet with packed
    item indices, one row group at a time. The temporary files live
    beside the bundle and are removed before returning.

    Returns a ShredResult; all its byte counts are bytes.
    """
    del seed  # No stochastic step exists here; see the docstring.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / "structured.parquet"
    tmp_parquet_path = out / f"structured{_TMP_SUFFIX}"
    # The temp parquet is a spill format, not an output: light compression
    # for speed, no fancy encodings. Its bytes never affect determinism
    # because only the rewritten final file survives.
    tmp_writer = pq.ParquetWriter(
        tmp_parquet_path, _SCHEMA, compression="zstd", compression_level=1
    )
    cols: dict[str, list] = {name: [] for name in _SCHEMA.names}
    packed = {
        "content": _PackedStore(out, "content", int(level)),
        "extra": _PackedStore(out, "extra", int(level)),
    }
    residual_store = BlockWriter(
        out,
        "residual",
        int(level),
        workers=_COMPRESS_WORKERS,
        block_bytes=_BLOCK_BYTES,
    )
    lines = events = malformed = residual_lines = raw_bytes = 0
    residual_seq = 0

    def _flush_rows(force: bool = False) -> None:
        n = len(cols["file"])
        if n == 0 or (not force and n < _ROW_GROUP_ROWS):
            return
        tmp_writer.write_table(pa.table(cols, schema=_SCHEMA))
        for column in cols.values():
            column.clear()

    def _emit(p: _Pending) -> None:
        nonlocal residual_seq
        content_uid = packed["content"].add(p.content) if p.content is not None else None
        extra_uid = packed["extra"].add(p.extra) if p.extra is not None else None
        if p.residual is not None:
            # Residual items are appended in row order, so the item index
            # is exactly the running residual count: the column becomes a
            # 0,1,2,... sequence over residual rows, which delta-packs to
            # almost nothing.
            residual_store.append(p.residual)
            residual_idx = residual_seq
            residual_seq += 1
        else:
            residual_idx = None
        cols["trace_id"].append(p.trace_id)
        cols["ts"].append(p.ts)
        cols["role"].append(p.role)
        cols["type"].append(p.type)
        cols["model"].append(p.model)
        cols["tool_name"].append(p.tool_name)
        cols["tokens_in"].append(p.tokens_in)
        cols["tokens_out"].append(p.tokens_out)
        cols["file"].append(p.file)
        cols["line_index"].append(p.line_index)
        cols["content_ref"].append(content_uid)
        cols["extra_ref"].append(extra_uid)
        cols["residual_ref"].append(residual_idx)
        _flush_rows()

    for parsed in parsed_iter:
        p = _build_pending(parsed)
        lines += 1
        raw_bytes += len(parsed.raw)
        if parsed.event is None:
            malformed += 1
        else:
            events += 1
        if p.residual is not None:
            residual_lines += 1
        _emit(p)

    _flush_rows(force=True)
    tmp_writer.close()

    store_bytes = {"residual": residual_store.finish()}
    perms: dict[str, pa.Array] = {}
    for prefix in ("content", "extra"):
        perm, compressed = packed[prefix].pack(workers=_COMPRESS_WORKERS)
        store_bytes[prefix] = compressed
        perms[prefix] = pa.array(perm, type=pa.int64())

    # Rewrite pass: map provisional uids to packed positions. pc.take
    # keeps nulls null, so rows without a payload stay untouched. Row
    # group boundaries depend only on the row count (the temp file was
    # written in _ROW_GROUP_ROWS groups and is read back the same way),
    # so the final file is bit-identical for a given input sequence.
    writer = pq.ParquetWriter(
        parquet_path,
        _SCHEMA,
        compression="zstd",
        compression_level=int(level),
        use_dictionary=list(_DICT_COLUMNS),
        column_encoding=dict(_COLUMN_ENCODING),
    )
    tmp_file = pq.ParquetFile(tmp_parquet_path)
    for batch in tmp_file.iter_batches(batch_size=_ROW_GROUP_ROWS):
        table = pa.Table.from_batches([batch])
        for prefix, column in (("content", "content_ref"), ("extra", "extra_ref")):
            idx = table.schema.get_field_index(column)
            remapped = pc.take(perms[prefix], table.column(column))
            table = table.set_column(idx, column, remapped)
        writer.write_table(table)
    writer.close()
    tmp_file.close()
    tmp_parquet_path.unlink(missing_ok=True)

    return ShredResult(
        lines=lines,
        events=events,
        malformed=malformed,
        residual_lines=residual_lines,
        raw_bytes=raw_bytes,
        structured_bytes=parquet_path.stat().st_size,
        store_bytes=store_bytes,
        out_dir=out,
    )


def unshred(dir_path: str | Path) -> Iterator[tuple[str, int, bytes]]:
    """Yield (file, line_index, raw_bytes) for every shredded line, in order.

    The yielded bytes are byte-identical to the original input line
    (newline excluded). Rows with a residual_ref replay straight from the
    residual store; every other row is re-serialized from its stored
    fields, which shred_events proved matches the original bytes.

    Ref columns hold item indices. Content and extra items are packed in
    similarity order, unrelated to row order, so this full scan reads
    them through SpilledItems: one bounded-memory decompression pass to a
    temp file, then a seek per row, instead of re-decompressing large
    frames on every out-of-order access. Residual items sit in row order
    and stream through the ordinary block cache.

    Raises StoreError when the bundle is missing files or a row marked
    canonical cannot be re-serialized (which means on-disk corruption).
    """
    out = Path(dir_path)
    parquet_path = out / "structured.parquet"
    if not parquet_path.exists():
        raise StoreError(f"not a shred bundle, missing structured.parquet: {out}")
    pf = pq.ParquetFile(parquet_path)
    with SpilledItems(out, "content") as content_store, SpilledItems(out, "extra") as extra_store:
        residual_store = BlockReader(out, "residual")
        for batch in pf.iter_batches(batch_size=_READ_BATCH_ROWS):
            for row in batch.to_pylist():
                residual_ref = row["residual_ref"]
                if residual_ref is not None:
                    raw = residual_store.get_item(residual_ref)
                else:
                    # A canonical row with no content_ref had an empty
                    # body: shred_events stores nothing for empty content.
                    content_ref = row["content_ref"]
                    body = (
                        content_store.get(content_ref).decode("utf-8")
                        if content_ref is not None
                        else ""
                    )
                    extra: dict = {}
                    if row["extra_ref"] is not None:
                        extra = json.loads(
                            extra_store.get(row["extra_ref"]).decode("utf-8")
                        )
                    raw_or_none = _canonical_line(
                        row["trace_id"],
                        row["ts"],
                        row["role"],
                        row["type"],
                        body,
                        row["model"],
                        row["tokens_in"],
                        row["tokens_out"],
                        row["tool_name"],
                        extra,
                    )
                    if raw_or_none is None:
                        raise StoreError(
                            "cannot re-serialize canonical row "
                            f"{row['file']}:{row['line_index']}: bundle is corrupt"
                        )
                    raw = raw_or_none
                yield row["file"], row["line_index"], raw
