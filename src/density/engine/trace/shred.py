"""Shred parsed trace lines into a byte-exact columnar bundle.

shred_events writes, under one output directory:

- ``structured.parquet``: one row per input line. Scalar columns are
  trace_id, ts, role, type, model, tool_name, tokens_in, tokens_out,
  file, line_index; ref columns are content_ref, extra_ref,
  residual_ref, each a 3-int list ``[block, offset, length]`` into the
  matching block store. ts is int64 microseconds since the Unix epoch,
  stored absolute and delta-encoded on disk (DELTA_BINARY_PACKED);
  low-cardinality strings are dictionary-encoded.
- three zstd block stores (see zdict): ``content`` holds the event text
  body, ``extra`` holds the compact JSON dump of unrecognized input
  fields, ``residual`` holds the exact original bytes of every line the
  canonical form cannot reproduce. A residual line writes only its raw
  bytes: its content and extra are not stored a second time.
  Byte-identical content and extra payloads are stored once and shared
  by ref (sha256-interned); residual items are never shared.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from density.engine.trace.zdict import LEVEL_COLD, BlockReader, BlockWriter, Ref
from density.errors import StoreError
from density.ingest.traces import ParsedLine

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# Rows per parquet row group. Boundaries depend only on the row count, so
# the output file is bit-identical for a given input sequence.
_ROW_GROUP_ROWS = 65_536
_READ_BATCH_ROWS = 8_192

# Block size for the text stores. Measured on the seed-1337 synth corpus
# at level 19: 16 MB blocks compress content 45 percent smaller than the
# zdict default of 4 MB, because repeated multi-KB payloads (resent
# system prompts) stop being re-learned at every block boundary. The cost
# is coarser random access: a reader decompresses one 16 MB block to
# serve a ref, which sequential replay amortizes fully.
#
# No shared dictionary is trained. Measured on the same corpus, a trained
# dictionary never repays its own stored bytes here: items are compressed
# inside large concatenated blocks, so the block itself already provides
# the context a dictionary would supply (110 KB dict saved only 60 KB of
# block bytes at 4 MB blocks, and made blocks larger at 16 MB). Skipping
# training also lets every line stream straight to the writers, so memory
# stays bounded regardless of corpus shape.
_BLOCK_BYTES = 16 << 20

# zdict guarantees byte-identical frames for any worker count, so a small
# compression pool is free speed, not a determinism risk.
_COMPRESS_WORKERS = 4

# Byte-identical content and extra payloads share one stored item, keyed
# by sha256 (the same interning rule the dedup contract states for
# bodies). Agent traces resend system prompts and pooled snippets
# constantly, so interning removes most items outright, which shrinks
# blocks, the per-item index, and the ref columns (repeated rows repeat
# the same ref triple). The map is capped and evicts oldest-first so
# memory stays bounded on corpora of unique payloads; eviction only
# costs a duplicate store later, never correctness. The residual store
# is never interned: its append order carries one item per residual
# line, which replay accounting and tests rely on.
_INTERN_CAP = 1 << 20

_STORE_PREFIXES = ("content", "extra", "residual")

_REF_TYPE = pa.list_(pa.int64())
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
_COLUMN_ENCODING = {"ts": "DELTA_BINARY_PACKED", "line_index": "DELTA_BINARY_PACKED"}


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

    Memory is bounded regardless of corpus shape: each line streams
    straight to the parquet writer and block stores, which buffer at most
    one row group and a few in-flight blocks.

    Returns a ShredResult; all its byte counts are bytes.
    """
    del seed  # No stochastic step exists here; see the docstring.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / "structured.parquet"
    writer = pq.ParquetWriter(
        parquet_path,
        _SCHEMA,
        compression="zstd",
        compression_level=int(level),
        use_dictionary=list(_DICT_COLUMNS),
        column_encoding=dict(_COLUMN_ENCODING),
    )
    cols: dict[str, list] = {name: [] for name in _SCHEMA.names}
    stores = {
        prefix: BlockWriter(
            out,
            prefix,
            int(level),
            workers=_COMPRESS_WORKERS,
            block_bytes=_BLOCK_BYTES,
        )
        for prefix in _STORE_PREFIXES
    }
    interns: dict[str, dict[bytes, Ref]] = {"content": {}, "extra": {}}
    lines = events = malformed = residual_lines = raw_bytes = 0

    def _put_interned(prefix: str, payload: bytes) -> Ref:
        cache = interns[prefix]
        key = hashlib.sha256(payload).digest()
        ref = cache.get(key)
        if ref is None:
            ref = stores[prefix].append(payload)
            if len(cache) >= _INTERN_CAP:
                # Oldest-first eviction: dict preserves insertion order.
                del cache[next(iter(cache))]
            cache[key] = ref
        return ref

    def _flush_rows(force: bool = False) -> None:
        n = len(cols["file"])
        if n == 0 or (not force and n < _ROW_GROUP_ROWS):
            return
        writer.write_table(pa.table(cols, schema=_SCHEMA))
        for column in cols.values():
            column.clear()

    def _emit(p: _Pending) -> None:
        content_ref = _put_interned("content", p.content) if p.content is not None else None
        extra_ref = _put_interned("extra", p.extra) if p.extra is not None else None
        residual_ref = stores["residual"].append(p.residual) if p.residual is not None else None
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
        cols["content_ref"].append(list(content_ref) if content_ref is not None else None)
        cols["extra_ref"].append(list(extra_ref) if extra_ref is not None else None)
        cols["residual_ref"].append(list(residual_ref) if residual_ref is not None else None)
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

    store_bytes = {prefix: stores[prefix].finish() for prefix in _STORE_PREFIXES}
    _flush_rows(force=True)
    writer.close()

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

    Raises StoreError when the bundle is missing files or a row marked
    canonical cannot be re-serialized (which means on-disk corruption).
    """
    out = Path(dir_path)
    parquet_path = out / "structured.parquet"
    if not parquet_path.exists():
        raise StoreError(f"not a shred bundle, missing structured.parquet: {out}")
    pf = pq.ParquetFile(parquet_path)
    content_store = BlockReader(out, "content")
    extra_store = BlockReader(out, "extra")
    residual_store = BlockReader(out, "residual")

    for batch in pf.iter_batches(batch_size=_READ_BATCH_ROWS):
        for row in batch.to_pylist():
            residual_ref = row["residual_ref"]
            if residual_ref is not None:
                raw = residual_store.get(_as_ref(residual_ref))
            else:
                # A canonical row with no content_ref had an empty body:
                # shred_events stores nothing for empty content.
                content_ref = row["content_ref"]
                body = (
                    content_store.get(_as_ref(content_ref)).decode("utf-8")
                    if content_ref is not None
                    else ""
                )
                extra: dict = {}
                if row["extra_ref"] is not None:
                    extra = json.loads(
                        extra_store.get(_as_ref(row["extra_ref"])).decode("utf-8")
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


def _as_ref(values: list[int]) -> Ref:
    """Convert a parquet ref column value back into a block store ref."""
    if len(values) != 3:
        raise StoreError(f"malformed ref in structured.parquet: {values!r}")
    return (values[0], values[1], values[2])
