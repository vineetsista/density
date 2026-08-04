"""Streaming trace ingestion: raw line iteration and malformed-line policy.

Reads JSONL trace logs line by line, never loading a whole file, and
preserves the exact input bytes of every line so that downstream storage
can replay the corpus byte-for-byte. Malformed lines never crash a
pipeline: they surface as ParsedLine records with ``event=None`` and an
error string, and IngestStats counts them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from density.errors import IngestError
from density.ingest.schemas import TraceEvent

# Trace files are *.jsonl plus numeric shard suffixes (*.jsonl.0,
# *.jsonl.17). Other suffixes like .jsonl.gz are deliberately excluded:
# this reader handles raw bytes only, and silently decompressing would
# break the byte-exact replay contract.
_TRACE_FILE_RE = re.compile(r"\.jsonl(\.\d+)?$")


@dataclass
class ParsedLine:
    """One physical input line with its parse outcome.

    raw holds the exact line bytes without the trailing newline (a CR from
    CRLF input is part of the payload). Exactly one of event or error is
    set: event for well-formed lines, error for malformed ones.
    """

    file: str
    line_index: int
    raw: bytes
    event: TraceEvent | None = None
    error: str | None = None


@dataclass
class IngestStats:
    """Counters aggregated over ParsedLine records.

    bytes_read counts payload bytes (raw line bytes, newlines excluded).
    """

    files: int = 0
    events: int = 0
    malformed: int = 0
    bytes_read: int = 0
    # Tracks which files this instance has already counted so files stays
    # a distinct count under repeated observe calls. Not part of the
    # public dataclass surface.
    _seen_files: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def observe(self, line: ParsedLine) -> None:
        """Fold one ParsedLine into the counters."""
        if line.file not in self._seen_files:
            self._seen_files.add(line.file)
            self.files += 1
        if line.event is None:
            self.malformed += 1
        else:
            self.events += 1
        self.bytes_read += len(line.raw)

    def merge(self, other: "IngestStats") -> None:
        """Add another stats object into this one, counter by counter."""
        self.files += other.files
        self.events += other.events
        self.malformed += other.malformed
        self.bytes_read += other.bytes_read
        self._seen_files |= other._seen_files


def _trace_files(root: Path) -> list[tuple[str, Path]]:
    """List (relpath, path) trace files under root, sorted by relpath.

    Sorting makes iteration order deterministic across filesystems, which
    determinism of every downstream artifact depends on.
    """
    if root.is_file():
        return [(root.name, root)]
    return sorted(
        (p.relative_to(root).as_posix(), p)
        for p in root.rglob("*")
        if p.is_file() and _TRACE_FILE_RE.search(p.name)
    )


def iter_raw_lines(path: str | Path) -> Iterator[tuple[str, int, bytes]]:
    """Yield (file_relpath, line_index, raw_line_bytes) for every line.

    path may be a single file or a directory searched recursively for
    *.jsonl and numeric shards *.jsonl.N. Lines are streamed, never a
    whole file at once. The yielded bytes exclude the trailing newline
    byte only; a CR from CRLF input is preserved, and a missing final
    newline is tolerated, so joining lines with b"\\n" reproduces the
    file byte-exactly (up to that final newline).

    Raises IngestError when path does not exist.
    """
    root = Path(path)
    if not root.exists():
        raise IngestError(f"trace path does not exist: {root}")
    for relpath, file_path in _trace_files(root):
        try:
            handle = file_path.open("rb")
        except OSError as exc:
            raise IngestError(f"cannot read trace file {file_path}: {exc}") from exc
        with handle:
            # Binary file iteration buffers one line at a time, so even a
            # multi-megabyte single line costs only its own size.
            for line_index, line in enumerate(handle):
                yield relpath, line_index, line.removesuffix(b"\n")


def _parse_line(raw: bytes) -> tuple[TraceEvent | None, str | None]:
    """Parse one raw line into (event, None) or (None, error string)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"invalid utf-8: {exc}"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(obj, dict):
        return None, f"not a json object: {type(obj).__name__}"
    try:
        return TraceEvent.from_raw(obj), None
    except Exception as exc:
        # Defensive: a canonicalization bug on one weird line must not
        # kill a multi-gigabyte ingest run. Quarantine and move on.
        return None, f"canonicalization failed: {exc}"


def iter_events(
    path: str | Path,
    on_malformed: str = "quarantine",
) -> Iterator[ParsedLine]:
    """Yield a ParsedLine for every physical line under path.

    on_malformed controls what happens to lines that are not valid JSON
    objects: "quarantine" (default) yields them with event=None and an
    error string, "skip" drops them, "raise" raises IngestError on the
    first one. Any other policy name raises IngestError.
    """
    if on_malformed not in ("quarantine", "skip", "raise"):
        raise IngestError(f"unknown on_malformed policy: {on_malformed!r}")
    for file_relpath, line_index, raw in iter_raw_lines(path):
        event, error = _parse_line(raw)
        if event is None:
            if on_malformed == "raise":
                raise IngestError(f"{file_relpath}:{line_index}: {error}")
            if on_malformed == "skip":
                continue
        yield ParsedLine(file=file_relpath, line_index=line_index, raw=raw, event=event, error=error)
