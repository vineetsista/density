"""Streaming line iteration and malformed-line policy for trace ingest."""

from __future__ import annotations

import json

import pytest

from density.errors import IngestError
from density.ingest.schemas import TraceEvent
from density.ingest.traces import IngestStats, ParsedLine, iter_events, iter_raw_lines


def test_iter_raw_lines_single_file(tmp_path) -> None:
    f = tmp_path / "a.jsonl"
    f.write_bytes(b'{"x":1}\n{"y":2}\n')
    got = list(iter_raw_lines(f))
    assert got == [("a.jsonl", 0, b'{"x":1}'), ("a.jsonl", 1, b'{"y":2}')]


def test_iter_raw_lines_missing_final_newline(tmp_path) -> None:
    f = tmp_path / "a.jsonl"
    f.write_bytes(b"x\ny")
    assert list(iter_raw_lines(f)) == [("a.jsonl", 0, b"x"), ("a.jsonl", 1, b"y")]


def test_iter_raw_lines_preserves_cr(tmp_path) -> None:
    f = tmp_path / "a.jsonl"
    f.write_bytes(b"a\r\nb\r\n")
    lines = [raw for _, _, raw in iter_raw_lines(f)]
    assert lines == [b"a\r", b"b\r"]
    # byte-exact reassembly must reproduce the file, CR included
    assert b"\n".join(lines) + b"\n" == f.read_bytes()


def test_iter_raw_lines_directory_with_shards(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jsonl").write_bytes(b"1\n")
    (tmp_path / "a.jsonl.1").write_bytes(b"2\n")
    (tmp_path / "sub" / "b.jsonl.10").write_bytes(b"3\n")
    (tmp_path / "notes.txt").write_bytes(b"ignored\n")
    (tmp_path / "a.jsonl.gz").write_bytes(b"ignored\n")
    got = list(iter_raw_lines(tmp_path))
    assert got == [
        ("a.jsonl", 0, b"1"),
        ("a.jsonl.1", 0, b"2"),
        ("sub/b.jsonl.10", 0, b"3"),
    ]


def test_iter_raw_lines_empty_and_newline_only_files(tmp_path) -> None:
    (tmp_path / "empty.jsonl").write_bytes(b"")
    (tmp_path / "newlines.jsonl").write_bytes(b"\n\n\n")
    got = list(iter_raw_lines(tmp_path))
    assert got == [
        ("newlines.jsonl", 0, b""),
        ("newlines.jsonl", 1, b""),
        ("newlines.jsonl", 2, b""),
    ]


def test_iter_raw_lines_missing_path_raises() -> None:
    with pytest.raises(IngestError):
        list(iter_raw_lines("/nonexistent/definitely/not/here"))


def test_ten_megabyte_single_line_streams(tmp_path) -> None:
    body = "x" * (10 * 1024 * 1024)
    line = json.dumps({"trace_id": "big", "ts": 1, "content": body}).encode()
    f = tmp_path / "big.jsonl"
    f.write_bytes(line)  # no trailing newline on purpose
    parsed = list(iter_events(f))
    assert len(parsed) == 1
    p = parsed[0]
    assert p.error is None
    assert p.raw == line
    assert isinstance(p.event, TraceEvent)
    assert p.event.trace_id == "big"
    assert len(p.event.content) == len(body)


def test_binary_garbage_is_malformed_not_exception(tmp_path) -> None:
    f = tmp_path / "junk.jsonl"
    f.write_bytes(b"\x00\x01\x02\xff\xfe binary garbage\n")
    parsed = list(iter_events(f))
    assert len(parsed) == 1
    assert parsed[0].event is None
    assert isinstance(parsed[0].error, str) and parsed[0].error


def test_non_object_json_is_malformed(tmp_path) -> None:
    f = tmp_path / "arr.jsonl"
    f.write_bytes(b"[1,2,3]\n\"str\"\n7\n")
    parsed = list(iter_events(f))
    assert [p.event for p in parsed] == [None, None, None]
    assert all(p.error for p in parsed)


def test_non_finite_ts_lines_are_events_not_malformed(tmp_path) -> None:
    # Python's json.loads accepts NaN/Infinity tokens, so these lines are
    # valid JSON objects. They must classify as events with ts 0 (the key
    # kept in extra as evidence), not as malformed lines.
    lines = [
        b'{"trace_id":"t","ts":NaN,"role":"user","type":"message","content":"a"}',
        b'{"trace_id":"t","ts":Infinity,"role":"user","type":"message","content":"b"}',
        b'{"trace_id":"t","ts":-Infinity,"role":"user","type":"message","content":"c"}',
        b'{"trace_id":"t","ts":"inf","role":"user","type":"message","content":"d"}',
    ]
    f = tmp_path / "nonfinite.jsonl"
    f.write_bytes(b"".join(raw + b"\n" for raw in lines))
    parsed = list(iter_events(f))
    assert all(p.error is None for p in parsed)
    assert all(p.event is not None for p in parsed)
    assert [p.event.ts for p in parsed] == [0, 0, 0, 0]
    assert all(list(p.event.extra) == ["ts"] for p in parsed)


def test_iter_events_on_tiny_corpus_counts(tiny_corpus) -> None:
    corpus, n_events, n_malformed = tiny_corpus
    parsed = list(iter_events(corpus / "traces"))
    assert len(parsed) == n_events + n_malformed
    ok = [p for p in parsed if p.event is not None]
    bad = [p for p in parsed if p.event is None]
    assert len(ok) == n_events
    assert len(bad) == n_malformed
    assert all(p.error is None for p in ok)
    assert all(p.error for p in bad)
    # canonical fields survive
    assert ok[0].event.trace_id == "tr-0000"
    assert ok[0].event.extra == {"custom_field": {"nested": 0}}


def test_ingest_stats_aggregation(tiny_corpus) -> None:
    corpus, n_events, n_malformed = tiny_corpus
    stats = IngestStats()
    total = 0
    for p in iter_events(corpus / "traces"):
        stats.observe(p)
        total += len(p.raw)
    assert stats.files == 1
    assert stats.events == n_events
    assert stats.malformed == n_malformed
    assert stats.bytes_read == total


def test_ingest_stats_merge() -> None:
    a = IngestStats(files=1, events=2, malformed=3, bytes_read=4)
    b = IngestStats(files=10, events=20, malformed=30, bytes_read=40)
    a.merge(b)
    assert (a.files, a.events, a.malformed, a.bytes_read) == (11, 22, 33, 44)


def test_on_malformed_skip_and_raise(tmp_path) -> None:
    f = tmp_path / "mix.jsonl"
    f.write_bytes(b'{"trace_id":"t"}\n{broken\n')
    kept = list(iter_events(f, on_malformed="skip"))
    assert len(kept) == 1 and kept[0].event is not None
    with pytest.raises(IngestError):
        list(iter_events(f, on_malformed="raise"))
    with pytest.raises(IngestError):
        list(iter_events(f, on_malformed="bogus-policy"))


def test_parsed_line_shape(tmp_path) -> None:
    f = tmp_path / "one.jsonl"
    f.write_bytes(b'{"trace_id":"t","ts":5}\n')
    (p,) = list(iter_events(f))
    assert isinstance(p, ParsedLine)
    assert p.file == "one.jsonl"
    assert p.line_index == 0
    assert p.raw == b'{"trace_id":"t","ts":5}'
    assert p.event is not None and p.error is None


def test_byte_exact_reassembly_of_corpus(tiny_corpus) -> None:
    corpus, _, _ = tiny_corpus
    src = corpus / "traces" / "part-0000.jsonl"
    raws = [raw for _, _, raw in iter_raw_lines(src)]
    assert b"\n".join(raws) + b"\n" == src.read_bytes()
