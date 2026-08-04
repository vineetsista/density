"""Shred and unshred: byte-exact columnar storage of parsed trace lines.

shred_events splits well-formed lines into structured parquet columns plus
zstd text stores, keeping the exact original bytes of every line that its
canonical re-serialization cannot reproduce. unshred must replay every
input line byte-for-byte in original order: that is the one property these
tests treat as non-negotiable, on hand-built corpora and on generated ones.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pyarrow.parquet as pq
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from density.engine.trace.shred import ShredResult, shred_events, unshred
from density.engine.trace.zdict import LEVEL_COLD, LEVEL_WARM, BlockReader
from density.ingest.traces import iter_events, iter_raw_lines

SEED = 1337

CONTRACT_COLUMNS = [
    "trace_id",
    "ts",
    "role",
    "type",
    "model",
    "tool_name",
    "tokens_in",
    "tokens_out",
    "file",
    "line_index",
    "content_ref",
    "extra_ref",
    "residual_ref",
]


def _canon(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _event_obj(i: int, **overrides) -> dict:
    """Canonical-order event dict; overrides update in place or append after."""
    obj = {
        "trace_id": f"tr-{i % 5:03d}",
        "ts": 1_700_000_000_000_000 + i * 250_000,
        "role": ["system", "user", "assistant", "tool"][i % 4],
        "type": "message",
        "content": f"event {i} body",
    }
    obj.update(overrides)
    return obj


def _shred(src: Path, out: Path, level: int = LEVEL_WARM) -> ShredResult:
    return shred_events(iter_events(src), out, level=level, seed=SEED)


def _mixed_corpus(src: Path) -> list[tuple[str, bytes, str]]:
    """Write a two-file corpus; returns (file, raw_line, kind) in ingest order.

    kind is "canonical" (residual_ref expected null), "residual" (well-formed
    JSON that canonical re-serialization cannot reproduce byte-exactly), or
    "malformed" (not a JSON object at all).
    """
    ev_tool = _event_obj(
        1,
        role="assistant",
        type="tool_call",
        model="gpt-x",
        tokens_in=5,
        tokens_out=9,
        tool_name="search",
    )
    ev_unicode = _event_obj(2, content="unicode café 日本語 \U0001f680 ☃")
    ev_extra = _event_obj(3, custom={"nested": [1, 2]}, flag=True, pi=3.5)
    spaced = json.dumps(_event_obj(4), ensure_ascii=False).encode("utf-8")
    reordered = _canon(
        {
            "ts": 1_700_000_000_000_000,
            "trace_id": "tr-x",
            "role": "user",
            "type": "message",
            "content": "swapped key order",
        }
    )
    cr_line = _canon(_event_obj(6)) + b"\r"
    seconds_ts = _canon(
        {
            "trace_id": "tr-sec",
            "ts": 1_700_000_000,
            "role": "user",
            "type": "message",
            "content": "epoch seconds get normalized to micros",
        }
    )
    # Valid JSON whose decoded strings contain lone surrogates: legal input,
    # but impossible to re-encode strictly, so the raw bytes must win.
    surrogate_content = (
        b'{"trace_id":"tr-sur","ts":1,"role":"user","type":"message","content":"\\ud800"}'
    )
    surrogate_id = b'{"trace_id":"\\ud800","ts":2,"role":"user","type":"message","content":"x"}'

    a_lines: list[tuple[bytes, str]] = [
        (_canon(_event_obj(0)), "canonical"),
        (_canon(ev_tool), "canonical"),
        (_canon(ev_unicode), "canonical"),
        (_canon(ev_extra), "canonical"),
        (spaced, "residual"),
        (reordered, "residual"),
        (cr_line, "residual"),
        (seconds_ts, "residual"),
        (b"{truncated", "malformed"),
        (b"\x00\x01\x02\xfe binary junk", "malformed"),
        (b"just some plain text", "malformed"),
        (b"[1,2,3]", "malformed"),
        (surrogate_content, "residual"),
        (surrogate_id, "residual"),
    ]
    b_lines: list[tuple[bytes, str]] = [
        (_canon(_event_obj(20)), "canonical"),
        (b"{oops", "malformed"),
        (_canon(_event_obj(21)), "canonical"),
    ]
    (src / "a.jsonl").write_bytes(b"".join(raw + b"\n" for raw, _ in a_lines))
    (src / "b.jsonl").write_bytes(b"".join(raw + b"\n" for raw, _ in b_lines))
    return [("a.jsonl", raw, kind) for raw, kind in a_lines] + [
        ("b.jsonl", raw, kind) for raw, kind in b_lines
    ]


def test_mixed_corpus_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    spec = _mixed_corpus(src)
    out = tmp_path / "out"
    result = _shred(src, out)

    # The builder and the reader must agree on what was written.
    counters: dict[str, int] = {}
    expected = []
    for file, raw, _ in spec:
        idx = counters.get(file, 0)
        counters[file] = idx + 1
        expected.append((file, idx, raw))
    originals = list(iter_raw_lines(src))
    assert originals == expected

    # The whole point: every line comes back byte-for-byte, in order.
    assert list(unshred(out)) == originals

    assert result.lines == len(spec)
    assert result.events == sum(1 for _, _, k in spec if k != "malformed")
    assert result.malformed == sum(1 for _, _, k in spec if k == "malformed")
    assert result.residual_lines == sum(1 for _, _, k in spec if k != "canonical")
    assert result.raw_bytes == sum(len(raw) for _, raw, _ in spec)

    # The residual store holds exactly the non-canonical lines, in order.
    residual_items = list(BlockReader(out, "residual").iter_all())
    assert residual_items == [raw for _, raw, k in spec if k != "canonical"]

    # residual_ref is null precisely for canonical lines.
    table = pq.read_table(out / "structured.parquet")
    refs = table.column("residual_ref").to_pylist()
    assert [r is None for r in refs] == [k == "canonical" for _, _, k in spec]


def test_cr_inside_line_is_preserved(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    line = _canon(_event_obj(0)) + b"\r"
    (src / "a.jsonl").write_bytes(line + b"\n")
    out = tmp_path / "out"
    _shred(src, out)
    assert list(unshred(out)) == [("a.jsonl", 0, line)]


def test_structured_parquet_columns_and_encodings(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    objs = []
    for i in range(24):
        if i % 3 == 0:
            objs.append(_event_obj(i, type="tool_call", tool_name=["search", "fetch"][i % 2]))
        else:
            objs.append(_event_obj(i))
    (src / "a.jsonl").write_bytes(b"".join(_canon(o) + b"\n" for o in objs))
    out = tmp_path / "out"
    _shred(src, out)

    pf = pq.ParquetFile(out / "structured.parquet")
    assert pf.schema_arrow.names == CONTRACT_COLUMNS

    md = pf.metadata
    col_meta = {
        md.row_group(0).column(i).path_in_schema: md.row_group(0).column(i)
        for i in range(md.num_columns)
    }
    # ts is stored as absolute int64 microseconds, delta-encoded on disk.
    assert col_meta["ts"].physical_type == "INT64"
    assert "DELTA_BINARY_PACKED" in col_meta["ts"].encodings
    assert any(
        e in ("RLE_DICTIONARY", "PLAIN_DICTIONARY") for e in col_meta["tool_name"].encodings
    )

    table = pf.read()
    assert table.column("ts").to_pylist() == [o["ts"] for o in objs]
    assert table.column("file").to_pylist() == ["a.jsonl"] * len(objs)
    assert table.column("line_index").to_pylist() == list(range(len(objs)))
    assert table.column("tool_name").to_pylist() == [o.get("tool_name") for o in objs]
    assert table.column("trace_id").to_pylist() == [o["trace_id"] for o in objs]


def test_content_and_extra_stores(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    with_extra = _event_obj(0, custom={"k": [1, 2]}, note="n")
    plain = _event_obj(1)
    unicode_ev = _event_obj(2, content="café \U0001f680")
    objs = [with_extra, plain, unicode_ev]
    (src / "a.jsonl").write_bytes(b"".join(_canon(o) + b"\n" for o in objs))
    out = tmp_path / "out"
    result = _shred(src, out, level=LEVEL_COLD)

    table = pq.read_table(out / "structured.parquet")
    content_refs = table.column("content_ref").to_pylist()
    extra_refs = table.column("extra_ref").to_pylist()

    content_reader = BlockReader(out, "content")
    for ref, obj in zip(content_refs, objs):
        assert content_reader.get(tuple(ref)) == obj["content"].encode("utf-8")

    extra_reader = BlockReader(out, "extra")
    expected_extra = {"custom": {"k": [1, 2]}, "note": "n"}
    assert extra_reader.get(tuple(extra_refs[0])) == _canon(expected_extra)
    # Events without unknown keys carry no extra item at all.
    assert extra_refs[1] is None
    assert extra_refs[2] is None

    # All three lines are canonical, so the residual store stays empty.
    assert list(BlockReader(out, "residual").iter_all()) == []
    assert result.residual_lines == 0
    assert result.store_bytes["residual"] == 0


def test_malformed_lines_quarantined_to_residual(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    junk = [b"{broken", b"\xff\xfe not utf8", b"7", b'"a string"', b"plain words"]
    (src / "junk.jsonl").write_bytes(b"".join(raw + b"\n" for raw in junk))
    out = tmp_path / "out"
    result = _shred(src, out)

    assert result.events == 0
    assert result.malformed == len(junk)
    assert result.residual_lines == len(junk)

    table = pq.read_table(out / "structured.parquet")
    for col in (
        "trace_id",
        "ts",
        "role",
        "type",
        "model",
        "tool_name",
        "tokens_in",
        "tokens_out",
        "content_ref",
        "extra_ref",
    ):
        assert table.column(col).to_pylist() == [None] * len(junk)

    refs = table.column("residual_ref").to_pylist()
    reader = BlockReader(out, "residual")
    assert [reader.get(tuple(r)) for r in refs] == junk
    assert list(unshred(out)) == [("junk.jsonl", i, raw) for i, raw in enumerate(junk)]


def test_tool_layout_lines_are_canonical(tmp_path: Path) -> None:
    """Token-less tool lines with tool_name before content stay canonical.

    Tool call and tool result lines are commonly emitted as "which tool,
    then its payload" and carry no token counts. The canonical layout is
    a deterministic function of the stored field values, so both this
    shape and the tokens-last shape round-trip without the residual store.
    """
    src = tmp_path / "src"
    src.mkdir()
    call = _canon(
        {
            "trace_id": "tr-t",
            "ts": 1_700_000_000_000_000,
            "role": "assistant",
            "type": "tool_call",
            "tool_name": "search",
            "content": '{"q":"density"}',
            "model": "gpt-x",
        }
    )
    result = _canon(
        {
            "trace_id": "tr-t",
            "ts": 1_700_000_000_000_001,
            "role": "tool",
            "type": "tool_result",
            "tool_name": "search",
            "content": "12 hits",
        }
    )
    lines = [call, result]
    (src / "a.jsonl").write_bytes(b"".join(raw + b"\n" for raw in lines))
    out = tmp_path / "out"
    shredded = _shred(src, out)

    assert shredded.residual_lines == 0
    table = pq.read_table(out / "structured.parquet")
    assert table.column("residual_ref").to_pylist() == [None, None]
    assert table.column("tool_name").to_pylist() == ["search", "search"]
    assert list(unshred(out)) == [("a.jsonl", i, raw) for i, raw in enumerate(lines)]


def test_residual_lines_do_not_double_store_content_and_extra(tmp_path: Path) -> None:
    """A residual line replays from raw bytes alone: storing its content
    or extra as well would keep the largest payloads twice."""
    src = tmp_path / "src"
    src.mkdir()
    spaced = json.dumps(
        _event_obj(0, content="a body worth not storing twice", note="n"),
        ensure_ascii=False,
    ).encode("utf-8")
    (src / "a.jsonl").write_bytes(spaced + b"\n")
    out = tmp_path / "out"
    shredded = _shred(src, out)

    assert shredded.residual_lines == 1
    table = pq.read_table(out / "structured.parquet")
    assert table.column("content_ref").to_pylist() == [None]
    assert table.column("extra_ref").to_pylist() == [None]
    assert list(BlockReader(out, "content").iter_all()) == []
    assert list(BlockReader(out, "extra").iter_all()) == []
    assert list(BlockReader(out, "residual").iter_all()) == [spaced]
    assert list(unshred(out)) == [("a.jsonl", 0, spaced)]


def test_identical_content_is_interned_once(tmp_path: Path) -> None:
    """Byte-identical content payloads share one stored item and one ref,
    which is what keeps resent system prompts from bloating the store."""
    src = tmp_path / "src"
    src.mkdir()
    prompt = "the same system prompt resent on every call " * 8
    objs = [
        _event_obj(i, role="system", content=prompt) if i % 2 == 0 else _event_obj(i)
        for i in range(6)
    ]
    (src / "a.jsonl").write_bytes(b"".join(_canon(o) + b"\n" for o in objs))
    out = tmp_path / "out"
    _shred(src, out)

    table = pq.read_table(out / "structured.parquet")
    refs = table.column("content_ref").to_pylist()
    prompt_refs = [r for r, o in zip(refs, objs) if o["content"] == prompt]
    assert len(set(map(tuple, prompt_refs))) == 1
    stored = list(BlockReader(out, "content").iter_all())
    assert stored.count(prompt.encode("utf-8")) == 1
    # Interning must never change what replay yields.
    assert [raw for _, _, raw in unshred(out)] == [_canon(o) for o in objs]


def test_empty_content_stores_nothing(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    line = _canon(_event_obj(0, content=""))
    (src / "a.jsonl").write_bytes(line + b"\n")
    out = tmp_path / "out"
    shredded = _shred(src, out)

    assert shredded.residual_lines == 0
    table = pq.read_table(out / "structured.parquet")
    assert table.column("content_ref").to_pylist() == [None]
    assert list(BlockReader(out, "content").iter_all()) == []
    assert list(unshred(out)) == [("a.jsonl", 0, line)]


def test_non_finite_ts_round_trips(tmp_path: Path) -> None:
    """NaN/Infinity ts tokens are valid JSON to json.loads: the lines must
    classify as events, never crash, and always replay byte-exactly."""
    src = tmp_path / "src"
    src.mkdir()
    lines = [
        b'{"trace_id":"t","ts":NaN,"role":"user","type":"message","content":"a"}',
        b'{"trace_id":"t","ts":Infinity,"role":"user","type":"message","content":"b"}',
        b'{"trace_id":"t","ts":"inf","role":"user","type":"message","content":"c"}',
    ]
    (src / "a.jsonl").write_bytes(b"".join(raw + b"\n" for raw in lines))
    out = tmp_path / "out"
    shredded = _shred(src, out)

    assert shredded.malformed == 0
    assert shredded.events == len(lines)
    assert list(unshred(out)) == [("a.jsonl", i, raw) for i, raw in enumerate(lines)]


def test_out_of_int64_range_ints_fall_back_to_residual(tmp_path: Path) -> None:
    """Parsed ints that overflow the int64 columns must not crash or lie.

    Such a line may even re-serialize byte-exactly, but the parquet row
    cannot hold the value, so replay must come from the residual store.
    """
    src = tmp_path / "src"
    src.mkdir()
    big_ts = _canon(
        {"trace_id": "tr-big", "ts": 10**20, "role": "user", "type": "message", "content": "x"}
    )
    neg_ts = _canon(
        {"trace_id": "tr-neg", "ts": -(10**20), "role": "user", "type": "message", "content": "x"}
    )
    big_tokens = _canon(_event_obj(0, tokens_in=2**70))
    lines = [big_ts, neg_ts, big_tokens]
    (src / "a.jsonl").write_bytes(b"".join(raw + b"\n" for raw in lines))
    out = tmp_path / "out"
    result = _shred(src, out)

    assert result.malformed == 0
    assert result.residual_lines == 3
    table = pq.read_table(out / "structured.parquet")
    assert table.column("ts").to_pylist()[:2] == [None, None]
    assert table.column("tokens_in").to_pylist()[2] is None
    assert all(r is not None for r in table.column("residual_ref").to_pylist())
    assert list(unshred(out)) == [("a.jsonl", i, raw) for i, raw in enumerate(lines)]


def test_stores_use_requested_level(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jsonl").write_bytes(b"".join(_canon(_event_obj(i)) + b"\n" for i in range(10)))
    out = tmp_path / "out"
    _shred(src, out, level=LEVEL_COLD)
    for prefix in ("content", "extra", "residual"):
        index = json.loads((out / f"{prefix}.index.json").read_text(encoding="utf-8"))
        assert index["level"] == LEVEL_COLD


def test_empty_input_produces_empty_bundle(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = shred_events(iter([]), out, level=LEVEL_WARM, seed=SEED)
    assert result.lines == 0
    assert result.events == 0
    assert result.malformed == 0
    assert (out / "structured.parquet").exists()
    assert list(unshred(out)) == []


def test_determinism_bit_identical_outputs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _mixed_corpus(src)
    outputs = []
    for name in ("out-a", "out-b"):
        out = tmp_path / name
        _shred(src, out, level=LEVEL_COLD)
        outputs.append({p.name: p.read_bytes() for p in sorted(out.iterdir())})
    assert outputs[0].keys() == outputs[1].keys()
    assert outputs[0] == outputs[1]


# Property test: any mix of canonical events, loosely spaced events, and
# arbitrary junk lines must survive shred -> unshred byte-for-byte.

_CASE_COUNTER = itertools.count()

_JSON_TEXT = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=30)
_LINE_TEXT = st.text(
    alphabet=st.characters(blacklist_characters="\n", blacklist_categories=("Cs",)),
    max_size=30,
)

_EVENT_DICTS = st.fixed_dictionaries(
    {
        "trace_id": st.text(alphabet="abc-0123", max_size=8),
        "ts": st.integers(min_value=-(2**62), max_value=2**62 - 1),
        "role": st.sampled_from(["system", "user", "assistant", "tool"]),
        "type": st.sampled_from(["message", "tool_call", "tool_result"]),
        "content": _JSON_TEXT,
    },
    optional={
        "tokens_in": st.integers(min_value=0, max_value=10**9),
        "note": _JSON_TEXT,
        "ratio": st.floats(allow_nan=False, allow_infinity=False),
    },
)

_LINE_BYTES = st.one_of(
    _EVENT_DICTS.map(_canon),
    _EVENT_DICTS.map(lambda o: json.dumps(o, ensure_ascii=False).encode("utf-8")),
    _LINE_TEXT.map(lambda t: t.encode("utf-8")),
)


@settings(
    max_examples=30,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(lines=st.lists(_LINE_BYTES, max_size=8))
def test_round_trip_property(tmp_path: Path, lines: list[bytes]) -> None:
    case = tmp_path / f"case-{next(_CASE_COUNTER)}"
    src = case / "src"
    src.mkdir(parents=True)
    (src / "gen.jsonl").write_bytes(b"".join(line + b"\n" for line in lines))
    out = case / "out"
    result = shred_events(iter_events(src), out, level=LEVEL_WARM, seed=SEED)
    assert result.lines == len(lines)
    assert list(unshred(out)) == [("gen.jsonl", i, line) for i, line in enumerate(lines)]
