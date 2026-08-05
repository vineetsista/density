"""TraceEvent.from_raw canonicalization: aliases, timestamps, extra preservation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from density.ingest.schemas import TraceEvent

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _micros(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - _EPOCH) // timedelta(microseconds=1)


# ---------------------------------------------------------------- alias matrix

TRACE_ALIASES = ["trace_id", "traceId", "conversation_id", "session_id", "run_id"]
ROLE_ALIASES = ["role", "speaker", "author"]
TYPE_ALIASES = ["type", "event_type", "kind"]
CONTENT_ALIASES = ["content", "text", "message", "output"]
MODEL_ALIASES = ["model", "model_name"]
TOKENS_IN_ALIASES = ["tokens_in", "prompt_tokens", "input_tokens"]
TOKENS_OUT_ALIASES = ["tokens_out", "completion_tokens", "output_tokens"]
TS_ALIASES = ["ts", "timestamp", "time", "created_at"]


@pytest.mark.parametrize("key", TRACE_ALIASES)
def test_trace_id_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: "tr-1"})
    assert ev.trace_id == "tr-1"
    assert key not in ev.extra


@pytest.mark.parametrize("key", ROLE_ALIASES)
def test_role_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: "assistant"})
    assert ev.role == "assistant"
    assert ev.extra == {}


@pytest.mark.parametrize("key", TYPE_ALIASES)
def test_type_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: "tool_call"})
    assert ev.type == "tool_call"
    assert ev.extra == {}


@pytest.mark.parametrize("key", CONTENT_ALIASES)
def test_content_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: "hello"})
    assert ev.content == "hello"
    assert ev.extra == {}


@pytest.mark.parametrize("key", MODEL_ALIASES)
def test_model_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: "gpt-x"})
    assert ev.model == "gpt-x"
    assert ev.extra == {}


@pytest.mark.parametrize("key", TOKENS_IN_ALIASES)
def test_tokens_in_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: 12})
    assert ev.tokens_in == 12
    assert ev.extra == {}


@pytest.mark.parametrize("key", TOKENS_OUT_ALIASES)
def test_tokens_out_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: 34})
    assert ev.tokens_out == 34
    assert ev.extra == {}


@pytest.mark.parametrize("key", TS_ALIASES)
def test_ts_aliases(key: str) -> None:
    ev = TraceEvent.from_raw({key: 1_700_000_000})
    assert ev.ts == 1_700_000_000_000_000
    assert ev.extra == {}


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("TraceId", "trace_id", "tr-2"),
        ("CONVERSATION_ID", "trace_id", "tr-3"),
        ("Session_Id", "trace_id", "tr-4"),
        ("Speaker", "role", "user"),
        ("AUTHOR", "role", "sys"),
        ("Event_Type", "type", "message"),
        ("KIND", "type", "message"),
        ("Text", "content", "abc"),
        ("OUTPUT", "content", "xyz"),
        ("Model_Name", "model", "m1"),
        ("PROMPT_TOKENS", "tokens_in", 7),
        ("Completion_Tokens", "tokens_out", 8),
        ("Timestamp", "ts", 0),
        ("CREATED_AT", "ts", 0),
    ],
)
def test_alias_case_insensitive(key: str, field: str, value: object) -> None:
    ev = TraceEvent.from_raw({key: value})
    assert getattr(ev, field) == value
    assert ev.extra == {}


def test_first_alias_wins_and_loser_stays_in_extra() -> None:
    ev = TraceEvent.from_raw({"content": "a", "text": "b"})
    assert ev.content == "a"
    assert ev.extra == {"text": "b"}


def test_case_variant_duplicate_stays_in_extra() -> None:
    ev = TraceEvent.from_raw({"trace_id": "a", "TraceId": "b"})
    assert ev.trace_id == "a"
    assert ev.extra == {"TraceId": "b"}


def test_tool_name_aliases_on_tool_event() -> None:
    ev = TraceEvent.from_raw({"type": "tool_call", "name": "search"})
    assert ev.tool_name == "search"
    assert ev.extra == {}
    ev = TraceEvent.from_raw({"type": "tool_result", "tool": "grep"})
    assert ev.tool_name == "grep"
    ev = TraceEvent.from_raw({"type": "message", "tool_name": "explicit"})
    assert ev.tool_name == "explicit"


def test_name_alias_ignored_on_non_tool_event() -> None:
    ev = TraceEvent.from_raw({"type": "message", "name": "bob"})
    assert ev.tool_name is None
    assert ev.extra == {"name": "bob"}


def test_null_alias_consumed_and_next_alias_wins() -> None:
    # A recognized key with a null value carries no information: it must not
    # leak into extra, and a later alias may still supply the value.
    ev = TraceEvent.from_raw({"model": None, "model_name": "m2"})
    assert ev.model == "m2"
    assert ev.extra == {}


# ------------------------------------------------------------ timestamp matrix


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1_700_000_000, 1_700_000_000_000_000),            # epoch seconds int
        (1_700_000_000.25, 1_700_000_000_250_000),          # epoch seconds float
        (1_700_000_000_000, 1_700_000_000_000_000_000 // 1000),  # epoch millis
        (1_700_000_000_123, 1_700_000_000_123_000),         # epoch millis
        (1_700_000_000_000_000, 1_700_000_000_000_000),     # epoch micros
        (1_700_000_000_123_456, 1_700_000_000_123_456),     # epoch micros
        (0, 0),
        ("1700000000", 1_700_000_000_000_000),              # numeric string
        ("2023-11-14T22:13:20Z", 1_700_000_000_000_000),    # ISO, Z suffix
        ("2023-11-14T22:13:20+00:00", 1_700_000_000_000_000),
        ("2023-11-14T23:13:20+01:00", 1_700_000_000_000_000),
        ("2023-11-14T22:13:20", 1_700_000_000_000_000),     # naive means UTC
        ("2023-11-14T22:13:20.123456Z", 1_700_000_000_123_456),
        (
            "2023-11-14 22:13:20.5",
            _micros(datetime(2023, 11, 14, 22, 13, 20, 500000)),
        ),
    ],
)
def test_ts_normalization_matrix(raw: object, expected: int) -> None:
    ev = TraceEvent.from_raw({"ts": raw})
    assert ev.ts == expected


def test_unparseable_ts_becomes_zero_and_stays_in_extra() -> None:
    ev = TraceEvent.from_raw({"ts": "not a date"})
    assert ev.ts == 0
    assert ev.extra == {"ts": "not a date"}


@pytest.mark.parametrize(
    "raw",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "nan",
        "inf",
        "-inf",
        "Infinity",
        "1e999",
    ],
)
def test_non_finite_ts_is_unparseable_not_a_crash(raw: object) -> None:
    # json.loads accepts the NaN/Infinity tokens by default, so these
    # reach canonicalization as ordinary floats. They must be treated as
    # uninterpretable (ts 0, original key kept in extra), never raised.
    ev = TraceEvent.from_raw({"ts": raw})
    assert ev.ts == 0
    assert list(ev.extra) == ["ts"]


# -------------------------------------------------------------------- defaults


def test_defaults_for_missing_fields() -> None:
    ev = TraceEvent.from_raw({})
    assert ev.trace_id == "unknown"
    assert ev.ts == 0
    assert ev.role == "unknown"
    assert ev.type == "unknown"
    assert ev.content == ""
    assert ev.model is None
    assert ev.tokens_in is None
    assert ev.tokens_out is None
    assert ev.tool_name is None
    assert ev.extra == {}


# ---------------------------------------------------------------- content dump


def test_non_string_content_json_dumped_compactly() -> None:
    ev = TraceEvent.from_raw({"content": {"a": 1, "b": [1, 2]}})
    assert ev.content == '{"a":1,"b":[1,2]}'


def test_non_string_content_keeps_unicode() -> None:
    ev = TraceEvent.from_raw({"content": {"e": "é\U0001f680"}})
    assert ev.content == '{"e":"é\U0001f680"}'


def test_numeric_content_dumped() -> None:
    ev = TraceEvent.from_raw({"content": 42})
    assert ev.content == "42"


# ------------------------------------------------------------ extra preserving


def test_extra_preserved_byte_for_byte_through_model_dump() -> None:
    unknown = {
        "custom_field": {"nested": [1, 2, {"deep": "é\U0001f680"}]},
        "weird key ": 3.5,
        "flag": True,
        "nothing": None,
    }
    raw = {"trace_id": "t", "ts": 1, **unknown}
    ev = TraceEvent.from_raw(raw)
    dumped = ev.model_dump()["extra"]
    assert list(dumped.keys()) == list(unknown.keys())
    before = json.dumps(unknown, separators=(",", ":"), ensure_ascii=False)
    after = json.dumps(dumped, separators=(",", ":"), ensure_ascii=False)
    assert before.encode() == after.encode()


def test_tiny_corpus_style_event_round_trip() -> None:
    raw = {
        "trace_id": "tr-0001",
        "ts": 1_700_000_000_000_000,
        "role": "assistant",
        "type": "message",
        "content": "body",
        "model": "gpt-x",
        "custom_field": {"nested": 7},
    }
    ev = TraceEvent.from_raw(raw)
    assert ev.trace_id == "tr-0001"
    assert ev.ts == 1_700_000_000_000_000
    assert ev.role == "assistant"
    assert ev.type == "message"
    assert ev.content == "body"
    assert ev.model == "gpt-x"
    assert ev.extra == {"custom_field": {"nested": 7}}
