"""Canonical trace event schema and alias canonicalization.

A TraceEvent is the normalized form of one JSON object from an agent trace
log. Vendors name the same concepts differently (traceId, conversation_id,
prompt_tokens, created_at, ...), so ``TraceEvent.from_raw`` folds a fixed
alias table into canonical field names. Every input key that was not
consumed by canonicalization is preserved verbatim, in original order, in
``extra`` so that replay and audit can reconstruct the original object.

Units and shapes: ``ts`` is microseconds since the Unix epoch (int64
range). All text is Python str, JSON is always UTF-8.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICRO = timedelta(microseconds=1)

# Alias tables in priority order, stored lowercase because input keys are
# matched case-insensitively. The first alias that yields a usable value
# wins. Any other input key that happens to alias the same field is left
# alone so it survives verbatim in ``extra``.
_TRACE_ID_ALIASES = ("trace_id", "traceid", "conversation_id", "session_id", "run_id")
_TS_ALIASES = ("ts", "timestamp", "time", "created_at")
_ROLE_ALIASES = ("role", "speaker", "author")
_TYPE_ALIASES = ("type", "event_type", "kind")
_CONTENT_ALIASES = ("content", "text", "message", "output")
_MODEL_ALIASES = ("model", "model_name")
_TOKENS_IN_ALIASES = ("tokens_in", "prompt_tokens", "input_tokens")
_TOKENS_OUT_ALIASES = ("tokens_out", "completion_tokens", "output_tokens")
_TOOL_NAME_ALIASES = ("tool_name", "tool", "name")


def _parse_epoch_number(value: int | float) -> int | None:
    """Normalize an epoch number of unknown unit to int microseconds.

    The unit is inferred from magnitude: below 1e11 the value is seconds
    (covers dates to year 5138), below 1e14 it is milliseconds, otherwise
    it is already microseconds. Integer inputs stay in exact integer
    arithmetic so large timestamps never lose precision to float math.

    Returns None for non-finite floats. Python's json.loads accepts the
    NaN/Infinity/-Infinity tokens by default, and round() on those raises,
    so they must be rejected here as uninterpretable rather than crash a
    pipeline deep inside canonicalization.
    """
    if isinstance(value, int):
        magnitude = abs(value)
        if magnitude < 100_000_000_000:
            return value * 1_000_000
        if magnitude < 100_000_000_000_000:
            return value * 1_000
        return value
    if not math.isfinite(value):
        return None
    magnitude = abs(value)
    if magnitude < 1e11:
        return round(value * 1_000_000)
    if magnitude < 1e14:
        return round(value * 1_000)
    return round(value)


def _parse_ts(value: Any) -> int | None:
    """Parse a timestamp of any accepted shape to int microseconds.

    Accepts epoch numbers (seconds, millis, or micros, inferred from
    magnitude), numeric strings, and ISO 8601 strings. Naive datetimes are
    taken as UTC. Returns None when the value cannot be interpreted,
    including NaN and infinite floats (numeric or string form), so the
    caller can leave the original key in ``extra`` as evidence.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _parse_epoch_number(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        return _parse_epoch_number(int(text))
    except ValueError:
        pass
    try:
        return _parse_epoch_number(float(text))
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    # Exact integer arithmetic on timedeltas, no float rounding.
    return (dt - _EPOCH) // _MICRO


def _parse_str(value: Any) -> str:
    """Coerce any value to str. Non-strings become their str() form."""
    return value if isinstance(value, str) else str(value)


def _parse_content(value: Any) -> str:
    """Strings pass through; anything else is JSON-dumped compactly.

    Compact separators and ensure_ascii=False keep the dump byte-stable
    and unicode-faithful, which matters for the byte-exact replay path.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _parse_int(value: Any) -> int | None:
    """Coerce token counts to int; return None for garbage values.

    bool is excluded because JSON true/false as a token count is noise,
    not a number.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _take(
    obj: dict,
    consumed: set[str],
    aliases: tuple[str, ...],
    parse: Callable[[Any], Any],
) -> Any:
    """Resolve one canonical field from ``obj`` using an alias table.

    Walks aliases in priority order and input keys in insertion order,
    matching case-insensitively. Three outcomes per matched key:
    - null value: consumed silently (a null alias carries no information
      and must not leak into extra), and later aliases may still win;
    - parseable value: consumed, resolution stops, later same-field
      aliases stay in extra untouched;
    - unparseable value (parse returned None): left unconsumed so the
      original key survives in extra as evidence of what was received.
    """
    for alias in aliases:
        for key in obj:
            if key in consumed or key.lower() != alias:
                continue
            value = obj[key]
            if value is None:
                consumed.add(key)
                continue
            parsed = parse(value)
            if parsed is None:
                continue
            consumed.add(key)
            return parsed
    return None


class TraceEvent(BaseModel):
    """One canonicalized trace event.

    ts is int microseconds since the Unix epoch. content is the main text
    body and may be empty. extra holds every unknown input field verbatim,
    in the original key order.
    """

    # A field literally named "model" is part of the contract; disable
    # pydantic's protected "model_" namespace so it raises no warnings.
    model_config = ConfigDict(protected_namespaces=())

    trace_id: str
    ts: int
    role: str
    type: str
    content: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_name: str | None = None
    extra: dict = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, obj: dict) -> TraceEvent:
        """Canonicalize a parsed JSON object into a TraceEvent.

        Unrecognized keys land in ``extra`` unchanged. Missing trace_id,
        role, and type become "unknown"; missing ts becomes 0; missing
        content becomes the empty string.
        """
        consumed: set[str] = set()

        trace_id = _take(obj, consumed, _TRACE_ID_ALIASES, _parse_str)
        ts = _take(obj, consumed, _TS_ALIASES, _parse_ts)
        role = _take(obj, consumed, _ROLE_ALIASES, _parse_str)
        type_ = _take(obj, consumed, _TYPE_ALIASES, _parse_str)
        content = _take(obj, consumed, _CONTENT_ALIASES, _parse_content)
        model = _take(obj, consumed, _MODEL_ALIASES, _parse_str)
        tokens_in = _take(obj, consumed, _TOKENS_IN_ALIASES, _parse_int)
        tokens_out = _take(obj, consumed, _TOKENS_OUT_ALIASES, _parse_int)

        type_ = type_ if type_ is not None else "unknown"
        # The bare "name" alias is ambiguous (people have names too), so
        # it only counts as a tool name when the event type says tool.
        is_tool_event = type_.lower().startswith("tool")
        tool_aliases = _TOOL_NAME_ALIASES if is_tool_event else _TOOL_NAME_ALIASES[:2]
        tool_name = _take(obj, consumed, tool_aliases, _parse_str)

        # Dict comprehension preserves the input insertion order, which is
        # what lets extra round-trip byte-for-byte through json.dumps.
        extra = {k: v for k, v in obj.items() if k not in consumed}

        return cls(
            trace_id=trace_id if trace_id is not None else "unknown",
            ts=ts if ts is not None else 0,
            role=role if role is not None else "unknown",
            type=type_,
            content=content if content is not None else "",
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tool_name=tool_name,
            extra=extra,
        )
