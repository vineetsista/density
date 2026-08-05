"""Deterministic synthetic agent corpus generator.

Writes a realistic agent-trace plus embedding corpus:

    out_dir/traces/part-0000.jsonl, part-0001.jsonl, ...   (UTF-8 JSONL)
    out_dir/embeddings/vectors.npy                          (float32 [n, dim])
    out_dir/embeddings/ids.npy                              (int64 [n])

Trace realism, all seeded and byte-reproducible per (gb, seed, dim):

- The stream is a GLOBAL TIME-ORDERED MERGE of many concurrent sessions,
  exactly like a production platform's append-only event log. One
  conversation advances slowly (seconds to minutes of model latency,
  tool time, and user think time) while the platform as a whole writes
  on the order of a megabyte per second across thousands of concurrent
  sessions, so consecutive events of one conversation land megabytes
  apart in the archive. Implemented as a streaming heapq.merge over
  per-lane monotonic generators; nothing buffers the corpus.
- Every conversation's system prompt is a persona BASE TEMPLATE (1.5 to
  4 KB, shared verbatim across that persona's conversations) plus an
  INJECTED PER-CONVERSATION CONTEXT BLOCK (roughly 200 to 1200 bytes of
  user profile, account state, session variables, and retrieved notes
  drawn from wide seeded spaces). The conversation resends its exact
  full prompt bytes on every model call, which is how real agent
  platforms behave and is the dominant deliberate redundancy of the
  corpus.
- Message and tool-output content is TEMPLATE PLUS SLOTS: names,
  emails, order ids, amounts, error codes, file paths, identifiers,
  and stack frames come from wide seeded spaces, so content bytes are
  mostly unique while template-level redundancy stays, matching real
  logs where the boilerplate repeats but the payload details never do.
- Idempotent tool re-reads: agents re-run the same tool with the same
  arguments inside one conversation (re-reading a file, re-fetching a
  customer record) and get byte-identical output minutes later.
- Kept from the original design: tool call JSON with jittered args,
  retry storms (2 to 6 near-identical calls), Poisson-bursty
  timestamps in integer microseconds, about 2 percent malformed lines,
  about 0.5 percent non-canonically spaced lines, unicode and emoji
  content, and occasional roughly 50 KB tool outputs.

Embedding realism: a low-rank manifold mixture. Real text embeddings
concentrate near low-dimensional manifolds (measured intrinsic
dimension of modern sentence encoders is roughly 10 to 40 despite
ambient dimensions of 768 or more), cluster scales differ by topic,
and real corpora carry heavy near-duplicate mass (retries, template
repeats). See sample_embeddings.

Design notes on speed: line tails are assembled from pre-encoded JSON
fragments plus slot fills, the per-conversation prompt is JSON-encoded
once and resent as bytes, and the k-way merge holds one pending event
per lane, so memory is O(concurrent sessions), never O(corpus).
"""

from __future__ import annotations

import heapq
import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = ["Persona", "PERSONAS", "SynthStats", "generate", "sample_embeddings"]

# Timestamp window: integer microseconds since epoch, mid 2024 to mid 2026.
# A capture is a contiguous slice of minutes, so it stays far inside the
# window; the margin below leaves room for the longest conversations.
_TS_LO = 1_720_000_000_000_000
_TS_HI = 1_780_000_000_000_000
_EPOCH_MARGIN_US = 7_200_000_000  # 2 hours

# Mean inter-event gaps in microseconds. Exponential inter-arrival times
# are exactly a Poisson process: small in-call gaps form the bursts, large
# inter-call and think gaps separate them, which is what gives the corpus
# its heavy-tailed gap distribution (max gap >> median gap).
_GAP_IN_CALL = 12_000
_GAP_TOOL_EXEC = 900_000
_GAP_RETRY = 1_800_000
_GAP_INTER_CALL = 20_000_000
_GAP_LONG_THINK = 240_000_000
_GAP_SESSION_IDLE = 45_000_000

_P_TOOL = 0.55        # a call uses a tool instead of answering directly
_P_STORM = 0.10       # a tool call turns into a retry storm of 2 to 6
_P_REREAD = 0.45      # a repeated tool is an idempotent re-read: same
                      # args, byte-identical output (file re-read,
                      # unchanged record re-fetch)
_P_LONG_THINK = 0.08  # an extra minutes-long pause before a call
_P_LARGE = 0.02       # a tool result is a roughly 50 KB dump
_P_MALFORMED = 0.02   # a malformed line is injected after an event line
_P_RESPACED = 0.005   # an event line is serialized with extra spaces

# Interleaving model. A platform serving thousands of concurrent
# sessions appends every event to one shared time-ordered log, writing
# on the order of a megabyte per second in aggregate while any single
# conversation waits seconds to minutes between calls. Each "lane" here
# is one concurrent session slot cycling through conversations, and the
# lane count is the trace byte budget divided by this constant, so
# concurrency grows linearly with corpus size exactly like a busier
# platform. Calibrated so that at 1 GB (about 9000 lanes, roughly 1.2
# MB/s of simulated aggregate write rate over a 10 to 15 minute
# capture) the median stream distance between consecutive calls of one
# conversation is about 16 MiB, far beyond the 8 MiB match window of
# zstd level 19. Smaller corpora interleave proportionally less, which
# is realistic: they model a quieter platform.
_LANE_TARGET_BYTES = 73_000
_LANE_STAGGER_US = 300_000_000  # sessions arrive over one mean cycle

# Embedding geometry. Real text embeddings live near low-dimensional
# manifolds: intrinsic-dimension estimates for modern sentence encoders
# land around 10 to 40 even at ambient dimension 768+. Each cluster is
# a low-rank patch (rank 12 to 24) with its own scale, plus a small
# isotropic residual for the off-manifold noise real encoders show.
# 35 percent of points are near-copies of a base point, mirroring the
# trace reality above (retry storms, template repeats) where many
# stored texts are trivial variants of one another.
_N_CLUSTERS = 200
_ZIPF_EXPONENT = 1.05
_RANK_LO = 10
_RANK_HI = 14
_TAU_LO = 0.60
_TAU_HI = 0.75
_ISO_FRACTION = 0.12
_DUP_FRACTION = 0.35
_DUP_JITTER = 0.08
_EMB_CHUNK = 65_536

_SHARD_CAP = 48 * 1024 * 1024   # stay well under the 64 MiB shard limit
_FLUSH_BYTES = 4 * 1024 * 1024

# Plain-text junk that agents really do leak into JSONL logs: none of it
# parses as JSON, so each line counts as malformed downstream.
_BARE_TEXT = (
    "Traceback (most recent call last):",
    '  File "agent_loop.py", line 214, in step',
    "ERROR upstream_timeout retry budget exhausted after 30000 ms",
    "<<<agent heartbeat 0x7f3a>>> stream closed by peer",
    "warn: tokenizer fallback engaged for span 4411..4508",
    "connection reset by peer while flushing event buffer",
    "OOM killer reaped worker 12, rss 7g, restarting from checkpoint",
    "partial write recovered, resuming at record boundary",
)

_MASK64 = (1 << 64) - 1


def _mix(j: int, salt: int) -> int:
    """splitmix64-style finalizer over (j, salt).

    One 62-bit rng draw per message feeds many decorrelated slot values:
    hashing (j, salt) is far cheaper than a numpy draw per slot and keeps
    the rng call sequence short and stable, which is what makes the
    generator's draw order easy to keep deterministic.
    """
    x = (j + (salt + 1) * 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _json_str(s: str) -> str:
    """Canonical JSON string literal for s (ensure_ascii=False)."""
    return json.dumps(s, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Slot value spaces. Wide and seeded: values are derived from mixed 64-bit
# integers, so the combinatorial spaces (thousands of names, 1e8 order
# ids, 16^12 session ids) make repeated values vanishingly rare while the
# surrounding template text repeats constantly, which is exactly the
# redundancy profile of real agent logs.
# ---------------------------------------------------------------------------

_FIRST_NAMES = (
    "Ava", "Liam", "Noah", "Mara", "Iris", "Hugo", "Nina", "Omar", "Priya",
    "Kenji", "Lucia", "Tomas", "Freya", "Dario", "Elif", "Anton", "Bea",
    "Caleb", "Dina", "Ewa", "Farid", "Greta", "Hana", "Ivo", "Jonas",
    "Kira", "Lena", "Marek", "Nadia", "Otis", "Paula", "Quinn", "Rosa",
    "Sven", "Tara", "Umar", "Vera", "Wren", "Ximena", "Yusuf", "Zoe",
    "Anders", "Bianca", "Ciro", "Delia", "Emeka", "Fanny", "Gustav",
)
_LAST_NAMES = (
    "Okafor", "Lindqvist", "Marchetti", "Novak", "Tanaka", "Fischer",
    "Alvarez", "Kowalski", "Haddad", "Bergstrom", "Costa", "Dubois",
    "Eriksen", "Farkas", "Grimaldi", "Horvat", "Ibrahim", "Jansen",
    "Kaur", "Lombardi", "Moreau", "Nielsen", "Oliveira", "Petrov",
    "Quispe", "Rossi", "Schmidt", "Takahashi", "Ueda", "Vasquez",
    "Weber", "Xu", "Yilmaz", "Zhang", "Amari", "Bruns", "Chen",
    "Dvorak", "Egede", "Fontaine", "Gruber", "Hansen", "Iversen",
    "Jonsson", "Keita", "Larsen", "Mbeki", "Nakamura", "Olsen",
    "Pavlov", "Quaranta", "Rahman", "Sato", "Toledo", "Unger",
    "Villanueva", "Wagner", "Xie", "Yamada", "Zapata", "Andrade",
    "Bishop", "Calder", "Dorsey",
)
_MAIL_DOMAINS = (
    "gmail.com", "outlook.com", "proton.me", "fastmail.com", "yahoo.com",
    "icloud.com", "gmx.net", "zoho.com", "hey.com",
)
_COMPANY_A = (
    "Blue", "North", "Clear", "Iron", "Swift", "Cedar", "Prime", "Quiet",
    "Vast", "Ember", "Solid", "Bright", "Deep", "Atlas", "Nova", "Delta",
    "Radial", "Summit", "Harbor", "Vector",
)
_COMPANY_B = (
    "metrics", "ledger", "signal", "forge", "harbor", "stack", "grid",
    "works", "path", "loop", "layer", "field", "pilot", "scale", "sense",
    "bridge", "current", "matter",
)
_REGIONS = ("us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-northeast", "sa-east")
_TIERS = ("free", "pro_monthly", "pro_annual", "team_monthly", "team_annual", "edu")
_LOCALES = ("en-US", "en-GB", "de-DE", "fr-FR", "es-MX", "pt-BR", "ja-JP", "zh-TW", "nl-NL", "sv-SE")
_ERROR_CODES = (
    "upstream_timeout", "rate_limited", "conn_reset", "schema_drift",
    "lock_timeout", "quota_exceeded", "checksum_mismatch", "stale_read",
    "oom_kill", "tls_handshake_fail", "dns_fail", "partial_commit",
)
_CODE_PKGS = ("core", "services", "ingest", "billing", "auth", "search", "export", "sync", "notify", "queue")
_CODE_MODS = (
    "scheduler", "worker", "session", "ledger", "parser", "mapper",
    "router", "cache", "batcher", "retry", "config", "uploader",
    "metrics", "tokenizer", "planner", "differ", "spooler", "monitor",
    "indexer", "compactor", "throttle", "resolver", "emitter",
    "checkpoint", "migrator", "auditor",
)
_CODE_VERBS = (
    "load", "flush", "merge", "split", "apply", "verify", "rebuild",
    "enqueue", "drain", "rotate", "compact", "resolve", "hydrate",
    "reconcile",
)
_CODE_NOUNS = (
    "batch", "cursor", "shard", "lease", "offset", "frame", "bucket",
    "epoch", "segment", "manifest", "quorum", "replica", "window",
    "ticket", "payload", "digest", "backlog", "snapshot",
)
_KB_TITLES = (
    "Refund windows by plan", "Overage credits",
    "Seat management for team plans", "Data export formats",
    "GDPR and account deletion", "Payment methods and currencies",
    "Workspace merge checklist", "Offline sync conflict rules",
    "API key rotation", "Invoice header customization",
    "Downgrade proration", "Duplicate charge investigation",
    "Export job troubleshooting", "Chargeback and dispute process",
)
_NOTE_PHRASES = (
    "customer confirmed by email", "escalated once before",
    "goodwill credit already used this year", "vip flag set by ae",
    "prior chargeback resolved amicably", "asked for invoice header change",
    "seat count disputed last quarter", "export retried successfully",
    "requested data residency info", "beta features enabled",
)
_WAREHOUSES = ("snowflake", "bigquery", "redshift", "databricks", "clickhouse")
_FUNDINGS = ("seed", "series_a", "series_b", "series_c", "bootstrapped")
_STAGES = ("inbound", "outreach", "nurture", "meeting_booked", "disqualified")


def _first(h: int) -> str:
    return _FIRST_NAMES[h % len(_FIRST_NAMES)]


def _last(h: int) -> str:
    return _LAST_NAMES[(h >> 8) % len(_LAST_NAMES)]


_SLOTS: dict[str, Callable[[int], str]] = {
    "name": lambda h: _first(h) + " " + _last(h),
    "email": lambda h: (
        _first(h).lower() + "." + _last(h).lower() + str(h % 97)
        + "@" + _MAIL_DOMAINS[(h >> 16) % len(_MAIL_DOMAINS)]
    ),
    "company": lambda h: (
        _COMPANY_A[h % len(_COMPANY_A)] + _COMPANY_B[(h >> 5) % len(_COMPANY_B)]
    ),
    "webdomain": lambda h: (
        _COMPANY_A[h % len(_COMPANY_A)].lower()
        + _COMPANY_B[(h >> 5) % len(_COMPANY_B)] + ".io"
    ),
    "order": lambda h: f"ord_{h % 100_000_000:08d}",
    "invoice": lambda h: f"inv_{h % 100_000_000:08d}",
    "ticket": lambda h: str(h % 90_000 + 10_000),
    "card": lambda h: str(h % 9_000 + 1_000),
    "amount": lambda h: f"{h % 480 + 3}.{(h >> 9) % 100:02d}",
    "pct": lambda h: f"{h % 89 + 10}",
    "prio": lambda h: str(h % 3 + 1),
    "date": lambda h: f"202{4 + h % 3}-{(h >> 4) % 12 + 1:02d}-{(h >> 9) % 28 + 1:02d}",
    "region": lambda h: _REGIONS[h % len(_REGIONS)],
    "tier": lambda h: _TIERS[h % len(_TIERS)],
    "locale": lambda h: _LOCALES[h % len(_LOCALES)],
    "sess": lambda h: f"sess_{h % (1 << 48):012x}",
    "errcode": lambda h: _ERROR_CODES[h % len(_ERROR_CODES)],
    "pkg": lambda h: _CODE_PKGS[h % len(_CODE_PKGS)],
    "mod": lambda h: _CODE_MODS[(h >> 3) % len(_CODE_MODS)],
    "path": lambda h: (
        "src/" + _CODE_PKGS[h % len(_CODE_PKGS)] + "/"
        + _CODE_MODS[(h >> 3) % len(_CODE_MODS)] + f"_{(h >> 9) % 40}.py"
    ),
    "ident": lambda h: (
        _CODE_VERBS[h % len(_CODE_VERBS)] + "_" + _CODE_NOUNS[(h >> 4) % len(_CODE_NOUNS)]
    ),
    "noun": lambda h: _CODE_NOUNS[h % len(_CODE_NOUNS)],
    "lineno": lambda h: str(h % 900 + 10),
    "sha": lambda h: f"{h % (1 << 40):010x}",
    "host": lambda h: (
        _CODE_NOUNS[h % len(_CODE_NOUNS)] + f"-{(h >> 5) % 800:03d}.internal"
    ),
    "lead": lambda h: f"lead_{h % 10_000_000:07d}",
    "version": lambda h: f"{1 + h % 4}.{(h >> 3) % 10}.{(h >> 7) % 20}",
    "ms": lambda h: str(h % 9_000 + 12),
    "count": lambda h: str(h % 240 + 2),
    "kbtitle": lambda h: _KB_TITLES[h % len(_KB_TITLES)],
    "note": lambda h: _NOTE_PHRASES[h % len(_NOTE_PHRASES)] + f" (ref {h % 100_000})",
    "warehouse": lambda h: _WAREHOUSES[h % len(_WAREHOUSES)],
    "funding": lambda h: _FUNDINGS[h % len(_FUNDINGS)],
    "stage": lambda h: _STAGES[h % len(_STAGES)],
}
_SLOT_INDEX = {name: i for i, name in enumerate(_SLOTS)}

# A compiled template alternates literal segments with (slot_fn, salt)
# fillers: literals[0] + fill0 + literals[1] + ... + literals[-1].
_Compiled = tuple[tuple[str, ...], tuple[tuple[Callable[[int], str], int], ...]]

_SLOT_RE = re.compile(r"\$\{([a-z]+)([0-9]*)\}")


def _compile(template: str) -> _Compiled:
    """Split a ${slot} template into literals and slot fillers once.

    The salt combines the slot's identity with its numeric suffix, so
    ${order} and ${order2} in one template draw different values while
    two plain ${order} occurrences agree (consistent entities inside a
    single message, like a real log line that repeats the same id).
    """
    literals: list[str] = []
    fillers: list[tuple[Callable[[int], str], int]] = []
    pos = 0
    for m in _SLOT_RE.finditer(template):
        name, suffix = m.group(1), m.group(2)
        fn = _SLOTS.get(name)
        if fn is None:
            raise KeyError(f"unknown slot {name!r} in template {template[:60]!r}")
        literals.append(template[pos : m.start()])
        fillers.append((fn, (_SLOT_INDEX[name] << 4) + (int(suffix) if suffix else 0)))
        pos = m.end()
    literals.append(template[pos:])
    return tuple(literals), tuple(fillers)


def _fill(compiled: _Compiled, j: int) -> str:
    """Instantiate a compiled template from message value j."""
    literals, fillers = compiled
    if not fillers:
        return literals[0]
    parts = [literals[0]]
    for i, (fn, salt) in enumerate(fillers):
        parts.append(fn(_mix(j, salt)))
        parts.append(literals[i + 1])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tool result builders. Each returns the content string of one
# tool_result event, sized like real tool output (records and search
# hits in the hundreds of bytes, file reads and test logs in the
# kilobytes) with template-level structure and slot-level entropy.
# About one in eight calls fails with a shared error shape, because
# real tools fail and error payloads are the most template-heavy text
# in real logs.
# ---------------------------------------------------------------------------

_T_TOOL_ERROR = _compile(
    '{"status":"error","code":"${errcode}","retry_in_ms":${ms},'
    '"request_id":"${sess}","hint":"transient, safe to retry"}'
)
_T_LOOKUP_HEAD = _compile(
    '{"status":"ok","customer":{"name":"${name}","email":"${email}",'
    '"plan":"${tier}","region":"${region}","since":"${date}",'
    '"mrr_usd":"${amount}"},"invoices":['
)
_T_INVOICE_ITEM = _compile(
    '{"id":"${invoice}","amount_usd":"${amount}","date":"${date}","paid":true}'
)
_T_LOOKUP_TAIL = _compile(
    '],"open_tickets":${count},"notes":["${note}","${note2}"],"sync_sha":"${sha}"}'
)
_T_KB_ITEM = _compile(
    '{"id":"kb-${ticket}","title":"${kbtitle}","updated":"${date}","score":"0.${pct2}"}'
)
_T_TICKET = _compile(
    '{"status":"ok","ticket_id":${ticket},"priority":"p${prio}","queue":"billing",'
    '"sla_hours":24,"assignee":"${name}"}'
)
_T_PATCH_OK = _compile(
    '{"ok":true,"file":"${path}","applied_hunks":${count},"rejected_hunks":0,'
    '"new_sha":"${sha}","fuzz":0}'
)
_T_PATCH_FAIL = _compile(
    '{"ok":false,"file":"${path}","error":"hunk ${count} failed to apply, '
    'context mismatch at line ${lineno}"}'
)
_T_ENRICH = _compile(
    '{"ok":true,"domain":"${webdomain}","company":"${company}","headcount":${count},'
    '"funding":"${funding}","warehouse":"${warehouse}","confidence":"0.${pct2}",'
    '"signals":["hiring ${count2} data roles as of ${date}",'
    '"careers page mentions ${warehouse}","contact ${name} <${email}>"]}'
)
_T_CRM = _compile(
    '{"ok":true,"lead_id":"${lead}","stage":"${stage}","next_step":"${date}",'
    '"updated_by":"piper","sync_sha":"${sha}"}'
)
_T_SEQUENCE = _compile(
    '{"ok":true,"sequence":"warm_intro_v${prio}","lead_id":"${lead}","touches":3,'
    '"started":true,"first_send":"${date}"}'
)
_T_TEST_PASS = _compile(
    "tests/${pkg}/test_${mod}.py::test_${ident} PASSED [ ${pct2}%]"
)
_T_TEST_FAIL = _compile(
    "FAILED tests/${pkg}/test_${mod}.py::test_${ident} :: ${errcode}"
)
_T_TEST_FRAME = _compile(
    '  File "src/${pkg}/${mod}.py", line ${lineno}, in ${ident}'
)
_T_TEST_ASSERT = _compile(
    "E   AssertionError: expected ${count} rows, got ${count2}"
)
_CODE_LINE_TMPLS = tuple(
    _compile(t)
    for t in (
        "def ${ident}(self, ${noun}, timeout_s=${count}):",
        "    lease = self._pool.acquire(lease_ms=${ms})",
        "    if lease is None:",
        '        raise LeaseTimeout("${errcode}")',
        "    for attempt in range(${count}):",
        "        batch = self._queue.pop_many(limit=${count2})",
        "        if not batch:",
        "            break",
        "        self._apply(batch, dry_run=False)",
        "    log.info('${ident} done', extra={'ms': ${ms}, 'rows': ${count2}})",
        "    return self._ledger.commit(${noun})",
        "# TODO(${name}): remove after ${date}, tracked in ${ticket}",
        "        except RetryBudgetExhausted:",
        "            self._metrics.incr('${ident}.retries')",
        "    assert ${noun} is not None, 'missing ${noun2}'",
        "    self._checkpoint.write(${noun}, sha='${sha}')",
    )
)


def _tool_failed(j: int) -> bool:
    """Shared failure coin: about one call in eight errors out."""
    return _mix(j, 399) % 8 == 0


def _res_lookup_customer(j: int) -> str:
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    h = _mix(j, 400)
    parts = [_fill(_T_LOOKUP_HEAD, j)]
    items = [_fill(_T_INVOICE_ITEM, _mix(j, 410 + i)) for i in range(2 + h % 4)]
    parts.append(",".join(items))
    parts.append(_fill(_T_LOOKUP_TAIL, j))
    return "".join(parts)


def _res_search_kb(j: int) -> str:
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    h = _mix(j, 420)
    items = [_fill(_T_KB_ITEM, _mix(j, 421 + i)) for i in range(2 + h % 3)]
    return '{"status":"ok","articles":[' + ",".join(items) + '],"query_ms":' + str(h % 900 + 8) + "}"


def _res_create_ticket(j: int) -> str:
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    return _fill(_T_TICKET, j)


def _res_read_file(j: int) -> str:
    """Pseudo file contents, 1 to 2.5 KB: real read_file output is text."""
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    h = _mix(j, 430)
    n_lines = 24 + h % 34
    path = _SLOTS["path"](_mix(j, 431))
    sha = _SLOTS["sha"](_mix(j, 432))
    out = [f"path {path} sha {sha} lines {n_lines}"]
    n_t = len(_CODE_LINE_TMPLS)
    for i in range(n_lines):
        hi = _mix(j, 440 + i)
        out.append(_fill(_CODE_LINE_TMPLS[hi % n_t], hi))
    return "\n".join(out)


def _res_run_tests(j: int) -> str:
    """A pytest-shaped log with 0 to 2 failure blocks."""
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    h = _mix(j, 450)
    n_pass = 6 + h % 16
    n_fail = (h >> 7) % 3
    out = ["============================= test session starts =============================="]
    for i in range(n_pass):
        out.append(_fill(_T_TEST_PASS, _mix(j, 460 + i)))
    for i in range(n_fail):
        hi = _mix(j, 480 + i)
        out.append(_fill(_T_TEST_FAIL, hi))
        out.append(_fill(_T_TEST_FRAME, hi))
        out.append(_fill(_T_TEST_FRAME, _mix(hi, 1)))
        out.append(_fill(_T_TEST_ASSERT, hi))
    out.append(f"{n_pass} passed, {n_fail} failed in {(h >> 9) % 90 + 4}.{h % 10}s")
    return "\n".join(out)


def _res_apply_patch(j: int) -> str:
    if _mix(j, 490) % 5 == 0:
        return _fill(_T_PATCH_FAIL, j)
    return _fill(_T_PATCH_OK, j)


def _res_enrich_lead(j: int) -> str:
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    return _fill(_T_ENRICH, j)


def _res_crm_update(j: int) -> str:
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    return _fill(_T_CRM, j)


def _res_send_sequence(j: int) -> str:
    if _tool_failed(j):
        return _fill(_T_TOOL_ERROR, j)
    return _fill(_T_SEQUENCE, j)


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Persona:
    """One synthetic agent persona.

    system_prompt is the BASE template, 1500 to 4000 UTF-8 bytes, shared
    verbatim across every conversation of the persona. Each conversation
    appends an injected context block (context_templates plus 0 to 4
    snippet_templates lines, roughly 200 to 1200 bytes) and resends the
    exact composed prompt on every model call: base bytes repeat across
    the whole corpus, injected bytes repeat only within one
    conversation. tools maps tool_name to an args builder
    (a, b, attempt) -> args JSON string (so retries can jitter args) and
    a result builder (j) -> content string. calls_lo..calls_hi bounds
    the model calls per conversation (coding agents run many more tool
    calls per task than chat-shaped agents).
    """

    key: str
    title: str
    model: str
    system_prompt: str
    context_templates: tuple[str, ...]
    snippet_templates: tuple[str, ...]
    user_templates: tuple[str, ...]
    assistant_templates: tuple[str, ...]
    tools: tuple[tuple[str, Callable[[int, int, int], str], Callable[[int], str]], ...]
    calls_lo: int
    calls_hi: int


_SUPPORT_PROMPT = (
    "You are Maya, the tier one customer support agent for Lumenote, a "
    "collaborative note taking product with free, pro, and team plans. You "
    "handle billing questions, refund requests, account access problems, and "
    "data export requests over chat.\n\n"
    "Ground rules:\n"
    "1. Verify the customer before discussing any account detail. Ask for the "
    "email on file, and the last four digits of the card only when billing is "
    "involved. Never ask for a full card number, a password, or a one time "
    "code, and never accept one if the customer volunteers it.\n"
    "2. Use the lookup_customer tool to pull the account record, the "
    "search_kb tool to find the current policy article, and the create_ticket "
    "tool to escalate. Never promise a refund before search_kb confirms the "
    "policy for that plan and region.\n"
    "3. Refund policy summary: monthly plans refund within 14 days of the "
    "charge, annual plans within 30 days, usage overage charges are never "
    "refunded but can be credited once per calendar year as a goodwill "
    "gesture up to 50 USD.\n"
    "4. Tone: calm, concrete, no jargon. Short sentences. One question at a "
    "time. Apologize once, then act. If the customer is angry, acknowledge "
    "the frustration in one sentence and move straight to the fix.\n"
    "5. Escalate to a human when you see chargeback threats, legal language, "
    "data loss claims, security reports, or any request you cannot resolve "
    "with the tools in two attempts. Escalations are priority p1 for team "
    "plan customers and p2 otherwise, and always include a one line summary "
    "plus the full tool output in the ticket body.\n"
    "6. Privacy: never reveal another user's data, never confirm whether an "
    "email address has an account, and redact any token or key the customer "
    "pastes into the chat, then tell them to rotate it immediately.\n"
    "7. Formatting: plain text only, no markdown tables. Currency always "
    "carries the ISO code, for example 12.00 USD.\n"
    "8. Close every conversation with a one line summary of what was done "
    "and what happens next, including the ticket number when one exists.\n"
    "9. If the tools error out twice in a row, tell the customer there is a "
    "temporary system issue, create a p2 ticket with the error text, and end "
    "the chat politely rather than guessing at account state."
)

_CODER_PROMPT = (
    "You are Rook, an autonomous coding agent working inside the Halcyon "
    "Labs monorepo. You plan, edit, run tests, and open changes for review. "
    "You never push directly to a release branch and you never edit files "
    "outside the task worktree.\n\n"
    "Operating procedure:\n"
    "1. Read before you write. Use read_file to inspect every file you plan "
    "to touch, plus its tests, before proposing a patch. Quote exact line "
    "numbers in your reasoning so reviewers can follow.\n"
    "2. Make the smallest change that satisfies the task. Do not reformat "
    "untouched code, do not rename public symbols, do not bump dependency "
    "versions unless the task says so explicitly.\n"
    "3. Every behavior change needs a test. Use run_tests with a narrow "
    "selector first, then the full suite for the affected package. A red "
    "test you wrote yourself is the only acceptable starting point for a "
    "bug fix.\n"
    "4. Use apply_patch for edits. Patches must apply cleanly with zero "
    "fuzz. If a patch fails to apply, re-read the file, never blind-retry "
    "the same hunk more than twice.\n"
    "5. Error handling policy for the codebase: library code raises typed "
    "exceptions, service code catches at the boundary and logs structured "
    "events, background jobs are idempotent and safe to re-run. Follow the "
    "existing pattern of the package you are in.\n"
    "6. Performance changes require numbers. Include the before and after "
    "measurement in the change description, with the exact command used, "
    "the machine class, and the variance across three runs.\n"
    "7. Security: never write secrets into code, config, tests, or logs. "
    "Flag any credential you encounter in the repo instead of using it. "
    "Treat all user input in the diff as hostile until validated.\n"
    "8. When tests are flaky, quarantine with a link to the tracking issue "
    "rather than retrying in a loop. Three consecutive identical failures "
    "means stop and report, not retry.\n"
    "9. Style: match the file you are editing. Comments explain why, not "
    "what. Public functions carry type hints and a docstring that states "
    "units and shapes for numeric data.\n"
    "10. Output discipline: your final message per task is a short summary, "
    "the list of files changed, the test evidence, and any follow-up work "
    "you deliberately left out of scope, each as its own line."
)

_SDR_PROMPT = (
    "You are Piper, a sales development representative for Datakite, a "
    "warehouse-native analytics platform sold to mid-market data teams. Your "
    "job is to qualify inbound leads, research accounts, and book meetings "
    "for account executives. You never quote custom pricing and you never "
    "send legal terms.\n\n"
    "Playbook:\n"
    "1. Research before outreach. Use enrich_lead to pull firmographics, "
    "then tailor the first line of every email to something true about the "
    "company. Generic openers are forbidden.\n"
    "2. Qualification bar: 20 or more employees, an existing cloud data "
    "warehouse, and a named data lead. Below the bar, send the self-serve "
    "starter link and mark the lead as nurture in crm_update.\n"
    "3. Sequences: use send_sequence with at most three touches over nine "
    "business days. Stop immediately on any reply, positive or negative, "
    "and log the outcome the same day.\n"
    "4. Meetings: offer two concrete slots in the prospect's timezone, "
    "thirty minutes, with the AE named in the invite. Confirm within one "
    "hour of booking.\n"
    "5. Tone: brief, specific, honest. No pressure tactics, no fake "
    "deadlines, no claiming features that are on the roadmap. If asked a "
    "technical question you cannot answer, say so and route it to the AE.\n"
    "6. Compliance: honor every unsubscribe instantly, never email personal "
    "addresses, never scrape sources marked internal, and keep all notes "
    "factual since prospects can request their data.\n"
    "7. Hygiene: every touch, call note, and status change lands in the CRM "
    "the same day via crm_update. A lead with no next step dated is a bug: "
    "fix it before end of day or hand it back to marketing."
)


def _support_tool_a(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"customer_id": "cus_{a % 100_000_000:08d}", '
        f'"include_invoices": true, "attempt": {attempt}}}'
    )


def _support_tool_b(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"query": "refund window annual plan region {b}", '
        f'"limit": {b + 3}, "attempt": {attempt}}}'
    )


def _support_tool_c(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"priority": "p{1 + b % 2}", "tags": ["billing", "escalation"], '
        f'"ref": {a % 1_000_000}, "attempt": {attempt}}}'
    )


def _coder_tool_a(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"path": "src/services/worker_{b}.py", '
        f'"start_line": {a % 4000}, "attempt": {attempt}}}'
    )


def _coder_tool_b(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"selector": "tests/unit/test_batch_{b}.py", '
        f'"timeout_s": {60 + a % 240}, "attempt": {attempt}}}'
    )


def _coder_tool_c(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"file": "src/core/scheduler.py", "hunk_id": {a % 10_000}, '
        f'"dry_run": false, "attempt": {attempt}}}'
    )


def _sdr_tool_a(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"domain": "example{a % 100_000}.io", '
        f'"fields": ["headcount", "funding"], "attempt": {attempt}}}'
    )


def _sdr_tool_b(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"lead_id": "lead_{a % 10_000_000:07d}", '
        f'"stage": "outreach_{b}", "attempt": {attempt}}}'
    )


def _sdr_tool_c(a: int, b: int, attempt: int) -> str:
    return (
        f'{{"template": "warm_intro_v{b}", '
        f'"lead_id": "lead_{a % 10_000_000:07d}", "attempt": {attempt}}}'
    )


PERSONAS: tuple[Persona, Persona, Persona] = (
    Persona(
        key="support",
        title="customer support agent",
        model="sable-large-2",
        system_prompt=_SUPPORT_PROMPT,
        context_templates=(
            "customer: ${name} <${email}>",
            "account: ${tier}, customer since ${date}, region ${region}, open invoices ${count}",
            "session: ${sess}, locale ${locale}, client web-${version}, csat last 90d ${pct2}",
        ),
        snippet_templates=(
            "note ${date}: refund of ${amount} USD on ${invoice} approved as a one "
            "time goodwill credit after a duplicate charge report was confirmed by billing.",
            "note ${date}: customer reported export job ${order} stuck at 90 percent, "
            "root cause was a stale worker lease, resolved by requeue on ${host}.",
            "kb excerpt kb-${ticket}: monthly plans refund within 14 days of the charge, "
            "annual within 30, overage is credit only, cap 50 USD per calendar year.",
            "ticket ${ticket} (${errcode}): seat sync between billing and workspace "
            "lagged one cycle, fixed by replaying events from ${date}.",
        ),
        user_templates=(
            "Hi, I was charged twice for invoice ${invoice} this month, ${amount} USD "
            "each, and I want one of them back.",
            "My export ${order} has been stuck at 90% since yesterday, can you check?",
            "I can't log in after changing my email to ${email}, the reset link 404s.",
            "Please cancel my annual plan and refund the unused months.",
            "Why did my invoice jump from ${amount} USD to ${amount2} USD?",
            "The team admin ${name} left the company and nobody can manage seats now.",
            "Je n'arrive pas à télécharger ma facture ${invoice}, ça bloque.",
            "請問可以把發票抬頭改成公司名稱嗎?謝謝!",
            "honestly this is the third time support ghosted me 😤 ticket ${ticket}, fix it",
            "love the product 🚀 but the ${amount} USD overage charge surprised me, help?",
            "I pasted my API key in a shared note by mistake, what do I do?",
            "Your status page says all green but nothing loads for me from ${region}.",
            "I need a GDPR export of everything you hold about ${email}.",
            "Charged in EUR but my card is in GBP, the fx fee on ${invoice} seems wrong.",
            "Can you merge my personal workspace into the ${company} team plan?",
            "The mobile app deleted my offline edits after sync on ${date}, recover them?",
        ),
        assistant_templates=(
            "I understand, and I can fix this. Could you confirm the email on the "
            "account so I can pull it up?",
            "Thanks ${name}. I see the duplicate ${amount} USD charge on ${invoice}, "
            "and per policy I can refund it since it is within 14 days.",
            "I have created ticket ${ticket} with priority p2. You will get an email "
            "update within one business day.",
            "The export job ${order} was stuck on our side. I restarted it and it "
            "should complete within 30 minutes. I will stay on the chat.",
            "That key should be treated as compromised. Please rotate it in Settings, "
            "then I will scrub it from the shared note history.",
            "Summary: refund of ${amount} USD issued to the card ending ${card}, "
            "confirmation email sent, ticket ${ticket} closed.",
            "I checked kb-${ticket} and annual plans refund within 30 days, so you "
            "are covered. Processing it now.",
            "I cannot see other accounts' data, but I can escalate this to the team "
            "that can, with your permission.",
        ),
        tools=(
            ("lookup_customer", _support_tool_a, _res_lookup_customer),
            ("search_kb", _support_tool_b, _res_search_kb),
            ("create_ticket", _support_tool_c, _res_create_ticket),
        ),
        calls_lo=2,
        calls_hi=8,
    ),
    Persona(
        key="coder",
        title="autonomous coding agent",
        model="sable-mini-1",
        system_prompt=_CODER_PROMPT,
        context_templates=(
            "repo: halcyon/${pkg}, branch task/${ident}-${count}, base sha ${sha}",
            "worktree: /work/${ident}, python 3.12, toolchain image build-${version}",
            "ci: last run ${date} on ${host}, ${count2} tests, quarantine list has ${count3} entries",
        ),
        snippet_templates=(
            "issue ${ticket}: ${errcode} in ${path} under concurrent retries, "
            "reproduces with selector tests/${pkg}/test_${mod}.py, owner ${name}.",
            "recent change ${sha}: ${ident} now takes an explicit deadline, callers "
            "in ${path} were migrated, ${path2} still uses the default.",
            "profiling note ${date}: ${ident} spends ${pct2} percent of wall time in "
            "${ident2}, candidate fix is batching the ${noun} lookups.",
            "review note: patches touching ${path} need a regression test since "
            "incident ${ticket} (${errcode}).",
        ),
        user_templates=(
            "The nightly batch job double-writes rows when a retry lands after a "
            "partial commit in ${path}. Fix it and prove it with a test.",
            "Please add a timeout to the ${ident} heartbeat, ${count}s default, "
            "configurable via env.",
            "Tests in tests/${pkg}/test_${mod}.py are flaky on CI, look into whether "
            "it is ordering or a real race.",
            "Migrate ${mod} from the legacy queue client, behavior must stay identical.",
            "Profile the ingest path, we regressed ${pct2}% since last release.",
            "Add structured logging to the retry loop in ${path}, include attempt "
            "count and backoff milliseconds.",
            "El linter falla en ${path} después del merge, ¿puedes arreglarlo?",
            "make the failing assertion in test_${ident} actually useful 🙏 it just "
            "says False is not True right now",
            "Why does apply_patch keep rejecting hunk ${count}? ${path} looks "
            "unchanged to me.",
            "Ship a fix for the off-by-one in pagination, page 2 repeats the last "
            "item of page 1.",
            "The config loader silently ignores unknown keys, that hid a typo in "
            "prod incident ${ticket}. Make it warn.",
            "Bump test coverage for the error paths in ${mod}, we only cover the "
            "happy path.",
        ),
        assistant_templates=(
            "Plan: reproduce with a narrow test, read ${path} and its tests, patch "
            "the smallest surface, then run the package suite. Starting with read_file.",
            "The root cause is a retry that re-enters commit() after the transaction "
            "already applied. The idempotency key in ${ident} is computed too late. "
            "Patch coming.",
            "Tests are green: ${count} passed, 0 failed, 2 skipped in ${ms}s. The "
            "new regression test fails on the old code as expected.",
            "The hunk failed because ${path} drifted upstream. Re-reading and "
            "rebasing the patch on the current contents.",
            "Measured before: ${ms} ms per 10k rows. After: ${ms2} ms. Command and "
            "variance are in the change description.",
            "Found a hardcoded token in ${path}. Not using it, flagging it per "
            "policy, and continuing with a fake.",
            "Quarantined test_${ident} with a link to issue ${ticket}, three "
            "identical failures in a row, per procedure.",
            "Done. Files changed: ${count}. Evidence: unit suite green, new test "
            "covers the ${errcode} retry path. Out of scope: ${ident} cleanup, "
            "noted in the task.",
        ),
        tools=(
            ("read_file", _coder_tool_a, _res_read_file),
            ("run_tests", _coder_tool_b, _res_run_tests),
            ("apply_patch", _coder_tool_c, _res_apply_patch),
        ),
        calls_lo=4,
        calls_hi=24,
    ),
    Persona(
        key="sdr",
        title="sales development representative",
        model="kestrel-8b",
        system_prompt=_SDR_PROMPT,
        context_templates=(
            "lead: ${name} <${email}>, ${company}",
            "firmographics: ${count} employees, warehouse ${warehouse}, region "
            "${region}, funding ${funding}",
            "crm: ${lead}, stage ${stage}, sequence warm_intro_v${prio}, last touch ${date}",
        ),
        snippet_templates=(
            "call note ${date}: ${name} asked about seat pricing versus usage "
            "pricing, no quote given per policy, AE to follow up.",
            "email reply ${date}: prospect said the current stack is ${warehouse} "
            "plus dbt, timeline is next quarter, budget owner is the data lead.",
            "research: ${company} is hiring ${count} data roles on their careers "
            "page as of ${date}, stack mentions ${warehouse}.",
        ),
        user_templates=(
            "New inbound from the pricing page: ${name}, data lead at a ${count} "
            "person fintech, warehouse is ${warehouse}. Qualify and book.",
            "Lead ${lead} replied 'not now, maybe Q3'. What is the right next step "
            "per playbook?",
            "The AE wants two slots offered for Thursday, prospect ${name} is in "
            "Europe, CET.",
            "Lead ${lead} unsubscribed mid-sequence, make sure everything stops.",
            "We got a batch of ${count} webinar signups, triage them against the "
            "qualification bar.",
            "Prospect asked whether we support dbt incremental models, I have no "
            "idea, handle it.",
            "México city based startup ${company}, ${count} personas, ¿califica "
            "para el self-serve o para un AE?",
            "the CEO of ${company} forwarded our email with just '???' 😅 what do "
            "we reply?",
            "CRM shows no next step on ${count} of your leads, clean that up today "
            "please.",
            "Competitor mentioned in the reply from ${company}, keep it honest and "
            "factual.",
        ),
        assistant_templates=(
            "Enriched: ${count} headcount, ${funding}, ${warehouse} confirmed via "
            "the careers page. Above the bar. Sending two Thursday slots, CET, AE "
            "named in the invite.",
            "Playbook says stop the sequence on any reply. Logged ${lead} as "
            "nurture with a ${date} follow-up, and thanked them briefly.",
            "Below the qualification bar at ${count} employees and no warehouse, "
            "so I sent the self-serve starter link and marked nurture.",
            "Unsubscribe honored for ${lead}, sequence halted, CRM updated with "
            "the timestamp. No further contact.",
            "I do not know the dbt answer, so per playbook I said exactly that and "
            "routed the thread to the AE with context.",
            "All ${count} leads now have a dated next step. Three moved to nurture, "
            "one is meeting-booked for Tuesday.",
            "Triage done: ${count} above the bar and queued for sequences, ${count2} "
            "sent the starter link. Notes are in the CRM.",
            "Reply kept factual: acknowledged the competitor does batch exports "
            "well, pointed at our warehouse-native difference, no roadmap claims.",
        ),
        tools=(
            ("enrich_lead", _sdr_tool_a, _res_enrich_lead),
            ("crm_update", _sdr_tool_b, _res_crm_update),
            ("send_sequence", _sdr_tool_c, _res_send_sequence),
        ),
        calls_lo=2,
        calls_hi=8,
    ),
)


@dataclass
class SynthStats:
    """Result of one generate() run. All byte counts are exact file bytes."""

    target_bytes: int
    bytes_traces: int
    bytes_embeddings: int
    bytes_total: int
    events: int
    malformed: int
    n_traces: int
    files: int
    n_vectors: int
    dim: int
    seed: int
    n_lanes: int
    elapsed_s: float

    def to_dict(self) -> dict:
        """Plain-type dict, safe for json.dumps."""
        return asdict(self)


class _PersonaPools:
    """Pre-encoded JSON fragments and compiled templates for one persona.

    Line tails start at the comma after the "ts" value and run to the
    closing brace, so a full line is head + str(ts) + tail. Tails are
    built with json.dumps using ensure_ascii=False and canonical
    separators, which guarantees that json.loads followed by a canonical
    json.dumps reproduces the line byte-for-byte: that keeps the
    downstream residual path quiet for 99.5 percent of lines.
    """

    def __init__(self, persona: Persona) -> None:
        model = _json_str(persona.model)
        self.persona = persona
        self.sys_pre = ',"role":"system","type":"message","content":'
        self.sys_post = ',"model":' + model + "}"
        self.user_pre = ',"role":"user","type":"message","content":'
        self.asst_pre = ',"role":"assistant","type":"message","content":'
        self.asst_mid = ',"model":' + model + ',"tokens_in":'
        self.user_c = tuple(_compile(t) for t in persona.user_templates)
        self.asst_c = tuple(_compile(t) for t in persona.assistant_templates)
        self.context_c = tuple(_compile(t) for t in persona.context_templates)
        self.snippet_c = tuple(_compile(t) for t in persona.snippet_templates)
        self.tool_names = tuple(name for name, _, _ in persona.tools)
        self.args_fns = tuple(fn for _, fn, _ in persona.tools)
        self.result_fns = tuple(fn for _, _, fn in persona.tools)
        self.call_pre = tuple(
            ',"role":"assistant","type":"tool_call","tool_name":'
            + _json_str(name)
            + ',"content":'
            for name in self.tool_names
        )
        self.call_post = ',"model":' + model + "}"
        self.result_pre = tuple(
            ',"role":"tool","type":"tool_result","tool_name":' + _json_str(name) + ',"content":'
            for name in self.tool_names
        )


def _session_context(pools: _PersonaPools, j: int) -> str:
    """The injected per-conversation context block, roughly 200 to 1200 B.

    Real platforms compose the system prompt per session: user profile,
    account state, session variables, and retrieved notes are stitched
    under the persona base. The block is unique per conversation (wide
    seeded slot spaces) but resent verbatim on every call within it.
    """
    parts = ["\n\nSession context (injected per conversation):"]
    for i, tmpl in enumerate(pools.context_c):
        parts.append(_fill(tmpl, _mix(j, 200 + i)))
    n_snips = _mix(j, 250) % 5
    n_avail = len(pools.snippet_c)
    for i in range(n_snips):
        h = _mix(j, 260 + i)
        parts.append("- " + _fill(pools.snippet_c[h % n_avail], h))
    return "\n".join(parts)


def _build_log_text(rng: np.random.Generator) -> str:
    """A deterministic pseudo build log used to slice large tool outputs.

    Returns roughly 200 KB of newline-separated log lines so that a 45 to
    55 K character slice at a random offset is always available.
    """
    n = 4000
    ms = rng.integers(2, 9000, size=n)
    rc = rng.integers(0, 3, size=n)
    verbs = ("compile", "link", "test", "upload", "verify", "cache-hit", "retry")
    parts = [
        f"[step {i:05d}] {verbs[i % len(verbs)]} module_{i % 97} in {ms[i]} ms rc={rc[i]}"
        for i in range(n)
    ]
    return "\n".join(parts)


def _respace(line: bytes) -> bytes:
    """Re-serialize one event line with non-canonical separators.

    The parsed object is unchanged, only inter-token spacing differs, so
    downstream canonical re-serialization will not match the raw bytes and
    the trace engine must take its residual path.
    """
    obj = json.loads(line)
    return json.dumps(obj, separators=(", ", ": "), ensure_ascii=False).encode("utf-8")


def _make_malformed(rng: np.random.Generator, prev: bytes, u: float) -> bytes:
    """One malformed line: truncated JSON, bare text, or binary garbage.

    u is a uniform value in [0, 1) reused both to pick the kind and to
    derive cheap jitter, keeping the RNG call count low. The returned
    bytes never contain a newline and never parse as a JSON object.
    """
    h = int(u * 1e9)
    if u < 0.40:
        # A strict prefix of a JSON object can never be valid JSON: the
        # top-level brace only closes at the final byte.
        span = min(len(prev) - 9, 380)
        return prev[: 8 + h % span]
    if u < 0.70:
        return _BARE_TEXT[h % len(_BARE_TEXT)].encode("utf-8")
    nb = 24 + h % 180
    raw = rng.integers(0, 256, size=nb, dtype=np.uint8).tobytes()
    # The 0xFF prefix guarantees a UnicodeDecodeError; newlines would split
    # the record so they are scrubbed.
    return b"\xff\xfe" + raw.replace(b"\n", b".")


# A lane stream item: (ts, lane, seq, line_bytes, is_event, trace_id).
# heapq.merge orders by tuple comparison; (ts, lane, seq) is unique per
# item, so the payload fields are never compared.
_StreamItem = tuple[int, int, int, bytes, bool, str]


def _lane_stream(
    lane: int,
    rng: np.random.Generator,
    pools_list: list[_PersonaPools],
    epoch_us: int,
    log_text: str,
) -> Iterator[_StreamItem]:
    """One concurrent session lane: an endless, time-monotonic event stream.

    The lane runs conversations back to back with idle gaps between
    sessions, like one slot of a platform's concurrent load. Yields
    _StreamItem tuples with strictly increasing (ts, seq) so heapq.merge
    can interleave thousands of lanes into one global time-ordered log.
    The generator is infinite: the consumer stops at its byte budget,
    which cuts the capture window exactly like a real archive slice.
    """
    log_span = len(log_text) - 60_000
    seq = 0
    # Stagger session arrival across one mean cycle so the simulated
    # platform reaches steady state instead of a thundering herd at epoch.
    ts = epoch_us + int(rng.random() * _LANE_STAGGER_US)
    conv_no = 0
    n_personas = len(pools_list)
    while True:
        pools = pools_list[int(rng.integers(0, n_personas))]
        persona = pools.persona
        trace_id = f"tr-{persona.key}-{lane:05d}-{conv_no:05d}"
        head = '{"trace_id":"' + trace_id + '","ts":'
        # The composed prompt is JSON-encoded once and resent as exact
        # bytes on every call of this conversation.
        prompt = persona.system_prompt + _session_context(pools, int(rng.integers(0, 1 << 62)))
        sys_tail = pools.sys_pre + _json_str(prompt) + pools.sys_post
        n_calls = int(rng.integers(persona.calls_lo, persona.calls_hi + 1))
        # Idempotent re-read cache: tool index -> (args_tail, result_tail).
        # Re-running the same tool with the same args returns the same
        # bytes (file re-read, unchanged record re-fetch), which is real
        # within-conversation redundancy separated by minutes of stream.
        cache: dict[int, tuple[str, str]] = {}
        for c in range(n_calls):
            if c > 0:
                ts += 1 + int(rng.exponential() * _GAP_INTER_CALL)
                if rng.random() < _P_LONG_THINK:
                    ts += int(rng.exponential() * _GAP_LONG_THINK)
            else:
                ts += 1
            j = int(rng.integers(0, 1 << 62))
            lines: list[tuple[int, str]] = [(ts, sys_tail)]
            ts += 1 + int(rng.exponential() * _GAP_IN_CALL)
            uh = _mix(j, 11)
            lines.append(
                (ts, pools.user_pre + _json_str(_fill(pools.user_c[uh % len(pools.user_c)], j)) + "}")
            )
            if rng.random() < _P_TOOL:
                t_idx = int(_mix(j, 12) % len(pools.tool_names))
                cached = cache.get(t_idx)
                storm = rng.random() < _P_STORM
                if cached is not None and not storm and rng.random() < _P_REREAD:
                    args_tail, result_tail = cached
                    ts += 1 + int(rng.exponential() * _GAP_IN_CALL)
                    lines.append((ts, args_tail))
                    ts += 1 + int(rng.exponential() * _GAP_TOOL_EXEC)
                    lines.append((ts, result_tail))
                else:
                    base_a = int(_mix(j, 13) % 100_000_000)
                    base_b = int(_mix(j, 14) % 9) + 1
                    retries = int(_mix(j, 15) % 5) + 2 if storm else 1
                    args_fn = pools.args_fns[t_idx]
                    pre = pools.call_pre[t_idx]
                    args_tail = ""
                    for r in range(retries):
                        ts += 1 + int(rng.exponential() * (_GAP_RETRY if r else _GAP_IN_CALL))
                        args = args_fn(base_a + r, base_b, r + 1)
                        args_tail = pre + _json_str(args) + pools.call_post
                        lines.append((ts, args_tail))
                    ts += 1 + int(rng.exponential() * _GAP_TOOL_EXEC)
                    if rng.random() < _P_LARGE:
                        off = _mix(j, 16) % log_span
                        length = 45_000 + _mix(j, 17) % 10_000
                        body = (
                            "tool output chunk ref "
                            + str(_mix(j, 18) % 10**12)
                            + "\n"
                            + log_text[off : off + length]
                        )
                    else:
                        body = pools.result_fns[t_idx](j)
                    result_tail = pools.result_pre[t_idx] + _json_str(body) + "}"
                    lines.append((ts, result_tail))
                    cache[t_idx] = (args_tail, result_tail)
            ts += 1 + int(rng.exponential() * _GAP_IN_CALL)
            ah = _mix(j, 19)
            tin = 900 + _mix(j, 20) % 2600
            tout = 40 + _mix(j, 21) % 700
            lines.append(
                (
                    ts,
                    pools.asst_pre
                    + _json_str(_fill(pools.asst_c[ah % len(pools.asst_c)], j))
                    + pools.asst_mid
                    + str(tin)
                    + ',"tokens_out":'
                    + str(tout)
                    + "}",
                )
            )
            # Post-pass per call: rare respacing plus malformed injection.
            # One vectorized draw covers the whole call, so the per-line
            # cost is an array lookup, and the draw count depends only on
            # the line count, which is itself a deterministic function of
            # earlier draws.
            r2 = rng.random((len(lines), 2))
            for i, (t_i, tail) in enumerate(lines):
                lb = (head + str(t_i) + tail).encode("utf-8")
                if r2[i, 0] < _P_RESPACED:
                    lb = _respace(lb)
                yield (t_i, lane, seq, lb, True, trace_id)
                seq += 1
                if r2[i, 1] < _P_MALFORMED:
                    mb = _make_malformed(rng, lb, r2[i, 1] / _P_MALFORMED)
                    yield (t_i, lane, seq, mb, False, trace_id)
                    seq += 1
        ts += 1 + int(rng.exponential() * _GAP_SESSION_IDLE)
        conv_no += 1


class _ShardWriter:
    """Buffered JSONL shard writer with a hard per-shard byte cap."""

    def __init__(self, traces_dir: Path, cap: int = _SHARD_CAP) -> None:
        self.dir = traces_dir
        self.cap = cap
        self.total = 0
        self.files = 0
        self._shard_bytes = 0
        self._buf = bytearray()
        self._fh = None

    def _open_next(self) -> None:
        self._flush()
        if self._fh is not None:
            self._fh.close()
        path = self.dir / f"part-{self.files:04d}.jsonl"
        self._fh = open(path, "wb")
        self.files += 1
        self._shard_bytes = 0

    def write(self, line: bytes) -> None:
        needed = len(line) + 1
        if self._fh is None or self._shard_bytes + needed > self.cap:
            self._open_next()
        self._buf += line
        self._buf += b"\n"
        self._shard_bytes += needed
        self.total += needed
        if len(self._buf) >= _FLUSH_BYTES:
            self._flush()

    def _flush(self) -> None:
        if self._buf and self._fh is not None:
            self._fh.write(self._buf)
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def sample_embeddings(n: int, dim: int, seed: int = 1337) -> np.ndarray:
    """Sample n synthetic text-embedding vectors, float32 [n, dim], unit norm.

    This is the single source of truth for the synth embedding
    distribution: generate() writes exactly this geometry (from a spawned
    seed) and gate measurements import this function directly.

    Geometry, modeled on real text-embedding corpora:
    - 200 clusters with Zipf-ish weights (topics are skewed).
    - Each cluster is a LOW-RANK patch: points spread along 12 to 24
      principal directions (real embedding manifolds have intrinsic
      dimension around 10 to 40, far below the ambient dim), with a
      per-cluster scale tau in [0.45, 0.65] and a 15 percent isotropic
      residual for off-manifold noise. This gives neighbors real
      distance contrast instead of the near-tie concentration a
      full-rank isotropic mixture produces.
    - 35 percent of points are near-copies of a base point (jitter 0.08
      before normalization) in groups of 2 to 6, mirroring the trace
      side of the corpus: retry storms and template repeats make many
      stored texts trivial variants of one another.

    Deterministic per (n, dim, seed): numpy.random.default_rng(seed),
    bit-identical across runs on one machine. Raises ValueError on
    non-positive n or dim.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    out = np.empty((n, dim), dtype=np.float32)
    _sample_embeddings_into(out, np.random.default_rng(seed))
    return out


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _sample_embeddings_into(out: np.ndarray, rng: np.random.Generator) -> None:
    """Fill a float32 [n, dim] array (or memmap) with the mixture.

    All rng draws happen in a fixed order independent of chunking, so the
    output is bit-identical whether out is in memory or a memmap.
    """
    n, dim = out.shape

    centers = rng.standard_normal((_N_CLUSTERS, dim), dtype=np.float32)
    centers = _normalize_rows(centers)
    # Per-cluster intrinsic rank and scale: topics differ in how varied
    # they are, so both are cluster-specific, not global constants.
    ranks = np.minimum(rng.integers(_RANK_LO, _RANK_HI + 1, size=_N_CLUSTERS), dim)
    taus = rng.uniform(_TAU_LO, _TAU_HI, size=_N_CLUSTERS).astype(np.float32)
    r_max = int(ranks.max())
    # Orthonormal in-manifold bases via QR: columns beyond a cluster's
    # rank stay zero so one padded tensor serves every cluster.
    basis = np.zeros((_N_CLUSTERS, dim, r_max), dtype=np.float32)
    for k in range(_N_CLUSTERS):
        rk = int(ranks[k])
        g = rng.standard_normal((dim, rk))
        q, _ = np.linalg.qr(g)
        basis[k, :, :rk] = q.astype(np.float32)

    ranks_zipf = np.arange(1, _N_CLUSTERS + 1, dtype=np.float64)
    weights = ranks_zipf ** (-_ZIPF_EXPONENT)
    weights /= weights.sum()

    n_dup = int(round(_DUP_FRACTION * n)) if n >= 8 else 0
    n_base = n - n_dup
    assign = rng.choice(_N_CLUSTERS, size=n_base, p=weights)

    low_coef = np.float32(np.sqrt(1.0 - _ISO_FRACTION))
    iso_coef = np.float32(np.sqrt(_ISO_FRACTION / dim))
    for start in range(0, n_base, _EMB_CHUNK):
        stop = min(start + _EMB_CHUNK, n_base)
        m = stop - start
        a = assign[start:stop]
        # Draw the chunk's randomness up front (fixed order), then apply
        # the per-cluster linear maps, which draw nothing.
        t = rng.standard_normal((m, r_max), dtype=np.float32)
        eps = rng.standard_normal((m, dim), dtype=np.float32)
        x = centers[a] + (taus[a][:, None] * iso_coef) * eps
        for k in np.unique(a):
            rows = np.nonzero(a == k)[0]
            rk = int(ranks[k])
            scale = np.float32(taus[k] * low_coef / np.sqrt(rk))
            x[rows] += scale * (t[rows, :rk] @ basis[k, :, :rk].T)
        out[start:stop] = _normalize_rows(x)

    if n_dup:
        # Duplicate groups of 2 to 6 (one base plus 1 to 5 copies) echo
        # retry storms and template repeats. Copies sit at cosine about
        # 1 - jitter^2 / 2 (~0.997) from their base, far above ordinary
        # neighbor similarity, so every such query has unambiguous
        # nearest neighbors, exactly like real near-duplicate text.
        group_bases = rng.integers(0, n_base, size=n_dup)
        group_copies = rng.integers(1, 6, size=n_dup)
        base_idx = np.repeat(group_bases, group_copies)[:n_dup]
        jitter = np.float32(_DUP_JITTER / np.sqrt(dim))
        for start in range(0, n_dup, _EMB_CHUNK):
            stop = min(start + _EMB_CHUNK, n_dup)
            m = stop - start
            src = np.asarray(out[base_idx[start:stop]], dtype=np.float32)
            y = src + jitter * rng.standard_normal((m, dim), dtype=np.float32)
            out[n_base + start : n_base + stop] = _normalize_rows(y)


def _write_embeddings(
    emb_dir: Path, n: int, dim: int, rng: np.random.Generator
) -> int:
    """Write vectors.npy (float32 [n, dim], L2-normalized) and ids.npy.

    The vectors are the sample_embeddings mixture (see its docstring),
    streamed into a memmap so peak memory stays one chunk. Returns total
    bytes written.
    """
    vec_path = emb_dir / "vectors.npy"
    out = np.lib.format.open_memmap(
        vec_path, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    _sample_embeddings_into(out, rng)
    out.flush()
    del out

    ids_path = emb_dir / "ids.npy"
    np.save(ids_path, np.arange(n, dtype=np.int64))
    return vec_path.stat().st_size + ids_path.stat().st_size


def generate(
    gb: float,
    out_dir: str | Path,
    seed: int = 1337,
    dim: int = 768,
    n_vectors: int | None = None,
) -> SynthStats:
    """Generate a synthetic agent corpus of roughly gb gigabytes.

    Arguments: gb is the total byte target in units of 1e9 bytes, dim is
    the embedding dimensionality, n_vectors defaults to
    max(5000, int(gb * 100_000)). Output lands under out_dir as
    traces/part-NNNN.jsonl shards plus embeddings/vectors.npy (float32
    [n_vectors, dim]) and embeddings/ids.npy (int64 [n_vectors]).

    The trace shards are one global time-ordered stream over many
    concurrent conversations (see the module docstring): a streaming
    k-way merge whose memory is O(concurrent lanes), never O(corpus).

    Deterministic per (gb, seed, dim, n_vectors): two runs with the same
    arguments produce byte-identical files. Total bytes land within 5
    percent of the target (embeddings are sized first, traces fill the
    remainder to within a few hundred bytes). Raises ValueError on
    non-positive gb, dim, or n_vectors, and when the embeddings would
    leave less than 64 KB of the byte target for traces.
    """
    if gb <= 0:
        raise ValueError(f"gb must be positive, got {gb}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if n_vectors is not None and n_vectors <= 0:
        raise ValueError(f"n_vectors must be positive, got {n_vectors}")

    t0 = time.perf_counter()
    target_bytes = int(gb * 1e9)
    if n_vectors is None:
        n_vectors = max(5000, int(gb * 100_000))

    # Embeddings are sized by (n_vectors, dim), so budget them first and
    # let the trace stream fill the remaining bytes precisely.
    emb_floor = n_vectors * dim * 4 + n_vectors * 8
    if target_bytes - emb_floor < 65536:
        raise ValueError(
            f"embeddings alone need about {emb_floor} bytes, which does not "
            f"leave room for traces inside the {target_bytes} byte target: "
            "lower dim or n_vectors, or raise gb"
        )

    root = Path(out_dir)
    traces_dir = root / "traces"
    emb_dir = root / "embeddings"
    traces_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    # Stale shards from a previous, larger run would corrupt the corpus.
    for old in traces_dir.glob("part-*.jsonl"):
        old.unlink()

    ss = np.random.SeedSequence(seed)
    ss_emb, ss_traces = ss.spawn(2)
    rng_emb = np.random.default_rng(ss_emb)

    bytes_embeddings = _write_embeddings(emb_dir, n_vectors, dim, rng_emb)
    budget = target_bytes - bytes_embeddings

    # Concurrency scales with the trace budget: a corpus twice the size
    # models a platform with twice the concurrent sessions over the same
    # capture window, which is how real archives grow.
    n_lanes = max(8, int(budget // _LANE_TARGET_BYTES))
    children = ss_traces.spawn(n_lanes + 1)
    rng_master = np.random.default_rng(children[0])
    log_text = _build_log_text(rng_master)
    epoch = int(rng_master.integers(_TS_LO, _TS_HI - _EPOCH_MARGIN_US))

    pools_list = [_PersonaPools(p) for p in PERSONAS]
    lanes = [
        _lane_stream(i, np.random.default_rng(children[i + 1]), pools_list, epoch, log_text)
        for i in range(n_lanes)
    ]

    writer = _ShardWriter(traces_dir)
    events = 0
    malformed = 0
    traces_seen: set[str] = set()
    consecutive_skips = 0
    # heapq.merge holds exactly one pending item per lane, so this loop
    # streams the global log without ever buffering the corpus.
    for _ts, _lane, _seq, lb, is_event, trace_id in heapq.merge(*lanes):
        needed = len(lb) + 1
        if writer.total + needed > budget:
            # Oversized tail lines are skipped, not trimmed: the stream
            # keeps flowing until a small line closes the gap, so the
            # corpus lands just under the byte budget with every written
            # line intact.
            if budget - writer.total < 512 or consecutive_skips >= 100_000:
                break
            consecutive_skips += 1
            continue
        consecutive_skips = 0
        writer.write(lb)
        if is_event:
            events += 1
            traces_seen.add(trace_id)
        else:
            malformed += 1
    writer.close()

    return SynthStats(
        target_bytes=int(target_bytes),
        bytes_traces=int(writer.total),
        bytes_embeddings=int(bytes_embeddings),
        bytes_total=int(writer.total + bytes_embeddings),
        events=int(events),
        malformed=int(malformed),
        n_traces=len(traces_seen),
        files=int(writer.files),
        n_vectors=int(n_vectors),
        dim=int(dim),
        seed=int(seed),
        n_lanes=int(n_lanes),
        elapsed_s=float(time.perf_counter() - t0),
    )
