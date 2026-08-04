"""Deterministic synthetic agent corpus generator.

Writes a realistic agent-trace plus embedding corpus:

    out_dir/traces/part-0000.jsonl, part-0001.jsonl, ...   (UTF-8 JSONL)
    out_dir/embeddings/vectors.npy                          (float32 [n, dim])
    out_dir/embeddings/ids.npy                              (int64 [n])

Realism targets, all seeded and byte-reproducible per (gb, seed, dim):
three personas whose 1.5 to 4 KB system prompts are resent on every model
call, tool calls with jittered JSON args, retry storms, Poisson-bursty
timestamps in integer microseconds, about 2 percent malformed lines, about
0.5 percent lines with non-canonical JSON spacing, unicode and emoji
content, and occasional roughly 50 KB tool outputs. Embeddings are a
200-cluster mixture of Gaussians on the unit sphere with Zipf-ish cluster
weights.

Design notes on speed: every event line is assembled from pre-encoded JSON
fragments (system prompt, pooled user and assistant messages, tool result
bodies), so the hot loop is string concatenation plus one UTF-8 encode,
not one json.dumps per line. Only the rare paths (jittered tool args,
large outputs, respaced lines) pay for a real json call. This keeps the
generator far above 10 MB/s of JSONL on a laptop CPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["Persona", "PERSONAS", "SynthStats", "generate"]

# Timestamp window: integer microseconds since epoch, mid 2024 to mid 2026.
# Conversations last minutes, so every trace stays far inside the window.
_TS_LO = 1_720_000_000_000_000
_TS_HI = 1_780_000_000_000_000

# Mean inter-event gaps in microseconds. Exponential inter-arrival times
# are exactly a Poisson process: small in-call gaps form the bursts, large
# inter-call and think gaps separate them, which is what gives the corpus
# its heavy-tailed gap distribution (max gap >> median gap).
_GAP_IN_CALL = 12_000
_GAP_TOOL_EXEC = 900_000
_GAP_RETRY = 1_800_000
_GAP_INTER_CALL = 20_000_000
_GAP_LONG_THINK = 240_000_000

_P_TOOL = 0.55        # a call uses a tool instead of answering directly
_P_STORM = 0.10       # a tool call turns into a retry storm of 2 to 6
_P_LONG_THINK = 0.08  # an extra minutes-long pause before a call
_P_LARGE = 0.02       # a tool result is a roughly 50 KB dump
_P_MALFORMED = 0.02   # a malformed line is injected after an event line
_P_RESPACED = 0.005   # an event line is serialized with extra spaces

_N_CLUSTERS = 200
_ZIPF_EXPONENT = 1.05
# Total noise norm around each unit cluster center. Same-cluster cosine is
# about 1 / (1 + tau^2), so 0.28 keeps neighbors near 0.92 similarity
# while random cross-cluster pairs stay near zero.
_NOISE_TAU = 0.28

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


@dataclass(frozen=True)
class Persona:
    """One synthetic agent persona.

    system_prompt is 1500 to 4000 UTF-8 bytes and is resent verbatim on
    every model call, which is the redundancy that trace dedup downstream
    is supposed to find. tools maps tool_name to a callable
    (a, b, attempt) -> args JSON string, so retries can jitter args.
    """

    key: str
    title: str
    model: str
    system_prompt: str
    tools: tuple[tuple[str, Callable[[int, int, int], str]], ...]
    user_pool: tuple[str, ...]
    assistant_pool: tuple[str, ...]
    result_pool: tuple[str, ...]


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
        tools=(
            ("lookup_customer", _support_tool_a),
            ("search_kb", _support_tool_b),
            ("create_ticket", _support_tool_c),
        ),
        user_pool=(
            "Hi, I was charged twice this month and I want one of them back.",
            "My export has been stuck at 90% since yesterday, can you check?",
            "I can't log in after changing my email, the reset link 404s.",
            "Please cancel my annual plan and refund the unused months.",
            "Why did my invoice jump from 12.00 USD to 31.50 USD?",
            "The team admin left the company and nobody can manage seats now.",
            "Je n'arrive pas à télécharger ma facture de mars, ça bloque.",
            "請問可以把發票抬頭改成公司名稱嗎?謝謝!",
            "honestly this is the third time support ghosted me 😤 fix it",
            "love the product 🚀 but the overage charge surprised me, help?",
            "I pasted my API key in a shared note by mistake, what do I do?",
            "Your status page says all green but nothing loads for me.",
            "I need a GDPR export of everything you hold about my account.",
            "Charged in EUR but my card is in GBP, the fx fee seems wrong.",
            "Can you merge my personal workspace into the company team plan?",
            "The mobile app deleted my offline edits after sync, recover them?",
        ),
        assistant_pool=(
            "I understand, and I can fix this. Could you confirm the email "
            "on the account so I can pull it up?",
            "Thanks for confirming. I see the duplicate charge on my side, "
            "and per policy I can refund it since it is within 14 days.",
            "I have created ticket 48231 with priority p2. You will get an "
            "email update within one business day.",
            "The export job was stuck on our side. I restarted it and it "
            "should complete within 30 minutes. I will stay on the chat.",
            "That key should be treated as compromised. Please rotate it in "
            "Settings, then I will scrub it from the shared note history.",
            "Summary: refund of 12.00 USD issued to the card ending 4417, "
            "confirmation email sent, no further action needed from you.",
            "I checked the knowledge base and annual plans refund within 30 "
            "days, so you are covered. Processing it now.",
            "I cannot see other accounts' data, but I can escalate this to "
            "the team that can, with your permission.",
        ),
        result_pool=(
            '{"status": "ok", "plan": "pro_annual", "invoices": 3, '
            '"last_payment": "2026-05-14", "region": "eu-west"}',
            '{"status": "ok", "articles": [{"id": "kb-1182", "title": '
            '"Refund windows by plan"}, {"id": "kb-0409", "title": '
            '"Overage credits"}]}',
            '{"status": "ok", "ticket_id": 48231, "priority": "p2", '
            '"queue": "billing"}',
            '{"status": "error", "code": "upstream_timeout", "retry_in_ms": '
            "2000}",
            '{"status": "ok", "matches": 0, "fallback": '
            '"manual_review_queue"}',
        ),
    ),
    Persona(
        key="coder",
        title="autonomous coding agent",
        model="sable-mini-1",
        system_prompt=_CODER_PROMPT,
        tools=(
            ("read_file", _coder_tool_a),
            ("run_tests", _coder_tool_b),
            ("apply_patch", _coder_tool_c),
        ),
        user_pool=(
            "The nightly batch job double-writes rows when a retry lands "
            "after a partial commit. Fix it and prove it with a test.",
            "Please add a timeout to the scheduler heartbeat, 30s default, "
            "configurable via env.",
            "Tests in tests/unit/test_batch_7.py are flaky on CI, look into "
            "whether it is ordering or a real race.",
            "Migrate worker_3 from the legacy queue client, behavior must "
            "stay identical.",
            "Profile the ingest path, we regressed 40% since last release.",
            "Add structured logging to the retry loop, include attempt "
            "count and backoff milliseconds.",
            "El linter falla en src/core/scheduler.py después del merge, "
            "¿puedes arreglarlo?",
            "make the failing assertion message actually useful 🙏 it just "
            "says False is not True right now",
            "Why does apply_patch keep rejecting hunk 3? The file looks "
            "unchanged to me.",
            "Ship a fix for the off-by-one in pagination, page 2 repeats "
            "the last item of page 1.",
            "The config loader silently ignores unknown keys, that hid a "
            "typo in prod. Make it warn.",
            "Bump test coverage for the error paths in the uploader, we "
            "only cover the happy path.",
        ),
        assistant_pool=(
            "Plan: reproduce with a narrow test, read the two files "
            "involved, patch the smallest surface, then run the package "
            "suite. Starting with read_file.",
            "The root cause is a retry that re-enters commit() after the "
            "transaction already applied. The idempotency key is computed "
            "too late. Patch coming.",
            "Tests are green: 214 passed, 0 failed, 2 skipped in 41.3s. "
            "The new regression test fails on the old code as expected.",
            "The hunk failed because the file drifted upstream. Re-reading "
            "and rebasing the patch on the current contents.",
            "Measured before: 1.84s per 10k rows. After: 1.11s. Command and "
            "variance are in the change description.",
            "Found a hardcoded token in the fixture file. Not using it, "
            "flagging it per policy, and continuing with a fake.",
            "Quarantined the flaky test with a link to issue 5521, three "
            "identical failures in a row, per procedure.",
            "Done. Files changed: 3. Evidence: unit suite green, new test "
            "covers the partial-commit retry. Out of scope: legacy queue "
            "cleanup, noted in the task.",
        ),
        result_pool=(
            '{"exit_code": 0, "passed": 214, "failed": 0, "skipped": 2, '
            '"duration_s": 41.3}',
            '{"exit_code": 1, "passed": 213, "failed": 1, "first_failure": '
            '"tests/unit/test_batch_7.py::test_retry_partial_commit"}',
            '{"ok": true, "applied_hunks": 2, "rejected_hunks": 0}',
            '{"ok": false, "error": "hunk 3 failed to apply, context '
            'mismatch at line 118"}',
            '{"path": "src/services/worker_3.py", "lines": 412, "sha": '
            '"9f31c2", "truncated": false}',
        ),
    ),
    Persona(
        key="sdr",
        title="sales development representative",
        model="kestrel-8b",
        system_prompt=_SDR_PROMPT,
        tools=(
            ("enrich_lead", _sdr_tool_a),
            ("crm_update", _sdr_tool_b),
            ("send_sequence", _sdr_tool_c),
        ),
        user_pool=(
            "New inbound from the pricing page: data lead at a 140 person "
            "fintech, warehouse is Snowflake. Qualify and book.",
            "This lead replied 'not now, maybe Q3'. What is the right next "
            "step per playbook?",
            "The AE wants two slots offered for Thursday, prospect is in "
            "Europe, CET.",
            "Lead unsubscribed mid-sequence, make sure everything stops.",
            "We got a batch of 30 webinar signups, triage them against the "
            "qualification bar.",
            "Prospect asked whether we support dbt incremental models, I "
            "have no idea, handle it.",
            "México city based startup, 25 personas, ¿califica para el "
            "self-serve o para un AE?",
            "the CEO forwarded our email with just '???' 😅 what do we "
            "reply?",
            "CRM shows no next step on 12 of your leads, clean that up "
            "today please.",
            "Competitor mentioned in the reply, keep it honest and factual.",
        ),
        assistant_pool=(
            "Enriched: 140 headcount, Series B, Snowflake confirmed via the "
            "careers page. Above the bar. Sending two Thursday slots, CET, "
            "AE named in the invite.",
            "Playbook says stop the sequence on any reply. Logged as "
            "nurture with a Q3 follow-up date, and thanked them briefly.",
            "Below the qualification bar at 12 employees and no warehouse, "
            "so I sent the self-serve starter link and marked nurture.",
            "Unsubscribe honored, sequence halted, CRM updated with the "
            "timestamp. No further contact.",
            "I do not know the dbt answer, so per playbook I said exactly "
            "that and routed the thread to the AE with context.",
            "All 12 leads now have a dated next step. Three moved to "
            "nurture, one is meeting-booked for Tuesday.",
            "Triage done: 9 above the bar and queued for sequences, 21 sent "
            "the starter link. Notes are in the CRM.",
            "Reply kept factual: acknowledged the competitor does batch "
            "exports well, pointed at our warehouse-native difference, no "
            "roadmap claims.",
        ),
        result_pool=(
            '{"ok": true, "headcount": 140, "funding": "series_b", '
            '"warehouse": "snowflake", "confidence": 0.87}',
            '{"ok": true, "lead_id": "lead_0048112", "stage": "nurture", '
            '"next_step": "2026-09-02"}',
            '{"ok": true, "sequence": "warm_intro_v3", "touches": 3, '
            '"started": true}',
            '{"ok": false, "error": "rate_limited", "retry_in_s": 30}',
            '{"ok": true, "unsubscribed": true, "halted_sequences": 1}',
        ),
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
    elapsed_s: float

    def to_dict(self) -> dict:
        """Plain-type dict, safe for json.dumps."""
        return asdict(self)


class _PersonaPools:
    """Pre-encoded JSON fragments for one persona.

    Every fragment starts at the comma after the "ts" value and runs to the
    closing brace (or to a splice point for variable tails), so a full line
    is head + str(ts) + fragment. Fragments are built with json.dumps using
    ensure_ascii=False and canonical separators, which guarantees that
    json.loads followed by a canonical json.dumps reproduces the line
    byte-for-byte: that is what keeps the residual path quiet for 99.5
    percent of lines.
    """

    def __init__(self, persona: Persona) -> None:
        enc = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
        model = enc(persona.model)
        self.persona = persona
        self.sys_frag = (
            ',"role":"system","type":"message","content":'
            + enc(persona.system_prompt)
            + ',"model":'
            + model
            + "}"
        )
        self.user_frags = tuple(
            ',"role":"user","type":"message","content":' + enc(u) + "}"
            for u in persona.user_pool
        )
        # Assistant fragments stop right before the token counts, which are
        # the only per-line variable part of an assistant message.
        self.asst_frags = tuple(
            ',"role":"assistant","type":"message","content":'
            + enc(a)
            + ',"model":'
            + model
            + ',"tokens_in":'
            for a in persona.assistant_pool
        )
        self.tool_names = tuple(name for name, _ in persona.tools)
        self.tool_fns = tuple(fn for _, fn in persona.tools)
        self.call_pre = tuple(
            ',"role":"assistant","type":"tool_call","tool_name":'
            + enc(name)
            + ',"content":'
            for name in self.tool_names
        )
        self.call_post = ',"model":' + model + "}"
        self.result_pre = tuple(
            ',"role":"tool","type":"tool_result","tool_name":' + enc(name) + ',"content":'
            for name in self.tool_names
        )
        # Full result fragments per (tool, pooled body): the common case
        # needs zero json work at emit time.
        self.result_frags = tuple(
            tuple(pre + enc(body) + "}" for body in persona.result_pool)
            for pre in self.result_pre
        )


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


def _conversation(
    rng: np.random.Generator,
    pools: _PersonaPools,
    conv_idx: int,
    log_text: str,
) -> list[tuple[bytes, bool]]:
    """Build one conversation as a list of (line_bytes, is_event) pairs.

    Timestamps are integer microseconds, strictly increasing within the
    conversation. Malformed lines carry is_event False.
    """
    n_calls = int(rng.integers(2, 9))
    u = rng.random((n_calls, 4))
    jit = rng.integers(0, 1 << 30, size=(n_calls, 8))
    gaps = rng.exponential(1.0, size=n_calls * 16 + 8)
    ts = int(rng.integers(_TS_LO, _TS_HI))

    head = '{"trace_id":"tr-' + pools.persona.key + "-" + format(conv_idx, "06d") + '","ts":'
    n_users = len(pools.user_frags)
    n_asst = len(pools.asst_frags)
    n_tools = len(pools.tool_names)
    n_results = len(pools.result_frags[0])
    log_span = len(log_text) - 60_000

    lines: list[str] = []
    g = 0
    for c in range(n_calls):
        if c > 0:
            ts += 1 + int(gaps[g] * _GAP_INTER_CALL)
            g += 1
            if u[c, 2] < _P_LONG_THINK:
                ts += int(gaps[g] * _GAP_LONG_THINK)
                g += 1
        else:
            ts += 1

        # The system prompt is resent on every call: this is the dominant,
        # deliberately redundant byte cost of the corpus.
        lines.append(head + str(ts) + pools.sys_frag)
        ts += 1 + int(gaps[g] * _GAP_IN_CALL)
        g += 1
        lines.append(head + str(ts) + pools.user_frags[jit[c, 0] % n_users])

        if u[c, 0] < _P_TOOL:
            t_idx = int(jit[c, 1]) % n_tools
            base_a = int(jit[c, 2])
            base_b = int(jit[c, 3]) % 9 + 1
            retries = 1
            if u[c, 1] < _P_STORM:
                retries = int(jit[c, 4]) % 5 + 2
            fn = pools.tool_fns[t_idx]
            pre = pools.call_pre[t_idx]
            for r in range(retries):
                ts += 1 + int(gaps[g] * (_GAP_RETRY if r else _GAP_IN_CALL))
                g += 1
                args = fn(base_a + r, base_b, r + 1)
                lines.append(
                    head
                    + str(ts)
                    + pre
                    + json.dumps(args, ensure_ascii=False)
                    + pools.call_post
                )
            ts += 1 + int(gaps[g] * _GAP_TOOL_EXEC)
            g += 1
            if u[c, 3] < _P_LARGE:
                off = int(jit[c, 5]) % log_span
                length = 45_000 + int(jit[c, 6]) % 10_000
                big = (
                    "tool output chunk ref "
                    + str(int(jit[c, 7]))
                    + "\n"
                    + log_text[off : off + length]
                )
                lines.append(
                    head
                    + str(ts)
                    + pools.result_pre[t_idx]
                    + json.dumps(big, ensure_ascii=False)
                    + "}"
                )
            else:
                lines.append(
                    head + str(ts) + pools.result_frags[t_idx][int(jit[c, 5]) % n_results]
                )
        ts += 1 + int(gaps[g] * _GAP_IN_CALL)
        g += 1
        tin = 900 + int(jit[c, 6]) % 2600
        tout = 40 + int(jit[c, 7]) % 700
        lines.append(
            head
            + str(ts)
            + pools.asst_frags[int(jit[c, 4]) % n_asst]
            + str(tin)
            + ',"tokens_out":'
            + str(tout)
            + "}"
        )

    # Post-pass: rare respacing plus malformed injection. Decisions come
    # from one vectorized draw so the per-line cost is an array lookup.
    r2 = rng.random((len(lines), 2))
    out: list[tuple[bytes, bool]] = []
    for i, text in enumerate(lines):
        lb = text.encode("utf-8")
        if r2[i, 0] < _P_RESPACED:
            lb = _respace(lb)
        out.append((lb, True))
        if r2[i, 1] < _P_MALFORMED:
            out.append((_make_malformed(rng, lb, r2[i, 1] / _P_MALFORMED), False))
    return out


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


def _write_embeddings(
    emb_dir: Path, n: int, dim: int, rng: np.random.Generator
) -> int:
    """Write vectors.npy (float32 [n, dim], L2-normalized) and ids.npy.

    Vectors are a 200-cluster mixture of Gaussians on the unit sphere with
    Zipf-ish cluster weights, so nearest neighbors are meaningful and the
    downstream recall measurements are not trivially satisfied by noise.
    Returns total bytes written.
    """
    centers = rng.standard_normal((_N_CLUSTERS, dim), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    ranks = np.arange(1, _N_CLUSTERS + 1, dtype=np.float64)
    weights = ranks ** (-_ZIPF_EXPONENT)
    weights /= weights.sum()
    assign = rng.choice(_N_CLUSTERS, size=n, p=weights)

    sigma = np.float32(_NOISE_TAU / np.sqrt(dim))
    vec_path = emb_dir / "vectors.npy"
    out = np.lib.format.open_memmap(
        vec_path, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    chunk = 65536
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x = centers[assign[start:stop]] + sigma * rng.standard_normal(
            (stop - start, dim), dtype=np.float32
        )
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out[start:stop] = x / norms
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

    Deterministic per (gb, seed, dim, n_vectors): two runs with the same
    arguments produce byte-identical files. Total bytes land within 5
    percent of the target (embeddings are sized first, traces fill the
    remainder to within a few kilobytes). Raises ValueError on
    non-positive gb, dim, or n_vectors, and when the embeddings alone
    would exceed the byte target.
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
    rng_tr = np.random.default_rng(ss_traces)

    bytes_embeddings = _write_embeddings(emb_dir, n_vectors, dim, rng_emb)
    budget = target_bytes - bytes_embeddings

    pools = [_PersonaPools(p) for p in PERSONAS]
    log_text = _build_log_text(rng_tr)

    writer = _ShardWriter(traces_dir)
    events = 0
    malformed = 0
    n_traces = 0
    conv_counter = 0
    stalls = 0
    while budget - writer.total >= 512:
        pp = pools[conv_counter % 3]
        conv = _conversation(rng_tr, pp, conv_counter // 3, log_text)
        conv_counter += 1
        wrote = 0
        has_event = False
        for lb, is_event in conv:
            needed = len(lb) + 1
            # Lines that no longer fit are skipped, not trimmed, so the
            # written corpus lands just under the byte budget while every
            # written line stays intact.
            if writer.total + needed > budget:
                continue
            writer.write(lb)
            wrote += needed
            if is_event:
                events += 1
                has_event = True
            else:
                malformed += 1
        if has_event:
            n_traces += 1
        if wrote == 0:
            stalls += 1
            if stalls >= 64:
                break
        else:
            stalls = 0
    writer.close()

    return SynthStats(
        target_bytes=int(target_bytes),
        bytes_traces=int(writer.total),
        bytes_embeddings=int(bytes_embeddings),
        bytes_total=int(writer.total + bytes_embeddings),
        events=int(events),
        malformed=int(malformed),
        n_traces=int(n_traces),
        files=int(writer.files),
        n_vectors=int(n_vectors),
        dim=int(dim),
        seed=int(seed),
        elapsed_s=float(time.perf_counter() - t0),
    )
