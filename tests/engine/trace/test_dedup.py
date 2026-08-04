"""Tests for density.engine.trace.dedup.

Covers: normalize idempotence and pattern stripping, minhash signature
determinism, near-duplicate clustering, exact sha256 grouping, cap
reporting, a 1000-text planted-family corpus, and degenerate inputs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from density.engine.trace.dedup import (
    BANDS,
    MINHASH_MIN_CHARS,
    NUM_PERM,
    ROWS,
    DedupResult,
    DedupStats,
    MinHashLSH,
    find_clusters,
    normalize,
)

SEED = 1337

BASE_SENTENCE = (
    "The deployment pipeline failed while provisioning the storage bucket "
    "because the retry budget was exhausted during the nightly sync job"
)


def _uuid_from(rng: np.random.Generator) -> str:
    hexdigits = "0123456789abcdef"
    s = "".join(hexdigits[int(i)] for i in rng.integers(0, 16, size=32))
    return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"


def _ts_from(rng: np.random.Generator) -> str:
    mm = int(rng.integers(1, 13))
    dd = int(rng.integers(1, 29))
    hh = int(rng.integers(0, 24))
    mi = int(rng.integers(0, 60))
    ss = int(rng.integers(0, 60))
    return f"2024-{mm:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{ss:02d}Z"


def _word(rng: np.random.Generator, lo: int = 4, hi: int = 10) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    n = int(rng.integers(lo, hi))
    return "".join(letters[int(i)] for i in rng.integers(0, 26, size=n))


def _build_family_corpus(
    seed: int = SEED, n_total: int = 1000, n_families: int = 40
) -> tuple[list[str], list[list[int]]]:
    """Deterministic corpus with planted near-duplicate families.

    Returns (texts, families) where families holds sorted member
    positions in the shuffled texts list. Family members differ only
    by uuid, timestamp, and casing, so they normalize identically.
    Fillers are random letter strings with negligible shingle overlap.
    """
    rng = np.random.default_rng(seed)
    flat: list[str] = []
    family_flat: list[list[int]] = []
    for _ in range(n_families):
        base = " ".join(_word(rng) for _ in range(20))
        size = int(rng.integers(3, 7))
        members: list[int] = []
        for j in range(size):
            u = _uuid_from(rng)
            ts = _ts_from(rng)
            # Alternate casing so members differ beyond the stripped ids.
            body = base.upper() if j % 2 else base
            members.append(len(flat))
            flat.append(f"request {u} {body} at {ts}")
        family_flat.append(members)
    n_fill = n_total - len(flat)
    assert n_fill > 0
    for _ in range(n_fill):
        flat.append(" ".join(_word(rng) for _ in range(15)))
    order = rng.permutation(len(flat))
    inv = np.empty(len(flat), dtype=np.int64)
    inv[order] = np.arange(len(flat))
    texts = [flat[int(j)] for j in order]
    families = [sorted(int(inv[j]) for j in fam) for fam in family_flat]
    return texts, families


# ---------------------------------------------------------------- normalize


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_normalize_idempotent_hypothesis(s: str) -> None:
    once = normalize(s)
    assert normalize(once) == once


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.one_of(
            st.text(alphabet="abcXYZ 09\t\n-:", max_size=12),
            st.just("550e8400-e29b-41d4-a716-446655440000"),
            st.just("2024-01-15T10:30:00Z"),
            st.just("1700000000000"),
        ),
        max_size=12,
    )
)
def test_normalize_idempotent_structured(parts: list[str]) -> None:
    s = " ".join(parts)
    once = normalize(s)
    assert normalize(once) == once


def test_normalize_strips_uuid() -> None:
    s = "Request 550E8400-E29B-41D4-A716-446655440000 failed"
    assert normalize(s) == "request failed"
    # A near-miss uuid (non-hex digit in the last group) is preserved.
    miss = "x 550e8400-e29b-41d4-a716-44665544000g y"
    assert normalize(miss) == miss


def test_normalize_strips_iso_timestamps() -> None:
    assert normalize("at 2024-01-15T10:30:00Z done") == "at done"
    assert normalize("at 2024-01-15 10:30:00.123456+00:00 done") == "at done"
    assert normalize("at 2024-01-15t10:30:00+0530 done") == "at done"
    # Date-only strings are not timestamps and survive.
    assert normalize("on 2024-01-15 we ship") == "on 2024-01-15 we ship"


def test_normalize_strips_epoch_integers() -> None:
    assert normalize("ts 1700000000 end") == "ts end"
    assert normalize("ms 1700000000123 end") == "ms end"
    assert normalize("us 1700000000000000 end") == "us end"
    # 9 and 17 digit runs are out of the epoch-like range.
    assert normalize("id 123456789 end") == "id 123456789 end"
    assert normalize("n 12345678901234567 end") == "n 12345678901234567 end"
    # Digits embedded in an identifier are not standalone integers.
    assert normalize("abc1234567890def") == "abc1234567890def"


def test_normalize_whitespace_and_case() -> None:
    assert normalize("  Foo\t\nBar   baz  ") == "foo bar baz"
    assert normalize("") == ""
    assert normalize(" \t \n ") == ""


def test_normalize_removal_adjacency_stays_idempotent() -> None:
    # Stripping the uuid must not fuse the digit runs into a new epoch.
    s = "12345 550e8400-e29b-41d4-a716-446655440000 67890"
    assert normalize(s) == "12345 67890"
    # A tab separator only becomes an ISO match after whitespace collapse.
    assert normalize("a 2024-01-15\t10:30:00 b") == "a b"
    assert normalize("2024-01-15\t10:30:00") == ""


# ---------------------------------------------------------------- MinHashLSH


def test_minhash_constants_match_contract() -> None:
    assert NUM_PERM == 128
    assert BANDS == 8
    assert ROWS == 16
    assert BANDS * ROWS == NUM_PERM


def test_minhash_signature_shape_and_determinism() -> None:
    text = BASE_SENTENCE
    sig_a = MinHashLSH(seed=SEED).signature(text)
    sig_b = MinHashLSH(seed=SEED).signature(text)
    assert sig_a.shape == (NUM_PERM,)
    assert sig_a.dtype == np.int64
    assert np.array_equal(sig_a, sig_b)
    sig_c = MinHashLSH(seed=SEED + 1).signature(text)
    assert not np.array_equal(sig_a, sig_c)


def test_minhash_signature_deterministic_across_processes() -> None:
    """Same seed must give bit-identical signatures in a fresh process.

    PYTHONHASHSEED is forced to a different value so any accidental use
    of the builtin salted hash would change the result.
    """
    local = MinHashLSH(seed=SEED).signature(BASE_SENTENCE)
    prog = (
        "from density.engine.trace.dedup import MinHashLSH\n"
        f"sig = MinHashLSH(seed={SEED}).signature({BASE_SENTENCE!r})\n"
        "print(','.join(str(int(v)) for v in sig))\n"
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "345"
    src = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", prog],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    remote = np.array([int(v) for v in out.stdout.strip().split(",")], dtype=np.int64)
    assert np.array_equal(local, remote)


def test_minhash_jaccard_estimates() -> None:
    lsh = MinHashLSH(seed=SEED)
    t1 = f"{BASE_SENTENCE} run 550e8400-e29b-41d4-a716-446655440000"
    t2 = f"{BASE_SENTENCE} RUN 123e4567-e89b-12d3-a456-426614174000"
    sim = MinHashLSH.estimated_jaccard(lsh.signature(t1), lsh.signature(t2))
    assert sim >= 0.85
    other = "completely unrelated content about gardening tips and tomato soil ph"
    dis = MinHashLSH.estimated_jaccard(lsh.signature(t1), lsh.signature(other))
    assert dis < 0.5


def test_minhash_empty_text_uses_sentinel() -> None:
    lsh = MinHashLSH(seed=SEED)
    sig = lsh.signature("")
    assert sig.shape == (NUM_PERM,)
    # Sentinel slots all agree, so two shingle-free texts estimate 1.0.
    assert MinHashLSH.estimated_jaccard(sig, lsh.signature(" \t ")) == 1.0


# ---------------------------------------------------------------- clustering


def test_near_dups_differing_by_uuid_and_timestamp_cluster() -> None:
    t1 = (
        f"{BASE_SENTENCE} run 550e8400-e29b-41d4-a716-446655440000 "
        "at 2024-01-15T10:30:00Z"
    )
    t2 = (
        f"{BASE_SENTENCE}  RUN 123e4567-e89b-12d3-a456-426614174000 "
        "at 2025-03-02T08:00:00Z"
    )
    assert t1 != t2
    result = find_clusters([t1, t2])
    assert result.clusters == [[0, 1]]
    assert result.exact_dup_groups == []
    assert result.stats.unique_texts == 2
    assert result.stats.near_dup_clusters == 1
    assert result.stats.exact_dup_bytes_saved == 0
    assert result.stats.skipped_over_cap == 0


def test_different_texts_do_not_cluster() -> None:
    texts = [
        "the quarterly report shows a steady increase in customer retention rates",
        "kubernetes pod eviction happens when node memory pressure crosses thresholds",
        "the recipe calls for two cups of flour and a pinch of baking soda overnight",
    ]
    assert all(len(t) > MINHASH_MIN_CHARS for t in texts)
    result = find_clusters(texts)
    assert result.clusters == []
    assert result.exact_dup_groups == []
    assert result.stats.unique_texts == 3
    assert result.stats.near_dup_clusters == 0


def test_exact_groups_and_bytes_saved() -> None:
    a = "x" * 70
    b = "y" * 70
    texts = [a, b, a, "short", a]
    result = find_clusters(texts)
    assert result.exact_dup_groups == [[0, 2, 4]]
    # Two extra copies of a 70-byte body are saved.
    assert result.stats.exact_dup_bytes_saved == 2 * 70
    assert result.stats.unique_texts == 3
    # Single-shingle texts "xxxxx" and "yyyyy" share nothing.
    assert result.clusters == []


def test_exact_groups_multibyte_lengths() -> None:
    emoji = "\U0001f680" * 20  # 4 bytes each in utf-8, 20 chars
    result = find_clusters([emoji, emoji, emoji])
    assert result.exact_dup_groups == [[0, 1, 2]]
    assert result.stats.exact_dup_bytes_saved == 2 * 80
    assert result.stats.unique_texts == 1


def test_minhash_length_threshold_is_strict() -> None:
    # 65 chars: eligible, byte-different, normalize-equal, so clustered.
    long_a = "a" * 63 + " B"
    long_b = "a" * 63 + " b"
    assert len(long_a) == MINHASH_MIN_CHARS + 1
    result = find_clusters([long_a, long_b])
    assert result.clusters == [[0, 1]]
    # Exactly 64 chars: not "longer than 64", so never minhashed.
    edge_a = "a" * 62 + " B"
    edge_b = "a" * 62 + " b"
    assert len(edge_a) == MINHASH_MIN_CHARS
    result = find_clusters([edge_a, edge_b])
    assert result.clusters == []
    assert result.stats.skipped_over_cap == 0


def test_cap_skips_and_reports() -> None:
    rng = np.random.default_rng(SEED)
    texts = [" ".join(_word(rng) for _ in range(15)) for _ in range(5)]
    assert all(len(t) > MINHASH_MIN_CHARS for t in texts)
    result = find_clusters(texts, cap=3)
    assert result.stats.skipped_over_cap == 2
    # Under the default cap nothing is skipped.
    assert find_clusters(texts).stats.skipped_over_cap == 0


def test_cluster_members_include_exact_duplicates() -> None:
    t1 = f"{BASE_SENTENCE} run 550e8400-e29b-41d4-a716-446655440000"
    t2 = f"{BASE_SENTENCE} run 123e4567-e89b-12d3-a456-426614174000"
    # t1 appears twice byte-identically; its copies join the near-dup cluster.
    result = find_clusters([t1, t2, t1])
    assert result.exact_dup_groups == [[0, 2]]
    assert result.clusters == [[0, 1, 2]]


def test_determinism_across_runs() -> None:
    texts, _ = _build_family_corpus(n_total=200, n_families=10)
    first = find_clusters(texts, seed=SEED)
    second = find_clusters(texts, seed=SEED)
    assert first == second
    assert isinstance(first, DedupResult)
    assert isinstance(first.stats, DedupStats)


def test_planted_families_recovered() -> None:
    texts, families = _build_family_corpus(n_total=1000, n_families=40)
    assert len(texts) == 1000
    result = find_clusters(texts, seed=SEED)
    got = {frozenset(c) for c in result.clusters}
    want = {frozenset(f) for f in families}
    assert got == want
    assert result.stats.near_dup_clusters == len(families)
    assert result.stats.unique_texts == 1000
    assert result.stats.skipped_over_cap == 0


def test_empty_input_safe() -> None:
    result = find_clusters([])
    assert result.clusters == []
    assert result.exact_dup_groups == []
    assert result.stats == DedupStats(
        unique_texts=0,
        exact_dup_bytes_saved=0,
        near_dup_clusters=0,
        skipped_over_cap=0,
    )


def test_short_text_input_safe() -> None:
    result = find_clusters(["", "", "a", "ab", "a"])
    assert result.exact_dup_groups == [[0, 1], [2, 4]]
    assert result.clusters == []
    assert result.stats.unique_texts == 3
    # One duplicate empty string saves 0 bytes, one duplicate "a" saves 1.
    assert result.stats.exact_dup_bytes_saved == 1
    assert result.stats.skipped_over_cap == 0


def test_find_clusters_accepts_any_iterable() -> None:
    result = find_clusters(iter(["z" * 70, "z" * 70]))
    assert result.exact_dup_groups == [[0, 1]]
    assert result.stats.unique_texts == 1


# ------------------------------------------------------------- per-text cap


def test_minhash_per_text_cap_bounds_cost_and_counts() -> None:
    """A multi-megabyte body is signed from a capped prefix, and counted.

    Bytes past the cap must not influence the signature, and the
    truncation must be observable so an audit can flag prefix-only
    comparison.
    """
    lsh = MinHashLSH(seed=SEED)
    big = "lorem ipsum dolor sit amet consectetur " * 60_000  # about 2.3 MB
    sig_big = lsh.signature(big)
    assert lsh.last_truncated_texts == 1
    sig_tail = lsh.signature(big + "a completely different trailing payload")
    assert lsh.last_truncated_texts == 1
    assert np.array_equal(sig_big, sig_tail)
    # Ordinary bodies are not truncated, and the counter resets per call.
    lsh.signature(BASE_SENTENCE)
    assert lsh.last_truncated_texts == 0


def test_find_clusters_reports_minhash_truncated() -> None:
    base = "retry storm payload chunk " * 60_000  # about 1.5 MB, repetitive
    t1 = f"{base} run 550e8400-e29b-41d4-a716-446655440000"
    t2 = f"{base} RUN 123e4567-e89b-12d3-a456-426614174000"
    result = find_clusters([t1, t2, BASE_SENTENCE])
    assert result.clusters == [[0, 1]]
    assert result.stats.unique_texts == 3
    assert result.stats.minhash_truncated == 2
    # A corpus of ordinary bodies reports zero truncation.
    assert find_clusters([t1[:200], t2[:200]]).stats.minhash_truncated == 0


def test_batch_and_singleton_signatures_agree_on_repetitive_bodies() -> None:
    """The unique-shingle reduction must not depend on batch layout."""
    texts = [
        "na " * 500 + "batman",
        BASE_SENTENCE,
        "ha " * 300 + BASE_SENTENCE,
        "",
        "spam eggs " * 200,
    ]
    lsh = MinHashLSH(seed=SEED)
    batched = lsh.signatures(texts)
    singles = np.stack([lsh.signature(t) for t in texts])
    assert np.array_equal(batched, singles)


def test_signatures_agree_under_tiny_batch_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from density.engine.trace import dedup as dedup_module

    texts = ["na " * 40 + w for w in ("alpha", "beta", "gamma")] + [""]
    want = MinHashLSH(seed=SEED).signatures(texts)
    monkeypatch.setattr(dedup_module, "_BATCH_SHINGLE_BUDGET", 17)
    got = MinHashLSH(seed=SEED).signatures(texts)
    assert np.array_equal(want, got)
