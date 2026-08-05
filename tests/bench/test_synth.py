"""Tests for density.bench.synth, the deterministic synthetic corpus generator.

Every test seeds explicitly. Corpus sizes are small (0.002 to 0.01 GB) so
the suite stays fast while still exercising every realism feature.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from density.bench.synth import PERSONAS, SynthStats, generate, sample_embeddings

SEED = 1337
DIM = 64

REQUIRED_KEYS = {
    "bytes_traces",
    "bytes_embeddings",
    "bytes_total",
    "target_bytes",
    "events",
    "malformed",
    "n_traces",
    "files",
    "n_vectors",
    "dim",
    "seed",
    "elapsed_s",
}


def _sha_map(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _raw_lines(root: Path) -> list[bytes]:
    lines: list[bytes] = []
    for p in sorted((root / "traces").glob("part-*.jsonl")):
        data = p.read_bytes()
        assert data.endswith(b"\n")
        lines.extend(data.split(b"\n")[:-1])
    return lines


def _split(lines: list[bytes]) -> tuple[list[tuple[bytes, dict]], int]:
    events: list[tuple[bytes, dict]] = []
    malformed = 0
    for ln in lines:
        try:
            obj = json.loads(ln.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            malformed += 1
            continue
        if not isinstance(obj, dict):
            malformed += 1
            continue
        events.append((ln, obj))
    return events, malformed


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, SynthStats]:
    out = Path(tmp_path_factory.mktemp("synth-corpus"))
    stats = generate(0.01, out, seed=SEED, dim=DIM)
    return out, stats


@pytest.fixture(scope="module")
def parsed(corpus) -> tuple[list[bytes], list[tuple[bytes, dict]], int]:
    out, _ = corpus
    lines = _raw_lines(out)
    events, malformed = _split(lines)
    return lines, events, malformed


def test_determinism_same_seed_bit_identical(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    sa = generate(0.002, a, seed=7, dim=32)
    sb = generate(0.002, b, seed=7, dim=32)
    sc = generate(0.002, c, seed=8, dim=32)

    map_a, map_b, map_c = _sha_map(a), _sha_map(b), _sha_map(c)
    assert map_a, "generator produced no files"
    assert map_a == map_b

    da, db = sa.to_dict(), sb.to_dict()
    da.pop("elapsed_s")
    db.pop("elapsed_s")
    assert da == db

    # A different seed must change content somewhere.
    assert map_a != map_c
    assert sc.seed == 8


def test_size_within_five_percent(corpus) -> None:
    out, stats = corpus
    target = int(0.01 * 1e9)
    assert stats.target_bytes == target
    assert abs(stats.bytes_total - target) <= 0.05 * target

    shard_files = sorted((out / "traces").glob("part-*.jsonl"))
    assert shard_files, "no trace shards written"
    disk_traces = sum(p.stat().st_size for p in shard_files)
    disk_emb = sum(p.stat().st_size for p in (out / "embeddings").iterdir())
    assert disk_traces == stats.bytes_traces
    assert disk_emb == stats.bytes_embeddings
    assert stats.bytes_total == stats.bytes_traces + stats.bytes_embeddings
    for p in shard_files:
        assert p.stat().st_size <= 64 * 1024 * 1024
    assert stats.files == len(shard_files)


def test_stats_to_dict(corpus) -> None:
    _, stats = corpus
    d = stats.to_dict()
    assert REQUIRED_KEYS <= set(d)
    assert d["events"] > 0
    assert d["n_traces"] > 0
    assert d["seed"] == SEED
    assert d["dim"] == DIM
    assert d["elapsed_s"] >= 0.0
    json.dumps(d)  # must be JSON-serializable


def test_malformed_fraction(corpus, parsed) -> None:
    _, stats = corpus
    lines, events, malformed = parsed
    assert malformed == stats.malformed
    assert len(events) == stats.events
    frac = malformed / len(lines)
    assert 0.01 <= frac <= 0.03


def test_all_three_personas_present(corpus, parsed) -> None:
    out, _ = corpus
    _, events, _ = parsed
    assert len(PERSONAS) == 3
    trace_ids = {obj["trace_id"] for _, obj in events}
    blob = b"".join(p.read_bytes() for p in sorted((out / "traces").glob("part-*.jsonl")))
    prompts = set()
    for persona in PERSONAS:
        prompt = persona.system_prompt
        prompts.add(prompt)
        size = len(prompt.encode("utf-8"))
        assert 1500 <= size <= 4000, f"{persona.key} prompt is {size} bytes"
        # Revised for the composed-prompt design: each conversation sends
        # base prompt plus an injected per-conversation context block, so
        # the serialized BASE bytes recur as a strict prefix of every
        # composed prompt (JSON string escaping is per-character, so the
        # escaping of a prefix is a prefix of the escaping of the whole).
        # The old pin matched the full JSON literal, closing quote
        # included, which encoded the bare-prompt design.
        enc = json.dumps(prompt, ensure_ascii=False).encode("utf-8")
        assert blob.count(enc[:-1]) >= 20
        assert any(t.startswith(f"tr-{persona.key}-") for t in trace_ids)
    assert len(prompts) == 3, "system prompts must be distinct"


def test_prompt_composition(parsed) -> None:
    """Base shared across a persona, injected block unique per conversation,
    and every resend inside one conversation is byte-identical."""
    _, events, _ = parsed
    base_by_key = {p.key: p.system_prompt for p in PERSONAS}
    sys_by_trace: dict[str, list[str]] = {}
    for _, obj in events:
        if obj.get("role") == "system":
            sys_by_trace.setdefault(obj["trace_id"], []).append(obj["content"])
    assert len(sys_by_trace) >= 50

    injected: list[str] = []
    resent_traces = 0
    for trace_id, contents in sys_by_trace.items():
        # The conversation resends its exact full prompt on every call.
        assert len(set(contents)) == 1, f"{trace_id} varied its prompt between calls"
        if len(contents) >= 2:
            resent_traces += 1
        key = trace_id.split("-")[1]
        base = base_by_key[key]
        assert contents[0].startswith(base), f"{trace_id} lacks the persona base prefix"
        block = contents[0][len(base):]
        # Injected context block: roughly 200 to 1200 bytes per the synth
        # contract, with slack for seed-to-seed spread at the edges.
        size = len(block.encode("utf-8"))
        assert 150 <= size <= 1400, f"{trace_id} injected block is {size} bytes"
        injected.append(block)
    assert resent_traces >= 20, "most conversations should resend the prompt"
    # Wide seeded slot spaces make injected blocks unique per conversation.
    assert len(set(injected)) == len(injected)


def test_stream_interleaving(corpus, parsed) -> None:
    """The trace stream is a global time-ordered log of concurrent sessions,
    so consecutive calls of one conversation land far apart in stream bytes."""
    out, _ = corpus
    lines, _, _ = parsed
    # Global time order: parsed event timestamps never decrease across the
    # whole stream, exactly like a platform's append-only event log.
    prev_ts = None
    sys_offsets: dict[str, list[int]] = {}
    pos = 0
    for raw in lines:
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            pos += len(raw) + 1
            continue
        if not isinstance(obj, dict):
            pos += len(raw) + 1
            continue
        ts = obj["ts"]
        assert prev_ts is None or ts >= prev_ts, "stream must be time-ordered"
        prev_ts = ts
        if obj.get("role") == "system":
            sys_offsets.setdefault(obj["trace_id"], []).append(pos)
        pos += len(raw) + 1

    # System events mark call starts: the byte distance between consecutive
    # calls of one conversation is the prompt-resend distance the trace
    # engine must bridge (and a plain zstd window cannot, at scale). At
    # 0.01 GB (about 120 lanes) the measured median is about 200 KB and it
    # grows linearly with corpus size; 100 KB is a safe deterministic floor.
    gaps = [
        b - a
        for offsets in sys_offsets.values()
        for a, b in zip(offsets, offsets[1:])
    ]
    assert len(gaps) >= 200
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    assert median_gap >= 100_000, f"median inter-call stream gap {median_gap} bytes"


def test_noncanonical_spacing_fraction(parsed) -> None:
    _, events, _ = parsed
    spaced = 0
    for raw, obj in events:
        canonical = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if canonical != raw:
            spaced += 1
    frac = spaced / len(events)
    assert 0.001 <= frac <= 0.015


def test_retry_storms(parsed) -> None:
    _, events, _ = parsed
    runs: list[list[dict]] = []
    current: list[dict] = []
    for _, obj in events:
        if (
            current
            and obj.get("type") == "tool_call"
            and current[-1].get("type") == "tool_call"
            and obj["trace_id"] == current[-1]["trace_id"]
            and obj.get("tool_name") == current[-1].get("tool_name")
        ):
            current.append(obj)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [obj] if obj.get("type") == "tool_call" else []
    if len(current) >= 2:
        runs.append(current)

    assert len(runs) >= 3, "expected several retry storms"
    assert max(len(r) for r in runs) >= 3
    # Args must be jittered inside a storm, not byte-identical repeats.
    assert any(len({e["content"] for e in r}) >= 2 for r in runs)
    # Timestamps inside a storm jitter forward.
    for r in runs:
        ts = [e["ts"] for e in r]
        assert ts == sorted(ts)


def test_large_tool_outputs(parsed) -> None:
    _, events, _ = parsed
    large = [obj for _, obj in events if len(obj.get("content", "")) >= 40_000]
    assert len(large) >= 3
    assert len(large) / len(events) <= 0.02
    for obj in large:
        assert obj["type"] == "tool_result"


def test_timestamps_bursty_and_monotonic(parsed) -> None:
    _, events, _ = parsed
    per_trace: dict[str, list[int]] = {}
    for _, obj in events:
        ts = obj["ts"]
        assert isinstance(ts, int)
        assert 1_700_000_000_000_000 < ts < 1_800_000_000_000_000  # micros, ~2026
        per_trace.setdefault(obj["trace_id"], []).append(ts)
    for series in per_trace.values():
        assert series == sorted(series)
        assert len(set(series)) == len(series), "timestamps must strictly increase"
    # Bursty arrival check: gaps span at least three orders of magnitude.
    gaps = np.concatenate(
        [np.diff(np.asarray(s, dtype=np.int64)) for s in per_trace.values() if len(s) > 1]
    )
    assert gaps.min() >= 1
    assert gaps.max() / max(np.median(gaps), 1) > 100


def test_unicode_and_emoji_present(parsed) -> None:
    _, events, _ = parsed
    contents = [obj.get("content", "") for _, obj in events]
    assert any(any(ord(ch) >= 0x1F000 for ch in c) for c in contents)
    assert any(any(0x80 <= ord(ch) < 0x1F000 for ch in c) for c in contents)


def test_embeddings_files_and_ids(corpus) -> None:
    out, stats = corpus
    X = np.load(out / "embeddings" / "vectors.npy")
    ids = np.load(out / "embeddings" / "ids.npy")
    assert X.dtype == np.float32
    assert ids.dtype == np.int64
    assert stats.n_vectors == 5000  # max(5000, int(0.01 * 100000))
    assert X.shape == (stats.n_vectors, DIM)
    assert ids.shape == (stats.n_vectors,)
    assert len(np.unique(ids)) == stats.n_vectors
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_embeddings_have_cluster_structure(corpus) -> None:
    out, _ = corpus
    X = np.load(out / "embeddings" / "vectors.npy")
    q, base = X[:400], X[400:3400]
    sims = q @ base.T
    nn = sims.max(axis=1)
    # Silhouette-like separation: nearest neighbors are far more similar
    # than average pairs, which random unit vectors would not produce.
    assert float(nn.mean()) > 0.5
    assert float(nn.mean() - sims.mean()) > 0.3


def test_sample_embeddings_deterministic_and_normalized() -> None:
    a = sample_embeddings(2000, 64, seed=5)
    b = sample_embeddings(2000, 64, seed=5)
    c = sample_embeddings(2000, 64, seed=6)
    assert a.dtype == np.float32
    assert a.shape == (2000, 64)
    assert np.array_equal(a, b), "same seed must be bit-identical"
    assert not np.array_equal(a, c)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)
    with pytest.raises(ValueError):
        sample_embeddings(0, 64)
    with pytest.raises(ValueError):
        sample_embeddings(100, 0)


def test_sample_embeddings_near_duplicate_mass() -> None:
    # The mixture plants about 35 percent near-copies (retry storms and
    # template repeats). Each copy sits at cosine about 1 - jitter^2 / 2
    # (~0.997) from its base, so a large fraction of points must have a
    # neighbor above 0.99 while random cluster-mates stay well below it.
    X = sample_embeddings(3000, 64, seed=11)
    sims = X @ X.T
    np.fill_diagonal(sims, -1.0)
    nn = sims.max(axis=1)
    frac_dup = float((nn > 0.99).mean())
    assert 0.25 <= frac_dup <= 0.75, f"near-duplicate fraction {frac_dup}"


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        generate(0.0, "/nonexistent-should-not-be-created")
    with pytest.raises(ValueError):
        generate(0.001, "/nonexistent-should-not-be-created", dim=0)
    with pytest.raises(ValueError):
        generate(0.001, "/nonexistent-should-not-be-created", n_vectors=0)
