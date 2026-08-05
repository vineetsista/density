"""Audit runner tests over a hand-built corpus.

The fixture is constructed by hand (small JSONL plus seeded numpy
vectors) instead of the synth generator, so these tests depend only on
the audit contract: planted exact duplicates (one boilerplate system
prompt per trace), a planted near-duplicate retry storm (same body up
to uuids and timestamps), malformed lines, one non-canonically spaced
line for the residual path, and clustered unit vectors with real
nearest-neighbor structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import density
from density.audit.runner import compress_to_store, human_bytes, run_audit, usd
from density.errors import AuditError
from density.recall.metrics import BytesAccount
from density.recall.verifier import CodecReport, VerifyResult

SEED = 1337

N_TRACES = 12
EVENTS_PER_TRACE = 6
N_VECTORS = 2000
DIM = 32

# Exact-duplicate body: byte-identical across every trace, so dedup must
# find one exact group of N_TRACES occurrences.
BOILER = (
    "You are the acme release helper agent. Follow the deploy runbook, cite "
    "ticket ids in every reply, and never skip a verification step."
)

# Near-duplicate body: differs per event only in the uuid and the ISO
# timestamp, both of which normalize() strips, so all copies form one
# near-duplicate cluster.
RETRY = (
    "Retry attempt {uid} at {ts} while fetching shard 7: upstream timeout, "
    "backing off with jitter before the next call."
)

MALFORMED_LINES = [
    b'{"trace_id": "tr-0000", "ts": ',
    b"\x80\x81 binary garbage",
    b"[1, 2, 3]",
]

# Well-formed JSON whose spacing differs from the compact canonical dump,
# forcing the residual path without being malformed.
SPACED_LINE = (
    b'{"trace_id": "tr-0000", "ts": 1700009999000000, "role": "user", '
    b'"type": "message", "content": "manually spaced line for the residual path"}'
)

WORDS = [
    "quartz", "meadow", "copper", "falcon", "harbor", "juniper",
    "marble", "onyx", "prairie", "saffron", "timber", "velvet",
]


def _trace_lines(t: int) -> list[bytes]:
    """The six event lines of one trace, written in canonical key order."""
    base_ts = 1_700_000_000_000_000 + t * 10_000_000
    tid = f"tr-{t:04d}"
    lines: list[bytes] = []

    def dump(obj: dict) -> bytes:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    lines.append(
        dump({"trace_id": tid, "ts": base_ts, "role": "system", "type": "message",
              "content": BOILER})
    )
    lines.append(
        dump({"trace_id": tid, "ts": base_ts + 250_000, "role": "user",
              "type": "message",
              "content": (
                  f"The {WORDS[t]} account shows drift é ✅ and the "
                  f"{WORDS[(t + 3) % 12]} ledger disagrees with region "
                  f"{WORDS[(t + 5) % 12]} exports, please reconcile batch {t}."
              )})
    )
    lines.append(
        dump({"trace_id": tid, "ts": base_ts + 500_000, "role": "assistant",
              "type": "message",
              "content": (
                  f"Rebased batch {t} onto the posting date column of the "
                  f"{WORDS[(t + 7) % 12]} ledger and re-ran the diff cleanly."
              ),
              "model": "sable-mini-1", "tokens_in": 120 + t, "tokens_out": 40 + t,
              "channel": f"deploy-{WORDS[t]}"})
    )
    for j in range(3):
        uid = f"{t:08d}-aaaa-4bbb-8ccc-{j:012d}"
        ts_iso = f"2024-06-01T12:{t:02d}:{j:02d}Z"
        lines.append(
            dump({"trace_id": tid, "ts": base_ts + 750_000 + j * 50_000,
                  "role": "tool", "type": "tool_call", "tool_name": "fetch_shard",
                  "content": RETRY.format(uid=uid, ts=ts_iso)})
        )
    return lines


def build_corpus(
    root: Path, with_traces: bool = True, with_embeddings: bool = True
) -> tuple[Path, dict]:
    """Write the fixture corpus under root/corpus and return (dir, meta)."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    events = 0
    malformed = 0
    if with_traces:
        tdir = corpus / "traces"
        tdir.mkdir()
        part0: list[bytes] = []
        for t in range(8):
            part0.extend(_trace_lines(t))
        part0.append(SPACED_LINE)
        part0.extend(MALFORMED_LINES)
        (tdir / "part-0000.jsonl").write_bytes(b"\n".join(part0) + b"\n")
        part1: list[bytes] = []
        for t in range(8, N_TRACES):
            part1.extend(_trace_lines(t))
        (tdir / "part-0001.jsonl").write_bytes(b"\n".join(part1) + b"\n")
        events = N_TRACES * EVENTS_PER_TRACE + 1  # plus the spaced line
        malformed = len(MALFORMED_LINES)
    if with_embeddings:
        edir = corpus / "embeddings"
        edir.mkdir()
        rng = np.random.default_rng(SEED)
        # 200 clusters of about 10 members each: a query's true top-10 is
        # essentially its own cluster, separated from every other cluster
        # by a wide score gap. That gap is what lets an 8-bit grid rank
        # neighbors correctly; with a few large dense clusters (16 at
        # noise 0.05) the rank-10 to rank-11 gap sinks below sq8
        # quantization error and the warm floor genuinely fails at dim 32.
        # Measured here: warm recall@10 0.998 vs floor 0.99, cold 0.998
        # vs floor 0.95.
        centers = rng.normal(size=(200, DIM)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        assign = rng.integers(0, 200, size=N_VECTORS)
        x = centers[assign] + 0.10 * rng.normal(size=(N_VECTORS, DIM)).astype(np.float32)
        x = x.astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        np.save(edir / "vectors.npy", x)
        np.save(edir / "ids.npy", np.arange(N_VECTORS, dtype=np.int64))
    meta = {
        "events": events,
        "malformed": malformed,
        "lines": events + malformed,
        "n_vectors": N_VECTORS if with_embeddings else 0,
        "dim": DIM if with_embeddings else 0,
    }
    return corpus, meta


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, dict]:
    return build_corpus(tmp_path_factory.mktemp("audit-corpus"))


@pytest.fixture(scope="module")
def audited(corpus, tmp_path_factory):
    """One full audit shared by every read-only assertion in this module."""
    corpus_dir, meta = corpus
    out = tmp_path_factory.mktemp("audit-out") / "report.html"
    result = run_audit(corpus_dir, out=str(out), seed=SEED)
    return result, meta, out


# -- full pipeline over the fixture -------------------------------------


def test_pipeline_counts_and_outputs(audited):
    result, meta, out = audited
    assert result.traces_present and result.embeddings_present
    assert result.trace_events == meta["events"]
    assert result.malformed == meta["malformed"]
    assert result.trace_files == 2
    assert result.vector_count == N_VECTORS
    assert result.vector_dim == DIM
    assert result.vectors_raw_bytes == N_VECTORS * DIM * 4
    assert out.exists()
    assert out.with_suffix(".md").exists()
    # Every stage ran and was timed; peak RSS was measured.
    for stage in ("ingest", "compress", "dedup", "verify", "round_trip", "price", "render"):
        assert stage in result.stage_seconds
    assert result.peak_rss_mb > 0


def test_number_consistency(audited):
    result, meta, _ = audited
    original = result.traces_raw_bytes + result.vectors_raw_bytes
    assert result.original_bytes == original
    for name in ("warm", "cold"):
        t = result.tier_results[name]
        assert t.total_bytes == t.traces_bytes + t.vectors_bytes + t.vectors_aux_bytes
        assert t.ratio == pytest.approx(original / t.total_bytes)
        assert t.footprint_fraction == pytest.approx(t.total_bytes / original)
        assert t.traces_bytes > 0
        assert t.recall is not None
        for value in t.recall.values():
            assert 0.0 <= value <= 1.0
        # Pass flags are the comparison, never a rounding of it.
        assert t.recall_pass == (t.recall["recall@10"] >= t.recall10_floor)
    # Codec arithmetic is exact: sq8 is one byte per dimension, pq one
    # byte per subquantizer (m = dim // 4).
    assert result.tier_results["warm"].vectors_bytes == N_VECTORS * DIM
    assert result.tier_results["cold"].vectors_bytes == N_VECTORS * (DIM // 4)


def test_savings_match_pricing_constants(audited):
    result, _, _ = audited
    p = result.savings.pricing
    assert p.s3_gb_month == 0.023 and p.vectordb_gb_month == 0.33
    gb = 1e9
    before = (
        result.traces_raw_bytes / gb * p.s3_gb_month
        + result.vectors_raw_bytes / gb * p.vectordb_gb_month
    )
    assert result.savings.before_monthly == pytest.approx(before)
    for name, t in result.tier_results.items():
        sv = result.savings.per_tier[name]
        after = (
            t.traces_bytes / gb * p.s3_gb_month
            + (t.vectors_bytes + t.vectors_aux_bytes) / gb * p.vectordb_gb_month
        )
        assert sv.after_monthly == pytest.approx(after)
        assert sv.monthly == pytest.approx(before - after)
        assert sv.yearly == pytest.approx(12 * sv.monthly)


def test_dedup_finds_planted_duplicates(audited):
    result, meta, _ = audited
    d = result.dedup
    assert d is not None
    # The boilerplate system prompt is byte-identical in all 12 traces.
    assert d.exact_group_count >= 1
    assert d.bytes_saved >= (N_TRACES - 1) * len(BOILER.encode("utf-8"))
    # The retry storm differs only in stripped tokens: one cluster of all
    # 36 tool bodies, and no larger cluster exists in the fixture.
    assert d.cluster_count >= 1
    assert d.top_clusters[0].size == N_TRACES * 3
    for cluster in d.top_clusters:
        assert 0 < len(cluster.sample) <= 120
    assert d.top_clusters[0].sample.startswith("Retry attempt")
    # Residual lines: 3 malformed plus the one non-canonically spaced line.
    assert d.residual_lines == meta["malformed"] + 1
    assert d.total_lines == meta["lines"]


def test_verifier_methodology_recorded(audited):
    result, _, _ = audited
    v = result.methodology["verifier"]
    assert v is not None
    assert v["n_queries"] == min(1000, N_VECTORS // 10)
    assert v["n_database"] == v["sample_size"] - v["n_queries"]
    assert v["k"] == 100
    assert v["rerank_depths"] == {"warm": 0, "cold": 200}
    assert result.methodology["seed"] == SEED
    assert isinstance(result.methodology["accel_active"], bool)
    assert "versions" in result.methodology
    rt = result.methodology["round_trip"]
    assert rt["sampled_lines"] >= 1
    assert sorted(rt["tiers"]) == ["cold", "warm"]


def test_summary_text_voice(audited):
    result, _, _ = audited
    text = result.summary_text()
    assert text.startswith("DENSITY audit:")
    assert "warm" in text and "cold" in text
    assert "recall@10" in text
    # Dash characters are asserted via escapes so this file itself never
    # contains one.
    assert "\u2014" not in text and "\u2013" not in text
    # A few terminal-friendly lines, not a wall of text.
    assert 4 <= len(text.splitlines()) <= 20


def test_to_dict_serializable_and_deterministic(corpus, tmp_path):
    corpus_dir, _ = corpus
    a = run_audit(corpus_dir, out=str(tmp_path / "a" / "report.html"), seed=SEED)
    b = run_audit(corpus_dir, out=str(tmp_path / "b" / "report.html"), seed=SEED)

    def strip(result) -> dict:
        d = json.loads(json.dumps(result.to_dict()))  # proves serializability
        # The only fields allowed to differ: machine measurements and the
        # caller-chosen output paths.
        for key in ("stage_seconds", "peak_rss_mb", "out_html", "out_md"):
            d.pop(key, None)
        return d

    assert strip(a) == strip(b)


# -- honesty and failure paths ------------------------------------------


def test_forced_recall_miss_is_stated_loudly(corpus, tmp_path, monkeypatch):
    corpus_dir, _ = corpus

    def fake_verify(X, codecs, seed=1337, n_queries=1000, sample_cap=200_000,
                    rerank_depths=None, reranker=None, reranker_factory=None):
        reports = {
            name: CodecReport(
                name=name,
                recall={"recall@1": 0.40, "recall@10": 0.50, "recall@100": 0.60},
                rerank_depth=(rerank_depths or {}).get(name, 0),
                bytes=BytesAccount(raw=1000, encoded=250, aux=10),
                bytes_per_vector=0.26,
                search_seconds=0.0,
            )
            for name in codecs
        }
        return VerifyResult(
            codecs=reports, n_vectors=X.shape[0], sample_size=X.shape[0],
            n_queries=n_queries, n_database=X.shape[0] - n_queries, k=100, seed=seed,
        )

    monkeypatch.setattr("density.audit.runner.verify_tiers", fake_verify)
    out = tmp_path / "report.html"
    result = run_audit(corpus_dir, out=str(out), seed=SEED)

    for name in ("warm", "cold"):
        assert result.tier_results[name].recall_pass is False
    flags = " ".join(result.honesty)
    assert "MISSED" in flags
    # A cold miss recommends warm, a warm miss recommends hot: never
    # silently averaged away.
    assert "Use the warm tier" in flags
    assert "Use the hot tier" in flags
    assert any("No requested tier passed" in f for f in result.honesty)
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    html = out.read_text(encoding="utf-8")
    assert "MISS" in md and "MISS" in html
    assert result.summary_text().count("MISS") >= 2


def test_round_trip_mismatch_raises_loud_audit_error(corpus, tmp_path, monkeypatch):
    corpus_dir, _ = corpus
    from density.engine.trace.shred import unshred as real_unshred

    def tampered(bundle):
        # Corrupt every replayed line by one byte: whichever lines the
        # seeded sample picks, the comparison must catch it.
        for file, idx, raw in real_unshred(bundle):
            yield file, idx, raw + b"?"

    monkeypatch.setattr("density.audit.runner.unshred", tampered)
    with pytest.raises(AuditError, match="round-trip byte mismatch"):
        run_audit(corpus_dir, out=str(tmp_path / "report.html"), seed=SEED)


def test_missing_embeddings_renders_honestly(tmp_path):
    corpus_dir, meta = build_corpus(tmp_path, with_embeddings=False)
    out = tmp_path / "report.html"
    result = run_audit(corpus_dir, out=str(out), seed=SEED)
    assert not result.embeddings_present
    assert result.vectors_raw_bytes == 0
    for t in result.tier_results.values():
        assert t.recall is None
        assert t.recall_pass is None
        assert t.vectors_bytes == 0
    assert any("Embeddings: not present" in f for f in result.honesty)
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "not present in this corpus" in md
    assert "not present in this corpus" in out.read_text(encoding="utf-8")
    # Trace numbers are still fully measured.
    assert result.trace_events == meta["events"]


def test_embeddings_only_corpus(tmp_path):
    corpus_dir, _ = build_corpus(tmp_path, with_traces=False)
    out = tmp_path / "report.html"
    result = run_audit(corpus_dir, out=str(out), seed=SEED)
    assert result.embeddings_present and not result.traces_present
    assert result.dedup is None
    assert result.traces_raw_bytes == 0
    assert result.methodology["round_trip"]["sampled_lines"] == 0
    for t in result.tier_results.values():
        assert t.traces_bytes == 0
        assert t.recall is not None  # recall is still measured
    assert any("Traces: not present" in f for f in result.honesty)
    assert "not present in this corpus" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_empty_corpus_raises_audit_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AuditError, match="nothing to audit"):
        run_audit(empty, out=str(tmp_path / "report.html"), seed=SEED)
    with pytest.raises(AuditError, match="does not exist"):
        run_audit(tmp_path / "missing", out=str(tmp_path / "r.html"), seed=SEED)


def test_pricing_toml_override(tmp_path):
    corpus_dir, _ = build_corpus(tmp_path, with_embeddings=False)
    toml = tmp_path / "pricing.toml"
    toml.write_text("[pricing]\ns3_gb_month = 0.05\nvectordb_gb_month = 1.5\n")
    result = run_audit(
        corpus_dir, out=str(tmp_path / "report.html"), pricing=toml, seed=SEED
    )
    p = result.savings.pricing
    assert p.s3_gb_month == 0.05 and p.vectordb_gb_month == 1.5
    assert p.source == str(toml)
    assert result.savings.before_monthly == pytest.approx(
        result.traces_raw_bytes / 1e9 * 0.05
    )
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "s3_gb_month = 0.05" in md


# -- compress_to_store ---------------------------------------------------


def test_compress_to_store_round_trips(corpus, tmp_path):
    corpus_dir, _ = corpus
    store_dir = compress_to_store(
        corpus_dir, out=str(tmp_path / "store.density"), seed=SEED
    )
    assert store_dir == str(tmp_path / "store.density")

    # Expected replay: the exact original bytes of one trace, in order.
    expected: list[bytes] = []
    for part in sorted((corpus_dir / "traces").glob("*.jsonl")):
        for raw in part.read_bytes().splitlines():
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict) and obj.get("trace_id") == "tr-0003":
                expected.append(raw)
    assert len(expected) == EVENTS_PER_TRACE

    with density.open(store_dir) as store:
        assert store.verify() == []
        assert store.replay_raw("tr-0003") == expected
        vec = np.load(corpus_dir / "embeddings" / "vectors.npy")[5]
        ids, scores = store.search(vec, k=5)
        assert 5 in ids.tolist()
        assert scores.shape == (5,)


def test_compress_to_store_nothing_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AuditError, match="nothing to compress"):
        compress_to_store(empty, out=str(tmp_path / "s.density"))


# -- formatting helpers --------------------------------------------------


def test_human_bytes_and_usd():
    assert human_bytes(0) == "0 B"
    assert human_bytes(999) == "999 B"
    assert human_bytes(1_230_000) == "1.23 MB"
    assert human_bytes(2_500_000_000) == "2.50 GB"
    assert usd(1234.5) == "$1,234.50"
    assert usd(-3.2) == "-$3.20"


# --- regression: never state a measurement that was not measured ----------


def test_hot_only_audit_reports_residual_as_not_measured(corpus, tmp_path):
    """A hot-only audit builds no bundle, so there is no residual rate.

    The old code printed "0.0 percent (0 of N lines)", which reads as a
    measured fact about a bundle that does not exist. In the one artifact
    this product sells as never estimating, that is the wrong default.
    """
    corpus_dir, _meta = corpus
    out = tmp_path / "hot.html"
    result = run_audit(corpus_dir, out=str(out), tiers=("hot",))
    dedup = result.to_dict()["dedup"]
    assert dedup["residual_lines"] is None
    assert dedup["residual_tier"] is None
    markdown = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "Residual rate: not measured" in markdown
    assert "0.0 percent" not in markdown.split("Residual rate:")[1][:80]


def test_warm_audit_names_the_tier_the_residual_rate_came_from(corpus, tmp_path):
    corpus_dir, _meta = corpus
    out = tmp_path / "warm.html"
    result = run_audit(corpus_dir, out=str(out), tiers=("warm",))
    dedup = result.to_dict()["dedup"]
    assert dedup["residual_lines"] is not None
    assert dedup["residual_tier"] == "warm"
    assert "warm bundle" in out.with_suffix(".md").read_text(encoding="utf-8")
