"""Store tests: the TDD spec for tiers/store.py.

Fixtures are hand-written JSONL lines plus seeded random vectors, never
bench.synth: the store contract must hold on data whose exact bytes the
test controls, including a CRLF-terminated line, a malformed line, and a
non-canonical spacing line that exercises the residual path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import density
from density.errors import StoreError
from density.tiers.manifest import Manifest

SEED = 1337
CREATED = "2026-01-01T00:00:00+00:00"
DIM = 32
N_VECTORS = 200
PLANTED_ROW = 17

# One line per shape the byte-exact contract must survive. _L_CRLF is
# written with a \r\n terminator, so its replayed payload must end in \r.
_L_UNICODE = (
    '{"trace_id":"t1","ts":1700000000000000,"role":"user",'
    '"type":"message","content":"hola é \U0001f680"}'
).encode("utf-8")
_L_SPACING = (
    b'{"trace_id": "t1", "ts": 1700000000250000, "role": "assistant", '
    b'"type": "message", "content": "non canonical spacing"}'
)
_L_CRLF = (
    b'{"trace_id":"t2","ts":1700000000500000,"role":"tool",'
    b'"type":"tool_result","tool_name":"grep","content":"crlf terminated"}'
)
_L_MALFORMED = b'{"trace_id":"t2","ts":170'
_L_TAIL = (
    b'{"trace_id":"t2","ts":1700000000750000,"role":"assistant",'
    b'"type":"message","content":"tail"}'
)

N_EVENTS = 4
N_MALFORMED = 1


def write_fixture(dir_path) -> object:
    """Write the 5-line fixture file and return its path.

    Line 3 is terminated with CRLF, so the CR byte belongs to its payload
    and must survive replay. The malformed line sits between t2's two
    events to prove replay skips it without losing order.
    """
    path = dir_path / "traces.jsonl"
    blob = (
        _L_UNICODE + b"\n"
        + _L_SPACING + b"\n"
        + _L_CRLF + b"\r\n"
        + _L_MALFORMED + b"\n"
        + _L_TAIL + b"\n"
    )
    path.write_bytes(blob)
    return path


def make_vectors(n: int = N_VECTORS, d: int = DIM, seed: int = SEED):
    """Seeded unit vectors plus a query planted next to row PLANTED_ROW.

    Returns (X float32 [n, d] L2-normalized, q float32 [d] normalized).
    The query is a small perturbation of one row, so exact search and
    every sane approximate tier must rank that row first.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = x[PLANTED_ROW] + 0.02 * rng.normal(size=d).astype(np.float32)
    q = (q / np.linalg.norm(q)).astype(np.float32)
    return x, q


def int_ids(n: int = N_VECTORS) -> np.ndarray:
    # Offset ids prove search maps row indices back to original ids.
    return np.arange(1000, 1000 + n, dtype=np.int64)


def open_store(path, **kwargs):
    kwargs.setdefault("created_at", CREATED)
    return density.open(path, **kwargs)


# -- traces ------------------------------------------------------------


def test_put_traces_counts_and_quarantine(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        stats = store.put_traces(fixture)
    assert stats.events == N_EVENTS
    assert stats.malformed == N_MALFORMED
    assert stats.files == 1
    m = Manifest.load(tmp_path / "s.density")
    assert m.datasets.traces.events == N_EVENTS
    assert m.datasets.traces.malformed == N_MALFORMED
    assert m.datasets.traces.files == ["traces.jsonl"]
    assert m.datasets.traces.raw_bytes == stats.bytes_read
    assert len(m.quarantine) == 1
    assert "json" in m.quarantine[0].error
    assert m.tiers["cold"].traces_dir == "tiers/cold/traces"
    assert m.tiers["cold"].bytes.traces > 0


def test_replay_raw_byte_exact(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture)
        assert store.replay_raw("t1") == [_L_UNICODE, _L_SPACING]
        # The CRLF line replays with its CR byte, and the malformed line
        # between t2's events neither appears nor breaks the order.
        assert store.replay_raw("t2") == [_L_CRLF + b"\r", _L_TAIL]


def test_replay_parses_events_in_original_order(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture)
        events = store.replay("t1")
    assert events == [
        json.loads(_L_UNICODE.decode("utf-8")),
        json.loads(_L_SPACING.decode("utf-8")),
    ]
    # Key order is part of "exactly as ingested".
    assert list(events[0]) == ["trace_id", "ts", "role", "type", "content"]


def test_replay_unknown_trace_id_raises(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture)
        with pytest.raises(StoreError, match="no-such-trace"):
            store.replay("no-such-trace")


def test_replay_without_traces_raises(tmp_path):
    with open_store(tmp_path / "s.density") as store:
        with pytest.raises(StoreError, match="no traces"):
            store.replay_raw("t1")


def test_put_traces_warm_tier(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        stats = store.put_traces(fixture, tier="warm")
        assert stats.events == N_EVENTS
        assert store.replay_raw("t2") == [_L_CRLF + b"\r", _L_TAIL]
    m = Manifest.load(tmp_path / "s.density")
    assert m.tiers["warm"].trace_zstd_level == 10


def test_put_traces_twice_same_tier_rejected(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture)
        with pytest.raises(StoreError, match="single put_traces"):
            store.put_traces(fixture)


def test_put_traces_hot_rejected(tmp_path):
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        with pytest.raises(StoreError, match="hot"):
            store.put_traces(fixture, tier="hot")


# -- embeddings and search ---------------------------------------------


def test_warm_search_recovers_planted_neighbor(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        ids, scores = store.search(q, k=5)
    assert ids.shape == (5,)
    assert scores.shape == (5,)
    assert ids[0] == 1000 + PLANTED_ROW
    assert np.all(np.diff(scores) <= 1e-5)  # best-first


def test_batch_query_shapes(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        ids, scores = store.search(np.stack([q, x[3]]), k=4)
    assert ids.shape == (2, 4)
    assert scores.shape == (2, 4)
    assert ids[0, 0] == 1000 + PLANTED_ROW
    assert ids[1, 0] == 1003


def test_string_ids_come_back(tmp_path):
    x, q = make_vectors()
    ids_in = np.asarray([f"vec-{i:04d}" for i in range(N_VECTORS)])
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(ids_in, x, tier="warm")
        ids, _ = store.search(q, k=3)
    assert ids[0] == f"vec-{PLANTED_ROW:04d}"


def test_cold_search_reranks_through_warm(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        store.put_embeddings(int_ids(), x, tier="cold")
        ids, scores = store.search(q, k=5, tier="cold")
    assert ids[0] == 1000 + PLANTED_ROW
    assert np.all(np.diff(scores) <= 1e-5)


def test_cold_search_alone_warns_and_still_works(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="cold")
        with pytest.warns(UserWarning, match="without rerank"):
            ids, _ = store.search(q, k=5)
    # A planted near-duplicate is far closer than anything else, so pure
    # ADC must still rank it first even without a reranker.
    assert ids[0] == 1000 + PLANTED_ROW


def test_cold_binary_codec_variant(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        store.put_embeddings(int_ids(), x, tier="cold", codec="binary")
        ids, _ = store.search(q, k=5, tier="cold")
    assert ids[0] == 1000 + PLANTED_ROW
    m = Manifest.load(tmp_path / "s.density")
    assert m.tiers["cold"].vector_codec == "binary"


def test_tier_none_prefers_highest_fidelity(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "a.density") as store:
        store.put_embeddings(int_ids(), x, tier="hot")
        store.put_embeddings(int_ids(), x, tier="warm")
        store.put_embeddings(int_ids(), x, tier="cold")
        default_ids, _ = store.search(q, k=5)
        hot_ids, hot_scores = store.search(q, k=5, tier="hot")
    assert np.array_equal(default_ids, hot_ids)
    assert hot_ids[0] == 1000 + PLANTED_ROW
    # Hot scores are exact cosine similarities of unit vectors.
    assert float(hot_scores[0]) == pytest.approx(float(x[PLANTED_ROW] @ q), abs=1e-5)

    with open_store(tmp_path / "b.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        store.put_embeddings(int_ids(), x, tier="cold")
        default_ids, _ = store.search(q, k=5)
        warm_ids, _ = store.search(q, k=5, tier="warm")
    assert np.array_equal(default_ids, warm_ids)


def test_second_put_embeddings_same_tier_rejected(tmp_path):
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        with pytest.raises(StoreError, match="single put_embeddings"):
            store.put_embeddings(int_ids(), x, tier="warm")


def test_put_embeddings_input_validation(tmp_path):
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        with pytest.raises(StoreError, match="ids"):
            store.put_embeddings(int_ids()[:-1], x, tier="warm")
        with pytest.raises(StoreError, match="codec"):
            store.put_embeddings(int_ids(), x, tier="warm", codec="pq")
        store.put_embeddings(int_ids(), x, tier="warm")
        with pytest.raises(StoreError, match="dim"):
            store.put_embeddings(int_ids(), np.ones((N_VECTORS, 16), np.float32), tier="cold")


def test_search_without_vectors_raises(tmp_path):
    with open_store(tmp_path / "s.density") as store:
        with pytest.raises(StoreError, match="no vectors"):
            store.search(np.ones(DIM, np.float32), k=3)


def test_search_query_dim_mismatch_raises(tmp_path):
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        with pytest.raises(StoreError, match="dim"):
            store.search(np.ones(DIM * 2, np.float32), k=3)


def test_text_search_needs_embedder(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        with pytest.raises(StoreError, match="set_embedder"):
            store.search("some text", k=3)
        store.set_embedder(lambda text: q)
        ids, _ = store.search("some text", k=3)
    assert ids[0] == 1000 + PLANTED_ROW


def test_search_cli_json_vector_and_demo_fallback(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        ids, scores = store.search_cli(json.dumps([float(v) for v in q]), k=3)
        assert ids[0] == 1000 + PLANTED_ROW
        assert len(ids) == len(scores) == 3
        with pytest.warns(UserWarning, match="demo-only"):
            text_ids, _ = store.search_cli("planted neighbor text", k=3)
        assert text_ids.shape == (3,)


# -- lifecycle, manifest, determinism ----------------------------------


def test_context_manager_flushes_and_reloads(tmp_path):
    fixture = write_fixture(tmp_path)
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture)
        store.put_embeddings(int_ids(), x, tier="warm")
    # A fresh process would see exactly this on-disk state.
    with open_store(tmp_path / "s.density") as store:
        ids, _ = store.search(q, k=3)
        assert ids[0] == 1000 + PLANTED_ROW
        assert store.replay_raw("t2") == [_L_CRLF + b"\r", _L_TAIL]


def test_closed_store_rejects_operations(tmp_path):
    x, q = make_vectors()
    store = open_store(tmp_path / "s.density")
    store.put_embeddings(int_ids(), x, tier="warm")
    store.close()
    store.close()  # idempotent
    with pytest.raises(StoreError, match="closed"):
        store.search(q, k=1)
    with pytest.raises(StoreError, match="closed"):
        store.put_traces(tmp_path)


def test_checksums_cover_every_referenced_file(tmp_path):
    fixture = write_fixture(tmp_path)
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture)
        store.put_embeddings(int_ids(), x, tier="warm")
    m = Manifest.load(tmp_path / "s.density")
    assert m.verify_checksums(tmp_path / "s.density") == []
    warm = m.tiers["warm"]
    assert warm.vectors_file in m.checksums
    assert warm.codec_state_file in m.checksums
    cold = m.tiers["cold"]
    assert any(rel.startswith(cold.traces_dir) for rel in m.checksums)
    assert warm.bytes.vectors > 0
    assert warm.bytes.total == warm.bytes.traces + warm.bytes.vectors + warm.bytes.vectors_aux


def test_identical_inputs_give_identical_manifests(tmp_path):
    """Same bytes in, same manifest out: checksums included.

    created_at is caller-supplied, so pinning it makes the whole manifest
    byte-comparable, which proves every written artifact is deterministic.
    """
    manifests = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        fixture = write_fixture(root)
        x, _ = make_vectors()
        with open_store(root / "s.density") as store:
            store.put_traces(fixture)
            store.put_embeddings(int_ids(), x, tier="warm")
            store.put_embeddings(int_ids(), x, tier="cold")
        manifests.append((root / "s.density" / "manifest.json").read_bytes())
    assert manifests[0] == manifests[1]


def test_open_rejects_file_path(tmp_path):
    blob = tmp_path / "not-a-dir"
    blob.write_text("x")
    with pytest.raises(StoreError, match="directory"):
        density.open(blob)


# -- adversarial review regressions ------------------------------------


def _jline(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_cold_search_k_beyond_rerank_depth(tmp_path):
    """Regression: cold PQ search crashed for any k > policy rerank depth.

    The rerank candidate pool must widen to at least k, so asking for
    250 of 400 rows (depth 200) returns 250 rows instead of a broadcast
    ValueError, and k > n clamps to n.
    """
    x, q = make_vectors(n=400)
    ids_in = int_ids(400)
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(ids_in, x, tier="warm")
        store.put_embeddings(ids_in, x, tier="cold")
        ids, scores = store.search(q, k=250, tier="cold")
        assert ids.shape == (250,)
        assert scores.shape == (250,)
        assert ids[0] == 1000 + PLANTED_ROW
        assert np.all(np.diff(scores) <= 1e-5)
        ids_all, _ = store.search(q, k=500, tier="cold")
        assert ids_all.shape == (400,)
        assert len(np.unique(ids_all)) == 400


def test_dataset_counters_count_distinct_corpus_once(tmp_path):
    """Regression: warm plus cold of one corpus doubled the dataset ledger.

    Per-call IngestStats stay per call; the dataset totals must describe
    the distinct corpus, or audit compression ratios inflate about 2x.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = [
        _jline({"trace_id": "t1", "ts": i, "role": "user", "type": "message",
                "content": f"c{i}"})
        for i in range(10)
    ]
    (corpus / "a.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    payload = sum(len(line) for line in lines)
    with open_store(tmp_path / "s.density") as store:
        stats_warm = store.put_traces(corpus, tier="warm")
        stats_cold = store.put_traces(corpus, tier="cold")
        # Identical holdings in both tiers replay fine (the fast path).
        assert store.replay_raw("t1") == lines
    assert stats_warm.events == stats_cold.events == 10
    m = Manifest.load(tmp_path / "s.density")
    assert m.datasets.traces.events == 10
    assert m.datasets.traces.malformed == 0
    assert m.datasets.traces.raw_bytes == payload
    assert len(m.datasets.traces.sources) == 1


def test_verify_reports_corrupted_file(tmp_path):
    """Store.verify() re-hashes checksummed files and names the bad ones."""
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        assert store.verify() == []
    codes = tmp_path / "s.density" / "tiers" / "warm" / "vectors" / "codes.npy"
    blob = bytearray(codes.read_bytes())
    blob[200] ^= 0xFF  # flip one code byte well past the npy header
    codes.write_bytes(bytes(blob))
    with open_store(tmp_path / "s.density") as store:
        assert store.verify() == ["tiers/warm/vectors/codes.npy"]


def test_corrupt_trace_blocks_surface_store_error(tmp_path):
    """Regression: zstd corruption leaked ZstdError out of replay_raw."""
    fixture = write_fixture(tmp_path)
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(fixture, tier="warm")
    blocks = tmp_path / "s.density" / "tiers" / "warm" / "traces" / "content.blocks.bin"
    blob = bytearray(blocks.read_bytes())
    blob[0] ^= 0xFF  # break the zstd frame magic: decode must fail
    blocks.write_bytes(bytes(blob))
    with open_store(tmp_path / "s.density") as store:
        with pytest.raises(StoreError, match="verify"):
            store.replay_raw("t1")


def test_replay_split_corpora_raises(tmp_path):
    """Regression: a trace split across tiers was silently truncated.

    put_traces(warm, A) then put_traces(cold, B) is legal; when a
    trace_id exists in both with different lines, replay must refuse
    instead of returning only the warm half.
    """
    a = tmp_path / "corpusA"
    b = tmp_path / "corpusB"
    a.mkdir()
    b.mkdir()
    la = [_jline({"trace_id": "t1", "ts": i, "role": "user", "type": "message",
                  "content": f"A{i}"}) for i in range(3)]
    lb = [_jline({"trace_id": "t1", "ts": 100 + i, "role": "user", "type": "message",
                  "content": f"B{i}"}) for i in range(3)]
    only_b = _jline({"trace_id": "t2", "ts": 200, "role": "user", "type": "message",
                     "content": "only in B"})
    (a / "a.jsonl").write_bytes(b"\n".join(la) + b"\n")
    (b / "b.jsonl").write_bytes(b"\n".join(lb) + b"\n" + only_b + b"\n")
    with open_store(tmp_path / "s.density") as store:
        store.put_traces(a, tier="warm")
        store.put_traces(b, tier="cold")
        with pytest.raises(StoreError, match="different"):
            store.replay_raw("t1")
        # A trace held by a single tier stays replayable.
        assert store.replay_raw("t2") == [only_b]


def test_search_k_below_one_raises(tmp_path):
    """Regression: k=0 returned empty on warm and crashed cold PQ."""
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        store.put_embeddings(int_ids(), x, tier="cold")
        for tier in (None, "warm", "cold"):
            with pytest.raises(StoreError, match="k must be"):
                store.search(q, k=0, tier=tier)
            with pytest.raises(StoreError, match="k must be"):
                store.search(q, k=-3, tier=tier)
        with pytest.raises(StoreError, match="k must be"):
            store.search_cli(json.dumps([float(v) for v in q]), k=0)


def test_search_cli_pathological_json_falls_back_to_text(tmp_path):
    """Regression: '[' * 100000 escaped as RecursionError from json.loads."""
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        with pytest.warns(UserWarning, match="demo-only"):
            ids, scores = store.search_cli("[" * 100000, k=3)
    assert ids.shape == (3,)
    assert scores.shape == (3,)


# -- matryoshka --------------------------------------------------------


def test_matryoshka_warm_tier_records_flag_and_searches(tmp_path):
    x, q = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm", matryoshka_dims=16)
        ids, scores = store.search(q, k=5)
    # The query is truncated into the tier's 16-dim space at search time,
    # where the planted near-duplicate is still by far the closest row.
    assert ids.shape == (5,)
    assert ids[0] == 1000 + PLANTED_ROW
    assert np.all(np.diff(scores) <= 1e-5)
    m = Manifest.load(tmp_path / "s.density")
    assert m.tiers["warm"].matryoshka_dims == 16
    meta = json.loads(
        (tmp_path / "s.density" / "tiers" / "warm" / "vectors" / "state.json")
        .read_text(encoding="utf-8")
    )
    assert meta["dim"] == 16
    # The store dim stays the input dim: full-width queries remain valid.
    assert m.datasets.embeddings.dim == DIM


def test_matryoshka_rejects_bad_dims(tmp_path):
    x, _ = make_vectors()
    with open_store(tmp_path / "s.density") as store:
        with pytest.raises(StoreError, match="matryoshka"):
            store.put_embeddings(int_ids(), x, tier="warm", matryoshka_dims=0)
        with pytest.raises(StoreError, match="matryoshka"):
            store.put_embeddings(int_ids(), x, tier="warm", matryoshka_dims=DIM + 1)


def test_matryoshka_rerank_requires_matching_dims(tmp_path):
    """Cross-tier rerank only applies when both tiers share the truncation.

    A full-width warm reranker cannot score a 16-dim cold query, so the
    cold search must fall back to no-rerank with the honesty warning;
    matching truncations rerank silently.
    """
    import warnings as _warnings

    x, q = make_vectors()
    with open_store(tmp_path / "a.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm")
        store.put_embeddings(int_ids(), x, tier="cold", matryoshka_dims=16)
        with pytest.warns(UserWarning, match="without rerank"):
            ids, _ = store.search(q, k=5, tier="cold")
        assert ids[0] == 1000 + PLANTED_ROW

    with open_store(tmp_path / "b.density") as store:
        store.put_embeddings(int_ids(), x, tier="warm", matryoshka_dims=16)
        store.put_embeddings(int_ids(), x, tier="cold", matryoshka_dims=16)
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any warning here is a failure
            ids, scores = store.search(q, k=5, tier="cold")
        assert ids[0] == 1000 + PLANTED_ROW
        assert np.all(np.diff(scores) <= 1e-5)


# --- regression: read paths must not conjure a store ----------------------


def test_open_with_create_false_refuses_a_missing_store(tmp_path) -> None:
    """A mistyped path used to leave a real but empty store behind."""
    missing = tmp_path / "typo.densty"
    with pytest.raises(StoreError, match="no store at"):
        density.open(missing, create=False)
    assert not missing.exists()


def test_open_with_create_false_opens_an_existing_store(tmp_path) -> None:
    path = tmp_path / "s.density"
    density.open(path, created_at="2024-01-01T00:00:00+00:00").close()
    with density.open(path, create=False) as store:
        assert store.path == path
