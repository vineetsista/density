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
    '{"trace_id": "t1", "ts": 1700000000250000, "role": "assistant", '
    '"type": "message", "content": "non canonical spacing"}'
).encode("utf-8")
_L_CRLF = (
    '{"trace_id":"t2","ts":1700000000500000,"role":"tool",'
    '"type":"tool_result","tool_name":"grep","content":"crlf terminated"}'
).encode("utf-8")
_L_MALFORMED = b'{"trace_id":"t2","ts":170'
_L_TAIL = (
    '{"trace_id":"t2","ts":1700000000750000,"role":"assistant",'
    '"type":"message","content":"tail"}'
).encode("utf-8")

N_EVENTS = 4
N_MALFORMED = 1


def write_fixture(dir_path) -> "object":
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
