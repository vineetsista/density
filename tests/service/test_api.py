"""Service endpoint tests: TestClient over a tiny hand-built store.

No sockets: fastapi.testclient drives the ASGI app in-process. The store
fixture is deliberately hand-built (density.open, a small JSONL, seeded
dim-32 vectors) so every expected value is visible in this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import density
from density.service.api import create_app

SEED = 1337
DIM = 32
N_VECTORS = 64
N_TRACES = 3
EVENTS_PER_TRACE = 4


def _make_lines() -> list[dict]:
    """Canonically serialized events across a few traces."""
    lines = []
    for i in range(N_TRACES * EVENTS_PER_TRACE):
        lines.append(
            {
                "trace_id": f"tr-{i % N_TRACES:02d}",
                "ts": 1_700_000_000_000_000 + i * 1_000,
                "role": ["user", "assistant"][i % 2],
                "type": "message",
                "content": f"turn {i} with unicode é \U0001f680",
            }
        )
    return lines


@pytest.fixture(scope="module")
def store_and_lines(tmp_path_factory):
    """One store holding warm traces and warm sq8 vectors, plus the raw lines."""
    tmp = tmp_path_factory.mktemp("service-store")
    lines = _make_lines()
    payload = b"\n".join(
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for obj in lines
    ) + b"\n"
    traces_path = tmp / "traces.jsonl"
    traces_path.write_bytes(payload)

    rng = np.random.default_rng(SEED)
    vectors = rng.normal(size=(N_VECTORS, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(N_VECTORS, dtype=np.int64)

    store = density.open(tmp / "store.density", seed=SEED, created_at="2026-01-01T00:00:00+00:00")
    store.put_traces(traces_path, tier="warm")
    store.put_embeddings(ids, vectors, tier="warm")
    yield store, lines, vectors
    store.close()


@pytest.fixture(scope="module")
def client(store_and_lines):
    store, _, _ = store_and_lines
    return TestClient(create_app(store))


@pytest.fixture(scope="module")
def raw_corpus(tmp_path_factory):
    """A tiny raw corpus directory (traces + embeddings) for /audit.

    256 vectors keep the audit's recall verification alive (queries are
    split off and at least 100 database rows must remain), while dim 16
    keeps the PQ fit fast.
    """
    root = tmp_path_factory.mktemp("service-raw") / "raw"
    traces = root / "traces"
    traces.mkdir(parents=True)
    lines = [
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for obj in _make_lines()
    ]
    (traces / "part-0000.jsonl").write_bytes(b"\n".join(lines) + b"\n")
    emb = root / "embeddings"
    emb.mkdir()
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(256, 16)).astype(np.float32)
    np.save(emb / "vectors.npy", x)
    return root


# -- /search ----------------------------------------------------------------


def test_search_vector_query(client, store_and_lines):
    _, _, vectors = store_and_lines
    q = json.dumps([float(v) for v in vectors[5]])
    resp = client.get("/search", params={"q": q, "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"ids", "scores"}
    assert len(body["ids"]) == 5 and len(body["scores"]) == 5
    # Searching a stored vector must return itself first: sq8 error is
    # far below the gap between self-similarity and random neighbors.
    assert body["ids"][0] == 5
    assert body["scores"] == sorted(body["scores"], reverse=True)
    assert all(isinstance(i, int) for i in body["ids"])


def test_search_text_query_uses_demo_embedder(client):
    # Free text is not a JSON vector, so search_cli falls back to the
    # demo HashingEmbedder (with its one-time warning) and still returns
    # k results. The endpoint itself does no parsing.
    resp = client.get("/search", params={"q": "refund window annual plan", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["ids"]) == 3
    assert len(body["scores"]) == 3


def test_search_explicit_tier(client, store_and_lines):
    _, _, vectors = store_and_lines
    q = json.dumps([float(v) for v in vectors[9]])
    resp = client.get("/search", params={"q": q, "k": 2, "tier": "warm"})
    assert resp.status_code == 200
    assert resp.json()["ids"][0] == 9


def test_search_bad_k_is_400(client):
    resp = client.get("/search", params={"q": "[1, 2]", "k": 0})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "StoreError"
    assert "k must be at least 1" in err["message"]


def test_search_unknown_tier_is_400(client):
    resp = client.get("/search", params={"q": "hello", "k": 1, "tier": "lukewarm"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "StoreError"


def test_search_empty_tier_without_vectors_is_400(client):
    # The store holds warm vectors only; forcing hot must fail cleanly.
    resp = client.get("/search", params={"q": "[1.0]", "k": 1, "tier": "hot"})
    assert resp.status_code == 400
    assert "holds no vectors" in resp.json()["error"]["message"]


def test_search_non_integer_k_is_400(client):
    resp = client.get("/search", params={"q": "hello", "k": "many"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "ValidationError"


# -- /replay ----------------------------------------------------------------


def test_replay_known_trace(client, store_and_lines):
    _, lines, _ = store_and_lines
    resp = client.get("/replay/tr-01")
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == "tr-01"
    expected = [obj for obj in lines if obj["trace_id"] == "tr-01"]
    assert body["events"] == expected


def test_replay_unknown_trace_is_404(client):
    resp = client.get("/replay/tr-nope")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["type"] == "StoreError"
    assert "tr-nope" in err["message"]
    # Clean JSON, no traceback text.
    assert "Traceback" not in resp.text


def test_replay_no_traces_is_404(tmp_path):
    # A store with vectors but no traces: replay cannot find anything,
    # which is a 404 (nothing exists), not a server error.
    store = density.open(tmp_path / "empty.density", seed=SEED)
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(8, 4)).astype(np.float32)
    store.put_embeddings(np.arange(8), x, tier="hot")
    c = TestClient(create_app(store))
    resp = c.get("/replay/anything")
    assert resp.status_code == 404
    store.close()


# -- /audit -----------------------------------------------------------------


def test_audit_endpoint_runs_and_writes_report(client, raw_corpus):
    resp = client.post("/audit", json={"path": str(raw_corpus)})
    assert resp.status_code == 200
    body = resp.json()
    # The report lands next to the corpus path, never inside it.
    report_html = Path(body["report_html"])
    report_md = Path(body["report_md"])
    assert report_html == raw_corpus.with_name(raw_corpus.name + ".report.html")
    assert report_html.exists() and report_md.exists()
    audit = body["audit"]
    assert audit["tiers"] == ["warm", "cold"]
    assert set(audit["tier_results"]) == {"warm", "cold"}
    for tier in ("warm", "cold"):
        t = audit["tier_results"][tier]
        assert t["total_bytes"] > 0
        assert t["recall"] is not None
        assert 0.0 <= t["recall"]["recall@10"] <= 1.0


def test_audit_endpoint_tier_subset(client, raw_corpus):
    resp = client.post("/audit", json={"path": str(raw_corpus), "tiers": ["warm"]})
    assert resp.status_code == 200
    assert set(resp.json()["audit"]["tier_results"]) == {"warm"}


def test_audit_missing_path_is_400(client, tmp_path):
    resp = client.post("/audit", json={"path": str(tmp_path / "nope")})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["type"] == "AuditError"
    assert "does not exist" in err["message"]


def test_audit_bad_tier_is_400(client, raw_corpus):
    resp = client.post("/audit", json={"path": str(raw_corpus), "tiers": ["glacial"]})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "AuditError"


def test_audit_invalid_body_is_400(client):
    resp = client.post("/audit", json={"tiers": ["warm"]})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "ValidationError"


# --- regression: /audit must stay inside the served root ------------------


def test_audit_rejects_a_path_outside_the_configured_root(tmp_path) -> None:
    """The service has no auth, so /audit must not be a filesystem primitive."""
    store_dir = tmp_path / "served" / "s.density"
    store_dir.parent.mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with density.open(store_dir, created_at="2024-01-01T00:00:00+00:00") as store:
        client = TestClient(create_app(store, audit_root=store_dir.parent))
        resp = client.post("/audit", json={"path": str(outside)})
        assert resp.status_code == 400
        assert "outside the served root" in resp.json()["error"]["message"]
        # A traversal attempt resolves before the prefix check.
        resp = client.post("/audit", json={"path": str(store_dir / ".." / ".." / "x")})
        assert resp.status_code == 400
