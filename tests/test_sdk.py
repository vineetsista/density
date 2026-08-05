"""The spec's SDK snippet, run verbatim against a tmp fixture.

Phase 3 gate: the store-related lines of the spec snippet must run
exactly as written. The density.audit line is Phase 4 and is kept as a
skip-marked stub below so the gate stays visible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import density


@pytest.fixture()
def corpus(tmp_path):
    """Hand-written traces plus seeded vectors, no bench.synth.

    Returns (traces_path, ids, vectors, query_vector, trace_id). The
    query is a perturbed copy of row 7, so row 7's id must rank first.
    """
    lines = [
        {"trace_id": "conv-42", "ts": 1_700_000_000_000_000 + i * 1000,
         "role": ["user", "assistant"][i % 2], "type": "message",
         "content": f"turn {i} with unicode é \U0001f680"}
        for i in range(6)
    ]
    payload = b"\n".join(
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for obj in lines
    ) + b"\n"
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_bytes(payload)

    rng = np.random.default_rng(1337)
    vectors = rng.normal(size=(128, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(500, 500 + 128, dtype=np.int64)
    query_vector = vectors[7] + 0.02 * rng.normal(size=32).astype(np.float32)
    query_vector = (query_vector / np.linalg.norm(query_vector)).astype(np.float32)
    return traces_path, ids, vectors, query_vector, "conv-42"


def test_sdk_snippet_runs_verbatim(tmp_path, corpus):
    traces_path, ids, vectors, query_vector, trace_id = corpus

    # The spec snippet, store-related lines, verbatim.
    store = density.open(tmp_path / "store.density")
    store.put_traces(traces_path)
    store.put_embeddings(ids, vectors, tier="warm")
    ids, scores = store.search(query_vector, k=10)
    events = store.replay(trace_id)

    assert ids.shape == (10,)
    assert scores.shape == (10,)
    assert ids[0] == 507
    assert len(events) == 6
    assert events[0]["trace_id"] == "conv-42"
    assert [e["content"] for e in events] == [
        f"turn {i} with unicode é \U0001f680" for i in range(6)
    ]
    store.close()


def test_sdk_audit_snippet(tmp_path):
    """The spec's `rep = density.audit("./raw", out="report.html")` line.

    Phase 4 gate: the snippet must run verbatim against a raw corpus dir
    and leave a self-contained HTML report plus its Markdown sibling.
    """
    raw = tmp_path / "raw"
    (raw / "traces").mkdir(parents=True)
    (raw / "embeddings").mkdir()
    prompt = "You are a support agent. Policies: " + "be kind. " * 40
    lines = []
    for conv in range(20):
        for turn in range(10):
            role = "system" if turn == 0 else ["user", "assistant"][turn % 2]
            content = prompt if turn == 0 else f"conv {conv} turn {turn} body"
            lines.append(
                {
                    "trace_id": f"conv-{conv}",
                    "ts": 1_700_000_000_000_000 + conv * 60_000_000 + turn * 1000,
                    "role": role,
                    "type": "message",
                    "content": content,
                }
            )
    payload = b"\n".join(
        json.dumps(o, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        for o in lines
    ) + b"\n"
    (raw / "traces" / "part-0000.jsonl").write_bytes(payload)
    rng = np.random.default_rng(1337)
    x = rng.standard_normal((2000, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(raw / "embeddings" / "vectors.npy", x)
    np.save(raw / "embeddings" / "ids.npy", np.arange(2000, dtype=np.int64))

    out = tmp_path / "report.html"
    rep = density.audit(str(raw), out=str(out))

    assert out.exists() and out.with_suffix(".md").exists()
    d = rep.to_dict()
    assert d["traces"]["raw_bytes"] > 0 and d["embeddings"]["raw_bytes"] > 0
    assert d["tier_results"] and rep.summary_text()


def test_import_density_is_light():
    """`import density` must work before Phase 4 modules exist."""
    assert density.__version__
    assert callable(density.open)
    assert callable(density.audit)
    assert callable(density.synth)


def test_density_audit_survives_submodule_import():
    """Importing a submodule rebinds density.audit to the package module.

    The package module is deliberately callable, so the SDK surface
    density.audit(...) keeps working regardless of import order.
    """
    import density.audit.pricing  # noqa: F401  (the rebinding trigger)

    assert callable(density.audit)
