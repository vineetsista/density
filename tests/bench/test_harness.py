"""Quick-mode smoke test of the bench harness: schema and sanity.

run_bench(quick=True) is executed once (module-scoped fixture, allow up
to about two minutes) and every test asserts against the JSON it wrote.
Wall-clock numbers are only checked for sign; measured recalls and byte
counts are checked for range and internal consistency.
"""

from __future__ import annotations

import json

import pytest

from density.bench.harness import SCHEMA_VERSION, run_bench
from density.errors import DensityError

SEED = 1337

QUICK_CONFIGS = {"sq8", "pq_adc", "pq_rerank", "binary_rerank"}


@pytest.fixture(scope="module")
def bench_payload(tmp_path_factory):
    """One quick bench run; returns (path, parsed payload)."""
    tmp = tmp_path_factory.mktemp("bench")
    path = run_bench(tmp / "out", quick=True, seed=SEED, corpus_cache=tmp / "cache")
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_writes_named_file(bench_payload):
    path, _ = bench_payload
    assert path.name == "bench_quick.json"
    assert path.is_file()


def test_top_level_schema(bench_payload):
    _, p = bench_payload
    assert p["schema_version"] == SCHEMA_VERSION
    assert p["mode"] == "quick"
    assert p["seed"] == SEED
    assert isinstance(p["accel_active"], bool)
    assert set(p["machine"]) == {"platform", "machine", "cpu_model", "cpu_count", "python"}
    assert set(p["versions"]) == {"density", "numpy", "pyarrow", "zstandard", "pydantic"}
    for key in ("vectors", "traces", "accel", "elapsed_seconds"):
        assert key in p
    assert p["elapsed_seconds"]["total"] > 0


def test_vector_matrix(bench_payload):
    _, p = bench_payload
    v = p["vectors"]
    assert v["n_vectors"] == 5_000
    assert v["n_queries"] == 200
    assert v["dims"] == [64]
    block = v["by_dim"]["64"]
    assert block["fp32_bytes_per_vector"] == 64 * 4
    assert block["n_database"] == 5_000 - 200
    assert set(block["codecs"]) == QUICK_CONFIGS
    for name, row in block["codecs"].items():
        for k in ("recall@1", "recall@10", "recall@100"):
            assert 0.0 <= row["recall"][k] <= 1.0, (name, k)
        assert row["bytes"]["encoded"] > 0
        assert row["bytes"]["raw"] > 0
        assert row["bytes_per_vector"] > 0
        assert row["reduction_x"] > 1.0
        assert row["fit_seconds"] >= 0
        assert row["add_seconds"] >= 0
        assert row["search_seconds"] > 0


def test_vector_matrix_config_shape(bench_payload):
    _, p = bench_payload
    codecs = p["vectors"]["by_dim"]["64"]["codecs"]
    # sq8 codes are exactly one byte per dim: 4.0x codes-only, no more.
    assert codecs["sq8"]["reduction_x_codes_only"] == pytest.approx(4.0)
    assert codecs["sq8"]["tier"] == "warm"
    assert codecs["sq8"]["rerank_depth"] == 0
    # pq ADC runs at the default m = d / 4; the rerank config scales the
    # m = 128 contract point down to d / 2 at this tiny dim.
    assert codecs["pq_adc"]["m"] == 16
    assert codecs["pq_adc"]["rerank_depth"] == 0
    assert codecs["pq_rerank"]["m"] == 32
    assert codecs["pq_rerank"]["rerank_depth"] == 200
    assert codecs["binary_rerank"]["rerank_depth"] == 500
    assert codecs["binary_rerank"]["codec"] == "binary"
    # binary codes are d / 8 bytes: 32x codes-only.
    assert codecs["binary_rerank"]["reduction_x_codes_only"] == pytest.approx(32.0)
    # Rerank must not hurt pq: the reranked config sees candidates the
    # plain ADC ranking would have kept anyway.
    assert (
        codecs["pq_rerank"]["recall"]["recall@10"]
        >= codecs["pq_adc"]["recall"]["recall@10"] - 0.05
    )


def test_trace_section(bench_payload):
    _, p = bench_payload
    t = p["traces"]
    assert t["corpus"]["gb"] == 0.01
    assert t["corpus"]["events"] > 0
    assert t["corpus"]["files"] > 0
    assert t["raw_file_bytes"] > 0
    assert t["raw_payload_bytes"] > 0
    assert t["cold_bundle_bytes"] > 0
    assert t["naive_zstd19_bytes"] > 0
    assert t["zstd_level"] == 19
    # The bundle must actually compress, and the ratios must be the
    # quotients of the recorded byte counts, not independent numbers.
    assert t["ratio_vs_raw"] > 1.0
    assert t["ratio_vs_raw"] == pytest.approx(t["raw_file_bytes"] / t["cold_bundle_bytes"])
    assert t["ratio_vs_naive_zstd19"] == pytest.approx(
        t["naive_zstd19_bytes"] / t["cold_bundle_bytes"]
    )


def test_accel_section(bench_payload):
    _, p = bench_payload
    a = p["accel"]
    assert isinstance(a["accel_active"], bool)
    assert a["accel_active"] == p["accel_active"]
    mk = a["measure_kernels"]
    assert mk["n"] == 200_000
    assert set(mk["kernels"]) == {"sq8_scores", "pq_adc_scan", "hamming_scan"}
    for entry in mk["kernels"].values():
        assert entry["fallback_s"] > 0
        # compiled_s and speedup are None when no compiled path loaded;
        # when present they must be positive.
        if entry["speedup"] is not None:
            assert entry["speedup"] > 0
            assert entry["compiled_s"] > 0


def test_corpus_cache_reused(bench_payload, tmp_path_factory):
    # The synth trace corpus is cached: the stamp file records the
    # parameters and byte count that make reuse safe.
    path, p = bench_payload
    cache = path.parent.parent / "cache"
    stamps = list(cache.glob("synth-*/stamp.json"))
    assert len(stamps) == 1
    meta = json.loads(stamps[0].read_text(encoding="utf-8"))
    assert meta["params"]["gb"] == 0.01
    assert meta["params"]["seed"] == SEED
    assert meta["bytes_traces"] > 0


def test_out_dir_must_be_a_directory(tmp_path):
    clash = tmp_path / "not-a-dir"
    clash.write_text("occupied")
    with pytest.raises(DensityError, match="not a directory"):
        run_bench(clash, quick=True, seed=SEED, corpus_cache=tmp_path / "cache")
