"""README benchmark injector tests: byte-exact splicing and the honesty guard.

The script is loaded from scripts/ by path (it is not a package module).
All tests run against a COPY of the repo README plus fake results JSON in
tmp dirs; the real README is never written.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_readme_bench.py"

spec = importlib.util.spec_from_file_location("update_readme_bench", SCRIPT)
urb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(urb)


def fake_payload(recall10: float = 0.9876, mode: str = "quick") -> dict:
    """A minimal results payload in the harness schema (schema_version 1)."""
    def row(codec: str, tier: str, m, depth: int, red: float, r10: float) -> dict:
        return {
            "codec": codec,
            "tier": tier,
            "m": m,
            "rerank_depth": depth,
            "recall": {"recall@1": r10 - 0.01, "recall@10": r10, "recall@100": r10},
            "bytes": {"raw": 1000, "encoded": 250, "aux": 10},
            "bytes_per_vector": 4 * 64 / red,
            "bytes_per_vector_codes_only": 4 * 64 / red,
            "reduction_x": red,
            "reduction_x_codes_only": red,
            "fit_seconds": 0.1,
            "add_seconds": 0.1,
            "search_seconds": 0.123,
        }

    return {
        "schema_version": 1,
        "mode": mode,
        "seed": 1337,
        "accel_active": False,
        "machine": {"platform": "x", "machine": "x86_64", "cpu_model": "Fake CPU",
                    "cpu_count": 8, "python": "3.12.0"},
        "versions": {"density": "0.1.0", "numpy": "2", "pyarrow": "15",
                     "zstandard": "0.22", "pydantic": "2"},
        "elapsed_seconds": {"total": 1.0},
        "vectors": {
            "n_vectors": 5000,
            "n_queries": 200,
            "sample_cap": 200000,
            "dims": [64],
            "corpus": "synth",
            "by_dim": {
                "64": {
                    "dim": 64,
                    "fp32_bytes_per_vector": 256,
                    "n_database": 4800,
                    "sample_size": 5000,
                    "codecs": {
                        "sq8": row("sq8", "warm", None, 0, 4.0, 0.999),
                        "pq_adc": row("pq", "cold", 16, 0, 16.0, 0.85),
                        "pq_rerank": row("pq", "cold", 32, 200, 8.0, recall10),
                        "binary_rerank": row("binary", "cold", None, 500, 32.0, 0.91),
                    },
                }
            },
        },
        "traces": {
            "corpus": {"gb": 0.01, "dim": 64, "n_vectors": 1000, "seed": 1337,
                       "files": 1, "events": 100, "malformed": 2, "residual_lines": 3},
            "raw_file_bytes": 10_000_000,
            "raw_payload_bytes": 9_900_000,
            "cold_bundle_bytes": 700_000,
            "naive_zstd19_bytes": 2_100_000,
            "zstd_level": 19,
            "ratio_vs_raw": 10_000_000 / 700_000,
            "ratio_vs_naive_zstd19": 3.0,
            "shred_seconds": 1.0,
            "naive_seconds": 1.0,
        },
        "accel": {
            "accel_active": False,
            "measure_kernels": {
                "n": 200000, "d": 768, "m": 192, "b": 96, "seed": 1337,
                "accel_active": False, "accel_available": False,
                "kernels": {
                    "sq8_scores": {"fallback_s": 0.5, "compiled_s": None,
                                   "speedup": None, "max_rel_diff": None},
                    "pq_adc_scan": {"fallback_s": 0.5, "compiled_s": None,
                                    "speedup": None, "max_rel_diff": None},
                    "hamming_scan": {"fallback_s": 0.5, "compiled_s": None,
                                     "speedup": None, "max_rel_diff": None},
                },
            },
        },
    }


@pytest.fixture()
def workspace(tmp_path):
    """A copy of the real README plus a results dir holding one fake quick file."""
    readme = tmp_path / "README.md"
    shutil.copyfile(ROOT / "README.md", readme)
    results = tmp_path / "results"
    results.mkdir()
    (results / "bench_quick.json").write_text(
        json.dumps(fake_payload()), encoding="utf-8"
    )
    return readme, results


def _args(readme: Path, results: Path, *extra: str) -> list[str]:
    return ["--readme", str(readme), "--results-dir", str(results), *extra]


def test_inject_replaces_only_between_markers(workspace):
    readme, results = workspace
    before = readme.read_bytes()
    assert urb.main(_args(readme, results)) == 0
    after = readme.read_bytes()

    s, e = urb.START.encode(), urb.END.encode()
    # Everything outside the markers is byte-identical.
    assert after[: after.index(s) + len(s)] == before[: before.index(s) + len(s)]
    assert after[after.index(e):] == before[before.index(e):]
    # And the markers still appear exactly once each.
    assert after.count(s) == 1 and after.count(e) == 1

    between = after[after.index(s) + len(s): after.index(e)].decode("utf-8")
    assert "| WARM / sq8 | 4.0x | 0.9990 | 0 | 0.123 |" in between
    assert "| COLD / pq (m=32) | 8.0x | 0.9876 | 200 | 0.123 |" in between
    assert "| COLD / binary | 32.0x | 0.9100 | 500 | 0.123 |" in between
    assert "Traces: 14.3x vs raw JSONL, 3.00x vs whole-file zstd level 19" in between
    assert "ACCEL_ACTIVE=False" in between
    assert "bench_quick.json" in between


def test_inject_is_idempotent(workspace):
    readme, results = workspace
    assert urb.main(_args(readme, results)) == 0
    first = readme.read_bytes()
    assert urb.main(_args(readme, results)) == 0
    assert readme.read_bytes() == first


def test_check_passes_after_inject(workspace):
    readme, results = workspace
    assert urb.main(_args(readme, results)) == 0
    assert urb.main(_args(readme, results, "--check")) == 0


def test_check_fails_on_stale_readme(workspace):
    # The unmodified scaffold README does not carry the measured table
    # yet, so --check must flag it.
    readme, results = workspace
    assert urb.main(_args(readme, results, "--check")) == 1


def test_check_fails_on_hand_edit(workspace):
    readme, results = workspace
    assert urb.main(_args(readme, results)) == 0
    text = readme.read_bytes().decode("utf-8")
    # A hand-flattered recall number: the exact fraud the guard exists for.
    readme.write_bytes(text.replace("0.9876", "0.9999").encode("utf-8"))
    assert urb.main(_args(readme, results, "--check")) == 1


def test_check_detects_newer_results(workspace):
    readme, results = workspace
    assert urb.main(_args(readme, results)) == 0
    (results / "bench_quick.json").write_text(
        json.dumps(fake_payload(recall10=0.9500)), encoding="utf-8"
    )
    assert urb.main(_args(readme, results, "--check")) == 1


def test_full_results_beat_quick(workspace):
    readme, results = workspace
    (results / "bench_full.json").write_text(
        json.dumps(fake_payload(recall10=0.9611, mode="full")), encoding="utf-8"
    )
    assert urb.main(_args(readme, results)) == 0
    between = readme.read_text(encoding="utf-8")
    assert "0.9611" in between
    assert "bench_full.json" in between
    assert "0.9876" not in between


def test_refuses_without_results(workspace):
    readme, results = workspace
    before = readme.read_bytes()
    (results / "bench_quick.json").unlink()
    assert urb.main(_args(readme, results)) == 2
    assert urb.main(_args(readme, results, "--check")) == 2
    assert readme.read_bytes() == before  # refusal writes nothing


def test_refuses_without_markers(workspace, tmp_path):
    _, results = workspace
    bare = tmp_path / "BARE.md"
    bare.write_bytes(b"# no markers here\n")
    assert urb.main(_args(bare, results)) == 2
    assert bare.read_bytes() == b"# no markers here\n"


def test_repo_readme_has_markers():
    raw = (ROOT / "README.md").read_text(encoding="utf-8")
    assert raw.count(urb.START) == 1
    assert raw.count(urb.END) == 1
    assert raw.index(urb.START) < raw.index(urb.END)
