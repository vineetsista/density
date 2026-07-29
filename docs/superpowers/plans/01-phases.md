# DENSITY implementation plan, all phases

Contracts live in 00-contracts.md. This file is the execution order, task
breakdown, and gate checklist. One plan file for all phases: the phases are
small enough that splitting files would add navigation, not clarity.

## Phase 0: skeleton and synth

Tasks
1. Scaffold: pyproject, LICENSE, README with BENCH markers, DECISIONS.md,
   GAPS.md, .gitignore, package tree with importable stubs, errors.py.
2. CLI: typer app with all commands registered; unimplemented ones exit
   with a clear message until their phase lands.
3. bench/synth.py per contract, with tests: determinism (same seed, same
   sha256 of outputs), size within 5 percent, malformed rate about 2
   percent, persona mix present, embedding cluster structure real
   (nearest-centroid accuracy far above chance).
4. Env: uv venv, uv pip install -e ".[dev]", pytest green.

Gate (measured, recorded in benchmarks/results/phase0.json)
- Fresh clone: uv venv && uv pip install -e ".[dev]" && pytest green.
- density synth --gb 0.05 completes in under 30 seconds.

## Phase 1: embedding engine and verifier

Tasks
1. engine/_accel/fallback.py (numpy kernels) plus _accel/__init__.py
   dispatch with ACCEL_ACTIVE (compiled path arrives in Phase 5).
2. SQ8, PQ, BinaryCodec, rerank, matryoshka per contract, TDD.
3. recall/metrics.py, recall/verifier.py.
4. Property tests per contract.
5. Bench script measures the Phase 1 gates on the standard corpus
   (100k vectors, dim 768, 1k held-out queries, seed 1337) and writes
   benchmarks/results/phase1.json.

Gate (all measured on the standard corpus)
- sq8: 4.0x vs fp32, recall@10 >= 0.99.
- pq standalone ADC recall@10 >= 0.80 (default m = d/4).
- pq + rerank(200 via sq8) recall@10 >= 0.95 at >= 16x total including
  codebook (measured at m = 128 for dim 768, see DECISIONS.md item 8).
- binary + rerank(500 via sq8) recall@10 >= 0.90 at 32x codes.

## Phase 2: trace engine

Tasks
1. ingest/schemas.py, ingest/traces.py, ingest/embeddings.py.
2. engine/trace/zdict.py, shred.py, dedup.py per contract, TDD.
3. Lossless round-trip proof: full byte equality over the whole synth
   corpus, in tests (small corpus, 100 percent) and in audit (1 percent
   sample of the real input).
4. Torture tests: truncated file, binary garbage file, 10 MB single line,
   empty file, file of only newlines. No crash, counts reported.
5. Gate measurement script writes benchmarks/results/phase2.json.

Gate
- Compression >= 12x vs raw JSONL on the synth corpus (COLD settings).
- Compression >= 2.5x vs whole-file zstd -19 on the same bytes.
- Round-trip byte equality 100 percent. Torture tests pass.

## Phase 3: tiers, store, SDK

Tasks
1. tiers/policy.py, manifest.py, store.py per contract.
2. density.open / audit / synth public API in __init__.py.
3. embedders.py demo hashing embedder, labeled demo-only.
4. Doctest for the exact SDK snippet from the spec.

Gate: SDK snippet runs verbatim as a doctest. Store round-trips
put_traces -> replay byte-exact and put_embeddings -> search sanely.

## Phase 4: audit runner and report

Tasks
1. audit/pricing.py, runner.py, report.py, templates/report.html.j2.
2. Streaming and memory bounds: peak RSS tracked in the result.
3. Report content per spec order, honesty section, methodology footnote.

Gate: density synth --gb 1 then density audit under 5 minutes for traces
and under 10 minutes for 1M x 768 vectors, laptop CPU. Report readable,
every number traceable. benchmarks/results/phase4.json records timings.

## Phase 5: bench, service, accel, README

Tasks
1. bench/harness.py full matrix, benchmarks/results/*.json.
2. scripts/update_readme_bench.py injects the table between
   BENCH:START and BENCH:END markers. Never hand-typed.
3. service/api.py FastAPI, tests via TestClient.
4. _accel/cpp kernels + pybind11 build-on-import, parity tests, 5x ADC
   speedup measurement on 1M PQ codes.
5. README quickstart, Roadmap (KV-cache compression: not in v1).

Gate: full MISSION terminal session from a fresh clone, real output
pasted into the final report. Accel speedup measured and recorded with
ACCEL_ACTIVE printed in every benchmark result.

## Verification before completion, every phase

Run: pytest (full suite), the phase gate script, and the demo commands of
all completed phases. Paste real measured output. Misses go to GAPS.md
with the measured number, never softened.
