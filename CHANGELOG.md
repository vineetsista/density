# Changelog

All notable changes to this project are documented here. This project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

First public release. Everything below is measured on the synthetic corpus
the repo ships; the numbers live in `benchmarks/results/` and the README
table is generated from them.

### Added

- **Vector codecs** (`density.engine.embed`): SQ8 per-dimension affine int8
  quantization, product quantization with anisotropic score-aware training
  (Guo et al. 2020) on top of k-means++ and full-batch Lloyd, sign-bit binary
  codes, exact and SQ8 rerankers, and Matryoshka truncation.
- **Trace engine** (`density.engine.trace`): schema-tolerant JSONL ingest,
  columnar shredding into Parquet plus content-addressed zstd block stores
  with similarity-sorted packing, MinHash LSH near-duplicate clustering, and
  byte-exact replay including malformed lines.
- **Tiers** (`density.tiers`): HOT, WARM, and COLD as measured contracts
  (a recall floor at a footprint target), a checksummed manifest, and a
  store directory that ties traces and vectors together.
- **Recall verifier** (`density.recall`): exact ground truth over held-out
  queries and per-codec recall@1/10/100, the referee behind every recall
  number the project prints.
- **Audit** (`density.audit`): one command that ingests a corpus, builds
  every tier, measures recall, re-verifies round-trip byte equality on a
  sample, prices before and after, and renders a self-contained HTML report
  plus its Markdown sibling.
- **C++20 kernels** (`density.engine._accel`): SQ8 scoring, PQ ADC scan, and
  hamming scan, compiled lazily on first import through pybind11 and cached
  per machine, with a numpy fallback that is always correct and always
  present.
- **Surfaces**: the `density` CLI (synth, audit, compress, search, replay,
  bench, serve), the Python SDK (`density.open`, `density.audit`,
  `density.synth`), and a local FastAPI service.
- **Corpus generator** (`density.bench.synth`): a deterministic, realistic
  agent corpus with concurrent session interleaving, resent system prompts,
  template-plus-slot content, retry storms, malformed lines, and low-rank
  manifold embeddings.
- **Benchmark harness** (`density.bench.harness`) plus
  `scripts/update_readme_bench.py`, which injects the measured table into
  the README and fails CI when the table drifts from the results.
- **Reproducibility**: per-phase gate scripts under `scripts/` that write
  `benchmarks/results/phase*.json`, each recording the machine and the exact
  dependency versions it measured with, and a `benchmarks/measured-versions.txt`
  pinning those versions.
- **CI**: a test matrix on Python 3.11, 3.12, and 3.13; a lint job with a
  prose check that machine-enforces the project's own style rule; the
  benchmark-table honesty guard; and a quickstart smoke test that installs a
  real wheel and runs the published commands against it.

### Known gaps

Two targets are missed and recorded with their measured values in
`GAPS.md`: COLD bundle compression against the whole-file zstd-19 baseline
(2.04x measured, 2.5x targeted) and SQ8 recall@10 on the 100k standard
corpus (0.9877 measured, 0.99 targeted). Neither number is softened
anywhere in the repo.
