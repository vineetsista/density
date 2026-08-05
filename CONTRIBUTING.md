# Contributing to DENSITY

Thanks for looking. This project has a small number of rules that are not
negotiable, because they are what the product claims about itself.

## The rules that matter

1. **No measured number is ever typed by hand.** Every figure in the README,
   the audit report, and `benchmarks/results/*.json` is produced by code that
   measured it. `python scripts/update_readme_bench.py --check` runs in CI and
   fails the build when the README table drifts from the newest results file.
2. **A missed target is recorded, never lowered.** If a gate misses, the
   measured value goes into `GAPS.md` with the analysis of why, and the target
   stays where it was. `GAPS.md` is the honest ledger; two targets are open in
   it today.
3. **Every open design choice gets one line in `DECISIONS.md`.** Newest at the
   bottom. If an implementation deviates from `docs/design/00-contracts.md`,
   both files change in the same commit.
4. **No network calls in the core library.** Not in ingest, not in the engine,
   not in the audit. The only socket in the repo is the one `density serve`
   binds, and that is the caller's explicit choice.
5. **Determinism.** Every stochastic step takes `seed: int = 1337` and uses
   `numpy.random.default_rng(seed)` or a `SeedSequence` child. Same seed, same
   machine, bit-identical output. Tests seed explicitly and never read the
   clock for logic; wall-clock assertions are allowed only in bench code.
6. **Byte-exactness.** `replay` returns the original bytes. Any change to the
   trace engine must keep the round-trip proof green, including for malformed
   lines, CRLF, lone surrogates, and non-canonical JSON spacing.
7. **No em dashes and no en dashes** in prose, docstrings, comments, or
   templates. Commas, colons, and periods do the job.
8. **Comments explain why, not what.** Type hints on public functions.
   Docstrings state units and shapes.
9. **Do not overstate.** If a docstring claims a bound, the code must hold it.
   Two docstrings in this repo already say where a stage is *not* bounded,
   which is the right kind of sentence to write.

## Setup

```bash
git clone https://github.com/vineetsista/density
cd density
uv venv && source .venv/bin/activate     # or: python -m venv .venv
uv pip install -e ".[dev]"               # or: pip install -e ".[dev]"
pytest -q
ruff check .
python scripts/check_prose.py
```

The `dev` extra pulls in `accel` and `service`, so one install covers the whole
suite. The `accel` extra only adds `pybind11`. The C++20 kernels compile
lazily on first import and cache under your user cache directory. If no
compiler is present the import falls back to the numpy kernels silently and
`density.ACCEL_ACTIVE` is `False`. Set `DENSITY_ACCEL_DEBUG=1` to see why a
build was skipped, or `DENSITY_ACCEL_DISABLE=1` to force the numpy path.

## Tests

`pytest -q` runs the whole suite and `ruff check .` runs the linter, both of
which CI enforces, along with `python scripts/check_prose.py`, which fails on
em and en dashes. Tests mirror the source tree
(`tests/engine/embed/test_sq8.py` for `src/density/engine/embed/sq8.py`).
Property tests use hypothesis: SQ8 reconstruction error bounds, ADC rank
correlation, hamming symmetry, dedup normalization idempotence, and shred
round-trip byte equality.

## Benchmarks and gates

```bash
density bench --quick                    # smoke configuration
density bench                            # the public matrix, tens of minutes
python scripts/update_readme_bench.py    # inject the table into the README
python scripts/measure_phase0.py         # per-phase gate scripts
python scripts/measure_phase1.py
python scripts/measure_phase2.py
python scripts/measure_phase4.py
```

Gate scripts write `benchmarks/results/phase*.json`. They are slow and
memory-hungry by design (Phase 4 part B holds 1M x 768 float32, about 3.1 GB)
and they record failures, including out-of-memory, rather than hiding them.

## Commits

Conventional commits: `feat(scope): ...`, `fix(scope): ...`,
`perf(scope): ...`, `test: ...`, `chore: ...`. One logical change per commit.

## Filing an issue

Include the output of:

```bash
python -c "import density, numpy, pyarrow, zstandard, sys; print(density.__version__, density.ACCEL_ACTIVE, numpy.__version__, pyarrow.__version__, zstandard.__version__, sys.version)"
```

plus the seed and the exact command. Because everything is seeded, a report
with a seed is usually a report that reproduces. For anything security
related, use [SECURITY.md](SECURITY.md) instead of a public issue.
