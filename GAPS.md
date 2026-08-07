# GAPS

Honest ledger of missed targets and deferred urges. A numeric target that is
missed gets its measured value recorded here, never silently lowered.

Open today:

- COLD bundle vs whole-file zstd-19: **2.0363x measured against a 2.5x gate**
  (Phase 2).
- SQ8 recall@10 on the 100k standard corpus: **0.9877 measured against the
  0.99 WARM floor** (Phase 1).
- The Phase 4 audit wall-clock gate is **not yet measured**: the reference
  machine was not idle. See below.

Scope limits that are not misses but bound where the measured numbers apply
are recorded at the bottom of this file.

## Phase 2: bundle_vs_naive gate, measured 2.04x against a 2.5x target

Measured (1 GB, seed 1337, benchmarks/results/phase2.json): COLD bundle
11.909 MB = structured.parquet 5.390 + content store 5.750 + residual
store 0.768 + extra 0. Naive whole-file zstd-19: 24.251 MB. Ratios:
bundle_vs_raw 58.11x (gate >= 12x, passes), bundle_vs_naive 2.0363x
(gate >= 2.5x, MISSED; the bundle would need to reach 9.70 MB). Full
round-trip byte equality holds on all 684,451 lines.

Floor analysis, measured per component, for structure-blind compression:
the content store is at 35.8x over 205.8 MB of unique payload bytes and
its remaining levers are exhausted (similarity-sorted packing already
bought 37 percent; a single 206 MB frame recovers 0.8 percent; level 22
recovers 0.2 percent). The ref stream sits 13 percent above its
zero-order entropy, trace_id is below its own raw-zstd ceiling, ts sits
10 percent above its empirical inter-arrival entropy. Summing measured
floors, a perfect entropy coder on every column reaches about 2.18x, so
2.5x is unreachable without parsing inside payloads. The honest next
levers are semantic template mining (slot extraction into columns) and
recency-model coding of the ref and trace_id streams: real roadmap work,
not tuning. Disk total including uncompressed index.json metadata is
13.27 MB; compacting that metadata is a worthwhile follow-up outside the
gate accounting.

## Phase 1: sq8 recall@10 gate, measured 0.9877 against a 0.99 target

Measured on the 100k standard corpus (benchmarks/results/phase1.json):
sq8 recall@10 is 0.9877. The rerank gates converge to the same 0.9877
because SQ8 is the reranker: this is SQ8's own top-10 ceiling at this
corpus density, not a PQ or binary defect. Int8 per-dim affine
quantization carries a fixed score-noise floor around 3e-4 cosine; at
100k points in 200 manifold clusters a fraction of neighbor lists still
have top-10 boundary gaps below that noise. The same codec on the
sparser 5,000-vector audit fixture measures 0.9930 and passes the WARM
floor: recall depends on the corpus, which is the product's own thesis.
The audit measures per corpus and recommends the higher tier whenever
the floor is missed. Levers inside the contracted format (per-dim
min/max int8 at exactly 4.0x) are exhausted; percentile-clipped ranges
or per-block scales are roadmap work that would change the format.

## Scope limit: the measured trace ratios assume canonical-form lines

Not a missed gate, but a bound on where the Phase 2 numbers apply, recorded
here because the ratios above are the headline claim.

The shredder stores a line in columnar form only when re-serializing its
canonical event reproduces the original bytes exactly: integer microsecond
`ts`, the canonical field order, compact separators, `ensure_ascii=False`.
Every other line keeps its raw bytes whole in the residual store, which is
correct (replay stays byte-exact) but gives up the two levers that produce
the measured ratio: sha256 interning of repeated payloads, and the
similarity-sorted 64 MB long-range frames that let zstd match near-duplicate
templates against each other.

So a corpus whose timestamps are epoch seconds or ISO strings, or whose
fields use vendor aliases (`conversation_id`, `created_at`, `prompt_tokens`),
or whose key order simply differs, lands on the residual path for every line
and compresses close to plain `zstd -19` rather than 58x. The synth corpus
emits the canonical shape, so the benchmark never exercises this; the audit
reports the measured residual rate per corpus, which is how you tell which
case you are in.

The fix is roadmap work, not tuning: persist a timestamp-unit tag and a
small key-order code so common vendor layouts re-serialize exactly, instead
of the two hardcoded layouts recognized today.

## Phase 4: the audit wall-clock gate is unmeasured, not passed and not missed

`benchmarks/results/` holds phase0, phase1, phase2, and the full bench. There
is no phase4.json, and this entry exists so that absence is not mistaken for
an oversight or for a quiet failure.

Phase 4's gate is the only one in the project that is a wall clock: an audit
of about 1 GB of traces under 300 seconds, and of 1M x 768 vectors under 600
seconds. Every other gate measures a property of the data or the format,
which a busy machine cannot change. A wall clock it can.

The run was attempted on the reference machine (12th Gen i7-1250U, 12 logical
cores) while that machine was also running unrelated workloads: another test
suite at 224 percent CPU, a Next.js dev server, and an ffmpeg encode, for a
one-minute load average near 19. A timing taken under that load measures the
contention, not the software. Recording it would have been worse than
recording nothing: it makes the code look slower than it is while still
carrying the authority of a measured number, which is exactly the failure
this project's first rule exists to prevent. The partial run was stopped and
its output discarded.

`scripts/measure_phase4.py` now samples the load average before and after
each part. Above 0.35 per core it records `timing_valid: false`, reports the
gate as "not measured (machine was not idle)" with the observed load, and
sets `pass` to null rather than to a verdict. Recall, honesty flags, and peak
RSS are still recorded, because those hold regardless of who else is using
the CPU. Run it on an idle machine to close this gap:

    python scripts/measure_phase4.py

What is known from the neighbouring measurements, offered as context and not
as a substitute: at 1 GB on this machine, Phase 2 measured dedup over content
bodies at 555 s and the naive whole-file zstd-19 baseline at 338 s. The audit
does strictly more than either (ingest, a shred per requested tier, dedup,
recall verification, and a round-trip sample). So the 300 second budget looks
unlikely to be met on a low-power mobile CPU of this class, and the honest
expectation is that this gate will be recorded as a miss once it is measured
properly. That sentence is a prediction, not a result, and nothing in the
repository reports it as one.
