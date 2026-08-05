# GAPS

Honest ledger of missed targets and deferred urges. A numeric target that is
missed gets its measured value recorded here, never silently lowered.

No known gaps yet.

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
