# DENSITY interface contracts

This file is the single source of truth for cross-module interfaces. Every
implementation agent reads this before writing code. If an implementation
needs to deviate, it updates this file and DECISIONS.md in the same commit.

## Global conventions

- Python 3.11+, numpy, pyarrow, zstandard, pydantic v2, typer, jinja2.
- Every stochastic step accepts `seed: int = 1337` and uses
  `numpy.random.default_rng(seed)` (or `SeedSequence` children). Same seed,
  same machine: bit-identical outputs.
- No network calls anywhere in the core library. No exceptions.
- All vectors are float32 and L2-normalized at ingest. Cosine similarity is
  therefore a plain dot product everywhere. Zero vectors are left as zeros.
- All on-disk binary is little-endian. All JSON is UTF-8.
- No em dashes in any prose, docstring, comment, or template. Use commas,
  colons, periods.
- Style: type hints on public functions, docstrings state units and shapes.
  Comments explain why, not what.
- Errors on user input never crash a pipeline: malformed input is counted,
  quarantined, and reported.
- Public exceptions: `density.errors.DensityError` base, with
  `IngestError`, `StoreError`, `AuditError` subclasses (src/density/errors.py).

## ingest/schemas.py

```python
class TraceEvent(pydantic.BaseModel):
    trace_id: str
    ts: int              # microseconds since epoch, int64 range
    role: str            # "system" | "user" | "assistant" | "tool" | other
    type: str            # "message" | "tool_call" | "tool_result" | other
    content: str         # main text body, may be empty
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_name: str | None = None
    extra: dict = {}     # every unknown input field, preserved verbatim
```

- `TraceEvent.from_raw(obj: dict) -> TraceEvent` canonicalizes a parsed JSON
  object. Recognized aliases (case-insensitive): trace_id from
  `trace_id | traceId | conversation_id | session_id | run_id`; ts from
  `ts | timestamp | time | created_at` (ISO 8601 strings, epoch seconds,
  millis, or micros all accepted, normalized to int micros); role from
  `role | speaker | author`; type from `type | event_type | kind`; content
  from `content | text | message | output` (non-string content is JSON-dumped
  compactly); model from `model | model_name`; tokens from
  `tokens_in | prompt_tokens | input_tokens` and
  `tokens_out | completion_tokens | output_tokens`; tool_name from
  `tool_name | tool | name` (the `name` alias applies only when type
  indicates a tool event). Every input key that was not consumed lands in
  `extra` unchanged. Missing trace_id becomes `"unknown"`, missing ts becomes
  0, missing role/type become `"unknown"`, missing content becomes `""`.

## ingest/traces.py

```python
@dataclass
class IngestStats:
    files: int
    events: int
    malformed: int
    bytes_read: int

def iter_raw_lines(path) -> Iterator[tuple[str, int, bytes]]
    # yields (file_relpath, line_index, raw_line_bytes_without_newline)
    # for every physical line in every *.jsonl / *.jsonl.* file under path
    # (or the single file if path is a file). Streaming, never loads a file
    # fully. Tolerates missing trailing newline and CRLF (CR is part of the
    # payload and must be preserved for byte-exact replay).

def iter_events(path, on_malformed="quarantine") -> Iterator[ParsedLine]
    # ParsedLine = dataclass(file, line_index, raw: bytes,
    #                        event: TraceEvent | None, error: str | None)
    # event is None for malformed lines (invalid JSON or non-object).
```

## ingest/embeddings.py

```python
def read_embeddings(path, expected_dim=None) -> EmbeddingSet
    # EmbeddingSet = dataclass(ids: np.ndarray[str or int64], X: float32 [n, d],
    #                          source: str, normalized: bool)
    # Accepts: .npy (2-D float array), .parquet (columns: id + fixed-size list
    #          or per-dim floats or binary blob), .jsonl ({"id": ..., "vector"
    #          or "embedding": [...]}), or a directory containing any mix.
    # Sniffs dim from data; validates all rows agree; L2-normalizes; ids
    # default to 0..n-1 when the format has none.
```

## engine/embed: shared codec interface

All vector codecs implement this informal protocol (defined as
`VectorCodec` Protocol in `engine/embed/__init__.py`):

```python
class VectorCodec(Protocol):
    name: str                    # "sq8" | "pq" | "binary"
    def fit(self, X: np.ndarray, seed: int = 1337) -> "VectorCodec"
    def encode(self, X: np.ndarray) -> np.ndarray      # codes, dtype/shape per codec
    def decode(self, codes: np.ndarray) -> np.ndarray  # float32 approximation
    def search(self, q: np.ndarray, k: int = 10,
               rerank_depth: int = 0,
               reranker: "Reranker | None" = None) -> tuple[np.ndarray, np.ndarray]
        # q: [d] or [nq, d] float32. Returns (ids int64 [nq, k], scores
        # float32 [nq, k]) sorted best-first. Ids are row indices into the
        # X passed to the most recent add()/fit-encode. rerank_depth == 0
        # means no rerank. rerank_depth > 0 requires a reranker.
    def add(self, X: np.ndarray) -> None               # encode and retain codes
    def encoded_nbytes(self) -> int                    # bytes of stored codes
    def aux_nbytes(self) -> int                        # codebooks, scales, etc.
    def to_state(self) -> dict[str, np.ndarray]        # for manifest persistence
    @classmethod
    def from_state(cls, state) -> "VectorCodec"
```

- sq8 (engine/embed/sq8.py): class `SQ8`. Per-dimension min/max affine
  quantization to uint8. codes: uint8 [n, d]. Scoring is exact int8-domain
  dot against the affine reconstruction: precompute `qa = q * scale` and
  `const = q @ mins`, score = codes @ qa + const, via the accel kernel.
  4.0x reduction vs fp32 (scales/mins are aux bytes).
- pq (engine/embed/pq.py): class `PQ(m=None, nbits=8)`, m defaults to d // 4,
  m must divide d. Trained with seeded mini-batch k-means per subspace
  (256 centroids, at least 8 epochs over a capped sample). codes: uint8
  [n, m]. Search builds an ADC lookup table lut float32 [m, 256] per query
  and scans via the accel kernel.
- binary (engine/embed/binary.py): class `BinaryCodec`. sign(x) packed to
  bits, codes: uint8 [n, d // 8]. Search is hamming distance via popcount
  kernel, scores returned as negative hamming so that higher is better.
- rerank (engine/embed/rerank.py):
  ```python
  class Reranker(Protocol):
      def score(self, q: np.ndarray, ids: np.ndarray) -> np.ndarray
  class SQ8Reranker:   # wraps a fitted SQ8 with stored codes
  class ExactReranker: # wraps a float32 matrix (HOT)
  def rerank_topk(q, candidate_ids, reranker, k) -> (ids, scores)
  ```
  Codec.search with rerank_depth=N takes its own top-N candidates and
  returns the top-k after rescoring with the reranker.
- matryoshka (engine/embed/matryoshka.py): `truncate(X, dims)` slices the
  first `dims` dimensions and re-normalizes. Off by default everywhere,
  exposed as an optional flag on audit and store APIs.

## engine/_accel

Three kernel functions, identical signatures in compiled and fallback paths:

```python
def sq8_scores(codes: np.uint8[n, d], qa: np.float32[d], const: float) -> np.float32[n]
def pq_adc_scan(codes: np.uint8[n, m], lut: np.float32[m, 256]) -> np.float32[n]
def hamming_scan(codes: np.uint8[n, b], q: np.uint8[b]) -> np.int32[n]
```

`density.engine._accel` exposes these plus `ACCEL_ACTIVE: bool` (True only
when the compiled path loaded), re-exported as `density.ACCEL_ACTIVE`.
Compiled and fallback results agree within 1e-5 relative.

## engine/trace

- shred.py: `shred_events(parsed_iter, out_dir, level, seed) -> ShredResult`
  writes a columnar bundle:
  - `structured.parquet`: trace_id, ts (delta-encoded int64 micros), role,
    type, model, tool_name (dictionary-encoded), tokens_in, tokens_out,
    file, line_index, content_ref, extra_ref, residual_ref.
  - text stores (content, extra, residual) as zstd blocks, see zdict.
  - Byte-exactness contract: at shred time each line is re-serialized from
    its canonical event with `json.dumps(obj, separators=(",", ":"),
    ensure_ascii=False)` preserving original key order; if the bytes match
    the original line, residual_ref is null; otherwise the exact original
    bytes go to the residual store and win at replay time. Malformed lines
    always go to the residual store. `unshred(dir) -> Iterator[(file,
    line_index, raw_bytes)]` reproduces every original line byte-for-byte.
- zdict.py:
  ```python
  def train_dict(samples: list[bytes], dict_size=110_000, sample_cap_bytes=100_000_000) -> bytes
  class BlockWriter:  # append bytes items into ~4 MB zstd frames with a
                      # shared dictionary; returns (block_id, offset, length)
                      # refs; writes blocks.bin + offsets
  class BlockReader:  # random access by ref
  # levels: COLD=19, WARM=10 (module constants LEVEL_COLD, LEVEL_WARM)
  ```
- dedup.py:
  ```python
  def normalize(text: str) -> str   # lowercase, collapse whitespace,
                                    # strip uuids and iso timestamps by regex
  class MinHashLSH:                 # 128 permutations, jaccard threshold 0.9,
                                    # over 5-gram shingles of normalize(text)
  def find_clusters(texts_iter) -> DedupResult
      # DedupResult: clusters (list of member lists), exact_dup_groups
      # (byte-identical groups, these actually share storage), stats.
  ```
  Byte-identical bodies are stored once (interned by content sha256).
  Near-duplicates (normalized-equal but byte-different) are clustered for
  reporting but stored in full, because replay is byte-exact.

## tiers

- policy.py:
  ```python
  class Tier(str, Enum): HOT = "hot"; WARM = "warm"; COLD = "cold"
  @dataclass(frozen=True)
  class TierSpec:
      tier: Tier
      vector_codec: str | None      # None (fp32) | "sq8" | "pq" | "binary"
      trace_zstd_level: int | None  # None = raw
      recall10_floor: float | None  # measured guarantee, None for HOT (exact)
      footprint_target: float       # fraction of original bytes
  TIER_SPECS: dict[Tier, TierSpec]
  # HOT: fp32 + raw traces, exact, 1.0
  # WARM: sq8 + level-10 columnar zstd, 0.99, 0.25
  # COLD: pq (default, binary optional) + level-19 dict zstd + dedup,
  #       0.95 pq / 0.90 binary with rerank, 0.08
  ```
- manifest.py: pydantic model `Manifest` with `format_version`, `created_at`
  (caller-supplied, deterministic in tests), `seed`, `datasets` (traces:
  files, events, malformed, raw bytes; embeddings: count, dim, source),
  `tiers` (per tier: paths, codec state file, bytes accounting), `dedup`
  (cluster count, exact dup groups, bytes saved, top clusters with samples
  truncated to 120 chars), `checksums` (sha256 of every referenced file),
  `versions` (density, numpy, pyarrow, zstandard). `save(dir)`, `load(dir)`;
  writes are atomic (tmp file, rename).
- store.py: class `Store`, opened by `density.open(path)`.
  ```python
  density.open(path) -> Store        # creates if missing (path.endswith
                                     # convention not required, any dir)
  Store.put_traces(jsonl_path, tier=Tier.COLD) -> IngestStats
  Store.put_embeddings(ids, vectors, tier="warm") -> None
  Store.search(query, k=10, tier=None) -> (ids, scores)
      # query: np.ndarray vector, or str if an embedder callable was
      # configured via Store.set_embedder(fn); tier=None picks the best
      # (highest-fidelity) vector tier present. Returns original ids.
  Store.replay(trace_id) -> list[dict]
      # parsed original events of that trace in original line order,
      # exactly as ingested (raw bytes parsed back to dicts; raw bytes
      # available via Store.replay_raw(trace_id) -> list[bytes]).
  Store.close(); context manager support.
  ```
  Text search: `density.embedders.HashingEmbedder(dim=768)` ships as a
  clearly labeled demo-only embedder (deterministic feature hashing).

  v1 store semantics (Phase 3):
  - One `put_traces` call and one `put_embeddings` call per tier. A
    second call on the same tier raises StoreError telling the caller to
    batch into a single call (`put_traces` accepts a directory).
    Incremental re-encoding is deferred until the audit can re-verify
    recall after appends.
  - `put_traces(tier="hot")` raises StoreError: the hot tier's contract
    is raw traces, and v1 does not build a raw passthrough bundle.
  - `put_embeddings(..., codec="binary")` is a cold-only override that
    selects COLD_BINARY_SPEC (0.90 floor, rerank depth 500). Vectors are
    L2-normalized float32 at ingest; ids (int64 or strings) are stored
    alongside the codes and search returns them, never row indices.
  - Cold `search` reranks at the policy depth (pq 200, binary 500)
    through the best aligned reranker: SQ8 over warm codes first, then
    ExactReranker over hot fp32. Aligned means the other tier stored the
    identical id sequence. With no aligned reranker the search runs
    without rerank and emits a one-time UserWarning naming the
    unapplied recall floor (the manifest schema has no notes field, so
    the warning is the honesty channel).
  - `Store.search_cli(query: str, k=10, tier=None)` backs the CLI: the
    query is parsed as a JSON array vector first, otherwise embedded as
    text via the configured embedder, or via the demo HashingEmbedder
    with a one-time demo-only warning when none is configured.
  - On disk, each tier's vector state persists as one little-endian
    .npy file per `to_state` array plus a `state.json` listing (npz is
    avoided: its zip container embeds timestamps and would break
    deterministic checksums). Every referenced file gets a sha256 entry
    in the manifest.

## recall/

- metrics.py:
  ```python
  def recall_at_k(gt: np.int64[nq, k], pred: np.int64[nq, >=k], k) -> float
  def compression_ratio(original_bytes, compressed_bytes) -> float
  @dataclass class BytesAccount: raw: int; encoded: int; aux: int
  ```
- verifier.py: referee module, imports only public codec interfaces.
  ```python
  def exact_ground_truth(X, Q, k=100, batch=4096) -> (ids, scores)
  def verify_tiers(X, codecs: dict[str, VectorCodec], seed=1337,
                   n_queries=1000, sample_cap=200_000,
                   rerank_depths: dict[str, int] | None = None,
                   reranker=None,
                   reranker_factory: Callable[[np.ndarray], Reranker] | None
                       = None) -> VerifyResult
      # Splits queries out of X (leave-out, seeded), samples the corpus to
      # sample_cap if larger, computes recall@1/10/100 per codec, bytes per
      # vector per codec. VerifyResult.to_dict() is JSON-serializable.
      # When a codec's rerank depth is positive, the reranker resolves in
      # order: the explicit reranker argument, reranker_factory(X_db) built
      # on the leave-out database vectors, then ExactReranker(X_db) as the
      # honest fallback. reranker_factory exists because a Reranker needs
      # the post-split database, which only verify_tiers knows.
  # default rerank depths: pq: 200, binary: 500, sq8: 0
  ```

## audit/

- pricing.py: `Pricing(s3_gb_month=0.023, vectordb_gb_month=0.33)`,
  `load_pricing(path | None)` reads pricing.toml (`[pricing]` table with the
  same keys) when present.
- runner.py:
  ```python
  def run_audit(path, out="report.html", tiers=("warm", "cold"),
                pricing=None, seed=1337, sample_events=0.01) -> AuditResult
  # AuditResult: sizes per tier (traces, vectors, total), measured recall
  # per tier, guarantees, pass/fail per guarantee, dedup stats, savings
  # per month/year, methodology (seeds, sample sizes, pricing constants),
  # honesty flags, elapsed seconds per stage. .to_dict() JSON-serializable.
  # Writes report.html and report.md next to it. Re-verifies trace
  # round-trip byte equality on a 1 percent event sample during every audit.
  ```
- report.py: `render(result: AuditResult) -> (markdown: str, html: str)`.
  HTML from templates/report.html.j2, fully self-contained, inline CSS only.

## bench/

- synth.py:
  ```python
  def generate(gb: float, out_dir, seed=1337, dim=768, n_vectors=None) -> SynthStats
  # Writes out_dir/traces/part-0000.jsonl ... and out_dir/embeddings/
  # vectors.npy + ids.npy (queries are held out by the verifier, not here).
  # Three personas: support agent, coding agent, SDR agent. Realism:
  # persona system prompts of 1.5 to 4 KB resent on every call, tool call
  # JSON with varying args, retry storms (same call repeated 2 to 6 times
  # with jittered ts and args), bursty timestamps (Poisson bursts), 2
  # percent malformed lines (truncated JSON, binary garbage, bare text),
  # unicode and emoji content, occasional 50 KB tool outputs, and 0.5
  # percent lines serialized with non-canonical spacing so the residual
  # path is exercised. Embeddings: 200-cluster mixture of Gaussians on the
  # unit sphere, cluster weights Zipf-ish. Deterministic per (gb, seed,
  # dim). Total bytes within 5 percent of the target.
  ```
- datasets.py: `load_corpus(dir) -> (trace_paths, EmbeddingSet)` for synth
  output or any user directory (reuses ingest readers).
- harness.py: `run_bench(out_json, quick=False, seed=1337)` runs the full
  matrix (codecs x dims where affordable), records measured numbers,
  machine info, ACCEL_ACTIVE, wall times, writes benchmarks/results/*.json.

## service/api.py

`create_app(store: Store) -> FastAPI` with POST /audit {path, tiers?},
GET /search?q=...&k=10 (q parsed as JSON array or text), GET
/replay/{trace_id}. Thin wrappers over SDK calls, JSON responses, no auth.

## cli.py

Typer app `app`, entry `main()`. Commands: synth (--gb, --out, --seed,
--dim), audit (PATH, --out, --tiers, --pricing, --seed), compress (PATH,
--out, --tiers: build a store without the report), search (STORE, QUERY,
-k, --tier), replay (STORE, TRACE_ID, --raw), bench (--quick, --out),
serve (STORE, --host, --port).

## Testing conventions

- tests mirror src (tests/engine/embed/test_sq8.py etc.).
- Property tests with hypothesis: sq8 reconstruction error bound
  (max abs error <= (max - min) / 254 / 2 per dim, allowing float fuzz),
  ADC distance approximates exact distance monotonically on toy data
  (Spearman rank correlation > 0.95), hamming symmetry and triangle
  sanity, dedup normalize idempotent, shred round-trip byte equality.
- Every test seeds explicitly. No test reads the network or the clock for
  logic (wall-clock timing assertions are allowed only in bench code).
- Shared fixtures in tests/conftest.py: tiny_corpus (200 events, dim 32
  vectors), rng.
