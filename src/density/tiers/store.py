"""The tiered store: a directory of tier bundles ruled by one manifest.

Layout of a populated store directory (every relative path below is
recorded, with a sha256 checksum, in manifest.json):

    mystore.density/
      manifest.json
      tiers/
        hot/
          vectors/
            ids.npy          original ids, int64 or unicode, [n]
            vectors.npy      float32 [n, d], L2-normalized originals
        warm/
          traces/            shred bundle (structured.parquet + block stores)
          vectors/
            ids.npy
            codes.npy        codec codes (sq8: uint8 [n, d])
            state.json       codec name plus array file listing
            state.mins.npy   remaining to_state arrays, one file each
            state.scale.npy
        cold/
          traces/
          vectors/ ...       same shape, pq or binary state arrays

Design choices, v1:

- One put_traces call and one put_embeddings call per tier. Re-encoding
  incrementally would force either codebook drift (silently breaking the
  recall guarantee) or full refits on every append; rejecting the second
  call with a clear error is the honest option until the audit can
  re-verify after appends.
- Vector state is stored as one .npy file per to_state array. .npy is
  deterministic byte-for-byte (unlike .npz, whose zip container embeds
  timestamps), and determinism is what makes manifest checksums a real
  integrity statement instead of noise.
- Traces are stored shredded for warm and cold only. The hot tier's
  contract is raw untouched files, which a copy would satisfy but not
  compress; v1 rejects put_traces(tier="hot") rather than pretend.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow.parquet as papq
import zstandard

from density.engine.embed import topk
from density.engine.embed.binary import BinaryCodec
from density.engine.embed.matryoshka import truncate as matryoshka_truncate
from density.engine.embed.pq import PQ
from density.engine.embed.rerank import ExactReranker, Reranker, SQ8Reranker
from density.engine.embed.sq8 import SQ8
from density.engine.trace.shred import shred_events, unshred
from density.errors import IngestError, StoreError
from density.ingest.embeddings import l2_normalize
from density.ingest.traces import IngestStats, ParsedLine, iter_events
from density.tiers.manifest import (
    MANIFEST_NAME,
    Manifest,
    QuarantineEntry,
    TierEntry,
    collect_versions,
)
from density.tiers.policy import Tier, spec_for

_CODEC_CLASSES: dict[str, type] = {"sq8": SQ8, "pq": PQ, "binary": BinaryCodec}

# Codecs a caller may request per tier (None means the tier default).
# Hot never encodes, warm is sq8 by contract, cold defaults to pq with
# binary as the documented size-over-recall variant.
_ALLOWED_CODECS: dict[Tier, set[str | None]] = {
    Tier.HOT: {None},
    Tier.WARM: {None, "sq8"},
    Tier.COLD: {None, "pq", "binary"},
}

# Manifest quarantine entries are a debugging sample, not the ledger: the
# full malformed count lives in datasets.traces.malformed. Capping keeps
# a garbage-heavy corpus from bloating manifest.json.
_QUARANTINE_CAP = 1000

_STATE_FILE = "state.json"

# search(tier=None) picks the first tier in this order that holds
# vectors: exact beats sq8 beats pq.
_FIDELITY_ORDER = (Tier.HOT, Tier.WARM, Tier.COLD)

# Trace replay checks warm before cold only because warm decompresses
# faster; both hold byte-identical lines when they hold the same trace.
_TRACE_TIER_ORDER = (Tier.WARM, Tier.COLD)


def _as_tier(tier: Tier | str) -> Tier:
    """Coerce a Tier or its string name, with a StoreError on junk."""
    try:
        return Tier(tier)
    except ValueError as exc:
        raise StoreError(
            f"unknown tier {tier!r}: expected one of 'hot', 'warm', 'cold'"
        ) from exc


def _little_endian(arr: np.ndarray) -> np.ndarray:
    """Return arr with a little-endian dtype, copying only when needed.

    The on-disk contract is little-endian everywhere; on the usual x86
    and arm hosts this is a no-op, the astype only runs on big-endian
    machines.
    """
    dt = arr.dtype
    if dt.itemsize == 1 or dt.byteorder == "<" or (
        dt.byteorder in "=|" and sys.byteorder == "little"
    ):
        return arr
    return arr.astype(dt.newbyteorder("<"))


def _parse_json_vector(text: str) -> np.ndarray | None:
    """Parse text as a JSON array of numbers, or None if it is not one.

    Returns float32 [d]. Anything that is not a non-empty all-numeric
    JSON list (including invalid JSON) returns None so the caller can
    fall back to treating the text as a query string: malformed input
    routes to a slower path, never to a crash.
    """
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        # ValueError covers JSONDecodeError (its subclass) plus other
        # parser rejections. RecursionError guards pathological nesting
        # like "[" * 100000: both json scanners recurse per bracket and
        # blow the stack long before deciding the text is invalid.
        return None
    if (
        isinstance(obj, list)
        and obj
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj)
    ):
        return np.asarray(obj, dtype=np.float32)
    return None


class Store:
    """A tiered trace-and-vector store rooted at one directory.

    Opened via density.open(path) or Store.open(path). All state lives in
    manifest.json (see tiers/manifest.py); mutating calls save the
    manifest immediately so a crash between calls never loses committed
    data, and close() flushes once more. Context manager support closes
    on exit.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seed: int = 1337,
        created_at: str | None = None,
    ) -> None:
        """Open an existing store or create a fresh one at path.

        path: store directory; created (with parents) when missing.
        seed: recorded in the manifest and used by every stochastic step
        (PQ training). created_at: ISO timestamp recorded on creation;
        None means the current UTC time. Tests pass a fixed value, which
        is what makes two identical builds byte-identical. Both are
        ignored when an existing manifest is loaded: the store's recorded
        values win.
        """
        self._dir = Path(path)
        if self._dir.exists() and not self._dir.is_dir():
            raise StoreError(f"store path is not a directory: {self._dir}")
        if (self._dir / MANIFEST_NAME).exists():
            self._manifest = Manifest.load(self._dir)  # verifies format_version
        else:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._manifest = Manifest(
                created_at=(
                    created_at
                    if created_at is not None
                    else datetime.now(timezone.utc).isoformat(timespec="seconds")
                ),
                seed=seed,
                versions=collect_versions(),
            )
            self._manifest.save(self._dir)
        self._seed = int(self._manifest.seed)
        self._closed = False
        self._embedder: Callable[[str], np.ndarray] | None = None
        # tier -> (payload, ids). payload is a fitted codec for encoded
        # tiers or the float32 [n, d] matrix for hot.
        self._vector_cache: dict[Tier, tuple[object, np.ndarray]] = {}
        self._warned: set[str] = set()

    # -- lifecycle -----------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        seed: int = 1337,
        created_at: str | None = None,
    ) -> "Store":
        """Open (or create) a store directory. See __init__ for params."""
        return cls(path, seed=seed, created_at=created_at)

    def close(self) -> None:
        """Flush the manifest and mark the store closed. Idempotent."""
        if self._closed:
            return
        self._manifest.save(self._dir)
        self._closed = True

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def path(self) -> Path:
        """The store directory."""
        return self._dir

    @property
    def manifest(self) -> Manifest:
        """The live manifest. Treat as read-only outside this class."""
        return self._manifest

    def _require_open(self) -> None:
        if self._closed:
            raise StoreError(f"store is closed: {self._dir}")

    # -- traces --------------------------------------------------------

    def put_traces(
        self, jsonl_path: str | Path, tier: Tier | str = Tier.COLD
    ) -> IngestStats:
        """Ingest and shred a JSONL corpus into one tier's trace bundle.

        jsonl_path: a .jsonl file or a directory searched recursively for
        *.jsonl and *.jsonl.N shards. tier: 'warm' (zstd level 10) or
        'cold' (level 19); 'hot' is rejected in v1, see the module
        docstring. Malformed lines never crash the ingest: they are
        counted, stored byte-exact for replay, and sampled into the
        manifest quarantine (first 1000).

        Returns IngestStats (files, events, malformed, bytes_read; bytes
        are payload bytes, newlines excluded). v1 allows a single
        put_traces call per tier: pass a directory to ingest many files
        at once.
        """
        self._require_open()
        t = _as_tier(tier)
        if t is Tier.HOT:
            raise StoreError(
                "the hot tier keeps traces raw and is not built by put_traces "
                "in v1: use tier='warm' or tier='cold'"
            )
        entry = self._manifest.tiers.get(t.value)
        if entry is not None and entry.traces_dir is not None:
            raise StoreError(
                f"tier {t.value!r} already holds traces: v1 supports a "
                "single put_traces call per tier, pass a directory to "
                "ingest every file in one call"
            )
        src = Path(jsonl_path)
        if not src.exists():
            raise IngestError(f"trace path does not exist: {src}")

        spec = spec_for(t)
        rel_dir = f"tiers/{t.value}/traces"
        out_dir = self._dir / rel_dir
        stats = IngestStats()
        quarantined: list[QuarantineEntry] = []
        seen_files: set[str] = set()
        # Identity of the corpus content, not its path: length-prefixed
        # (relpath, payload) records make the digest unambiguous, and the
        # sorted-relpath iteration order of iter_events makes it
        # deterministic for a given corpus.
        corpus_id = hashlib.sha256()

        def observed() -> Iterator[ParsedLine]:
            # One pass feeds the shredder, the counters, and the corpus
            # digest at once, so a multi-gigabyte corpus is read exactly
            # one time.
            for parsed in iter_events(src):
                stats.observe(parsed)
                seen_files.add(parsed.file)
                fb = parsed.file.encode("utf-8")
                corpus_id.update(len(fb).to_bytes(4, "little"))
                corpus_id.update(fb)
                corpus_id.update(len(parsed.raw).to_bytes(8, "little"))
                corpus_id.update(parsed.raw)
                if parsed.event is None and len(quarantined) < _QUARANTINE_CAP:
                    quarantined.append(
                        QuarantineEntry(
                            file=parsed.file,
                            line_index=parsed.line_index,
                            error=parsed.error or "unknown",
                        )
                    )
                yield parsed

        shred_events(observed(), out_dir, level=int(spec.trace_zstd_level), seed=self._seed)

        m = self._manifest
        tr = m.datasets.traces
        tr.files = sorted(set(tr.files) | seen_files)
        source_id = corpus_id.hexdigest()
        if source_id not in tr.sources:
            # First sighting of this corpus content: fold it into the
            # dataset ledger. A second tier ingesting the same corpus
            # keeps its own accurate per-call IngestStats, but must not
            # double the dataset totals the audit divides by, or every
            # compression ratio would be overstated about 2x.
            tr.sources = sorted([*tr.sources, source_id])
            tr.events += stats.events
            tr.malformed += stats.malformed
            tr.raw_bytes += stats.bytes_read
            m.quarantine.extend(quarantined)
            del m.quarantine[_QUARANTINE_CAP:]

        entry = self._tier_entry(t)
        entry.traces_dir = rel_dir
        entry.trace_zstd_level = int(spec.trace_zstd_level)
        # Accounting uses real on-disk sizes (parquet + blocks + indexes),
        # not the shredder's payload counters, so the report can never
        # claim a smaller footprint than ls shows.
        entry.bytes.traces = sum(
            f.stat().st_size for f in out_dir.rglob("*") if f.is_file()
        )
        self._retotal(entry)
        self._checksum_tree(rel_dir)
        m.save(self._dir)
        return stats

    def replay_raw(self, trace_id: str) -> list[bytes]:
        """Exact original line bytes of one trace, in original line order.

        Bytes are per line, trailing newline excluded; a CR from CRLF
        input is part of the payload and is preserved. Raises StoreError
        when the store holds no traces, the trace_id is unknown, a bundle
        file is corrupt (run verify() to identify it), or the trace_id
        exists in tiers loaded from different corpora with differing
        lines, because replay cannot know which sequence is the real
        trace. Known v1 limitation: a trace_id containing a lone
        surrogate is unreachable by its true id, because parquet stores
        the backslashreplace form of the id.
        """
        self._require_open()
        searched = False
        # Every tier holding traces is consulted, because put_traces may
        # legally have loaded warm and cold from different corpora. Tiers
        # whose parquet does not mention the trace_id cost one cheap
        # scalar-column read and are skipped without any decompression.
        holdings: list[tuple[Tier, list[bytes]]] = []
        for t in _TRACE_TIER_ORDER:
            entry = self._manifest.tiers.get(t.value)
            if entry is None or entry.traces_dir is None:
                continue
            searched = True
            bundle = self._dir / entry.traces_dir
            try:
                # The parquet scalar columns locate the trace's (file, line)
                # set cheaply; the bytes themselves always come from unshred,
                # which is the code path whose byte-exactness is proven.
                wanted = self._trace_rows(bundle, trace_id)
                if not wanted:
                    continue
                lines = [
                    raw
                    for file, line_index, raw in unshred(bundle)
                    if (file, line_index) in wanted
                ]
            except StoreError:
                raise
            except (zstandard.ZstdError, OSError, KeyError, ValueError) as exc:
                # Decode failures (zstd corruption, truncated parquet,
                # missing files) are store problems, not caller problems:
                # translate them so no foreign exception type leaks.
                raise StoreError(
                    f"cannot read {t.value} trace bundle {bundle}: {exc}. "
                    "A bundle file may be corrupt: run Store.verify() to "
                    "list files whose checksums no longer match."
                ) from exc
            holdings.append((t, lines))
        if not searched:
            raise StoreError("store holds no traces: call put_traces first")
        if not holdings:
            raise StoreError(f"unknown trace_id: {trace_id!r}")
        first_tier, first_lines = holdings[0]
        for other_tier, other_lines in holdings[1:]:
            # Identical holdings (the common same-corpus-in-both-tiers
            # case) keep the fast path: return the warm copy. Differing
            # holdings mean the tiers were loaded from different corpora
            # and truncating to one tier would silently drop lines.
            if other_lines != first_lines:
                raise StoreError(
                    f"trace {trace_id!r} exists in both the "
                    f"{first_tier.value} and {other_tier.value} tiers with "
                    "different lines: the tiers were loaded from different "
                    "corpora, and replay cannot know which sequence is the "
                    "real trace. Rebuild the store with every tier "
                    "ingesting the same corpus."
                )
        return first_lines

    def replay(self, trace_id: str) -> list[dict]:
        """Parsed original events of one trace, in original line order.

        Each dict is json.loads of the exact original line, so key order
        and values match the ingested bytes. Raises StoreError for an
        unknown trace_id.
        """
        # Lines that matched a trace_id were valid JSON objects at ingest
        # and replay byte-exact, so parsing here cannot fail on user data.
        return [json.loads(raw.decode("utf-8")) for raw in self.replay_raw(trace_id)]

    @staticmethod
    def _trace_rows(bundle: Path, trace_id: str) -> set[tuple[str, int]]:
        """The (file, line_index) set of one trace in a shred bundle."""
        table = papq.read_table(
            bundle / "structured.parquet", columns=["trace_id", "file", "line_index"]
        )
        tids = table.column("trace_id").to_pylist()
        files = table.column("file").to_pylist()
        lines = table.column("line_index").to_pylist()
        return {
            (f, i) for tid, f, i in zip(tids, files, lines) if tid == trace_id
        }

    # -- embeddings ----------------------------------------------------

    def put_embeddings(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        tier: Tier | str = "warm",
        codec: str | None = None,
        matryoshka_dims: int | None = None,
    ) -> None:
        """Encode and persist one tier's vectors with their original ids.

        ids: [n] integers or strings, returned verbatim by search.
        vectors: [n, d] float; rows are cast to float32 and L2-normalized
        here, per the global convention, so scores everywhere are cosine
        similarities. tier: 'hot' stores the fp32 originals, 'warm'
        encodes with SQ8, 'cold' with PQ. codec: optional cold-only
        override 'binary' (32x codes at a 0.90 recall floor with rerank).
        matryoshka_dims: optional, off by default; when set, vectors are
        truncated to their first matryoshka_dims dimensions and
        re-normalized (engine.embed.matryoshka) before fit/encode. The
        value is recorded in the tier entry, and search truncates query
        vectors for that tier the same way. Requires 1 <= matryoshka_dims
        <= d, plus the codec's own dim constraints in the truncated space.

        v1 allows a single put_embeddings call per tier: batch all
        vectors into one call. A second call raises StoreError.
        """
        self._require_open()
        t = _as_tier(tier)
        entry = self._manifest.tiers.get(t.value)
        if entry is not None and entry.vectors_file is not None:
            raise StoreError(
                f"tier {t.value!r} already holds vectors: v1 supports a "
                "single put_embeddings call per tier, batch all vectors "
                "into one call"
            )
        if codec not in _ALLOWED_CODECS[t]:
            raise StoreError(
                f"codec {codec!r} is not available for tier {t.value!r}: "
                f"allowed: {sorted(str(c) for c in _ALLOWED_CODECS[t])}"
            )

        ids_arr = np.asarray(ids)
        if ids_arr.ndim != 1:
            raise StoreError(f"ids must be a 1-D array, got shape {ids_arr.shape}")
        if ids_arr.dtype.kind in "iu":
            ids_arr = ids_arr.astype(np.int64)
        elif ids_arr.dtype.kind in "USO":
            # Unicode arrays round-trip through .npy without pickle, so
            # string ids stay as safe to load as integer ids.
            ids_arr = ids_arr.astype(str)
        else:
            raise StoreError(f"ids must be integers or strings, got dtype {ids_arr.dtype}")

        X = np.asarray(vectors)
        if X.ndim != 2 or X.shape[0] == 0:
            raise StoreError(f"vectors must be a non-empty [n, d] array, got shape {X.shape}")
        if ids_arr.shape[0] != X.shape[0]:
            raise StoreError(
                f"ids length {ids_arr.shape[0]} does not match vectors rows {X.shape[0]}"
            )
        n, d = X.shape
        m = self._manifest
        # The store dim is the input dim, pre-truncation: queries always
        # arrive full-width and are truncated per tier at search time, so
        # every tier must ingest the same original dimensionality.
        if m.datasets.embeddings.dim and m.datasets.embeddings.dim != d:
            raise StoreError(
                f"vectors dim {d} does not match store dim {m.datasets.embeddings.dim}"
            )
        Xn = l2_normalize(X)
        if matryoshka_dims is not None:
            try:
                Xn = matryoshka_truncate(Xn, int(matryoshka_dims))
            except ValueError as exc:
                raise StoreError(f"bad matryoshka_dims: {exc}") from exc

        codec_name = spec_for(t, codec).vector_codec  # None for hot
        rel_dir = f"tiers/{t.value}/vectors"
        out = self._dir / rel_dir
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "ids.npy", _little_endian(ids_arr))

        payload: object
        if codec_name is None:
            np.save(out / "vectors.npy", _little_endian(Xn))
            vectors_rel = f"{rel_dir}/vectors.npy"
            state_rel = None
            enc_bytes, aux_bytes = int(Xn.nbytes), 0
            payload = Xn
        else:
            codec_cls = _CODEC_CLASSES[codec_name]
            try:
                fitted = codec_cls().fit(Xn, seed=self._seed)
                fitted.add(Xn)
            except ValueError as exc:
                # Codec preconditions (PQ: m divides d, binary: d % 8)
                # surface as store misuse with the codec's own reason.
                raise StoreError(f"cannot encode with {codec_name}: {exc}") from exc
            arrays: dict[str, str] = {}
            for key, arr in fitted.to_state().items():
                fname = "codes.npy" if key == "codes" else f"state.{key}.npy"
                np.save(out / fname, _little_endian(np.ascontiguousarray(arr)))
                arrays[key] = fname
            # dim is the encoded dim: it equals d unless matryoshka
            # truncation shrank the vectors before the codec saw them.
            meta = {"codec": codec_name, "arrays": arrays, "count": n, "dim": int(Xn.shape[1])}
            (out / _STATE_FILE).write_text(
                json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
            )
            vectors_rel = f"{rel_dir}/codes.npy"
            state_rel = f"{rel_dir}/{_STATE_FILE}"
            enc_bytes, aux_bytes = int(fitted.encoded_nbytes()), int(fitted.aux_nbytes())
            payload = fitted

        # Dataset-level count and dim describe the latest vector set; dim
        # is enforced identical across tiers above.
        m.datasets.embeddings.count = n
        m.datasets.embeddings.dim = d
        m.datasets.embeddings.source = "put_embeddings"
        m.datasets.embeddings.normalized = True

        entry = self._tier_entry(t)
        entry.vector_codec = codec_name
        entry.vectors_file = vectors_rel
        entry.codec_state_file = state_rel
        entry.matryoshka_dims = (
            int(matryoshka_dims) if matryoshka_dims is not None else None
        )
        entry.bytes.vectors = enc_bytes
        entry.bytes.vectors_aux = aux_bytes
        self._retotal(entry)
        self._checksum_tree(rel_dir)
        m.save(self._dir)
        self._vector_cache[t] = (payload, ids_arr)

    def set_embedder(self, fn: Callable[[str], np.ndarray]) -> None:
        """Configure the text embedder used by search(str) queries.

        fn(text) must return a float32 [d] vector from the same model
        that produced the stored vectors; d must match the store dim.
        """
        self._embedder = fn

    # -- search --------------------------------------------------------

    def search(
        self,
        query: np.ndarray | str,
        k: int = 10,
        tier: Tier | str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k nearest vectors by cosine similarity, original ids back.

        query: float vector [d] or batch [nq, d], normalized here before
        searching; or a str when an embedder was configured via
        set_embedder. tier: None picks the highest-fidelity tier present
        (hot, then warm, then cold). Cold searches rerank at the policy
        depth (pq 200, binary 500) through the best aligned reranker:
        warm SQ8 codes first, hot fp32 second, none (with a warning) last.

        Returns (ids, scores): for a [d] query, ids [k'] in the original
        id dtype and scores float32 [k']; for a [nq, d] query, [nq, k'].
        k' = min(k, stored count). Scores are cosine similarities,
        best-first (negative hamming for a cold binary tier without a
        reranker). k < 1 raises StoreError. A tier ingested with
        matryoshka_dims truncates the query the same way before scoring.
        """
        self._require_open()
        self._validate_k(k)
        if isinstance(query, str):
            if self._embedder is None:
                raise StoreError(
                    "text query needs an embedder: call set_embedder(fn), "
                    "or use search_cli for the demo fallback"
                )
            query = self._embedder(query)
        Q = np.asarray(query, dtype=np.float32)
        single = Q.ndim == 1
        if single:
            Q = Q[None, :]
        if Q.ndim != 2:
            raise StoreError(f"query must be [d] or [nq, d], got shape {np.shape(query)}")
        t = self._resolve_search_tier(tier)
        loaded = self._load_tier_vectors(t)
        assert loaded is not None  # _resolve_search_tier guarantees presence
        payload, ids_arr = loaded
        dim = self._manifest.datasets.embeddings.dim
        if Q.shape[1] != dim:
            raise StoreError(f"query dim {Q.shape[1]} does not match store dim {dim}")
        Qn = l2_normalize(Q)
        tier_dims = self._matryoshka_dims(t)
        if tier_dims is not None:
            # The tier stored truncated-and-renormalized vectors, so the
            # query must be projected into the same space for scores to
            # remain cosines there. Truncation happens after normalizing
            # exactly as it did at ingest.
            Qn = matryoshka_truncate(Qn, tier_dims)

        if isinstance(payload, np.ndarray):
            row_ids, scores = self._exact_search(payload, Qn, k)
        else:
            codec = payload
            depth = spec_for(t, getattr(codec, "name", None)).rerank_depth
            reranker: Reranker | None = None
            if depth > 0:
                reranker = self._best_reranker(t, ids_arr)
                if reranker is None:
                    depth = 0
                    floor = spec_for(t, getattr(codec, "name", None)).recall10_floor
                    self._warn_once(
                        "no_reranker",
                        f"{t.value} search runs without rerank: no aligned warm "
                        f"or hot vectors are present, so the recall@10 floor of "
                        f"{floor} does not apply",
                    )
            row_ids, scores = codec.search(Qn, k=k, rerank_depth=depth, reranker=reranker)

        orig = ids_arr[row_ids]  # map row indices back to original ids
        scores = np.asarray(scores, dtype=np.float32)
        if single:
            return orig[0], scores[0]
        return orig, scores

    def search_cli(
        self, query: str, k: int = 10, tier: Tier | str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """CLI entry: query is a JSON array vector or free text.

        A parseable non-empty all-numeric JSON array is searched as a
        vector. Anything else is text: the configured embedder runs it,
        or, when none is configured, the demo HashingEmbedder does, with
        a one-time warning, because keyword hashing is not a semantic
        model. Returns (ids [k'], scores float32 [k']) as in search.
        k < 1 raises StoreError before any embedding work runs.
        """
        self._require_open()
        self._validate_k(k)
        vec = _parse_json_vector(query)
        if vec is None:
            fn = self._embedder
            if fn is None:
                dim = self._manifest.datasets.embeddings.dim
                if dim <= 0:
                    raise StoreError("store holds no vectors: call put_embeddings first")
                self._warn_once(
                    "demo_embedder",
                    "no embedder configured: falling back to the demo-only "
                    "HashingEmbedder (deterministic keyword hashing, not a "
                    "semantic model). Call set_embedder(fn) for real text "
                    "search.",
                )
                from density.embedders import HashingEmbedder

                fn = HashingEmbedder(dim=dim)
            vec = np.asarray(fn(query), dtype=np.float32)
        return self.search(vec, k=k, tier=tier)

    # -- integrity -----------------------------------------------------

    def verify(self) -> list[str]:
        """Re-hash every checksummed file, return the mismatched relpaths.

        An empty list means every file referenced by the manifest still
        matches its recorded sha256; a non-empty list names corrupted or
        missing files. This is an explicit call, not part of open(),
        because hashing a multi-gigabyte store on every open would tax
        the common path for a rare failure; Phase 4's audit calls it as
        part of every report, which is where the cost belongs.
        """
        self._require_open()
        return self._manifest.verify_checksums(self._dir)

    # -- internals -----------------------------------------------------

    @staticmethod
    def _validate_k(k: int) -> None:
        """Reject k < 1 uniformly, before any tier-specific code runs.

        Without this guard the failure mode depends on the tier: exact
        and sq8 searches would quietly return empty arrays while cold PQ
        crashes allocating a negative-width output. A k below 1 is always
        a caller bug, so it fails loudly and identically everywhere.
        """
        if int(k) < 1:
            raise StoreError(f"k must be at least 1, got {k}")

    def _matryoshka_dims(self, t: Tier) -> int | None:
        """The tier's recorded matryoshka truncation, None for full dim."""
        entry = self._manifest.tiers.get(t.value)
        return None if entry is None else entry.matryoshka_dims

    def _tier_entry(self, t: Tier) -> TierEntry:
        entry = self._manifest.tiers.get(t.value)
        if entry is None:
            entry = TierEntry(tier=t.value)
            self._manifest.tiers[t.value] = entry
        return entry

    @staticmethod
    def _retotal(entry: TierEntry) -> None:
        entry.bytes.total = (
            entry.bytes.traces + entry.bytes.vectors + entry.bytes.vectors_aux
        )

    def _checksum_tree(self, rel_dir: str) -> None:
        """Record sha256 for every file under one store subdirectory."""
        base = self._dir / rel_dir
        for f in sorted(base.rglob("*")):
            if f.is_file():
                self._manifest.record_checksum(
                    self._dir, f.relative_to(self._dir).as_posix()
                )

    def _has_vectors(self, t: Tier) -> bool:
        entry = self._manifest.tiers.get(t.value)
        return entry is not None and entry.vectors_file is not None

    def _resolve_search_tier(self, tier: Tier | str | None) -> Tier:
        if tier is None:
            for t in _FIDELITY_ORDER:
                if self._has_vectors(t):
                    return t
            raise StoreError("store holds no vectors: call put_embeddings first")
        t = _as_tier(tier)
        if not self._has_vectors(t):
            raise StoreError(f"tier {t.value!r} holds no vectors")
        return t

    def _load_tier_vectors(self, t: Tier) -> tuple[object, np.ndarray] | None:
        """Load (payload, ids [n]) for a tier, cached after first use.

        payload is the float32 [n, d] matrix for hot or a rebuilt fitted
        codec (codes included) for encoded tiers. Returns None when the
        tier holds no vectors.
        """
        cached = self._vector_cache.get(t)
        if cached is not None:
            return cached
        entry = self._manifest.tiers.get(t.value)
        if entry is None or entry.vectors_file is None:
            return None
        vec_dir = self._dir / f"tiers/{t.value}/vectors"
        try:
            ids_arr = np.load(vec_dir / "ids.npy")
            if entry.vector_codec is None:
                payload: object = np.ascontiguousarray(
                    np.load(self._dir / entry.vectors_file), dtype=np.float32
                )
            else:
                assert entry.codec_state_file is not None
                meta = json.loads(
                    (self._dir / entry.codec_state_file).read_text(encoding="utf-8")
                )
                state = {
                    key: np.load(vec_dir / fname)
                    for key, fname in meta["arrays"].items()
                }
                payload = _CODEC_CLASSES[meta["codec"]].from_state(state)
        except (OSError, KeyError, ValueError) as exc:
            raise StoreError(
                f"cannot load {t.value} vectors from {vec_dir}: {exc}. "
                "A file may be corrupt: run Store.verify() to list files "
                "whose checksums no longer match."
            ) from exc
        self._vector_cache[t] = (payload, ids_arr)
        return payload, ids_arr

    def _best_reranker(self, target: Tier, target_ids: np.ndarray) -> Reranker | None:
        """Best aligned reranker for a tier, or None.

        Alignment means the candidate tier stored the same ids in the
        same order, which is what makes its row indices meaningful for
        the target's candidates, and the same matryoshka_dims, because
        the reranker receives the target tier's (possibly truncated)
        query and scores it against its own stored vectors: mismatched
        truncation would crash on shape or silently score in the wrong
        space. Warm SQ8 codes win over hot fp32 by policy: they are 4x
        smaller to touch and already carry the 0.99 recall floor, which
        is plenty for rescoring a shortlist.
        """
        target_dims = self._matryoshka_dims(target)
        if target is not Tier.WARM and self._has_vectors(Tier.WARM):
            loaded = self._load_tier_vectors(Tier.WARM)
            if loaded is not None:
                payload, warm_ids = loaded
                if (
                    isinstance(payload, SQ8)
                    and self._ids_aligned(warm_ids, target_ids)
                    and self._matryoshka_dims(Tier.WARM) == target_dims
                ):
                    return SQ8Reranker(payload)
        if target is not Tier.HOT and self._has_vectors(Tier.HOT):
            loaded = self._load_tier_vectors(Tier.HOT)
            if loaded is not None:
                payload, hot_ids = loaded
                if (
                    isinstance(payload, np.ndarray)
                    and self._ids_aligned(hot_ids, target_ids)
                    and self._matryoshka_dims(Tier.HOT) == target_dims
                ):
                    return ExactReranker(payload)
        return None

    @staticmethod
    def _ids_aligned(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and bool(np.array_equal(a, b))

    @staticmethod
    def _exact_search(
        X: np.ndarray, Q: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact dot-product top-k over the hot fp32 matrix.

        X: float32 [n, d], Q: float32 [nq, d]. Returns (row ids int64
        [nq, k'], scores float32 [nq, k']), k' = min(k, n), best-first.
        """
        n = X.shape[0]
        kk = min(max(int(k), 0), n)
        ids = np.empty((Q.shape[0], kk), dtype=np.int64)
        scores = np.empty((Q.shape[0], kk), dtype=np.float32)
        for i, row in enumerate(Q):
            s = X @ row
            top = topk(s, kk)
            ids[i] = top
            scores[i] = s[top]
        return ids, scores

    def _warn_once(self, key: str, message: str) -> None:
        # Once per store instance: the condition is a property of the
        # store's contents, repeating it every call would just be noise.
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(message, UserWarning, stacklevel=3)
