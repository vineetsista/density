"""Audit runner: measure one corpus end to end and write the report.

run_audit is the product ceremony. It ingests a corpus, compresses it
into each requested tier, analyzes duplicate text bodies, measures
recall against exact ground truth, re-verifies trace round-trip byte
equality on a seeded sample, prices the before and after, and renders
the report. Every number in the report is computed here from the corpus
itself, never assumed.

Memory discipline: the trace corpus is never streamed into memory as a
whole. Every stage re-iterates the JSONL files from disk instead of
holding them, so ingest counters, per-tier shredding, dedup sampling,
and round-trip verification each cost one line plus writer buffers no
matter the corpus size. Two stages are deliberate exceptions and are
sized here so nobody has to discover them:

- dedup retains one copy of each distinct content body plus one index
  per event (see engine/trace/dedup.find_clusters), so its cost tracks
  unique text, not corpus size. On a corpus of mostly unique bodies
  that approaches the corpus itself.
- embeddings are held as one float32 matrix, about 3.1 GB at the
  1M x 768 design point, because codec fitting and exact ground truth
  both need random access to rows.

Determinism: with a fixed seed the whole result is bit-stable except
wall-clock stage timings and peak RSS, which measure the machine, not
the data. Those live in exactly two fields, stage_seconds and
peak_rss_mb, so callers can strip them and diff the rest.
"""

from __future__ import annotations

import resource
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from density.audit.pricing import Pricing, load_pricing
from density.engine.embed.binary import BinaryCodec
from density.engine.embed.pq import PQ
from density.engine.embed.rerank import Reranker, SQ8Reranker
from density.engine.embed.sq8 import SQ8
from density.engine.trace.dedup import find_clusters
from density.engine.trace.shred import ShredResult, shred_events, unshred
from density.errors import AuditError, DensityError, IngestError
from density.ingest.embeddings import EmbeddingSet, read_embeddings
from density.ingest.traces import (
    IngestStats,
    ParsedLine,
    iter_events,
    iter_raw_lines,
)
from density.recall.verifier import VerifyResult, verify_tiers
from density.tiers.manifest import collect_versions
from density.tiers.policy import Tier, spec_for

__all__ = [
    "AuditResult",
    "ClusterInfo",
    "DedupAudit",
    "SavingsAudit",
    "TierAudit",
    "TierSavings",
    "compress_to_store",
    "human_bytes",
    "run_audit",
    "usd",
]

_CODEC_CLASSES: dict[str, type] = {"sq8": SQ8, "pq": PQ, "binary": BinaryCodec}

# Pricing is quoted per decimal gigabyte-month (that is how S3 and the
# vector databases quote it), so all GB figures in the audit are decimal
# too, with the exact byte counts always printed beside them.
_GB = 1_000_000_000.0

# When a tier misses its recall floor, the honest recommendation is the
# next tier up: it keeps more fidelity at a larger footprint.
_HIGHER_TIER: dict[str, str] = {"cold": "warm", "warm": "hot"}

# Verifier sizing. sample_cap matches the verifier default; n_queries
# shrinks on small corpora so the leave-out split always leaves a
# database, and verification is skipped honestly below the point where
# recall@100 is even defined.
_VERIFY_SAMPLE_CAP = 200_000
_VERIFY_MAX_QUERIES = 1000
_VERIFY_MIN_DATABASE = 100

_TOP_CLUSTERS = 5
_SAMPLE_CHARS = 120


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TierAudit:
    """Measured outcome for one tier. All byte counts are bytes.

    recall maps "recall@1" / "recall@10" / "recall@100" to fractions in
    [0, 1], or is None when recall was not measured (no embeddings, too
    few vectors, or the hot tier, which is exact by definition).
    recall_pass is True/False against recall10_floor, or None when there
    is no measurement or no floor. ratio is original corpus bytes over
    this tier's total bytes; footprint_fraction is its reciprocal, shown
    beside the tier's footprint_target.
    """

    tier: str
    codec: str | None
    trace_zstd_level: int | None
    traces_bytes: int
    vectors_bytes: int
    vectors_aux_bytes: int
    total_bytes: int
    ratio: float | None
    footprint_fraction: float | None
    footprint_target: float
    recall: dict[str, float] | None
    recall10_floor: float | None
    recall_pass: bool | None
    rerank_depth: int

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "codec": self.codec,
            "trace_zstd_level": self.trace_zstd_level,
            "traces_bytes": int(self.traces_bytes),
            "vectors_bytes": int(self.vectors_bytes),
            "vectors_aux_bytes": int(self.vectors_aux_bytes),
            "total_bytes": int(self.total_bytes),
            "ratio": None if self.ratio is None else float(self.ratio),
            "footprint_fraction": (
                None if self.footprint_fraction is None else float(self.footprint_fraction)
            ),
            "footprint_target": float(self.footprint_target),
            "recall": (
                None
                if self.recall is None
                else {k: float(v) for k, v in self.recall.items()}
            ),
            "recall10_floor": (
                None if self.recall10_floor is None else float(self.recall10_floor)
            ),
            "recall_pass": self.recall_pass,
            "rerank_depth": int(self.rerank_depth),
        }


@dataclass
class ClusterInfo:
    """One reported duplicate cluster: its size and a truncated sample."""

    size: int
    sample: str

    def to_dict(self) -> dict:
        return {"size": int(self.size), "sample": self.sample}


@dataclass
class DedupAudit:
    """Duplicate analysis over event content bodies.

    cluster_count counts near-duplicate clusters (reported, stored in
    full because replay is byte-exact); exact_group_count counts
    byte-identical groups, which genuinely share storage; bytes_saved is
    the utf-8 bytes interning avoids. residual_lines over total_lines is
    the fraction of lines whose replay comes from raw stored bytes
    instead of the canonical columnar form; it is None when no bundle was
    built (a hot-only audit), because reporting 0 there would state a
    measurement about a bundle that does not exist. residual_tier names
    the tier the count came from, since the figure is level independent
    but the reader should still know which bundle produced it.
    """

    cluster_count: int
    exact_group_count: int
    bytes_saved: int
    top_clusters: list[ClusterInfo]
    residual_lines: int | None
    total_lines: int
    residual_tier: str | None = None

    def to_dict(self) -> dict:
        return {
            "cluster_count": int(self.cluster_count),
            "exact_group_count": int(self.exact_group_count),
            "bytes_saved": int(self.bytes_saved),
            "top_clusters": [c.to_dict() for c in self.top_clusters],
            "residual_lines": (
                None if self.residual_lines is None else int(self.residual_lines)
            ),
            "total_lines": int(self.total_lines),
            "residual_tier": self.residual_tier,
        }


@dataclass
class TierSavings:
    """Dollars for one tier: monthly cost after, and the delta vs before."""

    after_monthly: float
    monthly: float
    yearly: float

    def to_dict(self) -> dict:
        return {
            "after_monthly": float(self.after_monthly),
            "monthly": float(self.monthly),
            "yearly": float(self.yearly),
        }


@dataclass
class SavingsAudit:
    """The savings model with its constants echoed, never hidden.

    storage_cost = vectors_gb * vectordb_gb_month + traces_gb *
    s3_gb_month, before and after compression; savings is before minus
    after. GB is decimal (1e9 bytes) because that is how the prices are
    quoted.
    """

    pricing: Pricing
    before_monthly: float
    per_tier: dict[str, TierSavings]

    def to_dict(self) -> dict:
        return {
            "pricing": self.pricing.to_dict(),
            "before_monthly": float(self.before_monthly),
            "per_tier": {k: v.to_dict() for k, v in self.per_tier.items()},
        }


@dataclass
class AuditResult:
    """Everything the report renders, JSON-serializable via to_dict().

    stage_seconds and peak_rss_mb are the only fields that vary between
    identically seeded runs: they measure the machine. Everything else
    is a deterministic function of (corpus, seed, tiers, pricing).
    """

    path: str
    out_html: str
    out_md: str
    seed: int
    tiers: tuple[str, ...]
    traces_present: bool
    trace_files: int
    trace_events: int
    malformed: int
    traces_raw_bytes: int
    embeddings_present: bool
    vector_count: int
    vector_dim: int
    vectors_raw_bytes: int
    tier_results: dict[str, TierAudit]
    dedup: DedupAudit | None
    savings: SavingsAudit
    methodology: dict
    honesty: list[str]
    headline_tier: str | None
    stage_seconds: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float = 0.0

    @property
    def original_bytes(self) -> int:
        """Raw corpus bytes: trace payload bytes plus fp32 vector bytes."""
        return self.traces_raw_bytes + self.vectors_raw_bytes

    def to_dict(self) -> dict:
        """Plain-Python nested dict, safe for json.dumps."""
        return {
            "path": self.path,
            "out_html": self.out_html,
            "out_md": self.out_md,
            "seed": int(self.seed),
            "tiers": list(self.tiers),
            "traces": {
                "present": self.traces_present,
                "files": int(self.trace_files),
                "events": int(self.trace_events),
                "malformed": int(self.malformed),
                "raw_bytes": int(self.traces_raw_bytes),
            },
            "embeddings": {
                "present": self.embeddings_present,
                "count": int(self.vector_count),
                "dim": int(self.vector_dim),
                "raw_bytes": int(self.vectors_raw_bytes),
            },
            "tier_results": {name: t.to_dict() for name, t in self.tier_results.items()},
            "dedup": None if self.dedup is None else self.dedup.to_dict(),
            "savings": self.savings.to_dict(),
            "methodology": self.methodology,
            "honesty": list(self.honesty),
            "headline_tier": self.headline_tier,
            "stage_seconds": {k: float(v) for k, v in self.stage_seconds.items()},
            "peak_rss_mb": float(self.peak_rss_mb),
        }

    def summary_text(self) -> str:
        """A few terminal-friendly lines; the CLI prints this verbatim."""
        lines = [f"DENSITY audit: {self.path}"]
        parts = []
        if self.traces_present:
            parts.append(
                f"traces {human_bytes(self.traces_raw_bytes)} "
                f"({self.trace_events} events, {self.malformed} malformed)"
            )
        else:
            parts.append("traces not present")
        if self.embeddings_present:
            parts.append(
                f"vectors {human_bytes(self.vectors_raw_bytes)} "
                f"({self.vector_count} x {self.vector_dim} fp32)"
            )
        else:
            parts.append("vectors not present")
        lines.append(f"  original  {human_bytes(self.original_bytes)}: " + ", ".join(parts))
        for name in self.tiers:
            t = self.tier_results[name]
            bits = [human_bytes(t.total_bytes)]
            bits.append(f"{t.ratio:.1f}x" if t.ratio is not None else "no ratio")
            if t.recall is not None and t.recall10_floor is not None:
                tag = "PASS" if t.recall_pass else "MISS"
                bits.append(
                    f"recall@10 {t.recall['recall@10']:.4f} "
                    f"vs floor {t.recall10_floor:.2f}: {tag}"
                )
            elif name == Tier.HOT.value and self.embeddings_present:
                bits.append("recall exact by definition")
            else:
                bits.append("recall not measured")
            lines.append(f"  {name:<9} " + ", ".join(bits))
        if self.dedup is not None:
            lines.append(
                f"  dedup     {self.dedup.cluster_count} near-duplicate clusters, "
                f"{self.dedup.exact_group_count} exact groups, "
                f"{human_bytes(self.dedup.bytes_saved)} saved"
            )
        if self.headline_tier is not None:
            sv = self.savings.per_tier[self.headline_tier]
            lines.append(
                f"  savings   {usd(sv.monthly)} per month, {usd(sv.yearly)} per year "
                f"at the {self.headline_tier} tier"
            )
        lines.append(f"  report    {self.out_html} and {self.out_md}")
        for flag in self.honesty:
            lines.append(f"  note: {flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers (shared with report.py and summary_text)
# ---------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    """Humanize a byte count in decimal units: 1234567 -> '1.23 MB'.

    Decimal on purpose: the pricing model is quoted per decimal GB, and
    mixing decimal dollars with binary sizes would make the arithmetic
    unverifiable by eye. Exact byte counts are printed beside every
    humanized figure in the report.
    """
    n = int(n)
    for unit, div in (("GB", _GB), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def usd(amount: float) -> str:
    """Format dollars: 1234.5 -> '$1,234.50'. Negative amounts keep the sign."""
    if amount < 0:
        return f"-${-amount:,.2f}"
    return f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# Corpus location and streaming
# ---------------------------------------------------------------------------


def _locate(root: Path) -> tuple[Path | None, str | None, Path | None]:
    """Resolve (trace_root, exclude_prefix, embeddings_source) for a corpus.

    Layout rules, simplest first: a single .npy or .parquet file is an
    embeddings-only corpus; a single .jsonl file is a traces-only corpus
    (the audit is trace-centric, so a bare jsonl is read as traces). A
    directory uses the synth convention when present: traces under
    root/traces, embeddings under root/embeddings. Without a traces
    subdirectory the whole directory is scanned for *.jsonl, excluding
    the embeddings subdirectory by relpath prefix so embedding jsonl
    files are never double-counted as trace lines.
    """
    if root.is_file():
        suffix = root.suffix.lower()
        if suffix in (".npy", ".parquet"):
            return None, None, root
        if ".jsonl" in root.name:
            return root, None, None
        raise AuditError(
            f"unsupported corpus file: {root} (want a directory, a .jsonl "
            "trace file, or a .npy or .parquet embeddings file)"
        )
    trace_dir = root / "traces"
    emb_dir = root / "embeddings"
    if trace_dir.is_dir():
        trace_root: Path | None = trace_dir
        exclude: str | None = None
    else:
        trace_root = root
        exclude = "embeddings/" if emb_dir.is_dir() else None
    return trace_root, exclude, (emb_dir if emb_dir.is_dir() else None)


def _iter_parsed(trace_root: Path, exclude: str | None) -> Iterator[ParsedLine]:
    """Stream parsed lines from disk, honoring the embeddings exclusion.

    Called once per pipeline pass: re-reading from disk each time is the
    deliberate trade that keeps trace memory flat at one line, instead
    of holding a multi-gigabyte corpus for the later stages. The
    exclusion is applied to file names, so an embeddings .jsonl is never
    opened, let alone parsed into a TraceEvent whose content would be the
    JSON dump of a 768-float vector.
    """
    yield from iter_events(trace_root, exclude_prefix=exclude)


def _iter_raw(trace_root: Path, exclude: str | None) -> Iterator[tuple[str, int, bytes]]:
    """Stream raw lines without JSON parsing, for round-trip comparison.

    The replay side already yields exact bytes, so the source side needs
    no parse either: skipping json.loads roughly halves the cost of the
    verification pass.
    """
    yield from iter_raw_lines(trace_root, exclude_prefix=exclude)


@contextmanager
def _stage(name: str, sink: dict[str, float]) -> Iterator[None]:
    """Time one pipeline stage; records the elapsed time even on failure."""
    started = time.perf_counter()
    try:
        yield
    finally:
        sink[name] = time.perf_counter() - started


def _dir_bytes(path: Path) -> int:
    """Sum of on-disk file sizes under path: what ls would show, in bytes."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _normalize_tiers(tiers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate tier names, deduplicate, and keep the caller's order."""
    seen: list[str] = []
    for name in tiers:
        try:
            value = Tier(str(name)).value
        except ValueError as exc:
            raise AuditError(
                f"unknown tier {name!r}: expected 'hot', 'warm', or 'cold'"
            ) from exc
        if value not in seen:
            seen.append(value)
    if not seen:
        raise AuditError("no tiers requested: pass at least one of hot, warm, cold")
    return tuple(seen)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def run_audit(
    path: str | Path,
    out: str = "report.html",
    tiers: tuple[str, ...] = ("warm", "cold"),
    pricing: Pricing | str | Path | None = None,
    seed: int = 1337,
    sample_events: float = 0.01,
) -> AuditResult:
    """Audit a corpus: compress, verify recall, re-verify replay, price, report.

    path: corpus directory (traces/*.jsonl and embeddings/, or loose
    jsonl files), or a single trace or embeddings file. out: HTML report
    path; a sibling .md with the same stem is always written. tiers:
    tier names to measure. pricing: a Pricing, a pricing.toml path, or
    None for defaults. seed drives every stochastic step. sample_events
    is the fraction of trace lines round-trip-verified byte for byte, at
    least one line, on every audit.

    Returns an AuditResult. Raises AuditError when the corpus holds
    neither traces nor embeddings, and, loudly, when any sampled line
    fails byte-exact replay: that is a correctness failure, never a
    statistic.
    """
    root = Path(path)
    if not root.exists():
        raise AuditError(f"audit path does not exist: {root}")
    if not 0.0 < float(sample_events) <= 1.0:
        raise AuditError(
            f"sample_events must be in (0, 1], got {sample_events}"
        )
    tier_names = _normalize_tiers(tiers)
    out_path = Path(out)
    md_path = out_path.with_suffix(".md")
    pricing_obj = pricing if isinstance(pricing, Pricing) else load_pricing(pricing)

    stage_seconds: dict[str, float] = {}
    honesty: list[str] = []

    # -- stage: ingest -------------------------------------------------
    stats = IngestStats()
    emb: EmbeddingSet | None = None
    with _stage("ingest", stage_seconds):
        trace_root, exclude, emb_source = _locate(root)
        if trace_root is not None:
            for parsed in _iter_parsed(trace_root, exclude):
                stats.observe(parsed)
        total_lines = stats.events + stats.malformed
        traces_present = total_lines > 0
        if emb_source is not None:
            try:
                emb = read_embeddings(emb_source)
            except IngestError as exc:
                raise AuditError(f"cannot read embeddings at {emb_source}: {exc}") from exc
        if not traces_present and emb is None:
            raise AuditError(
                f"nothing to audit at {root}: no *.jsonl trace files and no "
                "embeddings directory were found"
            )
    if not traces_present:
        honesty.append("Traces: not present in this corpus. Trace sections are omitted.")
    if emb is None:
        honesty.append(
            "Embeddings: not present in this corpus. Recall verification skipped; "
            "vector sections state this instead of inventing numbers."
        )
    elif emb.nonfinite_rows > 0:
        honesty.append(
            f"{emb.nonfinite_rows} vectors contained NaN or Inf and were stored "
            "as zero rows. A zero row matches no query, so recall for those "
            "rows is unmeasurable and the measured recall below is depressed "
            "by their share of the corpus."
        )
    vectors_raw_bytes = 0 if emb is None else int(emb.X.nbytes)
    original_bytes = stats.bytes_read + vectors_raw_bytes

    tier_results: dict[str, TierAudit] = {}
    shred_results: dict[str, ShredResult] = {}
    verify_detail: dict | None = None

    # The working directory holds the per-tier shred bundles just long
    # enough to measure their on-disk size and replay them; it is
    # removed even when a stage raises.
    work = Path(tempfile.mkdtemp(prefix="density-audit-"))
    try:
        # -- stage: compress -------------------------------------------
        with _stage("compress", stage_seconds):
            for name in tier_names:
                spec = spec_for(name)
                traces_bytes = 0
                if traces_present and trace_root is not None:
                    if spec.trace_zstd_level is None:
                        # Hot keeps traces raw: the honest footprint is
                        # the raw payload itself, no bundle is built.
                        traces_bytes = int(stats.bytes_read)
                    else:
                        bundle = work / name / "traces"
                        shred_results[name] = shred_events(
                            _iter_parsed(trace_root, exclude),
                            bundle,
                            level=int(spec.trace_zstd_level),
                            seed=seed,
                        )
                        traces_bytes = _dir_bytes(bundle)
                vectors_bytes = 0
                vectors_aux = 0
                if emb is not None:
                    if spec.vector_codec is None:
                        vectors_bytes = int(emb.X.nbytes)
                    else:
                        codec_cls = _CODEC_CLASSES[spec.vector_codec]
                        try:
                            fitted = codec_cls().fit(emb.X, seed=seed)
                            fitted.add(emb.X)
                        except ValueError as exc:
                            raise AuditError(
                                f"cannot encode vectors for tier {name!r} with "
                                f"{spec.vector_codec}: {exc}"
                            ) from exc
                        vectors_bytes = int(fitted.encoded_nbytes())
                        vectors_aux = int(fitted.aux_nbytes())
                        # The fitted codec served its purpose (byte
                        # accounting); dropping it keeps only one codec
                        # alive at a time on large corpora.
                        del fitted
                total_bytes = traces_bytes + vectors_bytes + vectors_aux
                ratio = (original_bytes / total_bytes) if total_bytes > 0 else None
                fraction = (total_bytes / original_bytes) if original_bytes > 0 else None
                tier_results[name] = TierAudit(
                    tier=name,
                    codec=spec.vector_codec,
                    trace_zstd_level=spec.trace_zstd_level,
                    traces_bytes=traces_bytes,
                    vectors_bytes=vectors_bytes,
                    vectors_aux_bytes=vectors_aux,
                    total_bytes=total_bytes,
                    ratio=ratio,
                    footprint_fraction=fraction,
                    footprint_target=float(spec.footprint_target),
                    recall=None,
                    recall10_floor=spec.recall10_floor,
                    recall_pass=None,
                    rerank_depth=int(spec.rerank_depth),
                )
                if ratio is not None and ratio < 1.0:
                    honesty.append(
                        f"Tier {name} grew the data: compressed bytes exceed the "
                        "original. This corpus does not benefit from this tier."
                    )

        # -- stage: dedup ----------------------------------------------
        dedup_audit: DedupAudit | None = None
        with _stage("dedup", stage_seconds):
            if traces_present and trace_root is not None and stats.events > 0:
                bodies = (
                    p.event.content
                    for p in _iter_parsed(trace_root, exclude)
                    if p.event is not None
                )
                dd = find_clusters(bodies, seed=seed)
                # Largest clusters first; ties break on the earliest
                # member so the report is stable across runs.
                top = sorted(dd.clusters, key=lambda ms: (-len(ms), ms[0]))[:_TOP_CLUSTERS]
                wanted = {ms[0]: rank for rank, ms in enumerate(top)}
                samples: dict[int, str] = {}
                if wanted:
                    # One more disk pass fetches only the sample bodies:
                    # cluster indices refer to event iteration order, and
                    # holding five 120-char samples beats holding the
                    # whole corpus.
                    for idx, p in enumerate(
                        p
                        for p in _iter_parsed(trace_root, exclude)
                        if p.event is not None
                    ):
                        if idx in wanted:
                            samples[wanted[idx]] = p.event.content[:_SAMPLE_CHARS]
                            if len(samples) == len(wanted):
                                break
                residual_tier = next(iter(shred_results), None)
                any_shred = (
                    None if residual_tier is None else shred_results[residual_tier]
                )
                dedup_audit = DedupAudit(
                    cluster_count=dd.stats.near_dup_clusters,
                    exact_group_count=len(dd.exact_dup_groups),
                    bytes_saved=dd.stats.exact_dup_bytes_saved,
                    top_clusters=[
                        ClusterInfo(size=len(ms), sample=samples.get(rank, ""))
                        for rank, ms in enumerate(top)
                    ],
                    # Residual accounting is level independent, so any
                    # tier's bundle reports it. None when no bundle exists
                    # at all (a hot-only audit): the report says the figure
                    # was not measured rather than printing a zero.
                    residual_lines=(
                        any_shred.residual_lines if any_shred is not None else None
                    ),
                    total_lines=total_lines,
                    residual_tier=residual_tier,
                )
                if dd.stats.skipped_over_cap > 0:
                    honesty.append(
                        f"Dedup minhash cap reached: {dd.stats.skipped_over_cap} "
                        "eligible unique texts were not near-duplicate checked."
                    )
                if dd.stats.minhash_truncated > 0:
                    honesty.append(
                        f"Dedup compared {dd.stats.minhash_truncated} very large "
                        "bodies by a 1 MiB prefix, not their full length."
                    )

        # -- stage: verify ---------------------------------------------
        with _stage("verify", stage_seconds):
            codec_tiers = [
                n for n in tier_names if tier_results[n].codec is not None
            ]
            if emb is not None and codec_tiers:
                n_vec = int(emb.X.shape[0])
                n_queries = min(_VERIFY_MAX_QUERIES, n_vec // 10)
                pool = min(n_vec, _VERIFY_SAMPLE_CAP)
                if n_queries < 1 or pool - n_queries < _VERIFY_MIN_DATABASE:
                    honesty.append(
                        f"Recall verification skipped: {n_vec} vectors are too "
                        f"few to hold out queries and still measure recall@100 "
                        f"(needs at least {_VERIFY_MIN_DATABASE} database rows). "
                        "The recall guarantees are unverified on this corpus."
                    )
                else:
                    def _sq8_reranker(X_db: np.ndarray) -> Reranker:
                        # DECISIONS item 6: COLD recall is defined with
                        # rerank against the WARM int8 vectors, so the
                        # referee reranks through an SQ8 fitted on the
                        # leave-out database, never through exact fp32,
                        # which would flatter the cold tier.
                        sq8 = SQ8().fit(X_db, seed=seed)
                        sq8.add(X_db)
                        return SQ8Reranker(sq8)

                    vr: VerifyResult = verify_tiers(
                        emb.X,
                        {n: _CODEC_CLASSES[tier_results[n].codec]() for n in codec_tiers},
                        seed=seed,
                        n_queries=n_queries,
                        sample_cap=_VERIFY_SAMPLE_CAP,
                        rerank_depths={
                            n: tier_results[n].rerank_depth for n in codec_tiers
                        },
                        reranker_factory=_sq8_reranker,
                    )
                    verify_detail = {
                        "n_vectors": int(vr.n_vectors),
                        "sample_size": int(vr.sample_size),
                        "n_queries": int(vr.n_queries),
                        "n_database": int(vr.n_database),
                        "k": int(vr.k),
                        "rerank_depths": {
                            n: int(vr.codecs[n].rerank_depth) for n in codec_tiers
                        },
                    }
                    for n in codec_tiers:
                        t = tier_results[n]
                        t.recall = {
                            k: float(v) for k, v in vr.codecs[n].recall.items()
                        }
                        if t.recall10_floor is not None:
                            t.recall_pass = t.recall["recall@10"] >= t.recall10_floor
                            if not t.recall_pass:
                                higher = _HIGHER_TIER.get(n)
                                advice = (
                                    f" Use the {higher} tier for this corpus."
                                    if higher is not None
                                    else ""
                                )
                                honesty.append(
                                    f"Tier {n} MISSED its recall@10 floor: measured "
                                    f"{t.recall['recall@10']:.4f} against a floor of "
                                    f"{t.recall10_floor:.2f}.{advice}"
                                )

        # -- stage: round_trip -----------------------------------------
        round_trip_detail: dict = {"sampled_lines": 0, "total_lines": total_lines, "tiers": []}
        with _stage("round_trip", stage_seconds):
            if shred_results and trace_root is not None:
                n_sample = min(
                    total_lines, max(1, int(round(total_lines * float(sample_events))))
                )
                # A seeded choice makes the sample reproducible: the same
                # corpus and seed always re-verify the same lines.
                sampled = set(
                    np.random.default_rng(seed)
                    .choice(total_lines, size=n_sample, replace=False)
                    .tolist()
                )
                round_trip_detail["sampled_lines"] = n_sample
                for name, shred in shred_results.items():
                    _verify_round_trip(
                        name, shred.out_dir, trace_root, exclude, sampled
                    )
                    round_trip_detail["tiers"].append(name)

        # -- stage: price ----------------------------------------------
        with _stage("price", stage_seconds):
            before_monthly = pricing_obj.monthly_cost(
                traces_gb=stats.bytes_read / _GB, vectors_gb=vectors_raw_bytes / _GB
            )
            per_tier: dict[str, TierSavings] = {}
            for name, t in tier_results.items():
                after = pricing_obj.monthly_cost(
                    traces_gb=t.traces_bytes / _GB,
                    vectors_gb=(t.vectors_bytes + t.vectors_aux_bytes) / _GB,
                )
                monthly = before_monthly - after
                per_tier[name] = TierSavings(
                    after_monthly=after, monthly=monthly, yearly=12.0 * monthly
                )
            savings = SavingsAudit(
                pricing=pricing_obj, before_monthly=before_monthly, per_tier=per_tier
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Headline tier: biggest measured ratio among tiers that did not
    # miss a guarantee. A miss is never averaged away or promoted: if
    # every tier missed, the best ratio is shown with the miss stated.
    candidates = [n for n in tier_names if tier_results[n].ratio is not None]
    passing = [n for n in candidates if tier_results[n].recall_pass is not False]
    pool = passing or candidates
    headline_tier = (
        max(pool, key=lambda n: tier_results[n].ratio or 0.0) if pool else None
    )
    if candidates and not passing:
        honesty.append(
            "No requested tier passed its recall floor on this corpus: the "
            "headline shows the best ratio with its miss stated, not hidden."
        )

    methodology = {
        "seed": int(seed),
        "sample_events": float(sample_events),
        "round_trip": round_trip_detail,
        "verifier": verify_detail,
        "pricing": pricing_obj.to_dict(),
        "versions": collect_versions(),
        "accel_active": _accel_active(),
        "cold_rerank_reference": (
            "COLD recall is measured with rerank against SQ8 (WARM int8) "
            "reconstructions of the leave-out database, per DECISIONS item 6."
        ),
        "units": (
            "GB means decimal gigabytes (1e9 bytes) to match per-GB-month "
            "pricing; exact byte counts accompany every figure. Vector "
            "originals are counted as float32 bytes (count x dim x 4)."
        ),
    }

    result = AuditResult(
        path=str(root),
        out_html=str(out_path),
        out_md=str(md_path),
        seed=int(seed),
        tiers=tier_names,
        traces_present=traces_present,
        trace_files=int(stats.files),
        trace_events=int(stats.events),
        malformed=int(stats.malformed),
        traces_raw_bytes=int(stats.bytes_read),
        embeddings_present=emb is not None,
        vector_count=0 if emb is None else int(emb.X.shape[0]),
        vector_dim=0 if emb is None else int(emb.X.shape[1]),
        vectors_raw_bytes=vectors_raw_bytes,
        tier_results=tier_results,
        dedup=dedup_audit,
        savings=savings,
        methodology=methodology,
        honesty=honesty,
        headline_tier=headline_tier,
        stage_seconds=stage_seconds,
    )

    # -- stage: render -------------------------------------------------
    with _stage("render", stage_seconds):
        # Imported here, not at module top: report.py needs AuditResult
        # for typing, so the lazy import keeps the modules acyclic.
        from density.audit.report import render

        markdown, html = render(result)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")

    # ru_maxrss is kilobytes on Linux. Recorded after rendering so the
    # figure covers the whole audit; like stage timings it is a machine
    # measurement, excluded from determinism comparisons.
    result.peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return result


def _verify_round_trip(
    tier_name: str,
    bundle: Path,
    trace_root: Path,
    exclude: str | None,
    sampled: set[int],
) -> None:
    """Compare sampled replayed lines byte for byte against the source.

    The source and the shred bundle are streamed in lockstep: shredding
    preserves input order, so position i of unshred must be position i
    of the source stream. File and line_index agree on every line
    (cheap), raw bytes are compared at the sampled positions. Any
    mismatch raises AuditError naming the exact line: a replay
    divergence is a correctness failure, never a statistic to soften.
    """
    src = _iter_raw(trace_root, exclude)
    rep = unshred(bundle)
    pos = 0
    try:
        for (s_file, s_line, s_raw), (r_file, r_line, r_raw) in zip(
            src, rep, strict=True
        ):
            if s_file != r_file or s_line != r_line:
                raise AuditError(
                    f"round-trip order mismatch in tier {tier_name!r} at "
                    f"position {pos}: source {s_file}:{s_line} vs replay "
                    f"{r_file}:{r_line}"
                )
            if pos in sampled and s_raw != r_raw:
                raise AuditError(
                    f"round-trip byte mismatch in tier {tier_name!r} at "
                    f"{s_file}:{s_line}: replayed bytes differ from the "
                    "original line. The compressed bundle does not reproduce "
                    "the corpus; this audit cannot stand."
                )
            pos += 1
    except ValueError as exc:
        # zip(strict=True) raises ValueError when one side ends early:
        # a line-count mismatch is as fatal as a byte mismatch.
        raise AuditError(
            f"round-trip line count mismatch in tier {tier_name!r}: the "
            f"replay stream and the source disagree after {pos} lines ({exc})"
        ) from exc


def _accel_active() -> bool:
    """Whether the compiled kernels loaded; recorded in the methodology."""
    from density.engine._accel import ACCEL_ACTIVE

    return bool(ACCEL_ACTIVE)


# ---------------------------------------------------------------------------
# compress_to_store: build a real store, skip the report
# ---------------------------------------------------------------------------


def compress_to_store(
    path: str | Path,
    out: str | None = None,
    tiers: tuple[str, ...] = ("warm", "cold"),
    seed: int = 1337,
) -> str:
    """Build a tiered Store from a raw corpus and return the store directory.

    path: corpus directory or file, located exactly as run_audit does.
    out: store directory; None means PATH.density beside the corpus.
    Traces go to every requested warm or cold tier (the hot tier keeps
    traces raw and v1 builds no raw bundle, so hot receives vectors
    only); embeddings go to every requested tier when present.

    A flat layout whose traces sit loose beside an embeddings directory
    of .jsonl vectors excludes that directory here exactly as the audit
    does, so compress and audit always see the same corpus.

    Raises AuditError when the corpus holds neither traces nor
    embeddings.
    """
    root = Path(path)
    if not root.exists():
        raise AuditError(f"corpus path does not exist: {root}")
    tier_names = _normalize_tiers(tiers)
    trace_root, exclude, emb_source = _locate(root)

    traces_present = False
    if trace_root is not None:
        for _ in _iter_raw(trace_root, exclude):
            traces_present = True
            break
    emb: EmbeddingSet | None = None
    if emb_source is not None and emb_source.exists():
        try:
            emb = read_embeddings(emb_source)
        except IngestError as exc:
            raise AuditError(f"cannot read embeddings at {emb_source}: {exc}") from exc
    if not traces_present and emb is None:
        raise AuditError(
            f"nothing to compress at {root}: no *.jsonl trace files and no "
            "embeddings directory were found"
        )

    target = Path(out) if out is not None else Path(str(root) + ".density")
    import density

    store = density.open(target, seed=seed)
    try:
        for name in tier_names:
            spec = spec_for(name)
            if traces_present and trace_root is not None and spec.trace_zstd_level is not None:
                store.put_traces(trace_root, tier=name, exclude_prefix=exclude)
            if emb is not None:
                store.put_embeddings(emb.ids, emb.X, tier=name)
    except DensityError:
        raise
    finally:
        store.close()
    return str(target)
