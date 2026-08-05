"""Benchmark corpus loader: find traces and embeddings in one directory.

load_corpus resolves any corpus directory, synth output or user data,
into the two inputs every benchmark needs: the trace shard paths and the
embedding set. It reuses the ingest readers for both sides, so what
counts as a trace file or a readable embedding format is defined in
exactly one place (density.ingest) and can never drift between the bench
harness and the product pipeline.

Layout rules, matching the audit runner's corpus location logic:

- root/traces/ present: trace shards come from there (the synth layout).
- otherwise the whole directory is scanned for *.jsonl and *.jsonl.N,
  excluding root/embeddings/, so embedding JSONL files are never counted
  as trace lines.
- embeddings come from root/embeddings/ when it holds a supported file
  (.npy, .parquet, .jsonl); a missing or empty embeddings directory is
  simply "no embeddings", not an error.
"""

from __future__ import annotations

from pathlib import Path

from density.errors import IngestError
from density.ingest.embeddings import EmbeddingSet, read_embeddings

# Reused on purpose: _trace_files is the ingest reader's own file
# discovery (the *.jsonl / *.jsonl.N convention, sorted for determinism).
# Duplicating the pattern here would let the bench and the ingest pipeline
# silently disagree about what a trace file is.
from density.ingest.traces import _trace_files

__all__ = ["load_corpus"]

# Formats read_embeddings understands; used only to decide whether an
# embeddings directory is "present" versus merely an empty folder.
_EMBEDDING_SUFFIXES = (".npy", ".parquet", ".jsonl")


def _embeddings_present(emb_dir: Path) -> bool:
    """True when the directory holds at least one readable embedding file."""
    if not emb_dir.is_dir():
        return False
    return any(
        p.is_file() and p.suffix.lower() in _EMBEDDING_SUFFIXES
        for p in emb_dir.iterdir()
    )


def load_corpus(corpus_dir: str | Path) -> tuple[list[Path], EmbeddingSet | None]:
    """Locate the trace files and embeddings of one corpus directory.

    corpus_dir: a directory in the synth layout (traces/ + embeddings/)
    or any user directory of loose *.jsonl shards, optionally with an
    embeddings/ subdirectory. A single .jsonl file is accepted as a
    traces-only corpus and a single .npy or .parquet file as an
    embeddings-only corpus, mirroring the audit runner.

    Returns (trace_paths, embeddings): trace_paths is a sorted list of
    absolute Paths (possibly empty), embeddings is an EmbeddingSet with
    L2-normalized float32 vectors, or None when the corpus carries no
    embeddings. Missing embeddings is not an error; a corpus with neither
    traces nor embeddings raises IngestError, as does a missing path or
    an unreadable embedding file.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise IngestError(f"corpus path does not exist: {root}")

    if root.is_file():
        # Single-file conveniences, same shapes the audit accepts.
        suffix = root.suffix.lower()
        if ".jsonl" in root.name:
            return [root.resolve()], None
        if suffix in (".npy", ".parquet"):
            return [], read_embeddings(root)
        raise IngestError(
            f"unsupported corpus file: {root} (want a directory, a .jsonl "
            "trace file, or a .npy or .parquet embeddings file)"
        )

    trace_dir = root / "traces"
    emb_dir = root / "embeddings"
    if trace_dir.is_dir():
        # Synth layout: only traces/ holds trace shards, no exclusion
        # bookkeeping needed.
        trace_paths = [p.resolve() for _rel, p in _trace_files(trace_dir)]
    else:
        # Flat user layout: scan the whole directory but skip anything
        # under embeddings/, whose .jsonl files are vectors, not traces.
        trace_paths = [
            p.resolve()
            for rel, p in _trace_files(root)
            if not rel.startswith("embeddings/")
        ]

    embeddings: EmbeddingSet | None = None
    if _embeddings_present(emb_dir):
        # Read errors (ragged dims, corrupt files) propagate as
        # IngestError: present-but-broken embeddings are a real problem,
        # unlike absent ones.
        embeddings = read_embeddings(emb_dir)

    if not trace_paths and embeddings is None:
        raise IngestError(
            f"nothing to load at {root}: no *.jsonl trace files and no "
            "embeddings directory were found"
        )
    return trace_paths, embeddings
