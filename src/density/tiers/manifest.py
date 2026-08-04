"""The store manifest: the single source of truth for a .density directory.

Everything a store contains is declared here: datasets, tiers, codec state
files, dedup summary, checksums, versions, seeds. Readers trust the
manifest, and the manifest is written atomically (temp file then rename) so
a crash mid-write can never leave a store that parses but lies.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from density.errors import StoreError

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1


class TraceDataset(BaseModel):
    files: list[str] = Field(default_factory=list)
    events: int = 0
    malformed: int = 0
    raw_bytes: int = 0
    # sha256 identities (sorted hex digests) of every distinct corpus whose
    # counters were folded into the totals above. Content-derived rather
    # than path-derived on purpose: the same corpus ingested into a second
    # tier must not double the dataset ledger, and a path identity would
    # break byte-identical manifests across build directories.
    sources: list[str] = Field(default_factory=list)


class EmbeddingDataset(BaseModel):
    count: int = 0
    dim: int = 0
    source: str = ""
    normalized: bool = True


class Datasets(BaseModel):
    traces: TraceDataset = Field(default_factory=TraceDataset)
    embeddings: EmbeddingDataset = Field(default_factory=EmbeddingDataset)


class TierBytes(BaseModel):
    """Bytes accounting for one tier, split the way the report needs it."""

    traces: int = 0
    vectors: int = 0          # encoded vector payload
    vectors_aux: int = 0      # codebooks, scales, offsets
    total: int = 0


class TierEntry(BaseModel):
    tier: str
    vector_codec: str | None = None
    codec_state_file: str | None = None   # relative path to saved codec state
    vectors_file: str | None = None       # relative path to encoded vectors
    traces_dir: str | None = None         # relative path to shredded traces
    trace_zstd_level: int | None = None
    # Leading dimensions kept by matryoshka truncation at ingest, None for
    # the full input dim. Search must truncate queries for this tier the
    # same way, so the value is store state, not a transient option.
    matryoshka_dims: int | None = None
    bytes: TierBytes = Field(default_factory=TierBytes)


class DedupSummary(BaseModel):
    exact_dup_groups: int = 0
    bytes_saved: int = 0
    near_dup_clusters: int = 0
    top_clusters: list[dict] = Field(default_factory=list)  # sample truncated to 120 chars


class QuarantineEntry(BaseModel):
    file: str
    line_index: int
    error: str


class Manifest(BaseModel):
    format_version: int = FORMAT_VERSION
    created_at: str = ""            # caller-supplied ISO string, deterministic in tests
    seed: int = 1337
    datasets: Datasets = Field(default_factory=Datasets)
    tiers: dict[str, TierEntry] = Field(default_factory=dict)
    dedup: DedupSummary = Field(default_factory=DedupSummary)
    quarantine: list[QuarantineEntry] = Field(default_factory=list)
    checksums: dict[str, str] = Field(default_factory=dict)  # relpath -> sha256
    versions: dict[str, str] = Field(default_factory=dict)

    # -- persistence ---------------------------------------------------

    def save(self, store_dir: str | Path) -> Path:
        """Atomically write manifest.json into store_dir."""
        store = Path(store_dir)
        store.mkdir(parents=True, exist_ok=True)
        target = store / MANIFEST_NAME
        payload = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=store, prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target

    @classmethod
    def load(cls, store_dir: str | Path) -> "Manifest":
        path = Path(store_dir) / MANIFEST_NAME
        if not path.exists():
            raise StoreError(f"no manifest at {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StoreError(f"corrupt manifest at {path}: {exc}") from exc
        if data.get("format_version") != FORMAT_VERSION:
            raise StoreError(
                f"manifest format {data.get('format_version')} unsupported, "
                f"expected {FORMAT_VERSION}"
            )
        return cls.model_validate(data)

    # -- checksums -----------------------------------------------------

    def record_checksum(self, store_dir: str | Path, relpath: str) -> str:
        """Hash a store file and record it. Returns the hex digest."""
        digest = sha256_file(Path(store_dir) / relpath)
        self.checksums[relpath] = digest
        return digest

    def verify_checksums(self, store_dir: str | Path) -> list[str]:
        """Return relpaths whose current hash differs (or file missing)."""
        bad: list[str] = []
        for rel, expected in self.checksums.items():
            p = Path(store_dir) / rel
            if not p.exists() or sha256_file(p) != expected:
                bad.append(rel)
        return bad


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect_versions() -> dict[str, str]:
    """Library versions recorded in every manifest, for reproducibility."""
    import numpy
    import pyarrow
    import zstandard

    import density

    return {
        "density": density.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "zstandard": zstandard.__version__,
    }
