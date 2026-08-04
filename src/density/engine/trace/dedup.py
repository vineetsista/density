"""Duplicate detection for trace text bodies.

Two distinct notions of "duplicate" live here, with different storage
consequences:

- Exact duplicates are byte-identical bodies. They are interned by
  content sha256 and genuinely share storage, because replaying one copy
  replays them all byte-for-byte.
- Near duplicates differ in bytes but only by volatile tokens (uuids,
  timestamps, casing, whitespace). They are clustered for reporting so
  an audit can say "these 40k lines are one retry storm", but each copy
  is stored in full, because replay is byte-exact and must preserve
  every original byte.

Near-duplicate detection is MinHash LSH: NUM_PERM = 128 seeded
permutations over 5-byte shingles of the utf-8 encoding of
normalize(text), banded BANDS x ROWS = 8 x 16, candidate pairs verified
at a jaccard threshold of 0.9. All hashing is explicit 64-bit integer
arithmetic, never the builtin salted hash(), so signatures are
bit-identical across processes and machines for a given seed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "BANDS",
    "JACCARD_THRESHOLD",
    "MINHASH_CAP_DEFAULT",
    "MINHASH_MIN_CHARS",
    "NUM_PERM",
    "ROWS",
    "DedupResult",
    "DedupStats",
    "MinHashLSH",
    "find_clusters",
    "normalize",
]

NUM_PERM = 128
BANDS = 8
ROWS = 16

# Texts must be strictly longer than this many characters to enter the
# minhash path. Shorter bodies ("ok", tool ids, empty strings) produce so
# few shingles that jaccard estimates are noise; exact dedup still covers
# them.
MINHASH_MIN_CHARS = 64

JACCARD_THRESHOLD = 0.9

# Upper bound on how many eligible unique texts are minhashed in one
# find_clusters call. Texts beyond the cap are counted, not silently
# dropped, so audit reports can flag the truncation.
MINHASH_CAP_DEFAULT = 1_000_000

_SHINGLE = 5

# Whitespace collapse runs before pattern stripping so that, for example,
# a tab-separated date and time become a single ISO timestamp match.
_WS_RE = re.compile(r"\s+")

# The lookarounds reject uuids embedded in longer hex runs. A near miss
# (wrong group length, non-hex digit) must not match at all: partial
# matches would mangle identifiers that only resemble uuids.
_UUID_RE = re.compile(
    r"(?<![0-9a-f])"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"(?![0-9a-f])"
)

# Date plus time of day, with optional seconds, fraction, and zone.
# A bare date is deliberately not stripped: "2024-01-15" in prose is
# content, while a full timestamp is almost always log noise.
_ISO_TS_RE = re.compile(
    r"(?<!\d)"
    r"\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:z|[+-]\d{2}:?\d{2})?"
    r"(?!\d)",
    re.ASCII,
)

# Standalone digit runs of 10 to 16 digits cover epoch seconds (10),
# millis (13), and micros (16) plus the jittered values between. Runs of
# 9 or 17 digits are outside that window and survive, as do digits glued
# to letters (identifiers).
_EPOCH_RE = re.compile(r"(?<![0-9a-z])[0-9]{10,16}(?![0-9a-z])", re.ASCII)


def _normalize_pass(s: str) -> str:
    """One lowercase, collapse, strip sweep. Not idempotent on its own."""
    s = s.lower()
    s = _WS_RE.sub(" ", s).strip()
    s = _UUID_RE.sub(" ", s)
    s = _ISO_TS_RE.sub(" ", s)
    s = _EPOCH_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def normalize(text: str) -> str:
    """Canonicalize a text body for near-duplicate comparison.

    Lowercases, collapses runs of whitespace to single spaces, strips
    leading and trailing whitespace, and removes uuids, ISO timestamps,
    and standalone epoch-like integers (each replaced by a space, so
    neighboring tokens never fuse).

    The sweep runs to a fixpoint: stripping one pattern can join its
    neighbors into a fresh match (a uuid sitting between a date and a
    time, for example), and a single pass over such input would not be
    idempotent.

    Termination: the first pass may lengthen the string, because
    str.lower() can grow it (U+0130, latin capital I with dot above,
    lowercases to two codepoints). Every later pass starts from
    lowercase text, on which lower() is the identity (lower is
    idempotent for every codepoint), and the remaining transforms
    replace matches with a single space or collapse whitespace, so a
    later pass that changes the string strictly shrinks it. The length
    strictly decreases from the second changing pass on, so the loop
    terminates. Anyone reordering passes or adding a transform that can
    grow its input must re-establish this argument.
    """
    prev = text
    while True:
        cur = _normalize_pass(prev)
        if cur == prev:
            return cur
        prev = cur


def _utf8(text: str) -> bytes:
    # surrogatepass keeps lone surrogates (which malformed upstream JSON
    # can smuggle into content) from crashing the pipeline: they hash and
    # intern fine, they just never round-trip through strict utf-8.
    return text.encode("utf-8", errors="surrogatepass")


# Multiplier for the Horner sliding hash and band hashing (the FNV-1a
# 64-bit prime, odd so multiplication is a bijection mod 2**64).
_K = np.uint64(0x100000001B3)

# Placeholder fill for the per-position hash buffer. Every slot is
# overwritten before use: real windows by the scatter, shingle-free
# segments by _short_text_hash.
_FILL = np.uint64(0x9E3779B97F4A7C15)


def _short_text_hash(enc: bytes) -> np.uint64:
    """One stand-in shingle hash for a text with no complete 5-gram.

    enc: the full normalized utf-8 encoding, at most 4 bytes here. The
    value must be a function of those bytes, never a shared constant:
    with a shared sentinel, every shingle-free text would get the same
    signature and unrelated short bodies (trace lines that are mostly
    stripped uuids and timestamps around a short status token) would
    cluster at estimated jaccard 1.0. Seeding the accumulator with the
    length keeps b"\\x00" and b"\\x00\\x00" distinct under Horner.
    """
    # Python ints with an explicit 64-bit mask: numpy uint64 scalars warn
    # on overflow, and this loop runs over at most 4 bytes.
    acc = len(enc) + 1
    for byte in enc:
        acc = (acc * int(_K) + byte) & 0xFFFFFFFFFFFFFFFF
    return _mix64(np.array([acc], dtype=np.uint64))[0]

# Texts are batched so at most about this many shingle hashes are alive
# at once; the per-permutation temporaries are then one [shingles] uint64
# array, about 8 MB, independent of corpus size.
_BATCH_SHINGLE_BUDGET = 1 << 20

# Per-text bounds on minhash input. A single pathological body (a file
# dump, a base64 blob) must not be able to dominate a whole run or blow
# past the batch budget, so shingling sees at most this many bytes of
# the normalized encoding: exactly _BATCH_SHINGLE_BUDGET five-byte
# windows. A 128-slot signature over a 1 MiB prefix carries all the
# near-duplicate signal such a body can offer; affected texts are
# counted in DedupStats.minhash_truncated.
_MAX_TEXT_SHINGLE_BYTES = _BATCH_SHINGLE_BUDGET + _SHINGLE - 1

# normalize() itself is regex over the full string, so raw input is cut
# first as well. 4x the shingle-byte cap leaves ordinary prose enough
# raw characters to fill the shingle budget after normalization shrinks
# it; only whitespace-heavy monsters fall short, and those are compared
# by prefix like any other truncated body.
_MAX_TEXT_CHARS = 4 * _BATCH_SHINGLE_BUDGET


def _mix64(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, elementwise on uint64. Returns a new array.

    The Horner hash is linear in its inputs, which leaves the low bits
    weakly mixed; minhash takes minima over full 64-bit values, so every
    bit must look random. This finalizer is a well-studied bijection with
    full avalanche.
    """
    x = x.copy()
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


class MinHashLSH:
    """Seeded MinHash signatures with banded LSH clustering.

    Signatures are int64 [NUM_PERM] vectors. Permutation i is the affine
    map h_i(x) = (a_i * x + b_i) mod 2**64 with a_i odd (a bijection), and
    slot i of a signature is min over the text's shingle hashes, shifted
    right one bit so the value fits int64 without changing which shingle
    wins (the shift is monotonic).
    """

    def __init__(self, seed: int = 1337, threshold: float = JACCARD_THRESHOLD) -> None:
        self.seed = int(seed)
        self.threshold = float(threshold)
        rng = np.random.default_rng(seed)
        self._a = rng.integers(0, 2**64, size=NUM_PERM, dtype=np.uint64) | np.uint64(1)
        self._b = rng.integers(0, 2**64, size=NUM_PERM, dtype=np.uint64)
        # How many texts hit the per-text input cap in the most recent
        # signatures() call. Callers surface it as a truncation stat.
        self.last_truncated_texts = 0

    def signature(self, text: str) -> np.ndarray:
        """text: raw string, normalized internally. Returns int64 [NUM_PERM]."""
        return self.signatures([text])[0]

    def signatures(self, texts: Sequence[str]) -> np.ndarray:
        """Batch signatures. texts: n raw strings. Returns int64 [n, NUM_PERM].

        Shingling, hashing, and the per-text minima are all vectorized;
        texts are grouped into batches bounded by total shingle count,
        and each text contributes at most _BATCH_SHINGLE_BUDGET shingles
        (about 1 MiB of its normalized encoding, see _shingle_input), so
        peak memory and per-text cost stay flat no matter the corpus or
        body size. self.last_truncated_texts holds the number of texts
        the per-text cap truncated in this call.
        """
        texts = list(texts)
        out = np.empty((len(texts), NUM_PERM), dtype=np.int64)
        self.last_truncated_texts = 0
        start = 0
        # The encoding that overflowed the previous batch is carried
        # into the next one, so normalization runs exactly once per text.
        carry: bytes | None = None
        while start < len(texts):
            end = start
            total = 0
            batch: list[bytes] = []
            while end < len(texts):
                enc = carry if carry is not None else self._shingle_input(texts[end])
                carry = None
                cnt = max(len(enc) - (_SHINGLE - 1), 1)
                if batch and total + cnt > _BATCH_SHINGLE_BUDGET:
                    carry = enc
                    break
                batch.append(enc)
                total += cnt
                end += 1
            out[start:end] = self._signatures_batch(batch)
            start = end
        return out

    def _shingle_input(self, text: str) -> bytes:
        """Normalized utf-8 shingle input for one text, cost-capped.

        Without a cap, one multi-megabyte body (a file dump, a base64
        blob) would pay one hash per byte times NUM_PERM passes: minutes
        of work and gigabytes of temporaries for a single text. The raw
        text is cut at _MAX_TEXT_CHARS characters before normalization
        (bounding regex cost) and the normalized encoding at
        _MAX_TEXT_SHINGLE_BYTES bytes (bounding shingle count at exactly
        _BATCH_SHINGLE_BUDGET windows). Byte truncation may split a
        multibyte codepoint; that is harmless because shingles are byte
        5-grams, not characters. Truncation bumps
        self.last_truncated_texts.
        """
        truncated = False
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS]
            truncated = True
        enc = _utf8(normalize(text))
        if len(enc) > _MAX_TEXT_SHINGLE_BYTES:
            enc = enc[:_MAX_TEXT_SHINGLE_BYTES]
            truncated = True
        if truncated:
            self.last_truncated_texts += 1
        return enc

    def _signatures_batch(self, encoded: list[bytes]) -> np.ndarray:
        """encoded: normalized utf-8 bodies. Returns int64 [len(encoded), NUM_PERM]."""
        n = len(encoded)
        lens = np.array([len(e) for e in encoded], dtype=np.int64)
        n_windows = np.maximum(lens - (_SHINGLE - 1), 0)
        # Every text owns a non-empty hash segment: its windows, or one
        # stand-in slot, so reduceat segment starts are strictly increasing.
        seg_sizes = np.maximum(n_windows, 1)
        seg_starts = np.cumsum(seg_sizes) - seg_sizes
        hashes = np.full(int(seg_sizes.sum()), _FILL, dtype=np.uint64)

        buf = np.frombuffer(b"".join(encoded), dtype=np.uint8)
        if buf.size >= _SHINGLE:
            # Horner hash of every width-5 window of the concatenated
            # buffer in 5 vector ops; windows that straddle a text
            # boundary are computed too but never selected below.
            n_all = buf.size - (_SHINGLE - 1)
            acc = np.zeros(n_all, dtype=np.uint64)
            for i in range(_SHINGLE):
                acc *= _K
                acc += buf[i : i + n_all]
            acc = _mix64(acc)

            offsets = np.cumsum(lens) - lens
            has = n_windows > 0
            reps = n_windows[has]
            if reps.size:
                # Scatter each text's in-bounds window hashes into its
                # segment: pos enumerates 0..cnt-1 within each text.
                pos = np.arange(int(reps.sum()), dtype=np.int64)
                pos -= np.repeat(np.cumsum(reps) - reps, reps)
                src = np.repeat(offsets[has], reps) + pos
                dst = np.repeat(seg_starts[has], reps) + pos
                hashes[dst] = acc[src]

        # Shingle-free texts get a stand-in hash derived from their full
        # normalized bytes, so only normalized-equal texts can share a
        # signature. Rare (encoding under 5 bytes), so a Python loop is
        # fine here.
        for i in np.flatnonzero(n_windows == 0):
            hashes[seg_starts[i]] = _short_text_hash(encoded[i])

        # Collapse each segment to its distinct hash values before the
        # permutation loop: a permutation is a function of the hash, so
        # the min over a multiset equals the min over its set of values
        # and signatures stay bit-identical, while repetitive bodies
        # (log spam, padding runs) shrink dramatically before paying for
        # NUM_PERM passes. Skipped when every segment is a single hash,
        # where it could only be a no-op.
        if hashes.size > n:
            seg_ids = np.repeat(np.arange(n, dtype=np.int64), seg_sizes)
            order = np.lexsort((hashes, seg_ids))
            sorted_hashes = hashes[order]
            sorted_ids = seg_ids[order]
            keep = np.empty(sorted_hashes.size, dtype=bool)
            keep[0] = True
            keep[1:] = (sorted_hashes[1:] != sorted_hashes[:-1]) | (
                sorted_ids[1:] != sorted_ids[:-1]
            )
            hashes = sorted_hashes[keep]
            # Every segment keeps at least one value, so the reduceat
            # invariant (strictly increasing starts) still holds.
            seg_sizes = np.bincount(sorted_ids[keep], minlength=n)
            seg_starts = np.cumsum(seg_sizes) - seg_sizes

        # One permutation at a time keeps every op 1-D and contiguous:
        # measured about 4x faster than a 2-D [shingles, perms] reduceat,
        # whose axis-0 segments stride badly through memory.
        sig = np.empty((n, NUM_PERM), dtype=np.uint64)
        for j in range(NUM_PERM):
            permuted = hashes * self._a[j]
            permuted += self._b[j]
            sig[:, j] = np.minimum.reduceat(permuted, seg_starts)
        # The shift is monotonic, so shifting after the min picks the same
        # winners while making every value non-negative in int64.
        return (sig >> np.uint64(1)).astype(np.int64)

    @staticmethod
    def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Fraction of agreeing slots: an unbiased jaccard estimate.

        sig_a, sig_b: int64 [NUM_PERM]. Returns a float in [0, 1]; equal
        signatures give exactly 1.0.
        """
        a = np.asarray(sig_a)
        b = np.asarray(sig_b)
        if a.size == 0 or a.shape != b.shape:
            raise ValueError(
                f"signatures must be non-empty and same-shape, got {a.shape} and {b.shape}"
            )
        return float(np.count_nonzero(a == b)) / float(a.size)

    def cluster(self, sigs: np.ndarray) -> list[list[int]]:
        """Group signature rows into near-duplicate components.

        sigs: int64 [n, NUM_PERM]. Returns components with at least 2
        members as sorted row-index lists, ordered by first member.
        Banding proposes candidates (rows sharing all ROWS values in any
        band); every candidate pair is verified with estimated_jaccard
        against self.threshold before union. The output depends only on
        sigs and the threshold, never on dict or set iteration order.
        """
        sigs = np.ascontiguousarray(sigs, dtype=np.int64)
        n = sigs.shape[0]
        if n < 2:
            return []

        # One 64-bit Horner hash per (row, band): grouping by this value
        # stands in for grouping by the exact 16-value band. A 64-bit
        # collision is vanishingly rare and merely proposes a candidate
        # pair, which verification then rejects.
        v = sigs.astype(np.uint64).reshape(n, BANDS, ROWS)
        band_hash = np.zeros((n, BANDS), dtype=np.uint64)
        for r in range(ROWS):
            band_hash *= _K
            band_hash += v[:, :, r]
        band_hash = _mix64(band_hash)

        parent = np.arange(n, dtype=np.int64)

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = int(parent[root])
            while parent[x] != root:
                parent[x], x = root, int(parent[x])
            return root

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                # Smaller root wins: keeps the merge order-independent.
                if ry < rx:
                    rx, ry = ry, rx
                parent[ry] = rx

        need = self.threshold * NUM_PERM
        for band in range(BANDS):
            col = band_hash[:, band]
            order = np.argsort(col, kind="stable")
            svals = col[order]
            starts = np.flatnonzero(np.concatenate(([True], svals[1:] != svals[:-1])))
            ends = np.append(starts[1:], n)
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                members = np.sort(order[s:e])
                # Verify each member against one representative per
                # component already seen in this bucket. Buckets are
                # dominated by mutually similar texts, so this stays
                # near-linear where naive pairwise would be quadratic.
                reps: list[int] = []
                for m64 in members:
                    m = int(m64)
                    placed = False
                    for rep in reps:
                        if find(rep) == find(m):
                            placed = True
                            break
                        if np.count_nonzero(sigs[rep] == sigs[m]) >= need:
                            union(rep, m)
                            placed = True
                            break
                    if not placed:
                        reps.append(m)

        components: dict[int, list[int]] = {}
        for i in range(n):
            components.setdefault(find(i), []).append(i)
        out = [sorted(ms) for ms in components.values() if len(ms) >= 2]
        out.sort(key=lambda ms: ms[0])
        return out


@dataclass(frozen=True)
class DedupStats:
    """Counters from one find_clusters run.

    unique_texts: distinct byte contents seen.
    exact_dup_bytes_saved: utf-8 bytes not stored thanks to interning,
        sum over exact groups of (copies - 1) * body_bytes.
    near_dup_clusters: number of reported near-duplicate clusters.
    skipped_over_cap: eligible unique texts left un-minhashed by the cap.
    minhash_truncated: minhashed texts whose shingle input hit the
        per-text cap (about 1 MiB of normalized utf-8), so their
        near-duplicate comparison used a prefix of the body.
    """

    unique_texts: int
    exact_dup_bytes_saved: int
    near_dup_clusters: int
    skipped_over_cap: int
    minhash_truncated: int = 0


@dataclass
class DedupResult:
    """Outcome of find_clusters over one text corpus.

    clusters: near-duplicate clusters as sorted input-index lists
        (every occurrence of every member text, exact copies included),
        ordered by first member. Reporting only: storage is unaffected.
    exact_dup_groups: byte-identical groups with at least 2 occurrences,
        as sorted input-index lists in first-appearance order. These
        share storage via sha256 interning.
    stats: DedupStats counters.
    """

    clusters: list[list[int]]
    exact_dup_groups: list[list[int]]
    stats: DedupStats


def find_clusters(
    texts_iter: Iterable[str],
    seed: int = 1337,
    cap: int = MINHASH_CAP_DEFAULT,
    threshold: float = JACCARD_THRESHOLD,
) -> DedupResult:
    """Find exact and near-duplicate text bodies in one pass.

    texts_iter: any iterable of strings, consumed once; indices in the
    result refer to iteration order. seed drives the minhash
    permutations. cap bounds how many eligible unique texts (strictly
    longer than MINHASH_MIN_CHARS characters) enter the minhash path;
    the overflow is counted in stats.skipped_over_cap. threshold is the
    jaccard level a verified candidate pair must reach. Bodies whose
    normalized encoding exceeds the per-text shingle cap are compared
    by prefix and counted in stats.minhash_truncated.

    Memory holds one copy of each distinct body plus per-unique
    bookkeeping, never the full input stream.
    """
    interned: dict[bytes, int] = {}
    bodies: list[str] = []
    occurrences: list[list[int]] = []
    nbytes: list[int] = []
    for index, item in enumerate(texts_iter):
        # Defensive coercion: dedup runs deep inside audit over user
        # content, and a stray None or non-string must never crash it.
        text = item if isinstance(item, str) else ("" if item is None else str(item))
        raw = _utf8(text)
        digest = hashlib.sha256(raw).digest()
        uid = interned.get(digest)
        if uid is None:
            interned[digest] = len(bodies)
            bodies.append(text)
            occurrences.append([index])
            nbytes.append(len(raw))
        else:
            occurrences[uid].append(index)

    exact_dup_groups = [occ for occ in occurrences if len(occ) >= 2]
    bytes_saved = sum(
        (len(occ) - 1) * nb for occ, nb in zip(occurrences, nbytes) if len(occ) >= 2
    )

    eligible = [u for u in range(len(bodies)) if len(bodies[u]) > MINHASH_MIN_CHARS]
    cap = max(int(cap), 0)
    skipped = max(len(eligible) - cap, 0)
    kept = eligible[:cap]

    clusters: list[list[int]] = []
    truncated = 0
    if len(kept) >= 2:
        lsh = MinHashLSH(seed=seed, threshold=threshold)
        sigs = lsh.signatures([bodies[u] for u in kept])
        truncated = lsh.last_truncated_texts
        for component in lsh.cluster(sigs):
            members: list[int] = []
            for pos in component:
                members.extend(occurrences[kept[pos]])
            clusters.append(sorted(members))
        clusters.sort(key=lambda ms: ms[0])

    stats = DedupStats(
        unique_texts=len(bodies),
        exact_dup_bytes_saved=bytes_saved,
        near_dup_clusters=len(clusters),
        skipped_over_cap=skipped,
        minhash_truncated=truncated,
    )
    return DedupResult(
        clusters=clusters,
        exact_dup_groups=exact_dup_groups,
        stats=stats,
    )
