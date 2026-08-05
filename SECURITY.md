# Security

## Reporting a vulnerability

Report privately through GitHub Security Advisories:
<https://github.com/vineetsista/density/security/advisories/new>. Please do
not open a public issue for a vulnerability. Include the version, the exact
command or API call, and the seed if one was involved; everything in this
project is seeded, so a report with a seed usually reproduces.

## Supported versions

The `main` branch is the supported version. This project is pre-1.0 and
fixes land on `main` rather than in backported releases.

## Threat model

DENSITY is a local data-processing library. It makes no network calls
anywhere in the core library: not in ingest, not in the engine, not in the
audit, and the HTML report is self-contained with no external CSS, fonts, or
scripts. The only socket the project opens is the one `density serve` binds
when you run that command.

### `density serve` is unauthenticated by design

The HTTP service has **no authentication of any kind**. It is a localhost
development surface. It binds `127.0.0.1:8377` by default, and the `--host`
flag warns when you point it anywhere else.

`POST /audit` takes a filesystem path, reads that corpus, and writes
`PATH.report.html` and `PATH.report.md` next to it. The CLI confines this to
the served store's parent directory (`create_app(store, audit_root=...)`),
and a path resolving outside that root is rejected with a 400. Callers that
build the app themselves may pass `audit_root=None` to lift the confinement,
which is only appropriate behind their own authorization layer.

Audit responses embed dedup cluster samples, which are the first 120
characters of real content bodies from the audited corpus. Treat a report
and an audit response as carrying the same sensitivity as the corpus.

### Untrusted input

Trace and embedding files are treated as untrusted: malformed lines are
counted, quarantined, and reported rather than raised, and pathological
input (invalid UTF-8, truncated JSON, deeply nested JSON, multi-megabyte
single lines, binary garbage) is exercised by the test suite. Report any
input that crashes a pipeline, hangs it, or escapes the output directory.

Known resource characteristics, documented rather than defended against,
because they are properties of a local batch tool and not of a service:

- The audit holds the whole embedding matrix in memory, about 3.1 GB at the
  1M x 768 design point.
- Dedup retains one copy of each distinct content body plus one index per
  event, so a corpus of mostly unique text costs roughly its own unique
  bytes.

### Compiled kernels

The `accel` extra compiles C++20 kernels on first import with the system
compiler and caches the result under your user cache directory. The cache
key covers both source files, the interpreter version and ABI, the compiler
version, and the host CPU target, so a cache on shared storage cannot hand
one machine a binary built for another's instruction set. Set
`DENSITY_ACCEL_DISABLE=1` to skip the compiled path entirely.
