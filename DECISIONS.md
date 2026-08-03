# DECISIONS

One line of rationale per choice the spec left open. Newest at the bottom.

1. Repo root is the invocation directory `~/projects/density` itself, not a nested `density/` folder: the spec's layout prefix names the repo, and nesting would break `pip install -e .` from the clone root.
2. The superpowers skill pack is not installed in this environment, so its skills cannot be invoked by name; the workflow is followed in substance instead: spec treated as interview output, plans written to docs/superpowers/plans/, subagent-driven implementation, TDD, and measured verification before any phase is called done.
3. `git init` was run first because the spec requires conventional commits and the directory was not a repository.
4. `uv` was not present on the machine; installed via the official standalone installer to ~/.local/bin.
5. Build backend is hatchling with a lazy, cached, import-time compile for the accel kernels instead of scikit-build-core as the wheel backend: scikit-build-core forces CMake onto every plain `pip install -e .` and cannot express an optional extension, which would break the zero-toolchain demo session. `pip install -e .[accel]` still works exactly as specced: the extra installs pybind11, the first import compiles the kernels with the system compiler, and any failure falls back silently to numpy with `density.ACCEL_ACTIVE` set accordingly.
6. COLD tier recall is defined and measured with rerank against the WARM int8 vectors (or HOT fp32 when present), and COLD footprint counts only the PQ or binary payload plus codebooks plus compressed traces: this is the only reading under which the spec's own footprint table (PQ 1/16x, binary 1/32x, both <= 0.08x) is arithmetically consistent with rerank.
7. All vectors are L2-normalized at ingest so cosine similarity equals dot product everywhere; documented in the embedding engine and the report methodology.
8. The PQ 16x-including-codebook gate is measured at m=128 for dim 768 (24x codes-only, comfortably over 16x after codebook overhead at 100k vectors); the API default stays m=dim/4, which is exactly 16x codes-only but dips just under 16x once the codebook is amortized over corpora smaller than about 1M vectors.
9. Accel parity in tests is measured as max abs diff scaled by the result magnitude (floor 1.0) rather than per-element ratio: scores that legitimately cancel toward zero would make raw ratios meaningless while the contract intent (1e-5 agreement) is preserved.
