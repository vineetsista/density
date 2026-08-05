## What this changes

<!-- One paragraph. Link the issue if there is one. -->

## Checklist

- [ ] `pytest -q` passes.
- [ ] `ruff check .` passes.
- [ ] No measured number is typed by hand. Benchmark tables come from
      `python scripts/update_readme_bench.py`.
- [ ] A missed target, if any, is recorded in `GAPS.md` with its measured
      value. Targets are never lowered.
- [ ] Any deviation from `docs/design/00-contracts.md` updates that file and
      `DECISIONS.md` in the same commit.
- [ ] No em dashes in prose, docstrings, comments, or templates.
- [ ] Commit messages use conventional-commit prefixes.
