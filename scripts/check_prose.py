#!/usr/bin/env python3
"""Enforce the project's prose rules on every tracked text file.

One rule today, from docs/design/00-contracts.md: no em dashes and no en
dashes in prose, docstrings, comments, or templates. Commas, colons, and
periods do the job. The rule exists so the writing stays in one voice, and
it is checked by a script rather than remembered so it cannot rot.

Written in python rather than as a grep step because `grep -P '\\x{2014}'`
depends on the runner's locale being UTF-8, and a silently passing check is
worse than no check.

    python scripts/check_prose.py          # exit 1 on any violation

Only files git already tracks are scanned, so generated corpora, caches,
and virtualenvs are out of scope by construction.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Extensions whose contents are prose or carry prose (docstrings, comments,
# templates). Binary and data files are skipped: a measured result file may
# legitimately contain any byte.
TEXT_SUFFIXES = {".py", ".md", ".j2", ".toml", ".yml", ".yaml", ".cpp", ".h", ".txt"}

# Spelled as code points on purpose: writing the characters as literals
# would make this file the only violation of the rule it enforces, and the
# check would fail on itself forever.
BANNED = {
    chr(0x2014): "em dash",
    chr(0x2013): "en dash",
}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / name for name in out.split("\0") if name]


def main() -> int:
    violations: list[str] = []
    files = tracked_files()
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(ROOT)
        for line_no, line in enumerate(text.splitlines(), start=1):
            # Every occurrence, not just the first on the line: reporting one
            # at a time would make fixing a line an iterative game.
            for column, char in enumerate(line, start=1):
                name = BANNED.get(char)
                if name is not None:
                    violations.append(
                        f"{rel}:{line_no}:{column}: {name}: {line.strip()[:100]}"
                    )
    if violations:
        print("Prose rule violations (use commas, colons, periods):", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print(f"prose rules clean across {len(files)} tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
