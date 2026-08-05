"""Shared helpers for gate scripts: machine and dependency fingerprints.

Every published result file carries both, so a number can always be tied
to the machine and the dependency set that produced it. Dependencies are
declared with lower bounds only, so without the versions block a reader
resolving the project fresh has no way to know what was measured.
"""

from __future__ import annotations

import os
import platform
import sys

# Packages whose version can move a measured number: the numeric kernels,
# the columnar writer, the compressor, and the library itself.
_TRACKED_PACKAGES = ("density", "numpy", "pyarrow", "zstandard", "pydantic")


def machine_info() -> dict:
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }


def library_versions() -> dict:
    """Installed versions of every package that can move a measured number."""
    from importlib import metadata

    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions
