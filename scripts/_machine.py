"""Shared helper for gate scripts: machine and environment fingerprint."""

from __future__ import annotations

import os
import platform
import sys


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
