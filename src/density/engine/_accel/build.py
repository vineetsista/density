"""Compile-on-first-import loader for the C++ kernels.

Why build here instead of at wheel build time: DECISIONS.md item 5. A plain
`pip install -e .` must succeed on toolchain-less machines, so the extension
is compiled lazily by the first import that wants it, cached per user, and
every failure mode degrades silently to the numpy fallback (set
DENSITY_ACCEL_DEBUG=1 to see why a build or load failed).

Cache layout: <user cache dir>/density/accel/<key>/_density_kernels<ext>,
where <key> hashes both source files, the python version and ABI suffix,
and the compiler version. A changed source or toolchain therefore lands in
a fresh directory and stale builds are never picked up. Builds go to a
temporary name first and are moved into place with os.replace, so two
processes racing to build the same key both end up loading a complete .so.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from types import ModuleType

_MODULE_NAME = "_density_kernels"
_CPP_DIR = Path(__file__).resolve().parent / "cpp"
_SOURCES = (_CPP_DIR / "kernels.cpp", _CPP_DIR / "bindings.cpp")
_COMPILE_TIMEOUT_S = 300

# The first attempt's outcome is memoized for the process lifetime: success
# because loading twice is wasteful, failure because retrying cannot succeed
# without an environment change and would re-pay compiler probing each call.
_ATTEMPTED = False
_LOADED: ModuleType | None = None


def _debug(msg: str) -> None:
    # Silent fallback is a spec requirement, so diagnostics are opt-in.
    if os.environ.get("DENSITY_ACCEL_DEBUG") == "1":
        print(f"[density._accel] {msg}", file=sys.stderr)


def _cache_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "density" / "accel"


def _find_compiler() -> str | None:
    for candidate in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _compiler_version(cxx: str) -> str | None:
    try:
        proc = subprocess.run(
            [cxx, "--version"], capture_output=True, text=True, timeout=30
        )
    except Exception as exc:
        _debug(f"compiler probe failed: {exc!r}")
        return None
    if proc.returncode != 0:
        _debug(f"compiler probe exited {proc.returncode}")
        return None
    return proc.stdout.splitlines()[0] if proc.stdout else "unknown"


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _cache_key(compiler_version: str) -> str:
    # Any input that changes codegen or ABI participates in the key, so a
    # source edit, interpreter upgrade, or toolchain swap forces a rebuild.
    h = hashlib.sha256()
    for src in _SOURCES:
        h.update(src.read_bytes())
    h.update(sys.version.encode())
    h.update(_ext_suffix().encode())
    h.update(compiler_version.encode())
    return h.hexdigest()[:16]


def _include_dirs() -> list[str]:
    import pybind11

    paths = sysconfig.get_paths()
    dirs = [pybind11.get_include(), paths["include"]]
    platinclude = paths.get("platinclude")
    if platinclude and platinclude not in dirs:
        dirs.append(platinclude)
    return dirs


def _compile(cxx: str, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        cxx,
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        "-DNDEBUG",
    ]
    if sys.platform == "darwin":
        # Python extension modules on macOS resolve interpreter symbols at
        # load time instead of linking libpython.
        base_cmd += ["-undefined", "dynamic_lookup"]
    base_cmd += [f"-I{inc}" for inc in _include_dirs()]
    base_cmd += [str(src) for src in _SOURCES]

    tmp_dir = Path(tempfile.mkdtemp(prefix="build-", dir=target.parent))
    tmp_so = tmp_dir / target.name
    try:
        # -march=native buys SIMD width and single-instruction popcount, but
        # some toolchains reject it (cross compilers, odd platforms): retry
        # portable if the tuned build fails.
        for extra in (["-march=native"], []):
            cmd = base_cmd[:1] + extra + base_cmd[1:] + ["-o", str(tmp_so)]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=_COMPILE_TIMEOUT_S
                )
            except Exception as exc:
                _debug(f"compile invocation failed: {exc!r}")
                return False
            if proc.returncode == 0:
                os.replace(tmp_so, target)
                return True
            _debug(
                f"compile with {extra or 'portable flags'} exited "
                f"{proc.returncode}: {proc.stderr[-2000:]}"
            )
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _import_so(path: Path) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        _debug(f"no import spec for {path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("sq8_scores", "pq_adc_scan", "hamming_scan"):
        if not hasattr(module, name):
            _debug(f"loaded module lacks {name}")
            return None
    return module


def load_compiled() -> ModuleType | None:
    """Return the compiled kernel module, building it if needed, else None.

    Never raises: any missing prerequisite, compile error, or import error
    returns None so the dispatch layer keeps the numpy fallback.
    """
    global _ATTEMPTED, _LOADED
    if _ATTEMPTED:
        return _LOADED
    _ATTEMPTED = True
    try:
        _LOADED = _load_uncached()
    except Exception as exc:
        _debug(f"unexpected failure: {exc!r}")
        _LOADED = None
    return _LOADED


def _load_uncached() -> ModuleType | None:
    try:
        import pybind11  # noqa: F401  (probe only; used via _include_dirs)
    except Exception:
        _debug("pybind11 not importable, install the [accel] extra")
        return None
    for src in _SOURCES:
        if not src.is_file():
            _debug(f"missing kernel source {src}")
            return None
    cxx = _find_compiler()
    if cxx is None:
        _debug("no C++ compiler found (checked CXX, g++, clang++, c++)")
        return None
    version = _compiler_version(cxx)
    if version is None:
        return None
    target = _cache_root() / _cache_key(version) / f"{_MODULE_NAME}{_ext_suffix()}"
    if not target.is_file() and not _compile(cxx, target):
        return None
    module = _import_so(target)
    if module is not None:
        _debug(f"loaded compiled kernels from {target}")
    return module
