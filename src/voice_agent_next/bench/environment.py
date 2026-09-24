"""Reproducibility metadata for run manifests (research note 06, §8.6).

Collects versions, the git SHA of the voice-agent-next source tree (when running from a
checkout), the lockfile hash, hardware (CPU model and cores, RAM, NVIDIA GPUs, CPU
frequency governor), OS and Python. Every probe is best-effort: failures yield ``None``,
never an exception, and nothing here touches the network.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = ["collect_environment"]

_PACKAGES = ("voice-agent-next", "numpy", "soxr", "pydantic", "onnxruntime", "torch")
_TIMEOUT = 5.0


def _run(*cmd: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _source_root() -> Path | None:
    """Root of the git checkout containing this package (``None`` for installed wheels)."""
    pkg = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None:
        return None
    top = _run("git", "rev-parse", "--show-toplevel", cwd=pkg)
    if top is None:
        return None
    root = Path(top)
    try:
        if (root / "src" / "voice_agent_next").resolve() != pkg:
            return None  # e.g. a virtualenv inside some other repository
    except OSError:
        return None
    return root


def _git_info(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    sha = _run("git", "rev-parse", "HEAD", cwd=root)
    if sha is None:
        return None
    status = _run("git", "status", "--porcelain", "--untracked-files=no", cwd=root)
    return {
        "sha": sha,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        "dirty": bool(status),
    }


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        model = _run("sysctl", "-n", "machdep.cpu.brand_string")
        if model:
            return model
    return platform.processor() or None


def _cpu_governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _total_memory() -> int | None:
    if sys.platform == "win32":

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    if shutil.which("nvidia-smi"):
        out = _run(
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
        for line in (out or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                memory = int(parts[2]) if parts[2].isdigit() else None
                gpus.append({"name": parts[0], "driver": parts[1], "memory_mib": memory})
    if sys.platform == "darwin" and platform.machine() == "arm64":
        gpus.append({"name": "Apple Silicon GPU (integrated)"})
    return gpus


def _packages() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_environment() -> dict[str, Any]:
    """Versions, git state, hardware, OS and Python of the machine running the benchmark."""
    root = _source_root()
    lockfile = root / "uv.lock" if root is not None else None
    return {
        "packages": _packages(),
        "git": _git_info(root),
        "lockfile_sha256": _sha256_file(lockfile) if lockfile and lockfile.exists() else None,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
            "governor": _cpu_governor(),
        },
        "memory_bytes": _total_memory(),
        "gpus": _gpus(),
    }
