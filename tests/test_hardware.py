"""Hardware detection, CUDA library discovery/loading and backend selection.

Everything runs against fakes (NVML, ``nvidia-smi``, ``ctranslate2``, ``onnxruntime``,
site-packages trees, the dynamic loader), so the results do not depend on this machine.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from voice_agent_next import hardware
from voice_agent_next.cli.main import app
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.hardware import GPU, CudaLibrary, NvidiaInfo

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
RTX = GPU(0, "NVIDIA GeForce RTX 5070 Ti", 16303, (12, 0))


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    hardware.clear_cache()
    yield
    monkeypatch.undo()  # restore the cached functions before clearing their caches
    hardware.clear_cache()


def nvidia(*gpus: GPU, cuda: tuple[int, int] | None = (13, 4)) -> NvidiaInfo:
    return NvidiaInfo(gpus, "615.71.09" if gpus else None, cuda if gpus else None, "nvml")


def fake_ct2(devices: int = 1, version: str = "4.8.2") -> ModuleType:
    ct2 = ModuleType("ctranslate2")
    ct2.__version__ = version  # type: ignore[attr-defined]
    ct2.get_cuda_device_count = lambda: devices  # type: ignore[attr-defined]
    supported = {"cpu": {"int8", "int8_float32", "float32"}, "cuda": {"float16", "int8", "float32"}}
    ct2.get_supported_compute_types = lambda device: supported[device]  # type: ignore[attr-defined]
    return ct2


def fake_ort(*providers: str, cuda_version: str | None = None) -> ModuleType:
    ort = ModuleType("onnxruntime")
    ort.__version__ = "1.30.0"  # type: ignore[attr-defined]
    ort.get_available_providers = lambda: list(providers or (CPU,))  # type: ignore[attr-defined]
    if cuda_version is not None:
        ort.cuda_version = cuda_version  # type: ignore[attr-defined]
    return ort


def fake_loader(monkeypatch: pytest.MonkeyPatch, *, loaded: bool) -> list[tuple[Any, int]]:
    """Replace the CUDA library loader; returns the recorded requests."""
    calls: list[tuple[Any, int]] = []

    def load(components: Any, cuda_major: int) -> tuple[CudaLibrary, ...]:
        calls.append((tuple(components), cuda_major))
        found = hardware.find_cuda_libraries(components, cuda_major, search_path=[])
        return tuple(
            CudaLibrary(lib.component, lib.name, lib.files, lib.package, loaded=loaded)
            for lib in found
        )

    monkeypatch.setattr(hardware, "load_cuda_libraries", load)
    return calls


def touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


# ------------------------------------------------------------------------ detection
def test_parse_nvidia_smi() -> None:
    output = (
        "0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0\n"
        "1, NVIDIA A100-SXM4-80GB, 81920, 8.0\n"
        "2, Some GPU, [N/A], [N/A]\n"
        "garbage line\n"
    )
    gpus = hardware.parse_nvidia_smi(output)
    assert gpus[0] == RTX
    assert gpus[1].compute_capability == (8, 0)
    assert gpus[2] == GPU(2, "Some GPU", None, None)
    assert len(gpus) == 3
    assert str(RTX) == "NVIDIA GeForce RTX 5070 Ti (16 GB, compute capability 12.0)"


class FakeNvml:
    """The NVML calls :func:`hardware._query_nvml` makes, answered through ctypes refs."""

    def __init__(self, gpus: list[GPU], *, init_status: int = 0) -> None:
        self.gpus = gpus
        self.init_status = init_status
        self.shut_down = False

    def nvmlInit_v2(self) -> int:
        return self.init_status

    def nvmlShutdown(self) -> int:
        self.shut_down = True
        return 0

    def nvmlSystemGetDriverVersion(self, buf: Any, size: int) -> int:
        buf.value = b"615.71.09"
        return 0

    def nvmlSystemGetCudaDriverVersion_v2(self, ref: Any) -> int:
        ref._obj.value = 13040
        return 0

    def nvmlDeviceGetCount_v2(self, ref: Any) -> int:
        ref._obj.value = len(self.gpus)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index: int, ref: Any) -> int:
        ref._obj.value = index + 1
        return 0

    def _gpu(self, handle: Any) -> GPU:
        return self.gpus[handle.value - 1]

    def nvmlDeviceGetName(self, handle: Any, buf: Any, size: int) -> int:
        buf.value = self._gpu(handle).name.encode()
        return 0

    def nvmlDeviceGetMemoryInfo(self, handle: Any, ref: Any) -> int:
        memory = self._gpu(handle).memory_mib
        if memory is None:
            return 999  # NVML_ERROR_UNKNOWN
        ref._obj.total = memory * 2**20
        return 0

    def nvmlDeviceGetCudaComputeCapability(self, handle: Any, major: Any, minor: Any) -> int:
        capability = self._gpu(handle).compute_capability
        assert capability is not None
        major._obj.value, minor._obj.value = capability
        return 0


def test_detect_nvidia_through_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    nvml = FakeNvml([RTX, GPU(1, "Tesla T4", None, (7, 5))])
    monkeypatch.setattr(hardware, "_nvml_candidates", lambda: ["libnvidia-ml.so.1"])
    monkeypatch.setattr(ctypes, "CDLL", lambda name, *a, **k: nvml)
    monkeypatch.setattr(hardware, "_query_nvidia_smi", lambda: pytest.fail("NVML suffices"))
    info = hardware.detect_nvidia()
    assert info == NvidiaInfo((RTX, GPU(1, "Tesla T4", None, (7, 5))), "615.71.09", (13, 4), "nvml")
    assert nvml.shut_down
    assert hardware.detect_nvidia() is info  # cached


def test_detect_nvidia_falls_back_to_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_library(name: str, *args: Any, **kwargs: Any) -> Any:
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", no_library)
    monkeypatch.setattr(hardware.shutil, "which", lambda exe: "/usr/bin/nvidia-smi")
    outputs = {
        "--query-gpu=index,name,memory.total,compute_cap": "0, NVIDIA GeForce RTX 5070 Ti, 16303, 12.0\n",
        "--query-gpu=driver_version": "615.71.09\n",
    }  # fmt: skip

    def run(args: list[str], **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args, 0, stdout=outputs[args[1]])

    monkeypatch.setattr(hardware.subprocess, "run", run)
    info = hardware.detect_nvidia()
    assert (info.gpus, info.driver_version, info.source) == ((RTX,), "615.71.09", "nvidia-smi")


@pytest.mark.parametrize("failure", ["nvml-init", "smi-error", "nothing"])
def test_detect_nvidia_never_raises(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    nvml = FakeNvml([RTX], init_status=9)  # NVML_ERROR_DRIVER_NOT_LOADED
    monkeypatch.setattr(hardware, "_nvml_candidates", lambda: ["libnvidia-ml.so.1"])
    monkeypatch.setattr(ctypes, "CDLL", lambda name, *a, **k: nvml)
    which = None if failure == "nothing" else "/usr/bin/nvidia-smi"
    monkeypatch.setattr(hardware.shutil, "which", lambda exe: which)

    def run(args: list[str], **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(9, args, "NVIDIA-SMI has failed")

    monkeypatch.setattr(hardware.subprocess, "run", run)
    assert hardware.detect_nvidia() == NvidiaInfo()
    assert not nvml.shut_down  # only after a successful init


def test_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert hardware.detect_apple_silicon() is None
    hardware.clear_cache()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware, "_sysctl", {"machdep.cpu.brand_string": "Apple M3 Pro"}.get)
    monkeypatch.setattr(hardware, "is_installed", lambda module: module == "mlx")
    chip = hardware.detect_apple_silicon()
    assert chip == hardware.AppleSilicon("Apple M3 Pro", rosetta=False, mlx=True)
    hardware.clear_cache()
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    sysctl = {"sysctl.proc_translated": "1", "machdep.cpu.brand_string": "Apple M1"}
    monkeypatch.setattr(hardware, "_sysctl", sysctl.get)
    rosetta = hardware.detect_apple_silicon()
    assert rosetta is not None and rosetta.rosetta
    assert str(rosetta) == "Apple M1 (x86_64 Python under Rosetta 2)"
    hardware.clear_cache()
    monkeypatch.setattr(hardware, "_sysctl", {}.get)  # an Intel Mac
    assert hardware.detect_apple_silicon() is None


def test_runtime_info_from_fake_modules() -> None:
    ort = hardware.onnxruntime_info(fake_ort(CUDA, CPU, cuda_version="12.8"))
    assert ort is not None
    assert (ort.providers, ort.cuda_major) == ((CUDA, CPU), 12)
    assert "1.30.0 (CUDA 12.8)" in ort.build
    assert hardware.onnxruntime_info(fake_ort()).cuda_major is None  # type: ignore[union-attr]

    ct2 = hardware.ctranslate2_info(fake_ct2(devices=1))
    assert ct2 is not None
    assert (ct2.cuda_devices, ct2.cuda_major) == (1, 12)
    assert "float16" in ct2.cuda_compute_types
    broken = fake_ct2()
    broken.get_cuda_device_count = lambda: 1 / 0  # type: ignore[attr-defined]
    assert hardware.ctranslate2_info(broken).cuda_devices == 0  # type: ignore[union-attr]
    assert hardware.ctranslate2_info(fake_ct2(version="5.0.0")).cuda_major is None  # type: ignore[union-attr]


def test_compute_types() -> None:
    ct2 = fake_ct2()
    assert hardware.ctranslate2_compute_type(ct2, "cuda") == "float16"
    assert hardware.ctranslate2_compute_type(ct2, "cpu") == "int8"
    assert hardware.ctranslate2_compute_type(ct2, "cuda", "int8_float16") == "int8_float16"
    ct2.get_supported_compute_types = lambda device: {"float32"}  # type: ignore[attr-defined]
    assert hardware.ctranslate2_compute_type(ct2, "cuda") == "float32"
    ct2.get_supported_compute_types = lambda device: 1 / 0  # type: ignore[attr-defined]
    assert hardware.ctranslate2_compute_type(ct2, "cpu") == "default"


# --------------------------------------------------------------- CUDA library discovery
def test_find_cuda_12_wheels_linux_layout(tmp_path: Path) -> None:
    site = tmp_path / "site-packages"
    touch(
        site / "nvidia",
        "cublas/lib/libcublas.so.12",
        "cublas/lib/libcublasLt.so.12",
        "cudnn/lib/libcudnn.so.9",
        "cuda_runtime/lib/libcudart.so.12",
        "cufft/lib/libcufft.so.11",
    )
    libs = {
        lib.component: lib
        for lib in hardware.find_cuda_libraries(
            hardware.ONNXRUNTIME_CUDA_LIBRARIES, 12, search_path=[str(tmp_path / "x"), str(site)], windows=False
        )
    }  # fmt: skip
    cublas = libs["cublas"]
    assert cublas.files == ("libcublasLt.so.12", "libcublas.so.12")
    assert cublas.path == site / "nvidia" / "cublas" / "lib"
    assert (cublas.package, cublas.name, cublas.source) == (
        "nvidia-cublas-cu12",
        "cuBLAS 12",
        "pip",
    )
    assert libs["cudnn"].path == site / "nvidia" / "cudnn" / "lib"
    assert libs["cudnn"].package == "nvidia-cudnn-cu12"
    assert libs["cufft"].path is not None and libs["cufft"].name == "cuFFT 11"
    assert libs["curand"].path is None and not libs["curand"].found  # not installed
    assert str(libs["curand"]) == "cuRAND 10: missing"
    assert str(cublas) == "cuBLAS 12: pip (nvidia-cublas-cu12)"


def test_find_cuda_libraries_windows_layouts(tmp_path: Path) -> None:
    cu12, cu13 = tmp_path / "cu12", tmp_path / "cu13"
    touch(cu12 / "nvidia", "cublas/bin/cublas64_12.dll", "cublas/bin/cublasLt64_12.dll")
    touch(
        cu13 / "nvidia",
        "cu13/bin/x86_64/cublas64_13.dll",
        "cu13/bin/x86_64/cublasLt64_13.dll",
        "cudnn/bin/cudnn64_9.dll",
    )
    (cublas12,) = hardware.find_cuda_libraries(
        ("cublas",), 12, search_path=[str(cu12)], windows=True
    )
    assert cublas12.files == ("cublasLt64_12.dll", "cublas64_12.dll")
    assert cublas12.path == cu12 / "nvidia" / "cublas" / "bin"
    cublas13, cudnn = hardware.find_cuda_libraries(
        ("cublas", "cudnn"), 13, search_path=[str(cu13)], windows=True
    )
    assert cublas13.path == cu13 / "nvidia" / "cu13" / "bin" / "x86_64"
    assert cublas13.package == "nvidia-cublas"  # CUDA 13 wheels dropped the -cuNN suffix
    assert (cudnn.path, cudnn.package) == (cu13 / "nvidia" / "cudnn" / "bin", "nvidia-cudnn-cu13")
    # the Linux names are not found in a Windows tree, and vice versa
    (linux,) = hardware.find_cuda_libraries(("cublas",), 12, search_path=[str(cu12)], windows=False)
    assert linux.path is None


def test_find_needs_every_file_of_a_library(tmp_path: Path) -> None:
    touch(tmp_path / "nvidia", "cublas/lib/libcublas.so.12")  # libcublasLt.so.12 missing
    (cublas,) = hardware.find_cuda_libraries(("cublas",), 12, search_path=[str(tmp_path)], windows=False)  # fmt: skip
    assert cublas.path is None
    with pytest.raises(ValueError, match="unknown CUDA library"):
        hardware.find_cuda_libraries(("cusparse",), 12, search_path=[])


# ------------------------------------------------------------------ CUDA library loading
@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[str]]:
    """Fake dynamic loader: records loads; file names in ``missing`` fail to load."""
    record: dict[str, list[str]] = {"files": [], "names": [], "missing": []}

    def open_file(path: Path) -> object:
        if path.name in record["missing"]:
            raise OSError(f"{path}: cannot open shared object file")
        record["files"].append(str(path))
        return object()

    def open_name(name: str) -> object:
        if name in record["missing"]:
            raise OSError(f"{name}: cannot open shared object file")
        record["names"].append(name)
        return object()

    monkeypatch.setattr(hardware, "_open_file", open_file)
    monkeypatch.setattr(hardware, "_open_name", open_name)
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.setattr(sys, "platform", "linux")
    return record


def test_load_prefers_pip_wheels_and_caches(loader: dict[str, list[str]], tmp_path: Path) -> None:
    touch(tmp_path / "nvidia", "cublas/lib/libcublas.so.12", "cublas/lib/libcublasLt.so.12")
    (cublas,) = hardware.load_cuda_libraries(("cublas",), 12)
    assert cublas.loaded and cublas.source == "pip"
    lib_dir = tmp_path / "nvidia" / "cublas" / "lib"
    # cublasLt first: libcublas depends on it
    assert loader["files"] == [str(lib_dir / "libcublasLt.so.12"), str(lib_dir / "libcublas.so.12")]
    assert hardware.load_cuda_libraries(("cublas",), 12) == (cublas,)
    assert len(loader["files"]) == 2  # cached


def test_load_falls_back_to_the_system_library_path(loader: dict[str, list[str]]) -> None:
    (cublas,) = hardware.load_cuda_libraries(("cublas",), 12)
    assert cublas.loaded and cublas.source == "system" and cublas.path is None
    assert loader["names"] == ["libcublasLt.so.12", "libcublas.so.12"]
    assert str(cublas) == "cuBLAS 12: system library path"


def test_load_reports_missing_libraries(loader: dict[str, list[str]], tmp_path: Path) -> None:
    touch(tmp_path / "nvidia", "cudnn/lib/libcudnn.so.9")
    loader["missing"] += ["libcudnn.so.9", "libcublasLt.so.12"]  # broken wheel, no system copy
    cublas, cudnn = hardware.load_cuda_libraries(("cublas", "cudnn"), 12)
    assert not cublas.loaded and "libcublasLt.so.12" in (cublas.error or "")
    assert not cudnn.loaded and str(tmp_path) in (cudnn.error or "")
    assert cudnn.source == "missing"


def test_windows_loads_cudnn_sub_libraries(
    loader: dict[str, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    touch(
        tmp_path / "nvidia",
        "cudnn/bin/cudnn64_9.dll",
        "cudnn/bin/cudnn_ops64_9.dll",
        "cudnn/bin/cudnn_graph64_9.dll",
    )
    (cudnn,) = hardware.load_cuda_libraries(("cudnn",), 12)
    assert cudnn.loaded
    assert [Path(f).name for f in loader["files"]] == [
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn64_9.dll",
    ]


def test_real_loader_never_raises_for_missing_libraries() -> None:
    """The real ctypes path: an absent library is an error value, not an exception."""
    (lib,) = hardware.load_cuda_libraries(("cublas",), 99)
    assert not lib.loaded and lib.error


# ------------------------------------------------------------------- ctranslate2 choice
def test_ctranslate2_auto_uses_cuda_when_the_libraries_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    calls = fake_loader(monkeypatch, loaded=True)
    backend = hardware.select_ctranslate2_backend(ct2=fake_ct2())
    assert (backend.device, backend.compute_type, backend.fix) == ("cuda", "float16", None)
    assert backend.reason == RTX.name
    assert calls == [(("cublas",), 12)]
    assert str(backend) == f"cuda float16 ({RTX.name})"


def test_ctranslate2_auto_names_the_fix_when_cublas_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    fake_loader(monkeypatch, loaded=False)
    backend = hardware.select_ctranslate2_backend(ct2=fake_ct2())
    assert (backend.device, backend.compute_type) == ("cpu", "int8")
    assert backend.fix == hardware.CUDA_EXTRA_HINT
    assert RTX.name in backend.reason and "cublas" in backend.reason.lower()


@pytest.mark.parametrize(
    ("gpus", "env", "cuda", "reason", "fix"),
    [
        ((), None, None, "no CUDA GPU", None),
        ((RTX,), "-1", (13, 4), "hidden by CUDA_VISIBLE_DEVICES", None),
        ((RTX,), None, (11, 8), "only supports CUDA 11.8", "update the NVIDIA driver"),
        ((RTX,), None, (13, 4), "has no CUDA support here", None),
    ],
)
def test_ctranslate2_auto_without_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
    gpus: tuple[GPU, ...],
    env: str | None,
    cuda: tuple[int, int] | None,
    reason: str,
    fix: str | None,
) -> None:
    if env is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(*gpus, cuda=cuda))
    calls = fake_loader(monkeypatch, loaded=True)
    backend = hardware.select_ctranslate2_backend(ct2=fake_ct2(devices=0))
    assert (backend.device, backend.compute_type, backend.fix) == ("cpu", "int8", fix)
    assert reason in backend.reason
    assert calls == []  # CTranslate2 sees no GPU: nothing to load


def test_ctranslate2_explicit_devices_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia())
    calls = fake_loader(monkeypatch, loaded=False)
    cpu = hardware.select_ctranslate2_backend("cpu", ct2=fake_ct2())
    assert (cpu.device, cpu.compute_type, cpu.reason) == ("cpu", "int8", "requested")
    assert calls == []  # nothing loaded for CPU
    cuda = hardware.select_ctranslate2_backend("cuda", "int8_float16", ct2=fake_ct2(devices=0))
    assert (cuda.device, cuda.compute_type) == ("cuda", "int8_float16")
    assert calls == [(("cublas",), 12)]  # loaded, so that the explicit request can work


# ----------------------------------------------------------------------- ONNX choice
def test_onnx_auto_prefers_a_usable_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    calls = fake_loader(monkeypatch, loaded=True)
    ort = fake_ort("TensorrtExecutionProvider", CUDA, CPU, cuda_version="12.8")
    backend = hardware.select_onnx_backend(ort=ort)
    assert (backend.device, backend.providers, backend.reason) == ("cuda", (CUDA, CPU), RTX.name)
    assert calls == [(hardware.ONNXRUNTIME_CUDA_LIBRARIES, 12)]


def test_onnx_auto_on_a_cpu_build_with_an_nvidia_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    backend = hardware.select_onnx_backend(ort=fake_ort(CPU))
    assert (backend.device, backend.providers) == ("cpu", (CPU,))
    assert backend.fix == hardware.ONNXRUNTIME_GPU_HINT
    assert "has no CUDAExecutionProvider" in backend.reason


def test_onnx_auto_with_missing_cuda_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    fake_loader(monkeypatch, loaded=False)
    backend = hardware.select_onnx_backend(ort=fake_ort(CUDA, CPU, cuda_version="12.8"))
    assert backend.device == "cpu"
    assert backend.fix is not None and "nvidia-cudnn-cu12" in backend.fix
    assert "libcudnn.so.9" in backend.reason or "cudnn64_9.dll" in backend.reason


def test_onnx_auto_without_gpu_or_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia())
    plain = hardware.select_onnx_backend(ort=fake_ort(CPU))
    assert (plain.device, plain.fix, plain.reason) == ("cpu", None, "no accelerator available")
    dml = hardware.select_onnx_backend(ort=fake_ort("DmlExecutionProvider", CPU))
    assert dml.providers == ("DmlExecutionProvider", CPU)
    # models measured to run best on CPU pass no accelerators
    vad = hardware.select_onnx_backend(accelerators=(), cpu_reason="tiny model", ort=fake_ort(CUDA, CPU))  # fmt: skip
    assert (vad.device, vad.reason) == ("cpu", "tiny model")


def test_onnx_explicit_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia())
    assert hardware.select_onnx_backend("cpu").providers == (CPU,)  # imports nothing
    coreml = hardware.select_onnx_backend("coreml", ort=fake_ort("CoreMLExecutionProvider", CPU))
    assert (coreml.device, coreml.reason) == ("coreml", "requested")
    with pytest.raises(ConfigurationError, match="onnxruntime-directml"):
        hardware.select_onnx_backend("directml", ort=fake_ort(CPU))
    with pytest.raises(ConfigurationError, match="device must be one of"):
        hardware.select_onnx_backend("tpu", ort=fake_ort(CPU))


def test_session_providers() -> None:
    class Session:
        def get_providers(self) -> list[Any]:
            return [CUDA, (CPU, {})]

    assert hardware.session_providers(Session()) == [CUDA, CPU]
    assert hardware.session_providers(object()) == []


# ------------------------------------------------------------------------- PyTorch
def fake_torch(
    cuda: str | None = "13.0",
    devices: int = 1,
    arch: tuple[str, ...] = ("sm_80", "sm_90", "sm_120"),
    capability: tuple[int, int] = (12, 0),
    mps: bool = False,
) -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = "2.14.0"  # type: ignore[attr-defined]
    torch.version = SimpleNamespace(cuda=cuda)  # type: ignore[attr-defined]
    torch.cuda = SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: devices > 0,
        device_count=lambda: devices,
        get_arch_list=lambda: list(arch),
        get_device_capability=lambda i: capability,
    )
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))  # type: ignore[attr-defined]
    return torch


def test_torch_info_and_arch_support() -> None:
    info = hardware.torch_info(fake_torch())
    assert info is not None
    assert (info.version, info.cuda_version, info.cuda_devices) == ("2.14.0", "13.0", 1)
    assert info.capabilities == ((12, 0),)
    assert info.supports((12, 0)) and info.supports((8, 6))  # sm_80 runs on 8.6
    old = hardware.TorchInfo("2.6.0", "12.4", 1, ("sm_50", "sm_80", "sm_90"))
    assert not old.supports((12, 0))  # Blackwell needs torch >= 2.7
    assert hardware.TorchInfo("x", "12.4", 1, ("sm_80", "compute_90")).supports((12, 0))  # PTX
    assert hardware.TorchInfo("x", "12.4", 1, ()).supports((12, 0))  # unknown: assume yes


def test_torch_auto_uses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    backend = hardware.select_torch_backend(torch=fake_torch())
    assert (backend.device, backend.reason, backend.fix) == ("cuda", RTX.name, None)
    explicit = hardware.select_torch_backend("cuda:0", torch=fake_torch())
    assert explicit.device == "cuda:0" and explicit.reason.startswith("requested")
    assert hardware.select_torch_backend("cpu", torch=fake_torch()).device == "cpu"


def test_torch_auto_explains_a_build_without_kernels_for_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    old = fake_torch(cuda="12.4", arch=("sm_50", "sm_80", "sm_90"))
    backend = hardware.select_torch_backend(torch=old)
    assert backend.device == "cpu"
    assert "sm_120" in backend.reason and "not supported" in backend.reason
    assert backend.fix == hardware.TORCH_CUDA_HINT
    with pytest.raises(ConfigurationError, match="not supported"):
        hardware.select_torch_backend("cuda", torch=old)


def test_torch_auto_explains_a_cpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    backend = hardware.select_torch_backend(torch=fake_torch(cuda=None, devices=0))
    assert backend.device == "cpu" and "CPU build" in backend.reason
    assert backend.fix == hardware.TORCH_CUDA_HINT
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia())
    plain = hardware.select_torch_backend(torch=fake_torch(cuda=None, devices=0))
    assert (plain.device, plain.fix) == ("cpu", None)


def test_torch_auto_hidden_gpu_and_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    hidden = hardware.select_torch_backend(torch=fake_torch(devices=0))
    assert hidden.device == "cpu" and "CUDA_VISIBLE_DEVICES" in hidden.reason
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia())
    mac = fake_torch(cuda=None, devices=0, mps=True)
    assert hardware.select_torch_backend(torch=mac).device == "mps"
    assert hardware.select_torch_backend(accelerators=("cuda",), torch=mac).device == "cpu"
    assert hardware.select_torch_backend("mps", torch=mac).device == "mps"


@pytest.mark.parametrize("device", ["gpu", "cuda:x", "mps:1"])
def test_torch_rejects_unknown_devices(device: str) -> None:
    with pytest.raises(ConfigurationError, match="device must be one of"):
        hardware.select_torch_backend(device, torch=fake_torch())


# ------------------------------------------------------------------------- reporting
@pytest.fixture
def fake_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """An RTX 5070 Ti, CTranslate2 4.8.2 with cuBLAS 12 installed, CPU-only ONNX Runtime."""
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia(RTX))
    monkeypatch.setattr(hardware, "detect_apple_silicon", lambda: None)
    fake_loader(monkeypatch, loaded=True)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2())
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort(CPU))
    monkeypatch.setattr(hardware, "is_installed", lambda module: True)


@pytest.mark.usefixtures("fake_machine")
def test_report_rows() -> None:
    rows = dict(hardware.report())
    assert rows["NVIDIA GPU"].startswith(
        "NVIDIA GeForce RTX 5070 Ti (16 GB, compute capability 12.0)"
    )
    assert "driver 615.71.09 (CUDA 13.4)" in rows["NVIDIA GPU"]
    assert rows["ctranslate2"].startswith("4.8.2, 1 CUDA device(s)")
    assert rows["CUDA 12 libraries (ctranslate2)"] == "cuBLAS 12: system library path"
    assert rows["faster-whisper device=auto"] == f"cuda float16 ({RTX.name})"
    assert rows["onnxruntime device=auto (Kokoro)"].endswith(
        f"to use the GPU: {hardware.ONNXRUNTIME_GPU_HINT}"
    )


@pytest.mark.usefixtures("fake_machine")
def test_report_torch_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    rows = dict(hardware.report())
    assert rows["torch"] == "2.14.0 (CUDA 13.0, 1 CUDA device(s))"
    assert rows["torch device=auto (Chatterbox, Qwen3-TTS)"] == f"cuda ({RTX.name})"


def test_report_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: nvidia())
    monkeypatch.setattr(hardware, "detect_apple_silicon", lambda: None)
    monkeypatch.setattr(hardware, "onnxruntime_info", lambda: None)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2(devices=0))
    monkeypatch.setattr(hardware, "is_installed", lambda module: True)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(hardware, "select_ctranslate2_backend", broken)
    rows = dict(hardware.report())
    assert rows["NVIDIA GPU"] == "none"
    assert rows["faster-whisper device=auto"] == "error: driver exploded"


@pytest.mark.usefixtures("fake_machine")
def test_doctor_has_a_hardware_section() -> None:
    result = CliRunner().invoke(app, ["doctor"], env={"COLUMNS": "250"})
    assert result.exit_code == 0, result.output
    for text in (
        "NVIDIA GeForce RTX 5070 Ti",
        "faster-whisper device=auto",
        "cuda float16",
        "onnxruntime-gpu[cuda,cudnn]",
    ):
        assert text in result.output
