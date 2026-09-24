"""Deep environment diagnostics behind ``van doctor`` (see ``docs/cli/doctor.md``).

Every section returns a list of :class:`Check` rows (``ok`` / ``info`` / ``warn`` /
``fail`` / ``skip``) that the CLI prints as tables or as JSON. The passive sections only
query the machine; the interactive ones use the audio devices and only run when asked:

* ``system``: Python, platform, ML runtimes, API keys;
* ``audio``: PortAudio version, host APIs (judged per OS), the sound server, default
  devices with their supported sample rates and reported latencies, plus warnings such as
  a Bluetooth headset in hands-free (HFP) mode or a device opened in exclusive mode;
* ``hardware``: GPUs and where local models run (:func:`voice_agent_next.hardware.report`);
* ``presets``: which presets are ready (:func:`voice_agent_next.presets.check_preset`);
* ``models``: the model cache (:mod:`voice_agent_next.models`);
* ``network`` (opt-in): DNS, TCP connect and TLS handshake times of the cloud endpoints of
  the providers whose API key is set, and the fastest region where a provider has several.
  No request is sent, so no API call is made;
* ``mic`` (opt-in): records the microphone for a few seconds and judges its level:
  silence, clipping, noise floor, speech level (:func:`level_stats`, :func:`level_verdicts`);
* ``echo`` (opt-in): plays a chirp, records it and measures the round-trip delay and the
  echo return loss, then recommends an echo mode (:func:`analyze_echo`);
* ``latency`` (opt-in): plays a train of short pulses and measures the acoustic loopback
  latency and its jitter (:func:`analyze_latency`).

The signal analysis functions are pure numpy and are unit-tested on synthetic signals;
device access goes through :class:`SoundDeviceIO`, which tests replace with a fake.
"""

from __future__ import annotations

import os
import platform as _platform
import re
import socket
import ssl
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import numpy as np
import numpy.typing as npt

from . import __version__
from .utils.deps import is_installed

__all__ = [
    "DEFAULT_SECTIONS",
    "SCHEMA",
    "SECTIONS",
    "AudioIO",
    "Check",
    "DoctorReport",
    "EchoResult",
    "Endpoint",
    "LatencyResult",
    "LevelStats",
    "ProbeResult",
    "SoundDeviceIO",
    "analyze_echo",
    "analyze_latency",
    "audio_checks",
    "chirp",
    "cloud_endpoints",
    "echo_check",
    "echo_recommendation",
    "estimate_delay",
    "hardware_checks",
    "latency_check",
    "level_stats",
    "level_verdicts",
    "mic_check",
    "model_checks",
    "network_checks",
    "preset_checks",
    "probe_endpoint",
    "pulse_train",
    "system_checks",
]

Status = Literal["ok", "info", "warn", "fail", "skip"]
SCHEMA = "van-doctor/1"
"""Identifies the JSON layout of :meth:`DoctorReport.to_dict`."""
SECTIONS = ("system", "audio", "hardware", "presets", "models", "network", "mic", "echo", "latency")
DEFAULT_SECTIONS = ("system", "audio", "hardware", "presets", "models")
"""Sections that only query the machine; the others are opt-in."""

_FULL_SCALE = 32768.0
_FLOOR_DB = -120.0
"""Level reported for digital silence (JSON has no ``-inf``)."""


# ------------------------------------------------------------------------------ report
@dataclass
class Check:
    """One diagnostic row."""

    section: str
    name: str
    status: Status
    value: str
    hint: str | None = None
    """How to fix or improve it."""
    data: dict[str, Any] = field(default_factory=dict)
    """Measured values (JSON-serializable), e.g. ``{"delay_ms": 132.0}``."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    """All checks of one ``van doctor`` run."""

    checks: list[Check] = field(default_factory=list)

    def extend(self, checks: Iterable[Check]) -> None:
        self.checks.extend(checks)

    def section(self, name: str) -> list[Check]:
        return [c for c in self.checks if c.section == name]

    @property
    def sections(self) -> list[str]:
        return list(dict.fromkeys(c.section for c in self.checks))

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(("ok", "info", "warn", "fail", "skip"), 0)
        for c in self.checks:
            out[c.status] += 1
        return out

    def exit_code(self, *, strict: bool = False) -> int:
        """``1`` if a check failed (or, with ``strict``, warned), else ``0``."""
        counts = self.counts()
        return 1 if counts["fail"] or (strict and counts["warn"]) else 0

    def to_dict(self, *, strict: bool = False) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": __version__,
            "platform": sys.platform,
            "exit_code": self.exit_code(strict=strict),
            "summary": self.counts(),
            "sections": self.sections,
            "checks": [c.to_dict() for c in self.checks],
        }


def guarded(section: str, collect: Callable[[], list[Check]]) -> list[Check]:
    """Run a section; an unexpected exception becomes one ``fail`` row (a doctor must
    not crash on a broken install)."""
    try:
        return collect()
    except Exception as exc:
        return [Check(section, section, "fail", f"error: {type(exc).__name__}: {exc}")]


# ------------------------------------------------------------------------------ system
_PACKAGES = ("numpy", "soxr", "sounddevice", "onnxruntime", "torch", "mlx", "aiortc", "livekit")
_API_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY",
    "ELEVEN_API_KEY", "GROQ_API_KEY", "AWS_ACCESS_KEY_ID", "AZURE_OPENAI_API_KEY",
    "XAI_API_KEY",
)  # fmt: skip


def _package_version(module: str) -> str:
    try:
        imported = __import__(module)
    except Exception as exc:  # broken native libs (e.g. missing PortAudio)
        return f"installed but failed to import: {exc}"
    return str(getattr(imported, "__version__", "installed"))


def system_checks(environ: Mapping[str, str] | None = None) -> list[Check]:
    """voice-agent-next and Python versions, platform, ML runtimes, accelerators, API keys."""
    environ = os.environ if environ is None else environ
    s = "system"
    checks = [
        Check(s, "voice-agent-next", "ok", __version__),
        Check(
            s,
            "python",
            "ok" if sys.version_info >= (3, 11) else "fail",
            f"{_platform.python_version()} ({sys.executable})",
            data={"version": _platform.python_version(), "executable": sys.executable},
        ),
        Check(
            s,
            "platform",
            "ok",
            f"{_platform.system()} {_platform.release()} {_platform.machine()}",
            data={"sys_platform": sys.platform, "machine": _platform.machine()},
        ),
    ]
    for mod in _PACKAGES:
        if is_installed(mod):
            ver = _package_version(mod)
            status: Status = "warn" if ver.startswith("installed but failed") else "ok"
            checks.append(Check(s, mod, status, ver))
        else:
            checks.append(Check(s, mod, "info", "not installed"))
    if is_installed("onnxruntime"):
        try:
            import onnxruntime as ort

            providers = ", ".join(ort.get_available_providers())
            checks.append(Check(s, "onnxruntime providers", "info", providers))
        except Exception as exc:
            checks.append(Check(s, "onnxruntime providers", "warn", f"error: {exc}"))
    if is_installed("torch"):
        try:
            import torch

            accel = []
            if torch.cuda.is_available():
                accel.append(f"cuda ({torch.cuda.get_device_name(0)})")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                accel.append("mps")
            checks.append(Check(s, "torch accelerators", "info", ", ".join(accel) or "cpu only"))
        except Exception as exc:
            checks.append(Check(s, "torch accelerators", "warn", f"error: {exc}"))
    present = [k for k in _API_KEYS if environ.get(k)]
    checks.append(
        Check(s, "API keys set", "info", ", ".join(present) or "none", data={"set": present})
    )
    return checks


# ------------------------------------------------------------------------------- audio
_TEST_RATES = (8_000, 16_000, 22_050, 24_000, 44_100, 48_000)
_HOSTAPI_NOTES: dict[str, tuple[Status, str]] = {
    # Windows
    "Windows WASAPI": ("ok", "recommended: low latency, shared mode at the Windows mixer rate"),
    "MME": ("info", "compatible but adds 50-100 ms of latency"),
    "Windows DirectSound": ("info", "legacy: higher latency than WASAPI"),
    "Windows WDM-KS": ("info", "kernel streaming: opens devices in exclusive mode"),
    "ASIO": ("info", "pro-audio drivers: exclusive, lowest latency"),
    # macOS
    "Core Audio": ("ok", "the macOS audio system"),
    # Linux
    "ALSA": ("info", "use its 'pipewire'/'pulse'/'default' devices to share them"),
    "PulseAudio": ("ok", "sound server: devices are shared with other apps"),
    "PipeWire": ("ok", "sound server: devices are shared with other apps"),
    "JACK Audio Connection Kit": ("info", "JACK: needs a running JACK (or pipewire-jack) server"),
    "OSS": ("info", "legacy"),
}
_PREFERRED_HOSTAPIS = {
    "linux": ("PulseAudio", "PipeWire", "ALSA"),
    "darwin": ("Core Audio",),
    "win32": ("Windows WASAPI",),
}
_BLUETOOTH = re.compile(r"bluetooth|bluez|airpods|\bbt\b|buds|wh-1000|hands[- ]?free", re.I)
_HFP = re.compile(r"hands[- ]?free|\bhfp\b|\bhsp\b|headset gateway|\bag audio\b", re.I)


def _platform_key(platform: str) -> str:
    return "linux" if platform.startswith("linux") else platform


def _sound_server(environ: Mapping[str, str], exists: Callable[[str], bool]) -> str | None:
    """The Linux sound server whose socket is in ``$XDG_RUNTIME_DIR``, if any."""
    runtime = environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        return None
    if exists(os.path.join(runtime, "pipewire-0")):
        return "PipeWire"
    if exists(os.path.join(runtime, "pulse", "native")):
        return "PulseAudio"
    return None


def device_warnings(device: Any, kind: str, platform: str) -> list[tuple[Status, str, str]]:
    """``(status, problem, fix)`` for one default device (bluetooth, rate, latency, exclusive)."""
    out: list[tuple[Status, str, str]] = []
    rate = float(device.default_samplerate)
    latency = float(
        device.default_low_input_latency if kind == "input" else device.default_low_output_latency
    )
    name = f"{device.name} {device.hostapi}"
    bluetooth = bool(_BLUETOOTH.search(name))
    if _HFP.search(name) or (bluetooth and rate <= 16_000):
        out.append(
            (
                "warn",
                f"{device.name} looks like a Bluetooth headset in hands-free mode (HFP/HSP, "
                f"{rate:g} Hz): narrowband 8/16 kHz audio in both directions hurts speech "
                "recognition and voice quality",
                "use a wired/USB headset, or the Bluetooth headset for output only (A2DP) "
                "with another microphone",
            )
        )
    elif bluetooth:
        out.append(
            (
                "info",
                f"{device.name} is a Bluetooth device: expect 100-300 ms of extra latency",
                "a wired or USB device responds faster",
            )
        )
    if rate < 16_000:
        out.append(
            (
                "warn",
                f"{device.name} runs at {rate:g} Hz, below the 16 kHz speech models expect",
                "pick a device (or profile) running at 16 kHz or more",
            )
        )
    if latency > 0.1:
        out.append(
            (
                "warn",
                f"{device.name} reports {latency * 1000:.0f} ms of {kind} latency",
                "prefer a lower-latency host API (WASAPI on Windows, a sound-server device "
                "on Linux) or a wired device",
            )
        )
    if device.hostapi in ("Windows WDM-KS", "ASIO"):
        out.append(
            (
                "warn",
                f"{device.name} uses {device.hostapi}, which opens the device in exclusive "
                "mode: other apps lose it while the agent runs, and it fails if another app "
                "holds it exclusively",
                "select the 'Windows WASAPI' variant of the device (`van devices`)",
            )
        )
    if _platform_key(platform) == "linux" and device.hostapi == "ALSA" and "(hw:" in device.name:
        out.append(
            (
                "warn",
                f"{device.name} is a raw ALSA 'hw:' device: exclusive access, fixed rates, "
                "and it fails while PipeWire/PulseAudio holds it",
                "use the ALSA 'pipewire', 'pulse' or 'default' device instead",
            )
        )
    return out


def _supported_rates(sd: Any, index: int, kind: str) -> list[int] | None:
    """Rates PortAudio accepts for a mono int16 stream (``Pa_IsFormatSupported``: no stream
    is started). ``None`` when this ``sounddevice`` cannot check."""
    name = "check_input_settings" if kind == "input" else "check_output_settings"
    check = getattr(sd, name, None)
    if check is None:
        return None
    rates = []
    for rate in _TEST_RATES:
        try:
            check(device=index, samplerate=rate, channels=1, dtype="int16")
        except Exception:
            continue
        rates.append(rate)
    return rates


def _hostapi_counts(info: Any) -> dict[str, tuple[int, int]]:
    counts = dict.fromkeys(info.hostapis, (0, 0))
    for d in info.devices:
        n_in, n_out = counts.get(d.hostapi, (0, 0))
        counts[d.hostapi] = (n_in + (d.max_input_channels > 0), n_out + (d.max_output_channels > 0))
    return counts


def audio_checks(
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> list[Check]:
    """PortAudio, host APIs per OS, sound server, default devices, rates, latencies, warnings."""
    from .errors import MissingDependencyError, VoiceAgentError
    from .transports.local import describe_audio_system, import_sounddevice

    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    s = "audio"
    try:
        sd = import_sounddevice()
        info = describe_audio_system()
    except MissingDependencyError as exc:  # no sounddevice, or no PortAudio library
        return [
            Check(s, "sounddevice", "warn", str(exc), hint="pip install 'voice-agent-next[audio]'")
        ]
    except VoiceAgentError as exc:
        return [Check(s, "portaudio", "fail", f"error: {exc}")]
    version = {"version": info.portaudio_release}
    checks = [Check(s, "portaudio", "ok", info.portaudio_version, data=version)]
    key = _platform_key(platform)
    counts = _hostapi_counts(info)
    for api in info.hostapis:
        status, note = _HOSTAPI_NOTES.get(api, ("info", ""))
        n_in, n_out = counts.get(api, (0, 0))
        value = f"{n_in} input, {n_out} output device(s)" + (f"; {note}" if note else "")
        checks.append(
            Check(s, f"host API {api}", status, value, data={"inputs": n_in, "outputs": n_out})
        )
    if not info.hostapis:
        checks.append(Check(s, "host APIs", "fail", "PortAudio reports no host API"))
    preferred = _PREFERRED_HOSTAPIS.get(key, ())
    if preferred and info.hostapis and not any(api in info.hostapis for api in preferred):
        checks.append(
            Check(
                s,
                "host APIs",
                "warn",
                f"none of {', '.join(preferred)} is available on this {key} build of PortAudio",
                hint="reinstall sounddevice/PortAudio",
            )
        )
    if key == "linux":
        server = _sound_server(environ, path_exists)
        checks.append(
            Check(
                s,
                "sound server",
                "ok" if server else "info",
                f"{server} is running" if server else "no PipeWire/PulseAudio socket found",
                hint=None if server else "without a sound server, ALSA devices are exclusive",
                data={"server": server},
            )
        )
    for hint in info.hints(platform=platform):
        checks.append(Check(s, "audio setup", "warn", hint))
    for kind, device in (("input", info.default_input), ("output", info.default_output)):
        label = f"default {kind}"
        if device is None:
            checks.append(
                Check(
                    s,
                    label,
                    "warn",
                    "none",
                    hint=f"choose one with {kind}_device=... (`van devices` lists them)",
                )
            )
            continue
        latency = (
            device.default_low_input_latency
            if kind == "input"
            else device.default_low_output_latency
        )
        rates = _supported_rates(sd, device.index, kind)
        value = f"{device}, {device.default_samplerate:g} Hz, {latency * 1000:.0f} ms low latency"
        checks.append(
            Check(
                s,
                label,
                "ok",
                value,
                data={
                    "index": device.index,
                    "name": device.name,
                    "hostapi": device.hostapi,
                    "default_samplerate": device.default_samplerate,
                    "low_latency_ms": round(latency * 1000, 1),
                    "supported_rates": rates,
                },
            )
        )
        if rates is not None:
            checks.append(
                Check(
                    s,
                    f"{kind} sample rates",
                    "ok" if rates else "warn",
                    ", ".join(f"{r:g}" for r in rates) if rates else "no common rate accepted",
                    hint=None if rates else "the device may be busy or exclusive",
                    data={"rates": rates},
                )
            )
        for status, problem, fix in device_warnings(device, kind, platform):
            checks.append(Check(s, label, status, problem, hint=fix))
    din, dout = info.default_input, info.default_output
    if din is not None and dout is not None and din.default_samplerate != dout.default_samplerate:
        checks.append(
            Check(
                s,
                "rate mismatch",
                "info",
                f"input {din.default_samplerate:g} Hz, output {dout.default_samplerate:g} Hz: "
                "the session resamples; echo cancellation works at the capture rate",
            )
        )
    n_in = sum(1 for d in info.devices if d.max_input_channels > 0)
    n_out = sum(1 for d in info.devices if d.max_output_channels > 0)
    checks.append(
        Check(s, "audio devices", "info", f"{n_in} input, {n_out} output (details: `van devices`)")
    )
    return checks


# ---------------------------------------------------------------------------- hardware
def hardware_checks(report: Callable[[], list[tuple[str, str]]] | None = None) -> list[Check]:
    """GPUs, CUDA libraries and where local models run (``docs/hardware.md``)."""
    from . import hardware

    rows = (report or hardware.report)()
    checks = []
    for name, result in rows:
        status: Status = "ok"
        hint = None
        if result.startswith("error"):
            status = "warn"
        elif "to use the GPU:" in result:
            status = "warn"
            result, _, hint = result.partition("; to use the GPU: ")
            hint = f"to use the GPU: {hint}" if hint else None
        checks.append(Check("hardware", name, status, result, hint=hint))
    return checks


# ----------------------------------------------------------------------------- presets
def preset_checks(env: Any = None, transport: str = "local") -> list[Check]:
    """Readiness of every preset here (the same checks as ``van presets``)."""
    from . import presets

    env = env or presets.current_environment()
    chosen = presets.list_presets()
    results = {p.name: presets.check_preset(p, transport=transport, env=env) for p in chosen}
    ready = [name for name, r in results.items() if r.ready]
    auto = next((n for n in presets.AUTO_ORDER if n in results and results[n].ready), None)
    s = "presets"
    if ready:
        value = f"{len(ready)} of {len(results)} ready" + (
            f"; `van run` uses {auto}" if auto else ""
        )
        summary = Check(s, "presets", "ok", value, data={"ready": ready, "auto": auto})
    else:
        summary = Check(
            s,
            "presets",
            "warn",
            f"none of {len(results)} presets is ready here",
            hint="`van presets <name>` shows what each one needs",
            data={"ready": [], "auto": None},
        )
    checks = [summary]
    for name, r in results.items():
        checks.append(
            Check(
                s,
                name,
                "ok" if r.ready else "info",
                r.summary(),
                data={"ready": r.ready, "fixes": r.fixes()},
            )
        )
    return checks


# ------------------------------------------------------------------------------ models
def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"  # pragma: no cover


def model_checks() -> list[Check]:
    """Model cache location and which catalog models are cached (``van models list``)."""
    from . import models
    from .utils.download import cache_dir

    s = "models"
    statuses = [models.model_status(info) for info in models.catalog()]
    cached = [st for st in statuses if st.state == "cached"]
    partial = [st for st in statuses if st.state == "partial"]
    size = sum(st.size_on_disk for st in cached)
    checks = [
        Check(s, "model cache", "info", str(cache_dir())),
        Check(s, "Hugging Face cache", "info", str(models.hf_cache_dir())),
        Check(
            s,
            "catalog",
            "ok",
            f"{len(cached)} of {len(statuses)} models cached ({_human_size(size)})",
            hint=None if cached else "`van models download <name>` fetches one ahead of time",
            data={
                "models": len(statuses),
                "cached": [st.model.name for st in cached],
                "partial": [st.model.name for st in partial],
                "bytes": size,
            },
        ),
    ]
    if partial:
        checks.append(
            Check(
                s,
                "partial downloads",
                "warn",
                ", ".join(st.model.name for st in partial),
                hint="finish with `van models download <name>` or clean up with `van models prune`",
            )
        )
    if models.is_offline():
        checks.append(
            Check(
                s,
                "offline mode",
                "info",
                "VAN_OFFLINE/HF_HUB_OFFLINE is set: missing models cannot be downloaded",
            )
        )
    return checks


# ----------------------------------------------------------------------------- network
@dataclass(frozen=True)
class Endpoint:
    """A cloud endpoint to probe: ``host:port`` of ``provider`` (optionally a region)."""

    provider: str
    host: str
    port: int = 443
    region: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider} ({self.region})" if self.region else self.provider


_CLOUD_HOSTS: dict[str, tuple[str, ...]] = {
    "openai": ("api.openai.com",),
    "anthropic": ("api.anthropic.com",),
    "google": ("generativelanguage.googleapis.com",),
    "deepgram": ("api.deepgram.com",),
    "assemblyai": ("streaming.assemblyai.com",),
    "cartesia": ("api.cartesia.ai",),
    "elevenlabs": ("api.elevenlabs.io",),
    "groq": ("api.groq.com",),
    "xai": ("api.x.ai",),
    "cerebras": ("api.cerebras.ai",),
    "fireworks": ("api.fireworks.ai",),
    "together": ("api.together.ai",),
    "deepseek": ("api.deepseek.com",),
    "sambanova": ("api.sambanova.ai",),
    "openrouter": ("openrouter.ai",),
}
_REGIONS: dict[str, dict[str, str]] = {
    # region= options of the providers (their REGION_URLS); probed next to the default
    "assemblyai": {"us": "streaming.us.assemblyai.com", "eu": "streaming.eu.assemblyai.com"},
    "elevenlabs": {"us": "api.us.elevenlabs.io"},
}


def cloud_endpoints(
    environ: Mapping[str, str] | None = None, *, extra: Sequence[str] = ()
) -> list[Endpoint]:
    """Endpoints of the cloud providers whose API key is set, their regional variants, the
    Azure OpenAI resource (``AZURE_OPENAI_ENDPOINT``) and ``extra`` URLs or hosts."""
    from .registry import list_providers

    environ = os.environ if environ is None else environ
    env_vars: dict[str, set[str]] = {}
    for spec in list_providers():
        if not spec.local:
            env_vars.setdefault(spec.name, set()).update(spec.env)
    out: list[Endpoint] = []
    for provider, hosts in _CLOUD_HOSTS.items():
        if not any(environ.get(var) for var in env_vars.get(provider, ())):
            continue
        out += [Endpoint(provider, host) for host in hosts]
        out += [Endpoint(provider, h, region=r) for r, h in _REGIONS.get(provider, {}).items()]
    azure = environ.get("AZURE_OPENAI_ENDPOINT", "")
    if azure and any(environ.get(v) for v in env_vars.get("azure_openai", ())):
        host, port = _host_port(azure)
        if host:
            out.append(Endpoint("azure_openai", host, port))
    for target in extra:
        host, port = _host_port(target)
        if host:
            out.append(Endpoint(target if "://" in target else host, host, port))
    return list(dict.fromkeys(out))


def _host_port(target: str) -> tuple[str, int]:
    parts = urlsplit(target if "://" in target else f"https://{target}")
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = 80 if parts.scheme in ("http", "ws") else 443
    return parts.hostname or "", port


@dataclass
class ProbeResult:
    """DNS, TCP connect and TLS handshake times of one endpoint (milliseconds)."""

    host: str
    port: int
    address: str | None = None
    dns_ms: float | None = None
    connect_ms: float | None = None
    """≈ one network round trip."""
    tls_ms: float | None = None
    tls_version: str | None = None
    error: str | None = None


def probe_endpoint(host: str, port: int = 443, timeout: float = 5.0) -> ProbeResult:
    """Resolve ``host``, open a TCP connection and complete a TLS handshake; send nothing."""
    result = ProbeResult(host, port)
    t0 = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        result.error = f"DNS lookup failed: {exc}"
        return result
    result.dns_ms = (time.perf_counter() - t0) * 1000
    family, kind, proto, _, address = infos[0]
    result.address = str(address[0])
    sock = socket.socket(family, kind, proto)
    sock.settimeout(timeout)
    try:
        t1 = time.perf_counter()
        sock.connect(address)
        result.connect_ms = (time.perf_counter() - t1) * 1000
        context = ssl.create_default_context()
        t2 = time.perf_counter()
        with context.wrap_socket(sock, server_hostname=host) as tls:
            result.tls_ms = (time.perf_counter() - t2) * 1000
            result.tls_version = tls.version()
    except (OSError, ssl.SSLError) as exc:
        stage = "TCP connect" if result.connect_ms is None else "TLS handshake"
        result.error = f"{stage} failed: {exc}"
    finally:
        sock.close()
    return result


def network_checks(
    endpoints: Sequence[Endpoint],
    *,
    probe: Callable[[str, int, float], ProbeResult] = probe_endpoint,
    timeout: float = 5.0,
    slow_ms: float = 150.0,
) -> list[Check]:
    """Probe ``endpoints`` concurrently; ``fail`` when unreachable, ``warn`` when far away."""
    s = "network"
    if not endpoints:
        return [
            Check(
                s,
                "cloud endpoints",
                "skip",
                "no cloud provider is configured (no API key set)",
                hint="set a provider's API key or pass --endpoint URL",
            )
        ]
    with ThreadPoolExecutor(max_workers=min(16, len(endpoints))) as pool:
        results = list(pool.map(lambda e: probe(e.host, e.port, timeout), endpoints))
    checks = []
    for ep, r in zip(endpoints, results, strict=True):
        data = asdict(r)
        if r.error:
            checks.append(
                Check(
                    s,
                    ep.label,
                    "fail",
                    f"{ep.host}: {r.error}",
                    hint="check the connection, proxy, firewall or DNS",
                    data=data,
                )
            )
            continue
        value = (
            f"{ep.host}: DNS {r.dns_ms or 0:.0f} ms, connect {r.connect_ms or 0:.0f} ms, "
            f"TLS {r.tls_ms or 0:.0f} ms" + (f" ({r.tls_version})" if r.tls_version else "")
        )
        slow = (r.connect_ms or 0) > slow_ms
        checks.append(
            Check(
                s,
                ep.label,
                "warn" if slow else "ok",
                value,
                hint=(
                    f"a {r.connect_ms:.0f} ms round trip adds latency to every turn; a closer "
                    "region or provider helps"
                )
                if slow
                else None,
                data=data,
            )
        )
    checks += _region_advice(endpoints, results)
    return checks


def _region_advice(endpoints: Sequence[Endpoint], results: Sequence[ProbeResult]) -> list[Check]:
    by_provider: dict[str, list[tuple[Endpoint, float]]] = {}
    for ep, r in zip(endpoints, results, strict=True):
        if r.connect_ms is not None and not r.error:
            by_provider.setdefault(ep.provider, []).append((ep, r.connect_ms))
    checks = []
    for provider, measured in by_provider.items():
        if len(measured) < 2:
            continue
        best, best_ms = min(measured, key=lambda m: m[1])
        default_ms = next((ms for ep, ms in measured if ep.region is None), None)
        if best.region is not None and default_ms is not None and default_ms - best_ms > 20:
            value = (
                f"region {best.region!r} answers fastest ({best_ms:.0f} ms vs {default_ms:.0f} "
                "ms for the default endpoint)"
            )
            hint: str | None = f'{provider}: region="{best.region}"'
        else:
            value = f"the default endpoint is as fast as the regional ones ({best_ms:.0f} ms)"
            hint = None
        checks.append(Check("network", f"{provider} region", "info", value, hint=hint))
    return checks


# ------------------------------------------------------------------------ level meter
def _dbfs(samples: npt.NDArray[Any]) -> float:
    if samples.size == 0:
        return _FLOOR_DB
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return max(_FLOOR_DB, 20.0 * float(np.log10(rms / _FULL_SCALE))) if rms > 0 else _FLOOR_DB


@dataclass
class LevelStats:
    """Microphone level statistics (dBFS: 0 = full scale)."""

    duration: float
    peak_dbfs: float
    rms_dbfs: float
    noise_floor_dbfs: float
    """10th percentile of the 50 ms block levels: the level between words."""
    speech_dbfs: float
    """95th percentile of the 50 ms block levels: the loud (speech) parts."""
    clipped_ratio: float
    """Fraction of samples at (or within 1% of) full scale."""
    dc_offset: float
    """Mean sample value as a fraction of full scale."""


def level_stats(samples: npt.NDArray[np.int16], rate: int, block: float = 0.05) -> LevelStats:
    """Level statistics of mono int16 ``samples``."""
    x = np.asarray(samples).reshape(-1)
    n = max(1, round(block * rate))
    if x.size == 0:
        return LevelStats(0.0, _FLOOR_DB, _FLOOR_DB, _FLOOR_DB, _FLOOR_DB, 0.0, 0.0)
    blocks = [x[i : i + n] for i in range(0, x.size - n + 1, n)] or [x]
    levels = np.array([_dbfs(b) for b in blocks])
    peak = int(np.max(np.abs(x.astype(np.int32))))
    return LevelStats(
        duration=x.size / rate,
        peak_dbfs=max(_FLOOR_DB, 20.0 * float(np.log10(peak / _FULL_SCALE))) if peak else _FLOOR_DB,
        rms_dbfs=_dbfs(x),
        noise_floor_dbfs=float(np.percentile(levels, 10)),
        speech_dbfs=float(np.percentile(levels, 95)),
        clipped_ratio=float(np.mean(np.abs(x.astype(np.int32)) >= 32_440)),
        dc_offset=float(np.mean(x.astype(np.float64)) / _FULL_SCALE),
    )


SILENCE_DBFS = -70.0
NOISY_DBFS = -45.0
QUIET_SPEECH_DBFS = -40.0
CLIP_RATIO = 0.001


def level_verdicts(stats: LevelStats) -> list[tuple[Status, str, str | None]]:
    """``(status, verdict, fix)`` for a microphone recording where the user spoke."""
    if stats.peak_dbfs <= SILENCE_DBFS:
        return [
            (
                "fail",
                f"silence: the microphone peaks at {stats.peak_dbfs:.0f} dBFS",
                "unmute it, pick another input device (--input-device, `van devices`) or "
                "allow microphone access (macOS/Windows privacy settings)",
            )
        ]
    out: list[tuple[Status, str, str | None]] = []
    if stats.clipped_ratio > CLIP_RATIO:
        out.append(
            (
                "warn",
                f"clipping: {stats.clipped_ratio * 100:.2f}% of the samples hit full scale",
                "lower the input gain (aim for speech peaks between -20 and -6 dBFS)",
            )
        )
    if stats.noise_floor_dbfs > NOISY_DBFS:
        out.append(
            (
                "warn",
                f"noisy: the noise floor is {stats.noise_floor_dbfs:.0f} dBFS",
                "move away from fans/hum, lower the gain, or enable noise suppression "
                "(the 'aec' extra's WebRTC processor)",
            )
        )
    if stats.speech_dbfs < QUIET_SPEECH_DBFS:
        out.append(
            (
                "warn",
                f"quiet: the loudest parts reach only {stats.speech_dbfs:.0f} dBFS",
                "speak during the test, move closer or raise the input gain",
            )
        )
    elif stats.speech_dbfs - stats.noise_floor_dbfs < 10:
        out.append(
            (
                "warn",
                f"no speech detected: loud parts are only "
                f"{stats.speech_dbfs - stats.noise_floor_dbfs:.0f} dB above the noise floor",
                "speak during the test",
            )
        )
    if abs(stats.dc_offset) > 0.02:
        out.append(
            (
                "warn",
                f"DC offset of {stats.dc_offset * 100:.1f}% of full scale",
                "a faulty microphone or driver; enable the high-pass filter (AEC processor)",
            )
        )
    if not out:
        out.append(
            (
                "ok",
                f"good level: speech {stats.speech_dbfs:.0f} dBFS, noise floor "
                f"{stats.noise_floor_dbfs:.0f} dBFS, peak {stats.peak_dbfs:.0f} dBFS",
                None,
            )
        )
    return out


# ---------------------------------------------------------------------- echo analysis
def chirp(
    rate: int,
    duration: float = 0.5,
    f0: float = 300.0,
    f1: float | None = None,
    amplitude: float = 0.5,
) -> npt.NDArray[np.int16]:
    """Exponential sine sweep from ``f0`` to ``f1`` Hz with 10 ms fades (int16 mono)."""
    f1 = min(8_000.0, 0.45 * rate) if f1 is None else f1
    n = max(1, round(duration * rate))
    t = np.arange(n) / rate
    k = np.log(f1 / f0) / duration
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    signal = np.sin(phase)
    fade = min(n // 2, max(1, round(0.01 * rate)))
    ramp = np.linspace(0.0, 1.0, fade)
    signal[:fade] *= ramp
    signal[n - fade :] *= ramp[::-1]
    return np.round(signal * amplitude * 32767).astype(np.int16)


def estimate_delay(
    reference: npt.NDArray[Any], recorded: npt.NDArray[Any], max_lag: int | None = None
) -> tuple[int, float]:
    """Lag (samples) at which ``reference`` best matches ``recorded``, and the normalized
    correlation there (0..1; ~1 for a clean copy, small when absent)."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    rec = np.asarray(recorded, dtype=np.float64).reshape(-1)
    if ref.size == 0 or rec.size == 0 or not np.any(ref):
        return 0, 0.0
    n = 1 << int(np.ceil(np.log2(ref.size + rec.size)))
    corr = np.fft.irfft(np.fft.rfft(rec, n) * np.conj(np.fft.rfft(ref, n)), n)
    limit = rec.size if max_lag is None else min(rec.size, max_lag + 1)
    lags = corr[:limit]
    lag = int(np.argmax(np.abs(lags)))
    segment = rec[lag : lag + ref.size]
    denom = float(np.linalg.norm(ref[: segment.size]) * np.linalg.norm(segment))
    confidence = abs(float(lags[lag])) / denom if denom > 0 else 0.0
    return lag, min(1.0, confidence)


@dataclass
class EchoResult:
    """Round trip speaker -> microphone of a played chirp."""

    detected: bool
    delay_ms: float | None
    """Output + air + input latency: from writing the chirp to recording it."""
    erl_db: float | None
    """Echo return loss: played level minus echo level at the microphone (dB)."""
    played_dbfs: float
    echo_dbfs: float | None
    noise_dbfs: float
    confidence: float


def analyze_echo(
    played: npt.NDArray[Any],
    recorded: npt.NDArray[Any],
    rate: int,
    *,
    lead: float,
    probe_duration: float,
    max_delay: float = 1.0,
) -> EchoResult:
    """Find the chirp (played at ``lead`` s for ``probe_duration`` s) in the recording."""
    played = np.asarray(played).reshape(-1)
    recorded = np.asarray(recorded).reshape(-1)
    start = round(lead * rate)
    length = round(probe_duration * rate)
    probe = played[start : start + length]
    noise = recorded[: max(1, start)]
    noise_dbfs = _dbfs(noise)
    played_dbfs = _dbfs(probe)
    search = recorded[start:]
    lag, confidence = estimate_delay(probe, search, max_lag=round(max_delay * rate))
    echo = search[lag : lag + length]
    echo_dbfs = _dbfs(echo)
    detected = confidence >= 0.3 and echo_dbfs > noise_dbfs + 6.0
    return EchoResult(
        detected=detected,
        delay_ms=lag / rate * 1000 if detected else None,
        erl_db=played_dbfs - echo_dbfs if detected else None,
        played_dbfs=played_dbfs,
        echo_dbfs=echo_dbfs if detected else None,
        noise_dbfs=noise_dbfs,
        confidence=confidence,
    )


_AEC_MAX_DELAY_MS = 500


def echo_recommendation(
    result: EchoResult, *, aec_available: bool
) -> tuple[Status, str, str, dict[str, Any]]:
    """``(status, verdict, advice, settings)`` for :class:`LocalAudioTransport` echo control."""
    if not result.detected:
        return (
            "ok",
            f"no echo detected (noise floor {result.noise_dbfs:.0f} dBFS)",
            "headphones or no acoustic path: echo_mode='headphones' is safe; keep 'auto' "
            "when you switch to speakers",
            {"echo_mode": "headphones"},
        )
    delay = result.delay_ms or 0.0
    erl = result.erl_db or 0.0
    verdict = f"echo after {delay:.0f} ms, {erl:.0f} dB below the played level (ERL)"
    if delay > _AEC_MAX_DELAY_MS:
        tail = round(delay / 1000 + 0.3, 1)
        return (
            "warn",
            verdict,
            f"the delay exceeds the {_AEC_MAX_DELAY_MS} ms echo canceller range (Bluetooth?): "
            f"use echo_mode='half_duplex' with half_duplex_tail={tail}, or headphones",
            {"echo_mode": "half_duplex", "half_duplex_tail": tail},
        )
    settings: dict[str, Any] = {"echo_mode": "aec", "delay_ms": round(delay)}
    advice = (
        f"echo_mode='aec' (AEC3 finds the delay itself; WebRTCAudioProcessor(delay_ms="
        f"{round(delay)}) speeds up its first second)"
    )
    if not aec_available:
        advice += "; install it with pip install 'voice-agent-next[aec]' (without it 'auto' "
        advice += "falls back to half-duplex: no barge-in)"
    if erl < 6.0:
        return (
            "warn",
            verdict,
            "the echo is nearly as loud as the playback: lower the speaker volume or move "
            "the microphone away; " + advice,
            settings,
        )
    return ("ok", verdict, advice, settings)


# --------------------------------------------------------------------- latency analysis
def pulse_train(
    rate: int, *, count: int = 5, interval: float = 0.5, lead: float = 0.3
) -> tuple[npt.NDArray[np.int16], list[int], npt.NDArray[np.int16]]:
    """``count`` 40 ms chirps every ``interval`` s after ``lead`` s of silence, followed by
    ``interval`` s of silence: ``(signal, onsets in samples, pulse)``."""
    pulse = chirp(rate, duration=0.04, f0=1_000.0, amplitude=0.5)
    step = round(interval * rate)
    start = round(lead * rate)
    signal = np.zeros(start + step * (count + 1), dtype=np.int16)
    onsets = [start + i * step for i in range(count)]
    for onset in onsets:
        signal[onset : onset + pulse.size] = pulse
    return signal, onsets, pulse


@dataclass
class LatencyResult:
    """Acoustic loopback round trips of a pulse train (milliseconds)."""

    delays_ms: list[float]
    pulses: int
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    jitter_ms: float | None
    """Standard deviation of the round trips."""


def analyze_latency(
    recorded: npt.NDArray[Any],
    rate: int,
    onsets: Sequence[int],
    pulse: npt.NDArray[Any],
    *,
    max_delay: float = 0.45,
    min_confidence: float = 0.3,
) -> LatencyResult:
    """Find each pulse after its onset; ``max_delay`` must stay below the pulse interval."""
    rec = np.asarray(recorded).reshape(-1)
    window = round(max_delay * rate)
    noise_dbfs = _dbfs(rec[: max(1, onsets[0])]) if onsets else _FLOOR_DB
    delays = []
    for onset in onsets:
        segment = rec[onset : onset + window + pulse.size]
        lag, confidence = estimate_delay(pulse, segment, max_lag=window)
        if confidence >= min_confidence and _dbfs(segment[lag : lag + pulse.size]) > noise_dbfs + 6:
            delays.append(lag / rate * 1000)
    if not delays:
        return LatencyResult([], len(onsets), None, None, None, None)
    arr = np.array(delays)
    return LatencyResult(
        delays_ms=[round(d, 2) for d in delays],
        pulses=len(onsets),
        mean_ms=float(arr.mean()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        jitter_ms=float(arr.std()),
    )


# ------------------------------------------------------------------ device access (I/O)
LevelCallback = Callable[[float, float, float], None]
"""``on_level(elapsed_seconds, block_dbfs, peak_dbfs)`` for a live level meter."""


class AudioIO(Protocol):
    """The device operations the interactive checks need (faked in tests)."""

    def record(
        self, seconds: float, *, rate: int, device: int | None, on_level: LevelCallback | None
    ) -> npt.NDArray[np.int16]: ...

    def playrec(
        self,
        signal: npt.NDArray[np.int16],
        *,
        rate: int,
        input_device: int | None,
        output_device: int | None,
    ) -> tuple[npt.NDArray[np.int16], float | None]:
        """Play ``signal`` and record the same number of samples in one duplex stream;
        also returns the input + output latency PortAudio reports (seconds), if known."""
        ...


class SoundDeviceIO:
    """:class:`AudioIO` over ``sounddevice`` (PortAudio)."""

    def __init__(self, sd: Any = None) -> None:
        if sd is None:
            from .transports.local import import_sounddevice

            sd = import_sounddevice()
        self.sd = sd

    def record(
        self,
        seconds: float,
        *,
        rate: int,
        device: int | None,
        on_level: LevelCallback | None = None,
    ) -> npt.NDArray[np.int16]:
        total = round(seconds * rate)
        blocks: list[npt.NDArray[np.int16]] = []
        count = 0
        peak = _FLOOR_DB

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            nonlocal count, peak
            block = np.array(indata[:, 0], dtype=np.int16)
            blocks.append(block)
            count += block.size
            if on_level is not None:
                level = _dbfs(block)
                peak = max(peak, level)
                on_level(count / rate, level, peak)

        stream = self.sd.InputStream(
            device=device, samplerate=rate, channels=1, dtype="int16", callback=callback
        )
        deadline = time.perf_counter() + seconds + 5.0
        with stream:
            while count < total and time.perf_counter() < deadline:
                time.sleep(0.02)
        data = np.concatenate(blocks) if blocks else np.zeros(0, np.int16)
        return data[:total]

    def playrec(
        self,
        signal: npt.NDArray[np.int16],
        *,
        rate: int,
        input_device: int | None,
        output_device: int | None,
    ) -> tuple[npt.NDArray[np.int16], float | None]:
        recorded = self.sd.playrec(
            signal.reshape(-1, 1),
            samplerate=rate,
            channels=1,
            dtype="int16",
            device=(input_device, output_device),
        )
        self.sd.wait()
        reported: float | None = None
        try:
            latency = self.sd.get_stream().latency
            reported = float(sum(latency)) if isinstance(latency, (tuple, list)) else None
        except Exception:
            reported = None
        return np.asarray(recorded).reshape(-1).astype(np.int16), reported


def _device_rate(device: Any) -> int:
    return round(float(device.default_samplerate)) if device is not None else 48_000


def _playrec_any_rate(
    io: AudioIO,
    make: Callable[[int], npt.NDArray[np.int16]],
    rates: Sequence[int],
    input_device: int | None,
    output_device: int | None,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16], int, float | None]:
    """Try each rate until the duplex stream opens (devices may not share a rate)."""
    errors = []
    for rate in dict.fromkeys(rates):
        signal = make(rate)
        try:
            recorded, reported = io.playrec(
                signal, rate=rate, input_device=input_device, output_device=output_device
            )
        except Exception as exc:
            errors.append(f"{rate} Hz: {exc}")
            continue
        return signal, recorded, rate, reported
    raise RuntimeError("cannot open a duplex stream (" + "; ".join(errors) + ")")


def mic_check(
    io: AudioIO,
    *,
    device: Any,
    seconds: float = 5.0,
    on_level: LevelCallback | None = None,
) -> list[Check]:
    """Record ``seconds`` from ``device`` (an :class:`AudioDeviceInfo`) and judge the level."""
    rate = _device_rate(device)
    index = getattr(device, "index", None)
    samples = io.record(seconds, rate=rate, device=index, on_level=on_level)
    stats = level_stats(samples, rate)
    data = {k: round(v, 4) for k, v in asdict(stats).items()}
    data.update(device=str(device), rate=rate)
    if stats.duration < 0.5 * seconds:
        return [
            Check(
                "mic",
                "microphone",
                "fail",
                f"recorded only {stats.duration:.1f} of {seconds:g} s from {device}",
                hint="the device stopped delivering audio: is another app holding it?",
                data=data,
            )
        ]
    checks = [
        Check(
            "mic",
            "microphone",
            "info",
            f"{device}: {stats.duration:.1f} s at {rate} Hz",
            data=data,
        )
    ]
    for status, verdict, fix in level_verdicts(stats):
        checks.append(Check("mic", "level", status, verdict, hint=fix))
    return checks


def echo_check(
    io: AudioIO,
    *,
    input_device: Any,
    output_device: Any,
    aec_available: bool | None = None,
    lead: float = 0.3,
    probe_duration: float = 0.5,
    tail: float = 1.2,
) -> list[Check]:
    """Play a chirp on ``output_device``, record ``input_device``, measure delay and ERL."""
    if aec_available is None:
        aec_available = is_installed("livekit")

    def make(rate: int) -> npt.NDArray[np.int16]:
        probe = chirp(rate, duration=probe_duration)
        pad = np.zeros(round(lead * rate), np.int16)
        return np.concatenate([pad, probe, np.zeros(round(tail * rate), np.int16)])

    rates = (_device_rate(output_device), _device_rate(input_device), 48_000, 16_000)
    played, recorded, rate, reported = _playrec_any_rate(
        io, make, rates, getattr(input_device, "index", None), getattr(output_device, "index", None)
    )
    result = analyze_echo(played, recorded, rate, lead=lead, probe_duration=probe_duration)
    status, verdict, advice, settings = echo_recommendation(result, aec_available=aec_available)
    data = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(result).items()}
    data.update(rate=rate, reported_latency_ms=_ms(reported), recommended=settings)
    return [
        Check("echo", "echo path", status, verdict, data=data),
        Check("echo", "recommendation", "info", advice, data={"settings": settings}),
    ]


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


def latency_check(
    io: AudioIO,
    *,
    input_device: Any,
    output_device: Any,
    count: int = 5,
    interval: float = 0.5,
) -> list[Check]:
    """Acoustic loopback latency: pulses on ``output_device`` heard by ``input_device``."""
    onsets: list[int] = []
    pulse = np.zeros(0, np.int16)

    def make(rate: int) -> npt.NDArray[np.int16]:
        nonlocal onsets, pulse
        signal, onsets, pulse = pulse_train(rate, count=count, interval=interval)
        return signal

    rates = (_device_rate(output_device), _device_rate(input_device), 48_000, 16_000)
    _, recorded, rate, reported = _playrec_any_rate(
        io, make, rates, getattr(input_device, "index", None), getattr(output_device, "index", None)
    )
    result = analyze_latency(recorded, rate, onsets, pulse, max_delay=interval * 0.9)
    data: dict[str, Any] = asdict(result)
    reported_ms = _ms(reported)
    if reported_ms is None:
        low = getattr(input_device, "default_low_input_latency", 0.0) + getattr(
            output_device, "default_low_output_latency", 0.0
        )
        reported_ms = _ms(low) if low else None
    data.update(rate=rate, reported_latency_ms=reported_ms)
    if len(result.delays_ms) < max(1, count // 2) or result.mean_ms is None:
        return [
            Check(
                "latency",
                "loopback",
                "fail",
                f"heard {len(result.delays_ms)} of {count} pulses",
                hint="raise the speaker volume, unmute the microphone, or connect the output "
                "to the input with a loopback cable",
                data=data,
            )
        ]
    value = (
        f"round trip {result.mean_ms:.0f} ms (min {result.min_ms:.0f}, max {result.max_ms:.0f}, "
        f"jitter {result.jitter_ms:.1f} ms; {len(result.delays_ms)}/{count} pulses)"
    )
    if reported_ms is not None:
        value += f"; PortAudio reports {reported_ms:.0f} ms"
    status: Status = "ok"
    hint = None
    if result.mean_ms > 250:
        status = "warn"
        hint = "high: Bluetooth or a high-latency host API (MME, 'high' latency); see `van doctor`"
    elif (result.jitter_ms or 0) > 10:
        status = "warn"
        hint = "unstable: buffer underruns or clock drift; try latency='high' or another device"
    return [Check("latency", "loopback", status, value, hint=hint, data=data)]


# ------------------------------------------------------------------------- devices
def resolve_devices(input_device: int | str | None, output_device: int | str | None) -> Any:
    """The :class:`AudioDeviceInfo` pair for the interactive checks."""
    from .transports.local import describe_audio_system, select_audio_device

    devices = describe_audio_system().devices
    return (
        select_audio_device(devices, input_device, "input"),
        select_audio_device(devices, output_device, "output"),
    )
