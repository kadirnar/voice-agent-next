"""Local audio transport: talk to an agent through this computer's microphone and speakers.

Works on Linux, macOS and Windows through `sounddevice`_ (PortAudio), an optional
dependency that is imported lazily (``pip install 'voice-agent-next[audio]'``)::

    from voice_agent_next.transports.local import LocalAudioTransport

    transport = LocalAudioTransport(input_device="USB", echo_mode="headphones")
    await session.run(agent, transport)  # or: create_transport("local")

How it works (details and per-OS notes in ``docs/transports/local.md``):

* separate ``InputStream``/``OutputStream`` callbacks (``blocksize=0``, ``latency="low"``)
  at each device's default sample rate; the session resamples;
* capture: every block becomes an :class:`~voice_agent_next.audio.frame.AudioFrame`
  stamped with the capture time of its *first* sample and is handed to the event loop
  with ``loop.call_soon_threadsafe``;
* playback: a preallocated ring buffer drained by the output callback, which never waits
  for the event loop; :meth:`LocalAudioTransport.write_audio` applies back-pressure once
  ``max_buffered`` seconds are queued;
* echo: an optional :class:`~voice_agent_next.audio.processing.AudioProcessor` receives the
  exact audio sent to the speakers (``process_render``) and cleans the microphone
  (``process_capture``); without one, ``half_duplex`` mutes the microphone while the agent
  speaks.

.. _sounddevice: https://python-sounddevice.readthedocs.io
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import math
import sys
import threading
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import Any, Literal, cast, get_args

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFormat, AudioFrame
from ..audio.processing import AudioProcessor
from ..errors import ConfigurationError, MissingDependencyError, TransportError
from ..utils.aio import Chan
from ..utils.clock import now
from ..utils.deps import require
from ..utils.log import logger
from .base import Transport, TransportCapabilities

__all__ = [
    "AudioDeviceInfo",
    "AudioSystemInfo",
    "DeviceKind",
    "EchoMode",
    "LocalAudioTransport",
    "describe_audio_system",
    "find_audio_device",
    "import_sounddevice",
    "list_audio_devices",
    "portaudio_install_hint",
    "select_audio_device",
]

EchoMode = Literal["auto", "aec", "headphones", "half_duplex"]
"""How the transport keeps the agent from hearing (and interrupting) itself."""
DeviceKind = Literal["input", "output"]
_ActiveEchoMode = Literal["aec", "headphones", "half_duplex"]

_ECHO_MODES: tuple[str, ...] = get_args(EchoMode)
_DEFAULT_ECHO_CANCELLER_MODULE = "voice_agent_next.audio.aec"
_PCM16 = np.dtype("<i2")


# --------------------------------------------------------------------------- sounddevice
def portaudio_install_hint(platform: str | None = None) -> str:
    """How to get the PortAudio library that ``sounddevice`` loads, for ``platform``."""
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux"):
        return (
            "Install the PortAudio library: `sudo apt install libportaudio2` (Debian/Ubuntu), "
            "`sudo dnf install portaudio` (Fedora) or `sudo pacman -S portaudio` (Arch)."
        )
    return (
        "The sounddevice wheels bundle PortAudio on macOS and Windows; reinstall them with "
        "`pip install --force-reinstall sounddevice`."
    )


def import_sounddevice() -> ModuleType:
    """Import ``sounddevice`` or raise :class:`MissingDependencyError` saying how to fix it.

    Covers both a missing package and a missing PortAudio library (``sounddevice`` raises
    ``OSError`` at import time when it cannot load ``libportaudio``, typically on Linux).
    """
    try:
        return require("sounddevice", extra="audio")
    except OSError as exc:
        raise MissingDependencyError(
            f"sounddevice cannot load the PortAudio library ({exc}). {portaudio_install_hint()}"
        ) from exc


# ------------------------------------------------------------------------------- devices
@dataclass(frozen=True, slots=True)
class AudioDeviceInfo:
    """An audio device as reported by PortAudio (list them with ``van devices``)."""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    default_low_input_latency: float = 0.0
    default_low_output_latency: float = 0.0
    is_default_input: bool = False
    is_default_output: bool = False

    def max_channels(self, kind: DeviceKind) -> int:
        return self.max_input_channels if kind == "input" else self.max_output_channels

    def is_default(self, kind: DeviceKind) -> bool:
        return self.is_default_input if kind == "input" else self.is_default_output

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.hostapi})"


@dataclass(frozen=True, slots=True)
class AudioSystemInfo:
    """PortAudio version, host APIs and devices (used by ``van devices`` / ``van doctor``)."""

    portaudio_version: str
    hostapis: tuple[str, ...]
    devices: tuple[AudioDeviceInfo, ...]

    @property
    def default_input(self) -> AudioDeviceInfo | None:
        return next((d for d in self.devices if d.is_default_input), None)

    @property
    def default_output(self) -> AudioDeviceInfo | None:
        return next((d for d in self.devices if d.is_default_output), None)

    def hints(self, platform: str | None = None) -> list[str]:
        """Setup advice for this system: missing devices, PortAudio host API caveats."""
        platform = sys.platform if platform is None else platform
        hints: list[str] = []
        if not any(d.max_input_channels > 0 for d in self.devices):
            hints.append("no input device found: connect a microphone and allow this app to use it")
        if not any(d.max_output_channels > 0 for d in self.devices):
            hints.append("no output device found: connect speakers or headphones")
        if platform.startswith("linux"):
            apis = " ".join(self.hostapis).lower()
            if "pulse" not in apis and "pipewire" not in apis:
                hints.append(self._linux_hostapi_hint())
        elif platform == "win32":
            defaults = [d for d in (self.default_input, self.default_output) if d is not None]
            if any(d.hostapi == "MME" for d in defaults):
                hints.append(
                    "the default devices use MME, which adds latency; select the 'Windows "
                    "WASAPI' variant of your device by name for lower latency, e.g. "
                    "input_device='Microphone WASAPI' (see `van devices`)"
                )
        return hints

    def _linux_hostapi_hint(self) -> str:
        hint = (
            f"{self.portaudio_version} has no PulseAudio/PipeWire host API (Debian and "
            "Ubuntu ship PortAudio 19.6), so audio goes through ALSA"
        )
        preference = ("pipewire", "pulse", "default")
        routed = sorted(
            {d.name for d in self.devices if d.hostapi == "ALSA" and d.name in preference},
            key=preference.index,
        )
        if routed:
            names = ", ".join(repr(n) for n in routed)
            return (
                f"{hint}. With PipeWire or PulseAudio, use the ALSA {names} device "
                f"(e.g. input_device={routed[0]!r}), which the sound server routes, "
                "rather than a raw 'hw:' device that it may keep busy"
            )
        return (
            f"{hint}; no ALSA 'default'/'pipewire'/'pulse' device was found: install the "
            "ALSA plugin of your sound server (e.g. pipewire-alsa)"
        )


def describe_audio_system() -> AudioSystemInfo:
    """Query PortAudio for its version, host APIs and devices."""
    return _query_audio_system(import_sounddevice())


def list_audio_devices() -> list[AudioDeviceInfo]:
    """All audio devices, with the default input and output flagged."""
    return list(describe_audio_system().devices)


def find_audio_device(
    device: int | str | None = None, kind: DeviceKind = "input"
) -> AudioDeviceInfo:
    """Resolve an ``input``/``output`` device by index, name substring(s) or ``None``
    (the default device). See :func:`select_audio_device` for the matching rules."""
    return select_audio_device(list_audio_devices(), device, kind)


def select_audio_device(
    devices: Sequence[AudioDeviceInfo], device: int | str | None, kind: DeviceKind
) -> AudioDeviceInfo:
    """Pick the ``kind`` device that ``device`` designates among ``devices``.

    ``device`` is a PortAudio index (``3`` or ``"3"``), ``None`` for the default device, or
    space-separated name substrings matched case-insensitively, in order, against
    ``"<device name>, <host API>"`` (``"usb"``, ``"Microphone WASAPI"``), like
    ``sounddevice`` does. When a name matches several devices, an exact name match wins,
    then a device on the host API of the default device; anything else is ambiguous.

    Raises:
        ConfigurationError: no such device, not a ``kind`` device, or ambiguous name.
    """
    usable = [d for d in devices if d.max_channels(kind) > 0]
    if isinstance(device, bool):
        raise ConfigurationError(f"invalid {kind}_device: {device!r}")
    if isinstance(device, str) and device.strip().isdigit():
        device = int(device)
    if device is None:
        found = next((d for d in usable if d.is_default(kind)), None)
        if found is None:
            raise ConfigurationError(
                f"there is no default {kind} device; choose one with {kind}_device=... "
                f"{_available(usable, kind)}"
            )
        return found
    if isinstance(device, int):
        found = next((d for d in devices if d.index == device), None)
        if found is None:
            raise ConfigurationError(
                f"no audio device has index {device}. {_available(usable, kind)}"
            )
        if found.max_channels(kind) < 1:
            raise ConfigurationError(f"{found} is not an {kind} device. {_available(usable, kind)}")
        return found
    query = device.strip().lower()
    words = query.split()
    matches = [d for d in usable if _words_in_order(f"{d.name}, {d.hostapi}".lower(), words)]
    if not matches:
        raise ConfigurationError(f"no {kind} device matches {device!r}. {_available(usable, kind)}")
    if len(matches) > 1:
        matches = [d for d in matches if d.name.lower() == query] or matches
    if len(matches) > 1:
        default = next((d for d in usable if d.is_default(kind)), None) or next(
            (d for d in devices if d.is_default_input or d.is_default_output), None
        )
        if default is not None:
            matches = [d for d in matches if d.hostapi == default.hostapi] or matches
    if len(matches) > 1:
        listing = "\n".join(f"  {d}" for d in matches)
        raise ConfigurationError(
            f"{device!r} matches several {kind} devices; use an index or a more specific "
            f"name (host API words work too):\n{listing}"
        )
    return matches[0]


def _words_in_order(text: str, words: Sequence[str]) -> bool:
    pos = 0
    for word in words:
        pos = text.find(word, pos)
        if pos < 0:
            return False
        pos += len(word)
    return True


def _available(devices: Sequence[AudioDeviceInfo], kind: DeviceKind, limit: int = 12) -> str:
    if not devices:
        return f"No {kind} devices were found (see `van doctor`)."
    shown = ", ".join(str(d) for d in devices[:limit])
    more = f" and {len(devices) - limit} more" if len(devices) > limit else ""
    return f"{kind.capitalize()} devices: {shown}{more} (see `van devices`)."


def _query_audio_system(sd: Any) -> AudioSystemInfo:
    try:
        hostapis = tuple(str(api["name"]) for api in sd.query_hostapis())
        raw = list(sd.query_devices())
    except Exception as exc:
        raise TransportError(f"cannot list audio devices: {exc}") from exc
    try:
        version = str(sd.get_portaudio_version()[1])
    except Exception:
        version = "PortAudio (unknown version)"
    default_in = _default_device_index(sd, "input")
    default_out = _default_device_index(sd, "output")
    devices = []
    for position, info in enumerate(raw):
        index = int(info.get("index", position))
        api = int(info.get("hostapi", -1))
        devices.append(
            AudioDeviceInfo(
                index=index,
                name=str(info["name"]),
                hostapi=hostapis[api] if 0 <= api < len(hostapis) else "unknown",
                max_input_channels=int(info["max_input_channels"]),
                max_output_channels=int(info["max_output_channels"]),
                default_samplerate=float(info["default_samplerate"]),
                default_low_input_latency=float(info.get("default_low_input_latency", 0.0)),
                default_low_output_latency=float(info.get("default_low_output_latency", 0.0)),
                is_default_input=index == default_in,
                is_default_output=index == default_out,
            )
        )
    return AudioSystemInfo(version, hostapis, tuple(devices))


def _default_device_index(sd: Any, kind: DeviceKind) -> int | None:
    try:
        info = sd.query_devices(kind=kind)  # the default device of this kind
    except Exception:  # PortAudio has no default device of this kind
        return None
    index = info.get("index") if isinstance(info, dict) else None
    return index if isinstance(index, int) and index >= 0 else None


def _default_echo_canceller() -> AudioProcessor | None:
    """The library's default echo canceller (``voice_agent_next.audio.aec``), if available."""
    try:
        module = importlib.import_module(_DEFAULT_ECHO_CANCELLER_MODULE)
    except ImportError as exc:
        logger.debug("no default echo canceller: %s", exc)
        return None
    factory = getattr(module, "create_echo_canceller", None)
    if not callable(factory):
        return None
    try:
        processor = factory("auto")
    except Exception as exc:
        logger.warning("could not create the default echo canceller: %s", exc)
        return None
    return processor if isinstance(processor, AudioProcessor) else None


# ------------------------------------------------------------------------ playback buffer
class _PlaybackBuffer:
    """Fixed-capacity FIFO of int16 samples between the event loop and the output callback.

    The event loop writes and the PortAudio output callback reads. The storage is
    preallocated and each critical section is one or two small copies, so the callback
    never waits for anything slow (and never for the event loop).
    """

    def __init__(self, capacity: int, channels: int) -> None:
        self.capacity = capacity
        self._buf: npt.NDArray[np.int16] = np.zeros((capacity, channels), dtype=np.int16)
        self._read = 0  # total samples read (monotonic)
        self._write = 0  # total samples written (monotonic)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return self._write - self._read

    def write(self, samples: npt.NDArray[np.int16]) -> int:
        """Append as many of ``samples`` (shape ``(n, channels)``) as fit; return that count."""
        with self._lock:
            n = min(len(samples), self.capacity - (self._write - self._read))
            if n > 0:
                start = self._write % self.capacity
                first = min(n, self.capacity - start)
                self._buf[start : start + first] = samples[:first]
                if first < n:
                    self._buf[: n - first] = samples[first:n]
                self._write += n
            return max(n, 0)

    def read_into(self, out: npt.NDArray[np.int16]) -> int:
        """Move up to ``len(out)`` samples into ``out``; return how many were copied."""
        with self._lock:
            n = min(len(out), self._write - self._read)
            if n > 0:
                start = self._read % self.capacity
                first = min(n, self.capacity - start)
                out[:first] = self._buf[start : start + first]
                if first < n:
                    out[first:n] = self._buf[: n - first]
                self._read += n
            return n

    def clear(self) -> int:
        """Drop everything; return how many samples were dropped."""
        with self._lock:
            dropped = self._write - self._read
            self._read = self._write
            return dropped


# ----------------------------------------------------------------------------- transport
class LocalAudioTransport(Transport):
    """Talk to an agent through this computer's microphone and speakers.

    Devices are resolved when the transport is created and opened by :meth:`start`.

    Args:
        input_device: microphone: PortAudio index, name substring(s) (``"USB"``,
            ``"Microphone WASAPI"``) or ``None`` for the system default (see
            :func:`select_audio_device` and ``van devices``).
        output_device: speakers or headphones, same syntax.
        sample_rate: sample rate of both streams; default: each device's default rate
            (always supported; the session resamples, so there is rarely a reason to set it).
        input_sample_rate: overrides ``sample_rate`` for the microphone.
        output_sample_rate: overrides ``sample_rate`` for the speakers.
        input_channels: microphone channels (mono by default).
        output_channels: speaker channels (mono by default; PortAudio up-mixes).
        block_duration: callback period in seconds (e.g. ``0.01``). ``None`` lets the host
            API choose (``blocksize=0``), the most robust low-latency setting.
        latency: PortAudio latency hint: ``"low"`` (default), ``"high"`` or seconds.
        echo_canceller: an :class:`AudioProcessor` doing acoustic echo cancellation. The
            output callback feeds it the exact audio sent to the speakers
            (``process_render``, called from the output callback thread) and every
            microphone block goes through ``process_capture`` (input callback thread).
            Calls are serialized by the transport. Pass it here rather than to
            ``AgentSession(processors=...)``, which cannot know the playback timing.
        echo_mode: how the agent is kept from hearing itself:

            * ``"aec"``: echo cancellation with ``echo_canceller`` (or the library's default
              one, when available); barge-in works;
            * ``"headphones"``: no echo handling; use it only when the speakers cannot
              reach the microphone;
            * ``"half_duplex"``: the microphone is muted while the agent is heard and for
              ``half_duplex_tail`` seconds after. Safe without AEC, but the user cannot
              interrupt the agent;
            * ``"auto"`` (default): ``"aec"`` when an echo canceller is available,
              otherwise ``"half_duplex"`` (with a warning).
        half_duplex_tail: seconds the microphone stays muted after playback ends
            (half-duplex), covering room reverberation and latency estimation errors.
        max_buffered: seconds of queued agent audio above which :meth:`write_audio` waits.

    Raises:
        MissingDependencyError: ``sounddevice`` or the PortAudio library is missing.
        ConfigurationError: invalid options, unknown/ambiguous device.
    """

    capabilities = TransportCapabilities(pause=True, playback_position=True)

    MAX_INPUT_BACKLOG = 5.0
    """Seconds of microphone audio kept when nobody consumes :meth:`audio_input` (the
    oldest audio is dropped beyond that)."""

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        sample_rate: int | None = None,
        input_sample_rate: int | None = None,
        output_sample_rate: int | None = None,
        input_channels: int = 1,
        output_channels: int = 1,
        block_duration: float | None = None,
        latency: float | Literal["low", "high"] = "low",
        echo_canceller: AudioProcessor | None = None,
        echo_mode: EchoMode = "auto",
        half_duplex_tail: float = 0.3,
        max_buffered: float = 1.0,
    ) -> None:
        if echo_mode not in _ECHO_MODES:
            raise ConfigurationError(
                f"echo_mode must be one of {', '.join(_ECHO_MODES)}; got {echo_mode!r}"
            )
        if echo_canceller is not None and echo_mode not in ("auto", "aec"):
            raise ConfigurationError(
                f"echo_canceller is only used with echo_mode='aec' or 'auto', not {echo_mode!r}"
            )
        if block_duration is not None and block_duration <= 0:
            raise ConfigurationError(f"block_duration must be > 0 or None, got {block_duration}")
        if isinstance(latency, str) and latency not in ("low", "high"):
            raise ConfigurationError(f"latency must be 'low', 'high' or seconds, got {latency!r}")
        if not isinstance(latency, str) and latency <= 0:
            raise ConfigurationError(f"latency must be > 0 seconds, got {latency}")
        if half_duplex_tail < 0:
            raise ConfigurationError(f"half_duplex_tail must be >= 0, got {half_duplex_tail}")
        if max_buffered <= 0:
            raise ConfigurationError(f"max_buffered must be > 0, got {max_buffered}")
        if input_channels < 1 or output_channels < 1:
            raise ConfigurationError("input_channels and output_channels must be >= 1")

        self._sd = import_sounddevice()
        devices = _query_audio_system(self._sd).devices
        self.input_device = select_audio_device(devices, input_device, "input")
        self.output_device = select_audio_device(devices, output_device, "output")
        if input_channels > self.input_device.max_input_channels:
            raise ConfigurationError(
                f"{self.input_device} has {self.input_device.max_input_channels} input "
                f"channel(s); input_channels={input_channels}"
            )
        if output_channels > self.output_device.max_output_channels:
            raise ConfigurationError(
                f"{self.output_device} has {self.output_device.max_output_channels} output "
                f"channel(s); output_channels={output_channels}"
            )
        in_rate = sample_rate if input_sample_rate is None else input_sample_rate
        out_rate = sample_rate if output_sample_rate is None else output_sample_rate
        super().__init__(
            input_format=AudioFormat(_stream_rate(in_rate, self.input_device), input_channels),
            output_format=AudioFormat(_stream_rate(out_rate, self.output_device), output_channels),
        )
        self._rate_overridden = any(
            r is not None for r in (sample_rate, input_sample_rate, output_sample_rate)
        )
        self.block_duration = block_duration
        self.latency = latency
        self.echo_mode: EchoMode = echo_mode
        self.half_duplex_tail = half_duplex_tail
        self.max_buffered = max_buffered

        self._echo_canceller = echo_canceller
        self._owns_echo_canceller = False
        if echo_canceller is None and echo_mode in ("auto", "aec"):
            self._echo_canceller = _default_echo_canceller()
            self._owns_echo_canceller = self._echo_canceller is not None
        if echo_mode == "aec" and self._echo_canceller is None:
            raise ConfigurationError(
                "echo_mode='aec' needs an echo canceller: pass echo_canceller=... or install "
                "'voice-agent-next[aec]'. With headphones use echo_mode='headphones'; "
                "without either, echo_mode='half_duplex'."
            )
        active: str = echo_mode
        if echo_mode == "auto":
            active = "aec" if self._echo_canceller is not None else "half_duplex"
            if self._echo_canceller is None:
                logger.warning(
                    "LocalAudioTransport: no echo canceller available, using half-duplex mode: "
                    "the microphone is muted while the agent speaks, so it cannot be "
                    "interrupted. Install 'voice-agent-next[aec]' for echo cancellation, or "
                    "pass echo_mode='headphones' when using headphones."
                )
        self._active_mode = cast(_ActiveEchoMode, active)

        self._echo_lock = threading.Lock()
        self._buffer = _PlaybackBuffer(
            max(1, math.ceil(max_buffered * self.output_format.sample_rate)), output_channels
        )
        self._input: Chan[AudioFrame] = Chan()
        self._backlog = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._in_stream: Any = None
        self._out_stream: Any = None
        self._input_latency = 0.0
        self._output_latency = 0.0
        self._paused = False
        self._generation = 0  # bumped by clear_audio()/aclose() to abandon pending writes
        # (start, end) of the current stretch of agent audio at the speakers, in now() time
        self._render_span: tuple[float, float] = (-math.inf, -math.inf)
        self._started = False
        self._closing = False
        self._failure: TransportError | None = None
        self.input_overflows = 0
        """Microphone blocks PortAudio reported as overflowed (audio lost)."""
        self.output_underflows = 0
        """Speaker blocks PortAudio reported as underflowed (glitches)."""
        self.dropped_input_frames = 0
        """Microphone frames dropped because :meth:`audio_input` was not consumed."""

    # ---------------------------------------------------------------------- properties
    @property
    def active_echo_mode(self) -> _ActiveEchoMode:
        """The echo handling in effect (``"auto"`` resolved; may fall back at runtime)."""
        return self._active_mode

    @property
    def echo_canceller(self) -> AudioProcessor | None:
        """The echo canceller in use (given, or the library default in ``auto``/``aec``)."""
        return self._echo_canceller

    @property
    def input_latency(self) -> float:
        """Microphone latency reported by PortAudio for the open stream (seconds)."""
        return self._input_latency

    @property
    def output_latency(self) -> float:
        """Speaker latency reported by PortAudio for the open stream (seconds)."""
        return self._output_latency

    @property
    def paused(self) -> bool:
        return self._paused

    def __repr__(self) -> str:
        return (
            f"LocalAudioTransport(input={self.input_device} {self.input_format}, "
            f"output={self.output_device} {self.output_format}, echo={self._active_mode})"
        )

    # ----------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Open and start the microphone and speaker streams."""
        if self._closing:
            raise TransportError("LocalAudioTransport is closed")
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        if self._echo_canceller is not None and self._active_mode == "aec":
            self._echo_canceller.reset()
        try:
            self._out_stream = self._open_stream("output")
            self._in_stream = self._open_stream("input")
            self._output_latency = _stream_latency(self._out_stream)
            self._input_latency = _stream_latency(self._in_stream)
            for kind, stream in (("output", self._out_stream), ("input", self._in_stream)):
                try:
                    stream.start()
                except Exception as exc:
                    raise TransportError(f"cannot start the audio {kind} stream: {exc}") from exc
        except BaseException:
            self._close_streams()
            raise
        self._started = True
        logger.info(
            "local audio: input %s %s, output %s %s, latency %.0f/%.0f ms, echo: %s",
            self.input_device, self.input_format, self.output_device, self.output_format,
            self._input_latency * 1000, self._output_latency * 1000, self._active_mode,
        )  # fmt: skip
        self.emit("connected")

    async def aclose(self) -> None:
        """Stop the streams (queued audio is dropped) and end :meth:`audio_input`."""
        if self._closing:
            return
        self._closing = True
        self._generation += 1
        self._close_streams()
        self._buffer.clear()
        self._input.close()
        if self._echo_canceller is not None and self._owns_echo_canceller:
            try:
                self._echo_canceller.close()
            except Exception:
                logger.exception("closing the echo canceller failed")
        if self._started:
            self.emit("disconnected")

    def _open_stream(self, kind: DeviceKind) -> Any:
        sd = self._sd
        device = self.input_device if kind == "input" else self.output_device
        fmt = self.input_format if kind == "input" else self.output_format
        stream_cls = sd.InputStream if kind == "input" else sd.OutputStream
        callback = self._input_callback if kind == "input" else self._output_callback
        blocksize = 0  # let the host API choose: the most robust low-latency setting
        if self.block_duration is not None:
            blocksize = max(1, round(self.block_duration * fmt.sample_rate))
        try:
            return stream_cls(
                samplerate=fmt.sample_rate,
                blocksize=blocksize,
                device=device.index,
                channels=fmt.channels,
                dtype="int16",
                latency=self.latency,
                callback=callback,
                finished_callback=partial(self._stream_finished, kind),
            )
        except Exception as exc:
            hint = "check the device with `van devices`"
            if self._rate_overridden and fmt.sample_rate != round(device.default_samplerate):
                hint += f"; its default rate {device.default_samplerate:g} Hz always works"
            if sys.platform.startswith("linux"):
                hint += (
                    "; on Linux prefer the ALSA 'default'/'pipewire' device to raw 'hw:' "
                    "devices, which the sound server may hold (see `van doctor`)"
                )
            raise TransportError(
                f"cannot open the {kind} device {device} at {fmt}: {exc} ({hint})"
            ) from exc

    def _close_streams(self) -> None:
        for stream in (self._in_stream, self._out_stream):
            if stream is None:
                continue
            try:
                stream.close()  # aborts an active stream; PortAudio joins its callback thread
            except Exception as exc:
                logger.debug("closing an audio stream failed: %s", exc)
        self._in_stream = self._out_stream = None

    def _stream_finished(self, kind: DeviceKind) -> None:
        """PortAudio ``finished_callback``: the stream stopped (called from its thread)."""
        loop = self._loop
        if self._closing or loop is None:
            return
        device = self.input_device if kind == "input" else self.output_device
        error = TransportError(
            f"the audio {kind} stream ({device}) stopped unexpectedly; was the device disconnected?"
        )
        with contextlib.suppress(RuntimeError):  # the event loop is already closed
            loop.call_soon_threadsafe(self._fail, error)

    def _fail(self, error: TransportError) -> None:
        if self._closing or self._failure is not None:
            return
        self._failure = error
        logger.error("%s", error)
        self._input.close()

    # --------------------------------------------------------------------------- input
    async def audio_input(self) -> AsyncIterator[AudioFrame]:
        """Microphone frames (``input_format``), stamped with their capture time.

        Ends when the transport is closed; raises :class:`TransportError` if a device
        stops unexpectedly.
        """
        async for frame in self._input:
            self._backlog = max(0.0, self._backlog - frame.duration)
            yield frame
        if self._failure is not None:
            raise self._failure

    def _input_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio input callback (audio thread): must stay fast and never block."""
        t = now()
        if status and getattr(status, "input_overflow", False):
            self.input_overflows += 1
        fmt = self.input_format
        duration = frames / fmt.sample_rate
        start = t - duration - self._input_latency  # capture time of the first sample
        data = indata.tobytes()  # copy: PortAudio reuses the buffer
        frame = AudioFrame(data, fmt.sample_rate, fmt.channels, start)
        mode = self._active_mode
        if mode == "aec":
            processed = self._process_capture(frame)
            if processed is None:  # the echo canceller failed: now half-duplex
                mode = "half_duplex"
            else:
                frame = processed
        if mode == "half_duplex" and self._mic_gated(start, duration):
            frame = AudioFrame(bytes(len(data)), fmt.sample_rate, fmt.channels, start)
        loop = self._loop
        if frame and loop is not None:
            with contextlib.suppress(RuntimeError):  # the event loop is already closed
                loop.call_soon_threadsafe(self._deliver, frame)

    def _mic_gated(self, start: float, duration: float) -> bool:
        """Whether a microphone block overlaps agent playback (plus the tail)."""
        span_start, span_end = self._render_span
        return start < span_end + self.half_duplex_tail and start + duration > span_start

    def _deliver(self, frame: AudioFrame) -> None:
        """Event-loop side of the capture callback."""
        chan = self._input
        if chan.closed:
            return
        chan.send_nowait(frame)
        self._backlog += frame.duration
        while self._backlog > self.MAX_INPUT_BACKLOG and chan.qsize() > 1:
            dropped = chan.recv_nowait()
            self._backlog -= dropped.duration
            self.dropped_input_frames += 1
            if self.dropped_input_frames == 1:
                logger.warning(
                    "microphone audio is not being consumed; dropping audio older than %.0f s",
                    self.MAX_INPUT_BACKLOG,
                )

    # -------------------------------------------------------------------------- output
    async def write_audio(self, frame: AudioFrame) -> None:
        """Queue agent audio (``output_format``) for playback.

        Waits while more than ``max_buffered`` seconds are queued (back-pressure). Audio
        written after :meth:`aclose` is ignored.
        """
        if self._closing:
            return
        if self._failure is not None:
            raise TransportError(str(self._failure))
        if not self._started:
            raise TransportError("LocalAudioTransport is not started; call `await start()` first")
        fmt = self.output_format
        if frame.sample_rate != fmt.sample_rate:
            raise ValueError(f"expected {fmt.sample_rate} Hz audio, got {frame.format}")
        if not frame:
            return
        if frame.channels != fmt.channels:
            frame = frame.to_channels(fmt.channels)
        samples = np.frombuffer(frame.data, dtype=_PCM16).reshape(-1, fmt.channels)
        generation = self._generation
        written = self._buffer.write(samples)
        while written < len(samples):
            missing = (len(samples) - written) / fmt.sample_rate
            await asyncio.sleep(min(0.05, max(0.005, missing / 2)))
            if self._generation != generation or self._closing:
                return  # cleared or closed meanwhile: drop the rest of this frame
            if self._failure is not None:
                raise TransportError(str(self._failure))
            written += self._buffer.write(samples[written:])

    async def clear_audio(self) -> None:
        """Drop all queued agent audio immediately (barge-in).

        Also ends a pause: nothing is left to resume. Audio already inside the device
        buffer (at most the output latency) still plays.
        """
        self._generation += 1
        self._buffer.clear()
        self._paused = False

    async def pause_audio(self) -> None:
        """Pause playback, keeping queued audio (the speakers play silence meanwhile)."""
        self._paused = True

    async def resume_audio(self) -> None:
        """Resume paused playback where it stopped."""
        self._paused = False

    def buffered_duration(self) -> float:
        """Queued agent audio plus what the device has not played yet (seconds)."""
        queued = len(self._buffer) / self.output_format.sample_rate
        return queued + max(0.0, self._render_span[1] - now())

    def _output_callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio output callback (audio thread): must stay fast and never block."""
        t = now()
        if status and getattr(status, "output_underflow", False):
            self.output_underflows += 1
        played = 0 if self._paused else self._buffer.read_into(outdata)
        if played < frames:
            outdata[played:] = 0
        fmt = self.output_format
        if played:
            span_start, span_end = self._render_span
            if t > span_end + self.half_duplex_tail:
                span_start = t  # a new stretch of agent audio
            end = t + self._output_latency + played / fmt.sample_rate
            self._render_span = (span_start, max(span_end, end))
        if self._active_mode == "aec":
            # the echo reference is exactly what the speakers get, silence included
            reference = AudioFrame(
                outdata.tobytes(), fmt.sample_rate, fmt.channels, t + self._output_latency
            )
            self._process_render(reference)

    # ---------------------------------------------------------------------------- echo
    def _process_capture(self, frame: AudioFrame) -> AudioFrame | None:
        ec = self._echo_canceller
        if ec is None:
            return frame
        try:
            with self._echo_lock:
                return ec.process_capture(frame)
        except Exception:
            self._echo_failed("process_capture")
            return None

    def _process_render(self, frame: AudioFrame) -> None:
        ec = self._echo_canceller
        if ec is None:
            return
        try:
            with self._echo_lock:
                ec.process_render(frame)
        except Exception:
            self._echo_failed("process_render")

    def _echo_failed(self, method: str) -> None:
        if self._active_mode != "aec":
            return
        self._active_mode = "half_duplex"
        logger.exception(
            "echo canceller %s() failed; switching to half-duplex (the microphone is muted "
            "while the agent speaks)",
            method,
        )


def _stream_rate(rate: int | None, device: AudioDeviceInfo) -> int:
    if rate is not None:
        if rate <= 0:
            raise ConfigurationError(f"sample rate must be > 0, got {rate}")
        return int(rate)
    default = round(device.default_samplerate)
    return default if default > 0 else 48_000


def _stream_latency(stream: Any) -> float:
    try:
        latency = float(stream.latency)
    except (TypeError, ValueError, AttributeError):
        return 0.0
    return latency if 0.0 <= latency < 2.0 else 0.0
