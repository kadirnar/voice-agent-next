"""Echo cancellation, noise suppression and half-duplex gating for microphone audio.

When an agent talks through loudspeakers, the microphone hears the agent again; without
echo control the voice activity detector fires on that echo and the agent interrupts
itself. Three strategies, all :class:`~voice_agent_next.audio.processing.AudioProcessor`\\ s:

* :class:`WebRTCAudioProcessor`: WebRTC audio processing (AEC3 echo cancellation, noise
  suppression, high-pass filter, automatic gain control) through
  ``livekit.rtc.AudioProcessingModule`` (``pip install 'voice-agent-next[aec]'``). Barge-in
  keeps working.
* :class:`HalfDuplexGate`: mutes the microphone while the agent is audible. Needs no
  dependency, but the user cannot interrupt the agent.
* nothing (headphones): there is no acoustic echo path.

:func:`create_echo_canceller` picks one from a mode name. Every processor needs the audio
that is played as its reference: feed it to :meth:`AudioProcessor.process_render` from
exactly one place (the transport's playback callback when it has one). See
``docs/audio-processing.md``.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

from ..utils.clock import now
from ..utils.deps import require
from ..utils.log import logger
from .buffer import FrameChunker
from .frame import AudioFormat, AudioFrame
from .processing import AudioProcessor
from .resample import Resampler

__all__ = [
    "EchoMode",
    "HalfDuplexGate",
    "ReferenceTiming",
    "WebRTCAudioProcessor",
    "create_echo_canceller",
]

EchoMode = Literal["auto", "aec", "webrtc", "half_duplex", "headphones", "none"]
ReferenceTiming = Literal["played", "queued"]

_APM_RATES = (8_000, 16_000, 32_000, 48_000)
"""Native WebRTC APM rates, accepted for ``processing_rate``."""
_MAX_STREAM_DELAY_MS = 500
"""The APM rejects stream delays outside ``[0, 500]`` ms."""
_MAX_PENDING_RENDER = 1.0
"""Seconds of reference audio kept while no microphone audio is being processed."""
_MAX_PACED = 100
"""10 ms frames of ``"queued"`` reference kept waiting for the capture stream (1 s)."""
_PACED_LEAD = 3
"""10 ms reference frames a paced (``"queued"``) reference may run ahead of the capture.

AEC3 flushes its render buffer when the reference stays more than 8 blocks (32 ms) ahead
of the capture, after which any pause in the reference makes it lose its delay estimate.
"""


def _load_rtc() -> Any:
    """Import ``livekit.rtc`` (extra ``aec``); tests replace this with a fake module."""
    return require("livekit.rtc", extra="aec", package="livekit")


def _processing_rate(input_rate: int, forced: int | None) -> int:
    """APM rate for a stream: its own rate whenever 10 ms is a whole number of samples.

    The APM accepts any such rate and converts internally; other rates (22.05 or
    11.025 kHz) are resampled to 48 or 16 kHz here.
    """
    if forced is not None:
        return forced
    if input_rate % 100 == 0:
        return input_rate
    return 16_000 if input_rate <= 16_000 else 48_000


class _Converter:
    """Channel and sample-rate conversion of one stream to a fixed format (identity if equal).

    Deliberately uses the numpy polyphase backend: it adds < 1 ms of group delay, while
    soxr's streaming mode withholds 20-60 ms of audio, far beyond the 10 ms budget.
    """

    def __init__(self, output_rate: int, channels: int) -> None:
        self.output_rate = output_rate
        self.channels = channels
        self._rs: Resampler | None = None

    def push(self, frame: AudioFrame) -> AudioFrame:
        frame = frame.to_channels(self.channels)
        if frame.sample_rate == self.output_rate:
            return frame
        if self._rs is None or self._rs.input_rate != frame.sample_rate:
            self._rs = Resampler(
                frame.sample_rate, self.output_rate, self.channels, backend="numpy"
            )
        return self._rs.push(frame)


class _CapturePath:
    """Microphone stream state: input format <-> 10 ms APM frames, plus the output FIFO."""

    def __init__(self, fmt: AudioFormat, rate: int) -> None:
        self.format = fmt
        self.rate = rate
        self.to_apm = _Converter(rate, fmt.channels)
        self.from_apm = _Converter(fmt.sample_rate, fmt.channels)
        self.chunker = FrameChunker(rate, fmt.channels, samples_per_frame=rate // 100)
        self.out = bytearray()
        self.delay = 0
        """Samples (at the input rate) of silence inserted so far = latency added by framing."""


class _RenderPath:
    """Reference (speaker) stream state: any format -> mono 10 ms APM frames."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.samples = rate // 100
        self.silence = bytes(self.samples * 2)
        self.converter = _Converter(rate, 1)
        self.chunker = FrameChunker(rate, 1, samples_per_frame=self.samples)


class WebRTCAudioProcessor(AudioProcessor):
    """WebRTC audio processing: echo cancellation (AEC3), noise suppression, high-pass, AGC.

    Wraps ``livekit.rtc.AudioProcessingModule`` from the ``livekit`` wheel (extra ``aec``;
    Linux x86_64/aarch64, macOS x86_64/arm64, Windows x64). Frames of any size, sample
    rate and channel count are accepted:

    * capture audio is re-chunked into the exact 10 ms frames the APM needs, processed at
      its own rate (rates that are not a multiple of 100 Hz are resampled to 48/16 kHz and
      back) and returned as a frame of the **same format and duration** as the input.
      The only added latency is the framing remainder, at most 10 ms (zero when input
      frames are multiples of 10 ms) plus < 1 ms per resampling stage; it is reported by
      :attr:`latency` and subtracted from the returned frame's ``timestamp``;
    * :meth:`process_render` only queues the reference (cheap and non-blocking, safe to
      call from a real-time playback callback on another thread); it is converted to mono
      at the capture processing rate and handed to the APM on the capture path, right
      before the microphone audio that may contain its echo. While no reference arrives,
      silence is fed, so AEC3 keeps its delay estimate between agent turns.

    Args:
        echo_cancellation: cancel the played audio (:meth:`process_render`) from the mic.
        noise_suppression: suppress stationary background noise. Do not also enable noise
            suppression in the client, OS or engine: running it twice degrades speech and
            hurts transcription and turn detection.
        high_pass_filter: remove DC offset and low-frequency rumble.
        auto_gain_control: normalize the microphone level.
        reference: how :meth:`process_render` is called. ``"played"`` (default): with audio
            as it is being played, e.g. from a transport's playback callback; the reference
            goes to AEC3 as it arrives and AEC3 tracks delay and clock drift itself.
            ``"queued"``: with audio when it is queued for playback, possibly ahead of
            time, as :class:`~voice_agent_next.session.AgentSession`'s playout loop does
            (up to ``output_lookahead`` early); the reference is then released in step
            with the capture stream, like a playing device would.
        delay_ms: fixed stream-delay hint (0-500 ms) between playing audio and hearing its
            echo. ``None`` (default) uses the device latencies reported with
            :meth:`set_device_latency`, or no hint. AEC3 estimates the actual delay itself
            (up to ~500 ms); a wrong hint only slows down its first second.
        processing_rate: force the APM rate (8000, 16000, 32000 or 48000). ``None`` keeps
            the capture rate whenever possible (no resampling).

    Raises:
        MissingDependencyError: if the ``livekit`` package is not installed.
    """

    def __init__(
        self,
        *,
        echo_cancellation: bool = True,
        noise_suppression: bool = True,
        high_pass_filter: bool = True,
        auto_gain_control: bool = True,
        reference: ReferenceTiming = "played",
        delay_ms: int | None = None,
        processing_rate: int | None = None,
    ) -> None:
        if reference not in ("played", "queued"):
            raise ValueError(f"reference must be 'played' or 'queued', got {reference!r}")
        if delay_ms is not None and not 0 <= delay_ms <= _MAX_STREAM_DELAY_MS:
            raise ValueError(f"delay_ms must be within [0, {_MAX_STREAM_DELAY_MS}], got {delay_ms}")
        if processing_rate is not None and processing_rate not in _APM_RATES:
            raise ValueError(f"processing_rate must be one of {_APM_RATES}, got {processing_rate}")
        self._rtc = _load_rtc()
        self.echo_cancellation = echo_cancellation
        self.noise_suppression = noise_suppression
        self.high_pass_filter = high_pass_filter
        self.auto_gain_control = auto_gain_control
        self.reference: ReferenceTiming = reference
        self.delay_ms = delay_ms
        self.processing_rate = processing_rate
        self._lock = threading.Lock()  # capture path + every APM call
        self._pending_lock = threading.Lock()  # reference queue (touched by process_render)
        self._pending: deque[AudioFrame] = deque()
        self._pending_duration = 0.0
        self._input_latency: float | None = None
        self._output_latency: float | None = None
        self._init_state()

    def _init_state(self) -> None:
        self._apm: Any = None
        self._broken = False
        self._error_logged = False
        self._capture: _CapturePath | None = None
        self._render: _RenderPath | None = None
        self._render_lead = 0  # reference frames fed minus capture frames processed
        self._paced: deque[bytes] = deque()  # "queued" reference frames not released yet
        self._applied_delay_ms: int | None = None

    # ------------------------------------------------------------------ public API
    @property
    def latency(self) -> float:
        """Seconds of delay this processor currently adds to the capture stream."""
        cap = self._capture
        return 0.0 if cap is None else cap.delay / cap.format.sample_rate

    @property
    def stream_delay_ms(self) -> int | None:
        """Delay hint last passed to the APM (``None`` if none was set)."""
        return self._applied_delay_ms

    def set_device_latency(
        self, *, input_latency: float | None = None, output_latency: float | None = None
    ) -> None:
        """Report audio device latencies in seconds (e.g. ``sounddevice`` ``stream.latency``).

        Transports call this after opening their devices; unless ``delay_ms`` was given,
        the APM's stream-delay hint becomes ``input_latency + output_latency``.
        """
        for name, value in (("input_latency", input_latency), ("output_latency", output_latency)):
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value}")
        with self._lock:
            if input_latency is not None:
                self._input_latency = input_latency
            if output_latency is not None:
                self._output_latency = output_latency

    def process_render(self, frame: AudioFrame) -> None:
        """Queue audio that is played (the echo reference). Never blocks on processing."""
        if not frame or not self.echo_cancellation:
            return
        with self._pending_lock:
            self._pending.append(frame)
            self._pending_duration += frame.duration
            # nothing consumes the queue while the microphone is not processed: bound it
            while self._pending_duration > _MAX_PENDING_RENDER and len(self._pending) > 1:
                self._pending_duration -= self._pending.popleft().duration

    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        """Process microphone audio; returns a frame of the same format and duration."""
        if not frame or not self._enabled:
            return frame
        with self._lock:
            cap = self._capture
            if cap is None or cap.format != frame.format:
                cap = self._configure(frame.format)
            if self.echo_cancellation:
                self._drain_render()
            out = cap.out
            for chunk in cap.chunker.push(cap.to_apm.push(frame)):
                out += cap.from_apm.push(self._process_chunk(chunk)).data
            n = len(frame.data)
            if len(out) < n:
                # not enough processed audio yet (framing remainder, resampler warm-up): delay
                # the stream by the shortfall once; afterwards the FIFO always has enough
                missing = n - len(out)
                out[:0] = bytes(missing)
                cap.delay += missing // frame.format.bytes_per_sample
            data = bytes(out[:n])
            del out[:n]
            ts = None if frame.timestamp is None else frame.timestamp - self.latency
        return AudioFrame(data, frame.sample_rate, frame.channels, ts)

    def reset(self) -> None:
        """Drop all state; the APM is re-created (fresh echo estimate) on the next frame."""
        with self._lock:
            self._dispose_apm()
            self._init_state()
        with self._pending_lock:
            self._pending.clear()
            self._pending_duration = 0.0

    def close(self) -> None:
        """Release the native APM. The processor can still be used (it re-initializes)."""
        self.reset()

    # ------------------------------------------------------------------- internals
    @property
    def _enabled(self) -> bool:
        return (
            self.echo_cancellation
            or self.noise_suppression
            or self.high_pass_filter
            or self.auto_gain_control
        )

    def _configure(self, fmt: AudioFormat) -> _CapturePath:
        if self._capture is not None:
            logger.debug("capture format changed to %s: resetting audio processing", fmt)
            self._dispose_apm()
            self._init_state()
        rate = _processing_rate(fmt.sample_rate, self.processing_rate)
        self._capture = _CapturePath(fmt, rate)
        self._render = _RenderPath(rate)
        return self._capture

    def _ensure_apm(self) -> Any:
        if self._apm is None and not self._broken:
            try:
                self._apm = self._rtc.AudioProcessingModule(
                    echo_cancellation=self.echo_cancellation,
                    noise_suppression=self.noise_suppression,
                    high_pass_filter=self.high_pass_filter,
                    auto_gain_control=self.auto_gain_control,
                )
            except Exception:
                self._broken = True
                logger.exception(
                    "could not create the WebRTC audio processing module; "
                    "microphone audio passes through unprocessed"
                )
        return self._apm

    def _dispose_apm(self) -> None:
        apm, self._apm = self._apm, None
        # livekit frees the native module on garbage collection only, which fails once
        # interpreter shutdown has begun: release it explicitly when the handle is exposed
        dispose = getattr(getattr(apm, "_ffi_handle", None), "dispose", None)
        if callable(dispose):
            try:
                dispose()
            except Exception:
                logger.debug("failed to dispose the audio processing module", exc_info=True)

    def _call(self, fn: Callable[[Any], Any], arg: Any) -> bool:
        try:
            fn(arg)
            return True
        except Exception:
            if not self._error_logged:
                self._error_logged = True
                logger.exception("WebRTC audio processing failed; passing audio through")
            return False

    def _drain_render(self) -> None:
        with self._pending_lock:
            if not self._pending:
                return
            frames = list(self._pending)
            self._pending.clear()
            self._pending_duration = 0.0
        render = self._render
        assert render is not None
        for frame in frames:
            for chunk in render.chunker.push(render.converter.push(frame)):
                if self.reference == "queued":
                    self._paced.append(chunk.data)
                else:
                    self._feed_render(render, chunk.data)
        while len(self._paced) > _MAX_PACED:
            self._paced.popleft()

    def _feed_render(self, render: _RenderPath, data: bytes) -> None:
        apm = self._ensure_apm()
        if apm is None:
            return
        if len(data) != render.samples * 2:  # the native library aborts on other sizes
            raise AssertionError(f"reference frame must be 10 ms, got {len(data) // 2} samples")
        frame = self._rtc.AudioFrame(bytearray(data), render.rate, 1, render.samples)
        self._call(apm.process_reverse_stream, frame)
        self._render_lead += 1

    def _process_chunk(self, chunk: AudioFrame) -> AudioFrame:
        apm = self._ensure_apm()
        if apm is None:
            return chunk
        if chunk.samples_per_channel != chunk.sample_rate // 100:  # the native library aborts
            raise AssertionError(f"capture frame must be 10 ms, got {chunk.samples_per_channel}")
        if self.echo_cancellation:
            render = self._render
            assert render is not None
            while self._paced and self._render_lead < _PACED_LEAD:
                self._feed_render(render, self._paced.popleft())
            while self._render_lead <= 0:  # nothing is playing: the reference is silence
                self._feed_render(render, render.silence)
            self._update_stream_delay(apm)
        buf = bytearray(chunk.data)
        frame = self._rtc.AudioFrame(
            buf, chunk.sample_rate, chunk.channels, chunk.samples_per_channel
        )
        ok = self._call(apm.process_stream, frame)
        if self.echo_cancellation:
            self._render_lead -= 1
        if not ok:
            return chunk
        return AudioFrame(bytes(buf), chunk.sample_rate, chunk.channels, chunk.timestamp)

    def _update_stream_delay(self, apm: Any) -> None:
        if self.delay_ms is not None:
            delay_ms = self.delay_ms
        elif self._input_latency is not None or self._output_latency is not None:
            total = (self._input_latency or 0.0) + (self._output_latency or 0.0)
            delay_ms = min(_MAX_STREAM_DELAY_MS, round(total * 1000))
        else:
            return  # no hint: AEC3 finds the delay on its own
        if delay_ms != self._applied_delay_ms and self._call(apm.set_stream_delay_ms, delay_ms):
            self._applied_delay_ms = delay_ms


class HalfDuplexGate(AudioProcessor):
    """Mutes the microphone while the agent is audible and for ``tail`` seconds after.

    The fallback when no echo canceller is available: the agent never hears itself, but
    the user cannot interrupt it (no barge-in) while it speaks. Reference frames passed to
    :meth:`process_render` are treated as queued for playback back to back, so both
    real-time feeding (a playback callback) and bursts ahead of time (the session's
    look-ahead) are handled. Frames at or below ``threshold_db`` (silence) do not close the
    gate.

    Args:
        tail: seconds the gate stays closed after the last audible audio has played; must
            cover output + input latency and room reverberation (raise it for Bluetooth).
        threshold_db: level in dBFS above which a reference frame counts as audible.
        clock: time source (defaults to :func:`voice_agent_next.utils.now`).
    """

    def __init__(
        self,
        tail: float = 0.3,
        *,
        threshold_db: float = -60.0,
        clock: Callable[[], float] = now,
    ) -> None:
        if not (math.isfinite(tail) and tail >= 0):
            raise ValueError(f"tail must be a finite number >= 0, got {tail}")
        self.tail = tail
        self.threshold_db = threshold_db
        self._clock = clock
        self._lock = threading.Lock()
        self._play_end = -math.inf
        self._audible_end = -math.inf

    @property
    def gating(self) -> bool:
        """True while capture audio is being muted."""
        return self._clock() < self._audible_end + self.tail

    def process_render(self, frame: AudioFrame) -> None:
        if not frame:
            return
        audible = frame.dbfs() > self.threshold_db
        t = self._clock()
        with self._lock:
            self._play_end = max(self._play_end, t) + frame.duration
            if audible:
                self._audible_end = self._play_end

    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        if frame and self.gating:
            return AudioFrame(
                bytes(len(frame.data)), frame.sample_rate, frame.channels, frame.timestamp
            )
        return frame

    def reset(self) -> None:
        """Open the gate immediately (e.g. after the agent's audio was cleared)."""
        with self._lock:
            self._play_end = -math.inf
            self._audible_end = -math.inf


_MODES = ("auto", "aec", "webrtc", "half_duplex", "headphones", "none")


def create_echo_canceller(
    mode: str = "auto",
    *,
    noise_suppression: bool = True,
    high_pass_filter: bool = True,
    auto_gain_control: bool = True,
    reference: ReferenceTiming = "played",
    delay_ms: int | None = None,
    processing_rate: int | None = None,
    tail: float = 0.3,
) -> AudioProcessor | None:
    """Create the echo-control processor for audio played through loudspeakers.

    Modes (:data:`EchoMode`):

    * ``"auto"``: :class:`WebRTCAudioProcessor` when the ``aec`` extra (``livekit``) is
      installed, otherwise ``None`` with a warning;
    * ``"aec"`` (alias ``"webrtc"``): :class:`WebRTCAudioProcessor`; raises
      :class:`~voice_agent_next.errors.MissingDependencyError` without the extra;
    * ``"half_duplex"``: :class:`HalfDuplexGate` with ``tail`` (no barge-in);
    * ``"headphones"`` / ``"none"``: ``None``, there is no echo to remove.

    The other keyword arguments configure :class:`WebRTCAudioProcessor`.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown echo mode {mode!r}; expected one of {', '.join(_MODES)}")
    if mode in ("headphones", "none"):
        return None
    if mode == "half_duplex":
        return HalfDuplexGate(tail)
    options: dict[str, Any] = {
        "noise_suppression": noise_suppression,
        "high_pass_filter": high_pass_filter,
        "auto_gain_control": auto_gain_control,
        "reference": reference,
        "delay_ms": delay_ms,
        "processing_rate": processing_rate,
    }
    if mode in ("aec", "webrtc"):
        return WebRTCAudioProcessor(**options)
    try:
        _load_rtc()
    except ImportError as exc:  # MissingDependencyError is an ImportError
        logger.warning(
            "echo cancellation is unavailable (%s). The agent may hear itself through the "
            "speakers and interrupt itself: install 'voice-agent-next[aec]', use headphones, "
            "or use echo mode 'half_duplex'.",
            exc,
        )
        return None
    return WebRTCAudioProcessor(**options)
