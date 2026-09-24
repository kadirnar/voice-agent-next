"""Moonshine Streaming STT (``moonshine-voice``).

Unit tests run offline against a fake ``moonshine_voice`` package that mimics the real
``Transcriber`` / ``Stream`` API (lines with ``is_new`` / ``is_updated`` / ``is_complete``
flags; ``stop()`` completes every line, reports a failed final pass as an ``Error`` event
and returns ``None``; ``start()`` restarts line times at zero). The ``@pytest.mark.model``
tests load the real ``tiny-streaming`` model (45 MB) and transcribe the JFK clip::

    uv sync --extra moonshine && uv run pytest -m model tests/providers/test_moonshine.py
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import json
import logging
import math
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from voice_agent_next import AudioFrame, create
from voice_agent_next.audio.wav import read_wav
from voice_agent_next.engines import CascadeEngine
from voice_agent_next.errors import (
    ConfigurationError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
)
from voice_agent_next.metrics import STTMetrics
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.moonshine import MoonshineSTT, _map_error
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import STTEvent, STTEventType
from voice_agent_next.utils.download import DownloadError, download

WORDS = ["ask", "not", "what", "your", "country", "can", "do", "for", "you"]
ARCHS = {"TINY": 0, "BASE": 1, "TINY_STREAMING": 2, "BASE_STREAMING": 3}
ARCHS |= {"SMALL_STREAMING": 4, "MEDIUM_STREAMING": 5}
CATALOG = {"en": [5, 4, 1, 2, 0], "es": [4, 2, 1], "ko": [0]}
BASE_URL = "https://download.moonshine.ai/model/{arch}-{lang}/quantized"
FILES = ["encoder.ort", "decoder_kv.ort", "tokenizer.bin"]


# ------------------------------------------------------------------------ fakes
@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class FakeLine:
    text: str
    start_time: float
    duration: float
    line_id: int
    is_complete: bool
    is_updated: bool = False
    is_new: bool = False
    has_text_changed: bool = False
    words: list[FakeWord] | None = None


@dataclass
class FakeTranscript:
    lines: list[FakeLine]


@dataclass
class FakeError:
    error: Exception
    stream_handle: int
    line: Any = None


@dataclass
class Backend:
    """Stands in for ``moonshine_voice`` and records every call."""

    words_per_second: float = 4.0
    speech_start: float = 0.1
    complete_after: float | None = None
    """Seconds of audio after which the runtime's own VAD completes the line."""
    stop_error: Exception | None = None
    update_error: Exception | None = None
    load_error: Exception | None = None
    download_error: Exception | None = None
    word_timestamps: bool = False
    cache: Path = Path("/fake-cache")
    loads: list[dict[str, Any]] = field(default_factory=list)
    downloads: list[dict[str, Any]] = field(default_factory=list)
    updates: list[int] = field(default_factory=list)
    """Flags of every ``update_transcription`` call."""
    threads: set[str] = field(default_factory=set)
    streams: list[Any] = field(default_factory=list)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = self

        class ModelArch(int):
            def __new__(cls, value: int) -> ModelArch:
                if value not in ARCHS.values():
                    raise ValueError(f"{value} is not a valid ModelArch")
                return super().__new__(cls, value)

        class MoonshineError(Exception):
            pass

        class Stream:
            def __init__(self, transcriber: Transcriber, update_interval: float) -> None:
                self.update_interval = update_interval
                self.listeners: list[Any] = []
                self.started = False
                self.closed = False
                self.seconds = 0.0
                self.line_id = 1000 + 100 * len(backend.streams)
                self.state: FakeLine | None = None
                self.completed: list[FakeLine] = []
                backend.streams.append(self)

            def _track(self) -> None:
                backend.threads.add(threading.current_thread().name)

            def add_listener(self, listener: Any) -> None:
                self.listeners.append(listener)

            def start(self) -> None:
                self._track()
                self.started, self.seconds, self.state, self.completed = True, 0.0, None, []
                self.line_id += 1

            def add_audio(self, audio: list[float], sample_rate: int = 16_000) -> None:
                self._track()
                assert self.started and isinstance(audio, list) and sample_rate == 16_000
                self.seconds += len(audio) / sample_rate

            def _transcribe(self, complete: bool) -> FakeTranscript:
                speech = self.seconds - backend.speech_start
                n = min(len(WORDS), math.floor(speech * backend.words_per_second))
                if n <= 0 and self.state is None:
                    return FakeTranscript(list(self.completed))
                text = " ".join(WORDS[: max(n, 1)]).capitalize()
                prev = self.state
                done = complete or (
                    backend.complete_after is not None and self.seconds >= backend.complete_after
                )
                line = FakeLine(
                    text=text,
                    start_time=backend.speech_start,
                    duration=round(speech, 3),
                    line_id=self.line_id,
                    is_complete=done,
                    is_updated=prev is None or prev.text != text or done,
                    is_new=prev is None,
                    has_text_changed=prev is None or prev.text != text,
                    words=[
                        FakeWord(
                            f" {w}",
                            backend.speech_start + i * 0.25,
                            backend.speech_start + (i + 1) * 0.25,
                            0.9,
                        )
                        for i, w in enumerate(text.split())
                    ]
                    if backend.word_timestamps
                    else None,
                )
                self.state = line
                if done:
                    self.completed.append(line)
                    self.state = None
                    self.line_id += 1
                    self.seconds = -1e9  # nothing more is said in this stream
                    return FakeTranscript(list(self.completed))
                return FakeTranscript([*self.completed, line])

            def update_transcription(self, flags: int = 0) -> FakeTranscript:
                self._track()
                backend.updates.append(flags)
                if backend.update_error is not None:
                    raise backend.update_error
                return self._transcribe(complete=False)

            def stop(self) -> FakeTranscript | None:
                self._track()
                self.started = False
                if backend.stop_error is not None:
                    for listener in self.listeners:
                        listener(FakeError(error=backend.stop_error, stream_handle=1))
                    return None
                if self.state is None and self.seconds <= backend.speech_start:
                    return FakeTranscript([replace_updated(line) for line in self.completed])
                return self._transcribe(complete=True)

            def close(self) -> None:
                self.closed = True

        def replace_updated(line: FakeLine) -> FakeLine:
            return FakeLine(line.text, line.start_time, line.duration, line.line_id, True)

        class Transcriber:
            def __init__(
                self,
                model_path: str,
                model_arch: Any,
                update_interval: float = 0.5,
                options: dict[str, Any] | None = None,
            ) -> None:
                if backend.load_error is not None:
                    raise backend.load_error
                backend.loads.append(
                    {
                        "path": model_path,
                        "arch": int(model_arch),
                        "update_interval": update_interval,
                        "options": dict(options or {}),
                        "thread": threading.current_thread().name,
                    }
                )

            def create_stream(self, update_interval: float | None = None) -> Stream:
                return Stream(self, update_interval or 0.5)

        def find_model_info(language: str = "en", model_arch: Any = None) -> dict[str, Any]:
            if language not in CATALOG:
                raise ValueError(f"Language not found: {language}")
            if model_arch is not None and int(model_arch) not in CATALOG[language]:
                raise ValueError(f"Model not found for language: {language}")
            arch = model_arch if model_arch is not None else ModelArch(CATALOG[language][0])
            return {"model_arch": arch, "download_url": "https://x", "language": language}

        def download_model_from_info(info: dict[str, Any], **kwargs: Any) -> tuple[str, Any]:
            if backend.download_error is not None:
                raise backend.download_error
            backend.downloads.append({"info": info, **kwargs})
            return f"/fake-cache/{int(info['model_arch'])}-{info['language']}", info["model_arch"]

        def dependencies(language: str, options: dict[str, Any]) -> str:
            base = BASE_URL.format(arch=options["model_arch"], lang=language)
            files = FILES + (
                ["decoder_with_attention.ort"] if options.get("word_timestamps") else []
            )
            return json.dumps(
                {"groups": [{"base_url": base, "files": [{"name": f} for f in files]}]}
            )

        mv = _module("moonshine_voice")
        mv.Transcriber = Transcriber  # type: ignore[attr-defined]
        mv.ModelArch = ModelArch  # type: ignore[attr-defined]
        mv.MoonshineError = MoonshineError  # type: ignore[attr-defined]
        dl = _module("moonshine_voice.download")
        dl.find_model_info = find_model_info  # type: ignore[attr-defined]
        dl.download_model_from_info = download_model_from_info  # type: ignore[attr-defined]
        api = _module("moonshine_voice.moonshine_api")
        api.moonshine_get_stt_dependencies_string = dependencies  # type: ignore[attr-defined]
        files = _module("moonshine_voice.download_file")
        files.get_cache_dir = lambda: backend.cache  # type: ignore[attr-defined]
        for mod in (mv, dl, api, files):
            monkeypatch.setitem(sys.modules, mod.__name__, mod)


def _module(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> Backend:
    monkeypatch.delenv("VAN_OFFLINE", raising=False)
    fake = Backend()
    fake.install(monkeypatch)
    return fake


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = math.ceil(frame.duration / step - 1e-9)
    return [frame.slice(i * step, (i + 1) * step) for i in range(n)]


async def feed(stream: Any, frame: AudioFrame, *, pace: float = 0.0) -> None:
    for c in chunks(frame):
        stream.push_audio(c)
        await asyncio.sleep(pace)


async def collect(stream: Any, until: STTEventType, timeout: float = 5.0) -> list[STTEvent]:
    events: list[STTEvent] = []

    async def run() -> None:
        async for ev in stream:
            events.append(ev)
            if ev.type == until:
                return

    await asyncio.wait_for(run(), timeout)
    return events


def kinds(events: list[STTEvent]) -> list[str]:
    return [str(e.type) for e in events]


# ------------------------------------------------------------------ streaming
async def test_interims_while_speaking_and_a_final_on_flush(backend: Backend) -> None:
    stt = MoonshineSTT()
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    await feed(stream, synth_speech(1.2, 16_000), pace=0.002)
    await asyncio.sleep(0.2)
    stream.flush()
    events = await collect(stream, STTEventType.FINAL_TRANSCRIPT)
    types = kinds(events)
    assert types[0] == "start_of_speech"
    assert types[-1] == "final_transcript" and "end_of_speech" not in types
    interims = [e.text for e in events if e.type == STTEventType.INTERIM_TRANSCRIPT]
    assert interims and len(set(interims)) == len(interims)  # only changed text is emitted
    final = events[-1]
    assert final.text == "Ask not what your"
    assert final.transcript is not None and final.transcript.language == "en"
    assert final.transcript.start_time == pytest.approx(0.1)
    assert final.transcript.end_time == pytest.approx(1.2, abs=0.03)
    assert len({e.segment_id for e in events}) == 1
    (m,) = metrics
    assert m.provider == "moonshine" and m.model == "small-streaming" and m.streamed
    assert m.latency is not None and m.latency >= 0
    await stream.aclose()
    assert backend.streams[0].closed


async def test_updates_follow_update_interval_and_force_flag(backend: Backend) -> None:
    stt = MoonshineSTT(update_interval=0.5, force_updates=True)
    stream = stt.stream()
    await feed(stream, synth_speech(2.0, 16_000), pace=0.001)
    stream.end_input()
    [e async for e in stream]
    assert 2 <= len(backend.updates) <= 4  # ~every 0.5 s of audio, not every frame
    assert set(backend.updates) == {1}  # MOONSHINE_FLAG_FORCE_UPDATE
    assert backend.loads[0]["update_interval"] >= 1e6  # the package's own cadence is off
    await stream.aclose()


async def test_flush_without_speech_emits_an_empty_final(backend: Backend) -> None:
    stt = MoonshineSTT()
    stream = stt.stream()
    await feed(stream, AudioFrame.silence(0.05, 16_000))
    stream.flush()
    events = await collect(stream, STTEventType.FINAL_TRANSCRIPT)
    assert kinds(events) == ["final_transcript"]
    assert events[0].text == "" and events[0].segment_id is None
    await stream.aclose()


async def test_runtime_vad_completes_lines_and_times_continue_after_flush(
    backend: Backend,
) -> None:
    backend.complete_after = 0.8
    stt = MoonshineSTT(update_interval=0.1)
    stream = stt.stream()
    await feed(stream, synth_speech(1.0, 16_000), pace=0.002)
    first = await collect(stream, STTEventType.END_OF_SPEECH)
    assert kinds(first)[-2:] == ["final_transcript", "end_of_speech"]
    assert first[-1].text == first[-2].text == "Ask not"
    stream.flush()  # nothing left: the line was already final
    flushed = await collect(stream, STTEventType.FINAL_TRANSCRIPT)
    assert [e.text for e in flushed] == [""]
    # the native stream restarted at t=0; transcript times continue the session clock
    backend.complete_after = None
    await feed(stream, synth_speech(0.6, 16_000), pace=0.002)
    stream.end_input()
    rest = [e async for e in stream]
    finals = [e for e in rest if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert [e.text for e in finals] == ["Ask not"]
    assert finals[0].transcript is not None
    assert finals[0].transcript.start_time == pytest.approx(1.0 + 0.1, abs=0.03)
    assert finals[0].segment_id != first[-1].segment_id
    await stream.aclose()


async def test_interim_results_off(backend: Backend) -> None:
    stt = MoonshineSTT(interim_results=False)
    assert not stt.capabilities.interim_results
    stream = stt.stream()
    await feed(stream, synth_speech(1.0, 16_000), pace=0.002)
    stream.end_input()
    events = [e async for e in stream]
    assert "interim_transcript" not in kinds(events)
    assert [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT] == ["Ask not what"]
    assert backend.loads[0]["options"]["decode_incomplete_lines"] == "false"
    await stream.aclose()


async def test_word_timestamps_are_on_the_session_clock(backend: Backend) -> None:
    backend.word_timestamps = True
    stt = MoonshineSTT(word_timestamps=True)
    assert stt.capabilities.word_timestamps
    stream = stt.stream()
    await feed(stream, synth_speech(0.5, 16_000))
    stream.flush()
    await collect(stream, STTEventType.FINAL_TRANSCRIPT)
    await feed(stream, synth_speech(0.7, 16_000))
    stream.flush()
    events = await collect(stream, STTEventType.FINAL_TRANSCRIPT)
    words = events[-1].transcript.words if events[-1].transcript else None
    assert words is not None and [w.word for w in words] == ["Ask", "not"]
    assert words[0].start == pytest.approx(0.5 + 0.1, abs=0.03)
    assert words[0].confidence == pytest.approx(0.9)
    assert backend.loads[0]["options"]["word_timestamps"] == "true"
    assert backend.downloads[0]["include_word_timestamps"] is True
    await stream.aclose()


async def test_batch_transcribe_runs_a_stream(backend: Backend) -> None:
    stt = MoonshineSTT()
    result = await stt.transcribe(synth_speech(1.0, 48_000).to_channels(2))
    assert result.text == "Ask not what" and result.language == "en"


async def test_native_calls_run_off_the_event_loop(backend: Backend) -> None:
    stt = MoonshineSTT()
    stream = stt.stream()
    await feed(stream, synth_speech(0.6, 16_000))
    stream.end_input()
    [e async for e in stream]
    assert backend.threads and threading.current_thread().name not in backend.threads
    assert backend.loads[0]["thread"] != threading.current_thread().name
    await stream.aclose()


async def test_failed_final_pass_raises_from_the_stream(backend: Backend) -> None:
    backend.stop_error = RuntimeError("decoder exploded")
    stt = MoonshineSTT()
    stream = stt.stream()
    await feed(stream, synth_speech(0.6, 16_000))
    stream.flush()
    with pytest.raises(ProviderError, match="decoder exploded"):
        [e async for e in stream]
    await stream.aclose()
    assert backend.streams[0].closed


async def test_failed_update_raises_from_the_stream(backend: Backend) -> None:
    backend.update_error = backend_error("MoonshineInvalidHandleError", "Invalid handle")
    stream = MoonshineSTT().stream()
    await feed(stream, synth_speech(0.6, 16_000))
    with pytest.raises(ProviderError, match="Invalid handle"):
        await collect(stream, STTEventType.FINAL_TRANSCRIPT)
    await stream.aclose()


async def test_aclose_mid_stream_closes_the_native_stream(backend: Backend) -> None:
    stream = MoonshineSTT().stream()
    await feed(stream, synth_speech(0.4, 16_000))
    await asyncio.sleep(0.05)
    await stream.aclose()
    assert backend.streams and backend.streams[0].closed


# ------------------------------------------------------------ models and loading
async def test_model_loads_once_lazily_in_a_worker_thread(backend: Backend) -> None:
    stt = MoonshineSTT(
        model="medium-streaming",
        keyterms=["Kubernetes", " Moonshine "],
        cache_dir="model-cache",
        options={"vad_threshold": 0.6, "identify_speakers": False, "return_audio_data": True},
    )
    assert stt.model_path is None
    await asyncio.gather(stt.warmup(), stt.warmup())
    (load,) = backend.loads
    assert load["path"] == "/fake-cache/5-en" == stt.model_path
    assert load["arch"] == 5
    assert load["options"] == {
        "return_audio_data": "true",  # options= wins
        "keyterms": "Kubernetes,Moonshine",
        "vad_threshold": "0.6",
        "identify_speakers": "false",
    }
    (dl,) = backend.downloads
    assert int(dl["info"]["model_arch"]) == 5 and dl["info"]["language"] == "en"
    assert dl["cache_root"] == Path("model-cache") and dl["include_word_timestamps"] is False
    assert callable(dl["on_progress"])
    assert backend.streams[0].closed  # the warm-up stream


async def test_languages_and_non_english_licence_notice(
    backend: Backend, caplog: pytest.LogCaptureFixture
) -> None:
    stt = MoonshineSTT(language="es-ES")
    assert stt.language == "es"
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await stt.warmup()
    assert "Community License" in caplog.text
    assert backend.loads[0]["path"] == "/fake-cache/4-es"
    assert MoonshineSTT(language="auto").language is None
    with pytest.raises(ConfigurationError, match="no 'small-streaming' model for language 'ko'"):
        await MoonshineSTT(language="ko").warmup()
    with pytest.raises(ConfigurationError, match="language 'xx'"):
        await MoonshineSTT(language="xx").warmup()
    await MoonshineSTT(model="tiny", language="ko").warmup()
    with pytest.raises(ConfigurationError, match="another MoonshineSTT"):
        stt.stream(language="de")
    stream = stt.stream(language="es-MX")
    await stream.aclose()


async def test_local_model_directory(backend: Backend, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="arch="):
        MoonshineSTT(model=str(tmp_path))
    stt = MoonshineSTT(model=str(tmp_path), arch="tiny-streaming")
    await stt.warmup()
    assert backend.loads[0]["path"] == str(tmp_path) and backend.loads[0]["arch"] == 2
    assert backend.downloads == []


async def test_offline_uses_only_cached_files(
    backend: Backend, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend.cache = tmp_path
    monkeypatch.setenv("VAN_OFFLINE", "1")
    with pytest.raises(ProviderConnectionError, match=r"not cached \(encoder.ort"):
        await MoonshineSTT(model="tiny-streaming").warmup()
    root = tmp_path / "download.moonshine.ai/model/2-en/quantized"
    root.mkdir(parents=True)
    for name in FILES:
        (root / name).write_bytes(b"x")
    await MoonshineSTT(model="tiny-streaming").warmup()
    assert backend.loads[-1]["path"] == str(root) and backend.downloads == []
    with pytest.raises(ProviderConnectionError, match="decoder_with_attention"):
        await MoonshineSTT(model="tiny-streaming", word_timestamps=True).warmup()
    monkeypatch.delenv("VAN_OFFLINE")
    other = tmp_path / "other"
    with pytest.raises(ProviderConnectionError, match=re.escape(str(other))):
        await MoonshineSTT(model="tiny-streaming", local_files_only=True, cache_dir=other).warmup()


async def test_download_and_load_errors_are_mapped(backend: Backend) -> None:
    backend.download_error = backend_error("ConnectionError", "no route to host")
    with pytest.raises(ProviderConnectionError, match="no route to host"):
        await MoonshineSTT().warmup()
    backend.download_error = None
    backend.load_error = backend_error("MoonshineError", "Failed to load transcriber")
    with pytest.raises(ProviderError, match="Failed to load transcriber"):
        await MoonshineSTT().warmup()


def test_error_mapping() -> None:
    assert isinstance(_map_error(ValueError("bad"), "x"), ConfigurationError)
    assert isinstance(
        _map_error(backend_error("MoonshineInvalidArgumentError", "bad"), "x"), ConfigurationError
    )
    assert isinstance(
        _map_error(backend_error("RequestException", "x"), "x"), ProviderConnectionError
    )
    assert isinstance(_map_error(TimeoutError(), "x"), ProviderConnectionError)
    err = _map_error(backend_error("MoonshineError", "boom"), "transcription")
    assert type(err) is ProviderError and err.provider == "moonshine"
    assert "moonshine transcription failed: boom" in str(err)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "huge"}, "unknown model 'huge'"),
        ({"update_interval": 0}, "update_interval"),
        ({"keyterms": ["a,b"]}, "commas"),
    ],
)
def test_invalid_options(backend: Backend, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        MoonshineSTT(**kwargs)


def test_defaults(backend: Backend) -> None:
    stt = MoonshineSTT()
    assert (stt.model, stt.arch, stt.language, stt.sample_rate) == (
        "small-streaming",
        "small-streaming",
        None,
        16_000,
    )
    assert stt.streaming_model and not MoonshineSTT(model="base").streaming_model
    caps = stt.capabilities
    assert caps.streaming and caps.interim_results and not caps.word_timestamps
    assert not caps.end_of_turn and not caps.language_detection
    assert MoonshineSTT(model="TINY-streaming").arch == "tiny-streaming"


def test_missing_dependency_has_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "moonshine_voice", None)  # import fails
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[moonshine\]"):
        MoonshineSTT()


# ------------------------------------------------------ registry and pipeline
def test_registry_spec(backend: Backend) -> None:
    stt = create("stt", "moonshine/tiny-streaming", language="en")
    assert isinstance(stt, MoonshineSTT) and stt.model == "tiny-streaming"
    assert create("stt", "moonshine").model == "small-streaming"
    spec = get_provider("stt", "moonshine")
    assert spec.local and spec.env == () and spec.extra == "moonshine"
    assert spec.requires == ("moonshine_voice",) and "medium-streaming" in spec.models


async def test_cascade_uses_the_native_stream(backend: Backend) -> None:
    engine = CascadeEngine(stt="moonshine/tiny-streaming", vad="energy", llm="mock", tts="mock")
    assert isinstance(engine.stt, MoonshineSTT)  # no StreamAdapter: it streams natively


def backend_error(name: str, message: str) -> Exception:
    """An exception whose class carries a ``moonshine_voice`` / ``requests`` class name."""
    return type(name, (Exception,), {})(message)


# ------------------------------------------------------ real model (opt-in)
JFK_URL = (
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/"
    "b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav"
)
JFK_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


@pytest.fixture(scope="module")
def jfk() -> AudioFrame:
    """JFK, 1961 (public domain): "And so my fellow Americans, ask not what your country can
    do for you, ask what you can do for your country." 11 s, 16 kHz mono."""
    pytest.importorskip("moonshine_voice")
    try:
        return read_wav(download(JFK_URL, subdir="testdata", sha256=JFK_SHA256))  # shared cache
    except DownloadError as exc:
        pytest.skip(f"test clip unavailable: {exc}")


async def _loaded(stt: MoonshineSTT) -> MoonshineSTT:
    try:
        await stt.warmup()
    except ProviderConnectionError as exc:
        pytest.skip(f"model unavailable: {exc}")
    return stt


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower())


@pytest.mark.model
async def test_tiny_streaming_transcribes_the_jfk_clip(jfk: AudioFrame) -> None:
    stt = await _loaded(MoonshineSTT(model="tiny-streaming"))
    result = await stt.transcribe(jfk)
    text = _normalize(result.text)
    assert "ask not" in text and "what you can do for your country" in text
    await stt.aclose()


@pytest.mark.model
async def test_tiny_streaming_streams_in_real_time_with_a_fast_flush(jfk: AudioFrame) -> None:
    """Interims while JFK speaks; the flush right after "... for your country" is fast."""
    stt = await _loaded(MoonshineSTT(model="tiny-streaming"))
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    events: list[STTEvent] = []

    async def consume() -> None:
        async for ev in stream:
            events.append(ev)

    task = asyncio.create_task(consume())
    await feed(stream, jfk.slice(0, 10.9), pace=0.004)  # ~5x real time
    stream.flush()
    stream.end_input()
    await asyncio.wait_for(task, 30)
    await stream.aclose()
    finals = [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert "ask what you can do for your country" in _normalize(" ".join(finals))
    assert any(e.type == STTEventType.INTERIM_TRANSCRIPT for e in events)
    assert kinds(events)[0] == "start_of_speech"
    flush = [m.latency for m in metrics if m.latency is not None]
    assert flush and flush[0] < 2.0  # typically ~0.1 s on a desktop CPU
