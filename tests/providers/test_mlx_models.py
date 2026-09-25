"""Real MLX models on Apple silicon (``pytest -m model``; skipped elsewhere).

Always (about 315 MB of downloads, cached in CI): Whisper tiny through mlx-whisper (74 MB)
and Pocket TTS through mlx-audio (240 MB). Larger models are opt-in:

* ``VAN_TEST_MLX_PARAKEET=parakeet-tdt_ctc-110m`` (459 MB; or any ``mlx/`` model) streams
  a clip through parakeet-mlx;
* ``VAN_TEST_MLX_KOKORO=1`` synthesizes with Kokoro (355 MB, plus spaCy's English model);
* ``VAN_TEST_MLX_LM_MODEL=mlx-community/Qwen3-0.6B-4bit`` starts ``mlx_lm.server`` with
  that model and measures time to first token and a tool call.

Each test prints its latency numbers (run with ``-s`` to see them).
"""

from __future__ import annotations

import asyncio
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from voice_agent_next import AudioFrame
from voice_agent_next.errors import ProviderConnectionError
from voice_agent_next.metrics import LLMMetrics, STTMetrics, TTSMetrics
from voice_agent_next.stt import STT, STTEventType
from voice_agent_next.utils.clock import now
from voice_agent_next.utils.download import DownloadError, download

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="MLX needs macOS on Apple silicon",
    ),
]

JFK_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
JFK_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
JFK_TEXT = "ask not what your country can do for you"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower())


def _jfk_clip() -> AudioFrame:
    sf = pytest.importorskip("soundfile")  # a librosa dependency (parakeet-mlx)
    try:
        path = download(JFK_URL, filename="jfk.flac", subdir="test-audio", sha256=JFK_SHA256)
    except DownloadError as exc:
        pytest.skip(f"test clip unavailable: {exc}")
    data, rate = sf.read(str(path), dtype="float32")
    return AudioFrame.from_numpy(data, rate)


def _chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = math.ceil(frame.duration / step - 1e-9)
    return [frame.slice(i * step, (i + 1) * step) for i in range(n)]


async def _loaded(component: object) -> None:
    try:
        await component.warmup()  # type: ignore[attr-defined]
    except ProviderConnectionError as exc:
        pytest.skip(f"model unavailable: {exc}")


# ------------------------------------------------------------------------ STT
@pytest.mark.timeout(900)
async def test_whisper_tiny_transcribes_a_public_domain_clip() -> None:
    pytest.importorskip("mlx_whisper")
    from voice_agent_next.providers.mlx_whisper import MLXWhisperSTT

    clip = _jfk_clip()
    stt = MLXWhisperSTT(model="tiny", language="en", word_timestamps=True)
    t0 = now()
    await _loaded(stt)
    load = now() - t0
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    result = await stt.transcribe(clip)
    assert JFK_TEXT in _normalize(result.text)
    assert result.words and _normalize(result.words[0].word) == "and"
    # a typical voice turn: the first 3 s, transcribed after the end of speech
    turn = clip.slice(0.0, 3.0)
    times = []
    for _ in range(3):
        t0 = now()
        await stt.transcribe(turn)
        times.append(now() - t0)
    await stt.aclose()
    print(
        f"\nmlx-whisper tiny: load+warm-up {load:.2f} s; {clip.duration:.1f} s clip in "
        f"{metrics[0].duration * 1000:.0f} ms; final for a 3 s turn: "
        f"{min(times) * 1000:.0f} ms (best of 3), {np.median(times) * 1000:.0f} ms median"
    )


@pytest.mark.timeout(900)
async def test_parakeet_streams_a_clip() -> None:
    model = os.environ.get("VAN_TEST_MLX_PARAKEET")
    if not model:
        pytest.skip("set VAN_TEST_MLX_PARAKEET=<model> (e.g. parakeet-tdt_ctc-110m)")
    pytest.importorskip("parakeet_mlx")
    from voice_agent_next.providers.mlx import ParakeetMLXSTT

    clip = _jfk_clip()
    stt = ParakeetMLXSTT(model=model)
    t0 = now()
    await _loaded(stt)
    load = now() - t0
    batch = await stt.transcribe(clip)
    assert JFK_TEXT in _normalize(batch.text)

    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    interims: list[float] = []
    finals: list[str] = []

    async def consume() -> None:
        async for ev in stream:
            if ev.type == STTEventType.INTERIM_TRANSCRIPT:
                interims.append(now())
            elif ev.type == STTEventType.FINAL_TRANSCRIPT:
                finals.append(ev.text)

    reader = asyncio.create_task(consume())
    start = now()
    for i, frame in enumerate(_chunks(clip)):  # real time, like a microphone
        stream.push_audio(frame)
        await asyncio.sleep(max(0.0, start + (i + 1) * 0.02 - now()))
    stream.end_input()
    await reader
    await stream.aclose()
    await stt.aclose()
    assert JFK_TEXT in _normalize(" ".join(finals))
    [m] = [m for m in metrics if m.streamed]
    assert m.latency is not None and m.latency < 1.0
    print(
        f"\nparakeet-mlx {model}: load+warm-up {load:.2f} s; streaming {clip.duration:.1f} s "
        f"in real time: {len(interims)} interim results, final {m.latency * 1000:.0f} ms "
        "after the flush"
    )


# ------------------------------------------------------------------------ TTS
async def _speak(tts: object, text: str) -> tuple[AudioFrame, float, float, int]:
    t0 = now()
    ttfb: float | None = None
    frames = []
    async for item in tts.synthesize(text):  # type: ignore[attr-defined]
        if item.frame:
            ttfb = ttfb if ttfb is not None else now() - t0
            frames.append(item.frame)
    assert ttfb is not None
    return AudioFrame.concat(frames), ttfb, now() - t0, len(frames)


@pytest.mark.timeout(900)
async def test_pocket_tts_streams_speech_that_whisper_understands() -> None:
    pytest.importorskip("mlx_audio")
    from voice_agent_next.providers.mlx_audio import MLXAudioTTS

    tts = MLXAudioTTS(model="pocket-tts", streaming_interval=0.24)
    t0 = now()
    await _loaded(tts)
    load = now() - t0
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    text = "Hello! This is Pocket TTS, streaming speech from the Apple silicon GPU."
    runs = [await _speak(tts, text) for _ in range(3)]
    audio, _, elapsed, n = runs[-1]
    assert audio.sample_rate == 24_000 and 2.0 < audio.duration < 12.0
    assert audio.rms() > 0.01 and n > 3  # streamed, not one block
    assert metrics[-1].error is None
    ttfbs = [r[1] for r in runs]
    print(
        f"\nmlx-audio pocket-tts: load+warm-up {load:.2f} s; first audio "
        f"{min(ttfbs) * 1000:.0f} ms (best of 3), {np.median(ttfbs) * 1000:.0f} ms median; "
        f"RTF {elapsed / audio.duration:.3f} ({audio.duration:.2f} s of audio in {n} chunks)"
    )
    await tts.aclose()

    if not __import__("importlib").util.find_spec("mlx_whisper"):
        return
    from voice_agent_next.providers.mlx_whisper import MLXWhisperSTT

    stt: STT = MLXWhisperSTT(model="tiny", language="en")
    await _loaded(stt)
    heard = _normalize((await stt.transcribe(audio)).text)
    await stt.aclose()
    assert "pocket" in heard or "streaming speech" in heard, heard


@pytest.mark.timeout(900)
async def test_kokoro_synthesizes() -> None:
    if not os.environ.get("VAN_TEST_MLX_KOKORO"):
        pytest.skip("set VAN_TEST_MLX_KOKORO=1 (355 MB model and spaCy's English model)")
    pytest.importorskip("mlx_audio")
    pytest.importorskip("misaki")
    from voice_agent_next.providers.mlx_audio import MLXAudioTTS

    tts = MLXAudioTTS(model="kokoro", voice="af_heart")
    t0 = now()
    await _loaded(tts)
    load = now() - t0
    runs = [await _speak(tts, "Hello! How can I help you today?") for _ in range(3)]
    audio = runs[-1][0]
    assert 1.0 < audio.duration < 6.0 and audio.rms() > 0.01
    ttfbs = [r[1] for r in runs]
    print(
        f"\nmlx-audio kokoro: load+warm-up {load:.2f} s; first audio "
        f"{min(ttfbs) * 1000:.0f} ms (best of 3), RTF {runs[-1][2] / audio.duration:.3f}"
    )
    await tts.aclose()


# ------------------------------------------------------------------------ LLM
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def mlx_lm_server(tmp_path: Path) -> Iterator[tuple[str, str]]:
    """``(model, base_url)`` of a running ``mlx_lm.server``."""
    model = os.environ.get("VAN_TEST_MLX_LM_MODEL")
    if not model:
        pytest.skip("set VAN_TEST_MLX_LM_MODEL=<mlx-community model> (e.g. Qwen3-0.6B-4bit)")
    pytest.importorskip("mlx_lm")
    pytest.importorskip("openai")
    port = _free_port()
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm.server", "--model", model, "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except OSError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        pytest.fail(log_path.read_text()[-2000:])
                    time.sleep(1.0)
            yield model, f"http://127.0.0.1:{port}/v1"
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


@pytest.mark.timeout(1200)
async def test_mlx_lm_server_streams_and_calls_tools(mlx_lm_server: tuple[str, str]) -> None:
    from voice_agent_next.chat import ChatContext
    from voice_agent_next.providers.mlx_lm import MLXLMServerLLM
    from voice_agent_next.tools import function_tool

    model, base_url = mlx_lm_server
    llm = MLXLMServerLLM(base_url=base_url)
    await llm.warmup()  # loads the model (1-token completion)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)

    ctx = ChatContext()
    ctx.add_message("system", "You are a friendly voice assistant. Answer in one sentence.")
    ctx.add_message("user", "Say hello.")
    for _ in range(3):
        result = await llm.chat(ctx, max_tokens=40).collect()
        assert result.text.strip()

    @function_tool
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"sunny in {city}"

    ctx = ChatContext()
    ctx.add_message("system", "Use the tools to answer.")
    ctx.add_message("user", "What's the weather in Paris?")
    call = await llm.chat(ctx, tools=[get_weather], max_tokens=100).collect()
    await llm.aclose()
    ttfts = [m.ttft for m in metrics[:3] if m.ttft is not None]
    print(
        f"\nmlx_lm.server {model}: TTFT {min(ttfts) * 1000:.0f} ms (best of 3), "
        f"{np.median(ttfts) * 1000:.0f} ms median; tool call: "
        f"{[(c.name, c.arguments) for c in call.tool_calls]}"
    )
    assert [c.name for c in call.tool_calls] == ["get_weather"]
    assert "paris" in call.tool_calls[0].arguments.lower()
