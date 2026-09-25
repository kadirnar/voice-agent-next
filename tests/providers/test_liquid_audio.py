"""``liquid-audio`` provider against a fake ``llama-liquid-audio-server``.

The fake replays the real server's behaviour, as observed on the Linux x64 runner
(LFM2.5-Audio-1.5B Q4_0):

* SSE lines ``data: {"object":"chat.completion.chunk",...}`` with ``delta.content`` text and
  ``delta.audio_chunk`` = ``{"data": <base64 float32 PCM>, "format": "pcm",
  "sample_rate": 24000}`` (1920 samples = 80 ms per chunk; 6 text tokens, then 12 audio
  chunks), a final ``{"delta": {}, "finish_reason": "stop"}`` chunk and ``data: [DONE]``;
* a stateful context: ``reset_context`` (default true) clears it, otherwise messages are
  appended after the server's own reply;
* HTTP 400 for any role but ``system``/``user``; an in-stream error for an unsupported
  system prompt;
* after a request aborted by the client, a request without ``reset_context`` fails
  (``failed to run prefill``): the server's stop flag is only cleared by a reset.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np
import pytest

from tests.test_session import Recorder, speak, wait_for
from voice_agent_next import Agent, AgentSession, AgentState, CascadeOptions, ChatContext
from voice_agent_next.audio.frame import AudioFrame
from voice_agent_next.audio.wav import read_wav
from voice_agent_next.chat import AudioContent, ChatMessage
from voice_agent_next.errors import ConfigurationError, ProviderError
from voice_agent_next.metrics import LLMMetrics
from voice_agent_next.providers import liquid_audio
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.liquid_audio import (
    INTERLEAVED_PROMPT,
    LiquidAudioLLM,
    LiquidAudioModel,
    LiquidAudioServer,
)
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.registry import create
from voice_agent_next.transports import LoopbackTransport

SUPPORTED_PROMPTS = {
    "Perform ASR.",
    "Perform TTS. Use the US male voice.",
    "Respond with interleaved text and audio.",
}
CHUNK = 1920  # float32 samples per audio chunk (80 ms at 24 kHz)


def chunk_line(delta: dict[str, Any], finish: str | None = None) -> bytes:
    obj = {
        "object": "chat.completion.chunk",
        "created": 1790275208,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def audio_samples(n_chunks: int, seed: int = 0) -> np.ndarray:
    t = np.arange(n_chunks * CHUNK) / 24_000
    return (0.25 * np.sin(2 * np.pi * (220 + seed) * t)).astype(np.float32)


class FakeLiquidServer:
    def __init__(self, replies: list[str] | None = None, *, delay: float = 0.0) -> None:
        self.replies = list(replies or [])
        self.delay = delay
        self.context: list[Any] = []
        self.stopped = False
        self.bodies: list[dict[str, Any]] = []
        self.aborted = 0
        self.fail_next: str | None = None

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    async def handle(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        self.bodies.append(body)
        for msg in body["messages"]:
            if msg["role"] not in ("system", "user"):
                err = {
                    "error": {
                        "message": "role must be system or user",
                        "type": "server_error",
                        "code": 400,
                    }
                }
                return httpx.Response(400, json=err)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=self.stream(body)
        )

    async def stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if body.get("reset_context", True):
            self.context, self.stopped = [], False
        for msg in body["messages"]:
            if msg["role"] == "system" and msg["content"] not in SUPPORTED_PROMPTS:
                yield b'data: {"error":{"message":"Unsupported system prompt.","type":"server_error"}}\n\n'
                return
        if self.stopped or self.fail_next:
            message, self.fail_next = self.fail_next or "failed to run prefill", None
            err = {"error": {"message": message, "type": "server_error"}}
            yield b"data: " + json.dumps(err).encode() + b"\n\n"
            return
        self.context.extend(body["messages"])
        if body.get("max_tokens") == 4:  # the provider's warm-up request
            reply = "Hi"
        else:
            reply = self.replies.pop(0) if self.replies else "Hello there, how can I help you?"
        tokens = [t for t in reply.replace(" ", "\x00 ").split("\x00") if t]
        complete = False
        try:
            for i in range(0, len(tokens), 6):
                for token in tokens[i : i + 6]:
                    yield chunk_line({"content": token})
                samples = audio_samples(2, seed=i)
                for k in range(2):
                    data = base64.b64encode(samples[k * CHUNK : (k + 1) * CHUNK].tobytes())
                    audio = {"data": data.decode(), "format": "pcm", "sample_rate": 24000}
                    yield chunk_line({"audio_chunk": audio})
                    await asyncio.sleep(self.delay)
            complete = True  # (clients stop reading at [DONE])
            self.context.append({"role": "assistant", "content": reply})
            yield chunk_line({}, "stop")
            yield b"data: [DONE]\n\n"
        finally:
            if not complete:
                self.aborted += 1
                self.stopped = True


def llm_for(server: FakeLiquidServer, **kw: Any) -> LiquidAudioLLM:
    return LiquidAudioLLM(base_url="http://fake/v1", http_client=server.client(), **kw)


def user_audio(seconds: float = 0.5) -> AudioContent:
    return AudioContent(synth_speech(seconds, 16_000))


async def reply(llm: LiquidAudioLLM, ctx: ChatContext) -> tuple[str, list[AudioFrame]]:
    text: list[str] = []
    frames: list[AudioFrame] = []
    async for chunk in llm.chat(ctx):
        text.append(chunk.delta)
        if chunk.audio:
            frames.append(chunk.audio)
    return "".join(text), frames


def roles(body: dict[str, Any]) -> list[str]:
    return [m["role"] for m in body["messages"]]


# ------------------------------------------------------------------------- protocol


async def test_registry_spec_and_capabilities() -> None:
    llm = create("llm", "liquid-audio/lfm2.5-audio-1.5b")
    assert isinstance(llm, LiquidAudioLLM) and llm.model == "lfm2.5-audio-1.5b"
    caps = llm.capabilities
    assert caps.audio_input and caps.audio_output and caps.audio_sample_rate == 24_000
    assert not caps.tool_calling
    assert llm.base_url == "http://127.0.0.1:8080/v1"
    await llm.aclose()


async def test_stream_shape_audio_is_float32_at_24k_and_text_is_the_transcript() -> None:
    server = FakeLiquidServer(["I can help with that today."])
    llm = llm_for(server)
    metrics: list[LLMMetrics] = []
    llm.on("metrics", metrics.append)
    ctx = ChatContext()
    ctx.add_message("system", "You are a receptionist.")  # instructions: not sendable
    ctx.add_message("user", user_audio(0.7))
    text, frames = await reply(llm, ctx)
    await llm.aclose()

    assert text == "I can help with that today."
    assert all(f.sample_rate == 24_000 and f.channels == 1 for f in frames)
    assert sum(f.duration for f in frames) == pytest.approx(2 * 0.08)
    got = AudioFrame.concat(frames[:2]).to_float32()
    assert np.allclose(got, audio_samples(2, seed=0), atol=1 / 32768 + 1e-6)
    [body] = server.bodies
    assert body["stream"] is True and body["reset_context"] is True
    assert body["messages"][0] == {"role": "system", "content": INTERLEAVED_PROMPT}
    [part] = body["messages"][1]["content"]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    wav = read_wav(base64.b64decode(part["input_audio"]["data"]))
    assert wav.sample_rate == 16_000 and wav.duration == pytest.approx(0.7, abs=0.01)
    [m] = metrics
    assert m.ttft is not None and m.ttfb is not None and m.ttft <= m.ttfb
    assert m.completion_tokens == 6 + 2  # text tokens + audio chunks
    assert llm.resets == 1


async def test_only_new_turns_are_sent_while_the_server_context_matches() -> None:
    server = FakeLiquidServer(["First answer.", "Second answer.", "Third answer."])
    llm = llm_for(server)
    ctx = ChatContext()
    ctx.add_message("user", user_audio())
    first, _ = await reply(llm, ctx)
    ctx.add_message("assistant", first)
    ctx.add_message("user", "And a typed question?")
    ctx.add_message("assistant", "")  # the cascade's placeholder for the coming reply
    second, _ = await reply(llm, ctx)
    ctx.items.pop()
    ctx.add_message("assistant", second)
    ctx.add_message("user", user_audio(0.3))
    await reply(llm, ctx)
    await llm.aclose()

    assert [b["reset_context"] for b in server.bodies] == [True, False, False]
    assert server.bodies[1]["messages"] == [{"role": "user", "content": "And a typed question?"}]
    assert roles(server.bodies[2]) == ["user"]
    assert "input_audio" in json.dumps(server.bodies[2]["messages"])
    # the server holds the whole conversation, its own replies included
    assert [m["role"] for m in server.context] == ["system", "user", "assistant"] * 1 + [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert llm.resets == 1


async def test_a_truncated_reply_resets_and_replays_the_heard_history() -> None:
    server = FakeLiquidServer(["One two three four five six seven eight.", "Okay."])
    llm = llm_for(server)
    ctx = ChatContext()
    ctx.add_message("user", "Count to eight.")
    full, _ = await reply(llm, ctx)
    answer = ctx.add_message("assistant", full)
    answer.content, answer.interrupted = ["One two three"], True  # barge-in: heard part
    ctx.add_message("user", user_audio())
    await reply(llm, ctx)
    await llm.aclose()

    body = server.bodies[1]
    assert body["reset_context"] is True
    assert body["messages"] == [
        {"role": "system", "content": INTERLEAVED_PROMPT},
        {"role": "user", "content": "Count to eight."},
        {"role": "user", "content": "(Earlier you said: One two three)"},
        body["messages"][3],
    ]
    assert body["messages"][3]["content"][0]["type"] == "input_audio"
    assert all(m["role"] != "assistant" for b in server.bodies for m in b["messages"])
    assert llm.resets == 2


async def test_replay_options() -> None:
    server = FakeLiquidServer()
    llm = llm_for(server, assistant_note=None, max_replay_turns=3)
    ctx = ChatContext()
    for i in range(4):
        ctx.add_message("user", f"question {i}")
        ctx.add_message("assistant", f"answer {i}")
    ctx.add_message("user", "last")
    await reply(llm, ctx)
    await llm.aclose()
    # the last 3 messages, starting at a user turn; replies dropped
    assert [m["content"] for m in server.bodies[0]["messages"][1:]] == ["question 3", "last"]
    server2 = FakeLiquidServer()
    llm2 = llm_for(server2, max_replay_turns=4)
    await reply(llm2, ctx)
    await llm2.aclose()
    assert [m["content"] for m in server2.bodies[0]["messages"][1:]] == [
        "question 3",
        "(Earlier you said: answer 3)",
        "last",
    ]


async def test_a_cancelled_reply_is_followed_by_a_reset() -> None:
    server = FakeLiquidServer(["A long answer " * 20, "Short."], delay=0.01)
    llm = llm_for(server)
    ctx = ChatContext()
    ctx.add_message("user", user_audio())
    stream = llm.chat(ctx)
    async for chunk in stream:
        if chunk.audio:
            break  # barge-in: the reply is cancelled mid-way
    await stream.aclose()
    await wait_for(lambda: server.aborted == 1, 2)
    ctx.add_message("assistant", "A long")
    ctx.add_message("user", user_audio())
    text, frames = await reply(llm, ctx)
    await llm.aclose()
    assert text == "Short." and frames
    assert server.bodies[1]["reset_context"] is True  # the stop flag would fail it otherwise


async def test_an_error_before_output_retries_once_from_scratch() -> None:
    server = FakeLiquidServer(["Fine.", "Recovered."])
    llm = llm_for(server)
    ctx = ChatContext()
    ctx.add_message("user", "hello")
    first, _ = await reply(llm, ctx)
    ctx.add_message("assistant", first)
    ctx.add_message("user", "again")
    server.fail_next = "failed to run prefill"  # e.g. the context is full
    text, _ = await reply(llm, ctx)
    assert text == "Recovered."
    assert [b["reset_context"] for b in server.bodies] == [True, False, True]
    # an error on a fresh context is reported
    server.fail_next = "failed to run prefill"
    llm._held = None
    with pytest.raises(ProviderError, match="failed to run prefill"):
        await reply(llm, ctx)
    await llm.aclose()


async def test_http_errors_are_mapped() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "busy"}})

    llm = LiquidAudioLLM(
        base_url="http://fake/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    ctx = ChatContext()
    ctx.add_message("user", "hi")
    with pytest.raises(ProviderError, match="503"):
        await reply(llm, ctx)
    await llm.aclose()


def test_int16_audio_of_the_upstream_server_is_parsed() -> None:
    pcm = (np.arange(480) * 10).astype("<i2")
    event = {
        "choices": [
            {
                "delta": {
                    "audio": {
                        "data": base64.b64encode(pcm.tobytes()).decode(),
                        "format": "pcm",
                        "sample_rate": 24000,
                    }
                },
                "finish_reason": None,
            }
        ]
    }
    [(text, frame, reason)] = list(liquid_audio._parse_event(event))
    assert text == "" and reason is None and frame is not None
    assert np.array_equal(frame.to_numpy(), pcm)


async def test_warmup_primes_the_server_and_forces_a_reset() -> None:
    server = FakeLiquidServer(["Real answer."])
    llm = llm_for(server)
    await llm.warmup()
    ctx = ChatContext()
    ctx.add_message("user", "hello")
    assert (await reply(llm, ctx))[0] == "Real answer."
    await llm.aclose()
    assert server.bodies[0]["max_tokens"] == 4 and server.bodies[0]["reset_context"] is True
    assert server.bodies[1]["reset_context"] is True


def test_leading_silence_is_trimmed() -> None:
    def chunk(level_db: float) -> AudioFrame:
        return AudioFrame.from_numpy(
            np.full(CHUNK, 10 ** (level_db / 20), dtype=np.float32), 24_000
        )

    # LFM2.5-Audio's measured reply start: 0.4-0.9 s at -52...-67 dBFS, then speech
    trim = liquid_audio._LeadingSilence(-45.0, 1.5)
    quiet = [chunk(-57), chunk(-55), chunk(-64), chunk(-66), chunk(-59)]
    out = [f for c in quiet for f in trim.push(c)]
    assert out == []
    speech = chunk(-25)
    assert trim.push(speech) == [quiet[-1], speech]  # 80 ms of lead-in kept
    assert trim.push(quiet[0]) == [quiet[0]]  # later pauses are untouched
    assert trim.trimmed == pytest.approx(4 * 0.08)
    # never more than max_trim: a quiet reply is played as it is
    trim = liquid_audio._LeadingSilence(-45.0, 0.2)
    got = [f for c in quiet for f in trim.push(c)]
    assert got == quiet
    assert liquid_audio._LeadingSilence(None, 1.5).push(quiet[0]) == [quiet[0]]


# -------------------------------------------------------------------------- session


async def test_native_speech_to_speech_session_without_stt_and_tts() -> None:
    server = FakeLiquidServer(["Sure, we open at nine."])
    session = AgentSession(
        llm=llm_for(server),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.0),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("You are a receptionist."), transport)
    await speak(transport, 0.8, 0.6)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 5)
    await session.aclose()
    assert "".join(e.delta for e in rec.of("agent_transcript")) == "Sure, we open at nine."
    assert sum(p.frame.duration for p in transport.played_log) == pytest.approx(0.16, abs=0.05)
    [turn] = rec.turn_metrics()
    assert turn.voice_to_voice is not None
    answers = [
        i for i in session.history.items if isinstance(i, ChatMessage) and i.role == "assistant"
    ]
    assert [a.text for a in answers] == ["Sure, we open at nine."]


# --------------------------------------------------------------- downloads / server


def test_download_model_pins_revision_and_checksums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def fake_download(
        url: str, *, filename: str, subdir: str, sha256: str | None, **kw: Any
    ) -> Any:
        calls.append((url, filename, sha256))
        return tmp_path / filename

    monkeypatch.setattr(liquid_audio, "download", fake_download)
    model = liquid_audio.download_model("q4_0")
    assert model.model.name == "LFM2.5-Audio-1.5B-Q4_0.gguf"
    assert model.tokenizer.name == "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"
    assert len(calls) == 4 and all(sha and len(sha) == 64 for _, _, sha in calls)
    assert all(f"/resolve/{liquid_audio.HF_REVISION}/" in url for url, _, _ in calls)
    assert model.server_args()[:2] == ["-m", str(model.model)]
    with pytest.raises(ConfigurationError):
        liquid_audio.download_model("Q2_K")


@pytest.mark.parametrize(
    ("system", "machine", "zip_name"),
    [
        ("linux", "x86_64", "llama-liquid-audio-ubuntu-x64.zip"),
        ("linux", "aarch64", "llama-liquid-audio-ubuntu-arm64.zip"),
        ("darwin", "arm64", "llama-liquid-audio-macos-arm64.zip"),
        ("win32", "AMD64", None),
        ("darwin", "x86_64", None),
    ],
)
def test_runner_per_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, system: str, machine: str, zip_name: str | None
) -> None:
    monkeypatch.setattr(liquid_audio.sys, "platform", system)
    monkeypatch.setattr(liquid_audio.platform, "machine", lambda: machine)
    urls: list[str] = []

    def fake_archive(url: str, *, sha256: str, subdir: str) -> Any:
        urls.append(url)
        (tmp_path / "llama-liquid-audio-server").write_text("")
        return tmp_path

    monkeypatch.setattr(liquid_audio, "download_archive", fake_archive)
    if zip_name is None:
        assert not liquid_audio.runner_supported()
        with pytest.raises(ConfigurationError, match="base_url"):
            liquid_audio.download_runner()
        return
    assert liquid_audio.runner_supported()
    assert liquid_audio.download_runner() == tmp_path / "llama-liquid-audio-server"
    assert urls[0].endswith(f"/runners/{zip_name}")


FAKE_SERVER = """
import asyncio, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
async def main():
    async def handle(reader, writer):
        writer.close()
    await asyncio.sleep(0.3)  # "loading the model"
    server = await asyncio.start_server(handle, "127.0.0.1", port)
    print("Server ready", flush=True)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""


def fake_model(tmp_path: Any) -> LiquidAudioModel:
    return LiquidAudioModel(*(tmp_path / f"{n}.gguf" for n in ("m", "mm", "mv", "tok")))


async def test_managed_server_starts_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    server = LiquidAudioServer(
        executable=sys.executable,
        model=fake_model(tmp_path),
        threads=2,
        ctx_size=8192,
        startup_timeout=30,
    )
    original = server.command

    def command(exe: Any, model: LiquidAudioModel, port: int) -> list[str]:
        cmd = original(exe, model, port)
        assert cmd[1:3] == ["-m", str(model.model)]
        assert cmd[-4:] == ["-t", "2", "-c", "8192"]
        return [sys.executable, str(script), *cmd[1:]]

    monkeypatch.setattr(server, "command", command)
    async with server:
        assert server.running
        assert server.base_url == f"http://127.0.0.1:{server.port}/v1"
        assert await server.start() == server.base_url  # already running: no-op
    assert not server.running


async def test_managed_server_reports_a_failed_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    server = LiquidAudioServer(
        executable=sys.executable, model=fake_model(tmp_path), startup_timeout=30
    )
    monkeypatch.setattr(
        server,
        "command",
        lambda exe, model, port: [
            sys.executable,
            "-c",
            "print('ERR: model not found'); raise SystemExit(3)",
        ],
    )
    with pytest.raises(ProviderError, match="model not found"):
        await server.start()
    assert not server.running


async def test_serve_true_owns_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    async def start(self: LiquidAudioServer) -> str:
        self.port = 18123
        started.append("start")
        return self.base_url

    async def stop(self: LiquidAudioServer) -> None:
        started.append("stop")

    monkeypatch.setattr(LiquidAudioServer, "start", start)
    monkeypatch.setattr(LiquidAudioServer, "stop", stop)
    llm = LiquidAudioLLM(serve=True, server_options={"threads": 4})
    assert llm.server is not None and llm.server.threads == 4
    assert llm.server.ctx_size == llm.context_size == liquid_audio.MANAGED_CONTEXT_SIZE
    await llm._ensure_server()
    assert llm.base_url == "http://127.0.0.1:18123/v1"
    await llm.aclose()
    assert started == ["start", "stop"]


async def test_a_managed_server_that_died_is_restarted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLiquidServer(["First.", "After the restart."])
    alive = {"up": False}
    starts: list[int] = []

    async def start(self: LiquidAudioServer) -> str:
        self.port = 18124
        alive["up"] = True
        fake.context = []  # a new process: empty context
        starts.append(1)
        return self.base_url

    monkeypatch.setattr(LiquidAudioServer, "start", start)
    monkeypatch.setattr(LiquidAudioServer, "running", property(lambda self: alive["up"]))

    async def handler(request: httpx.Request) -> httpx.Response:
        if not alive["up"]:
            raise httpx.ConnectError("connection refused")
        return await fake.handle(request)

    llm = LiquidAudioLLM(
        server=LiquidAudioServer(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    ctx = ChatContext()
    ctx.add_message("user", "hello")
    first, _ = await reply(llm, ctx)
    ctx.add_message("assistant", first)
    ctx.add_message("user", "still there?")
    alive["up"] = False  # the server exits (out of context, killed...)
    text, _ = await reply(llm, ctx)
    await llm.aclose()
    assert text == "After the restart." and len(starts) == 2
    # the new process knows nothing: the conversation was replayed from scratch
    assert fake.bodies[-1]["reset_context"] is True
    assert [m["content"] for m in fake.bodies[-1]["messages"][1:]] == [
        "hello",
        "(Earlier you said: First.)",
        "still there?",
    ]


async def test_the_context_is_reset_before_it_overflows() -> None:
    server = FakeLiquidServer()
    llm = llm_for(server, context_size=400, max_tokens=100)  # room for ~236 positions
    ctx = ChatContext()
    for i in range(12):
        ctx.add_message("user", f"question number {i}")
        text, _ = await reply(llm, ctx)
        ctx.add_message("assistant", text)
        assert llm._held_tokens <= 400 - 100 - 64 + 100  # prompt budget + one reply
    await llm.aclose()
    resets = [b["reset_context"] for b in server.bodies]
    assert resets[0] and not resets[1] and 2 < sum(resets) < 12
    # a reset replays the recent turns that fit, ending with the new question
    body = next(b for b in server.bodies[1:] if b["reset_context"])
    assert len(body["messages"]) > 2 and body["messages"][-1]["content"].startswith("question")


# ---------------------------------------------------------------------- real server


@pytest.mark.model
async def test_real_server_round_trip() -> None:
    """Against a running server: ``LIQUID_AUDIO_BASE_URL=http://127.0.0.1:8080/v1``."""
    import os

    if not os.environ.get("LIQUID_AUDIO_BASE_URL"):
        pytest.skip("set LIQUID_AUDIO_BASE_URL to a running llama-liquid-audio-server")
    llm = LiquidAudioLLM()
    ctx = ChatContext()
    ctx.add_message("user", "Say hello in one short sentence.")
    text, frames = await reply(llm, ctx)
    await llm.aclose()
    assert text.strip() and sum(f.duration for f in frames) > 0.3
