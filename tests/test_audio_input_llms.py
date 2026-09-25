"""Audio-input LLMs for half-cascades: host profiles (OpenAI gpt-audio, DashScope
Qwen-Omni, vLLM, vLLM-Omni, llama.cpp), the known-model table, audio encoding per host,
and the user's transcript for the history (``CascadeOptions.input_transcriber``).

Every host is exercised through the ``openai`` SDK against an ``httpx.MockTransport``
server replaying Chat Completions SSE streams (no network)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

pytest.importorskip("openai")

import httpx

from tests.test_openai_llm import (
    FakeServer,
    SSEStream,
    chunk,
    collect_metrics,
    make_llm,
    usage_chunk,
    wait_for,
)
from tests.test_session import Recorder, speak
from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions, ChatContext
from voice_agent_next.chat import AudioContent, ChatMessage
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.llm import LLMCapabilities
from voice_agent_next.providers.dashscope import DashScopeLLM, dashscope_base_url
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.llamacpp import LlamaCppLLM
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS
from voice_agent_next.providers.openai._format import (
    AudioInputFormat,
    audio_content_part,
    to_chat_messages,
)
from voice_agent_next.providers.openai._models import is_audio_input_model
from voice_agent_next.providers.openai.llm import TRANSCRIBE_PROMPT, OpenAILLM
from voice_agent_next.providers.vllm import VllmLLM
from voice_agent_next.providers.vllm_omni import VllmOmniLLM
from voice_agent_next.registry import create, get_provider
from voice_agent_next.transports import LoopbackTransport


@pytest.fixture(autouse=True)
def _no_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DASHSCOPE_BASE_URL", "DASHSCOPE_WORKSPACE_ID", "DASHSCOPE_REGION",
                "DASHSCOPE_API_KEY", "VLLM_BASE_URL", "VLLM_OMNI_BASE_URL",
                "LLAMACPP_BASE_URL", "OPENAI_BASE_URL"):  # fmt: skip
        monkeypatch.delenv(var, raising=False)


def tone(seconds: float, rate: int) -> AudioFrame:
    t = np.arange(int(seconds * rate)) / rate
    return AudioFrame((np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2").tobytes(), rate, 1)


def decode_wav(b64: str) -> tuple[int, int]:
    """(sample rate, frames) of a base64 WAV."""
    with wave.open(io.BytesIO(base64.b64decode(b64))) as w:
        return w.getframerate(), w.getnframes()


def audio_parts(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        p["input_audio"]
        for m in body["messages"]
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if p.get("type") == "input_audio"
    ]


# ------------------------------------------------------------------------ encoding
def test_audio_input_format_encodings() -> None:
    frame = tone(0.5, 24_000)
    wav = AudioInputFormat().encode(frame)
    assert wav["format"] == "wav" and decode_wav(wav["data"]) == (24_000, 12_000)
    assert audio_content_part(frame)["input_audio"] == wav  # the default is unchanged

    at16 = AudioInputFormat("wav", 16_000).encode(frame)
    rate, n = decode_wav(at16["data"])
    assert rate == 16_000 and abs(n - 8_000) <= 2

    url = AudioInputFormat("wav", 16_000, data_url=True).encode(frame)
    assert url["data"].startswith("data:;base64,")
    assert decode_wav(url["data"].removeprefix("data:;base64,"))[0] == 16_000

    pcm = AudioInputFormat("pcm16", 16_000).encode(frame)
    raw = base64.b64decode(pcm["data"])
    assert pcm["format"] == "pcm16" and not raw.startswith(b"RIFF")
    assert abs(len(raw) // 2 - 8_000) <= 2

    stereo = AudioFrame(np.repeat(np.frombuffer(frame.data, "<i2"), 2).tobytes(), 24_000, 2)
    assert decode_wav(AudioInputFormat().encode(stereo)["data"]) == (24_000, 12_000)  # mono


def conversation() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("system", "Be brief.")
    ctx.add_message("user", AudioContent(tone(0.2, 16_000), "first question"))
    ctx.add_message("assistant", "First answer.")
    ctx.add_message("user", AudioContent(tone(0.2, 16_000)))  # never transcribed
    ctx.add_message("assistant", "Second answer.")
    ctx.add_message("user", AudioContent(tone(0.2, 16_000), "third question"))
    return ctx


def test_audio_history_sends_older_turns_as_their_transcripts() -> None:
    ctx = conversation()
    every = to_chat_messages(ctx, audio_input=True)
    assert [type(m["content"]).__name__ for m in every] == ["str", "list", "str", "list",
                                                           "str", "list"]  # fmt: skip
    last = to_chat_messages(ctx, audio_input=True, audio_history=1)
    # the first turn becomes text; the second has no transcript and stays audio
    assert last[1] == {"role": "user", "content": "first question"}
    assert last[3]["content"][0]["type"] == "input_audio"
    assert last[5]["content"][0]["type"] == "input_audio"
    none = to_chat_messages(ctx, audio_input=True, audio_history=0)
    assert none[5] == {"role": "user", "content": "third question"}
    assert none[3]["content"][0]["type"] == "input_audio"
    # a text model gets the transcripts and skips the audio it cannot hear
    text = to_chat_messages(ctx, audio_input=False)
    assert [m["content"] for m in text if m["role"] == "user"] == ["first question",
                                                                   "third question"]  # fmt: skip


# ------------------------------------------------------------------ known models
@pytest.mark.parametrize(
    "model",
    [
        "gpt-audio", "gpt-audio-mini", "gpt-4o-audio-preview", "gpt-4o-mini-audio-preview",
        "Qwen/Qwen2-Audio-7B-Instruct", "Qwen/Qwen2.5-Omni-7B", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "qwen3.5-omni-flash", "qwen-omni-turbo", "qwen3.8-omni-flash",
        "fixie-ai/ultravox-v0_5-llama-3_2-1b", "ultravox-v0_5-llama-3_2-1b-Q4_K_M.gguf",
        "mistralai/Voxtral-Mini-3B-2507", "ggml-org/Voxtral-Mini-3B-2507-GGUF",
        "google/gemma-3n-E4B-it", "gemma-4-E2B-it-Q4_K_M.gguf", "microsoft/Phi-4-multimodal-instruct",
        "openbmb/MiniCPM-o-4_5", "ibm-granite/granite-speech-3.3-8b", "moonshotai/Kimi-Audio-7B-Instruct",
        "LFM2.5-Audio-1.5B-Q4_0.gguf", "gemini-2.5-flash",
    ],
)  # fmt: skip
def test_known_audio_input_models(model: str) -> None:
    assert is_audio_input_model(model)


@pytest.mark.parametrize(
    "model",
    [None, "", "gpt-4.1-mini", "Qwen/Qwen3-8B", "llama-3.3-70b", "gemma-3-12b-it",
     "gemma-4-31b-it", "mistralai/Voxtral-Mini-4B-Realtime-2602", "gpt-realtime"],
)  # fmt: skip
def test_text_models_are_not_audio_input(model: str | None) -> None:
    assert not is_audio_input_model(model)


def test_audio_input_option_and_table() -> None:
    server = FakeServer([])
    assert make_llm(server, model="gpt-4o-audio-preview").capabilities.audio_input
    assert not make_llm(server, model="gpt-4.1-mini").capabilities.audio_input
    # explicit option wins over the table, also over given capabilities
    assert make_llm(server, model="gpt-4.1-mini", audio_input=True).capabilities.audio_input
    assert not make_llm(server, model="qwen3.5-omni-flash", base_url="http://h/v1",
                        audio_input=False).capabilities.audio_input  # fmt: skip
    caps = make_llm(server, capabilities=LLMCapabilities(tool_calling=False), audio_input=True)
    assert caps.capabilities.audio_input and not caps.capabilities.tool_calling
    # a model discovered from the server is unknown at construction: explicit option
    assert not make_llm(server, cls=LlamaCppLLM).capabilities.audio_input
    assert make_llm(server, cls=LlamaCppLLM, audio_input=True).capabilities.audio_input
    assert make_llm(
        server, cls=VllmLLM, model="Qwen/Qwen2-Audio-7B-Instruct"
    ).capabilities.audio_input
    assert make_llm(server, cls=VllmOmniLLM).capabilities.audio_input  # on for the host
    # the audio format can be given as a mapping (YAML config)
    llm = make_llm(server, cls=LlamaCppLLM, audio_format={"format": "pcm16", "sample_rate": 8000})
    assert llm.audio_format == AudioInputFormat("pcm16", 8000)


# ------------------------------------------------------------------ host profiles
def test_new_hosts_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = get_provider("llm", "dashscope")
    assert spec.factory is DashScopeLLM and spec.env == ("DASHSCOPE_API_KEY",)
    assert spec.default_model == "qwen3.5-omni-flash" and not spec.local
    omni = get_provider("llm", "vllm_omni")
    assert omni.factory is VllmOmniLLM and omni.local and omni.env == ()
    with pytest.raises(ConfigurationError):
        create("llm", "dashscope")  # no key
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ds")
    llm = create("llm", "dashscope")
    assert llm.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert llm.capabilities.audio_input and not llm.capabilities.audio_output
    assert create("llm", "vllm-omni").base_url == "http://127.0.0.1:8091/v1"
    monkeypatch.setenv("VLLM_OMNI_BASE_URL", "http://gpu:9000/v1")
    assert create("llm", "vllm_omni").base_url == "http://gpu:9000/v1"


def test_dashscope_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        dashscope_base_url(region="cn-beijing")
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert dashscope_base_url("ws123", "ap-southeast-1") == (
        "https://ws123.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    )
    with pytest.raises(ConfigurationError):
        dashscope_base_url("bad/host")
    with pytest.raises(ConfigurationError):
        dashscope_base_url(region="eu-central-9")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws9")
    monkeypatch.setenv("DASHSCOPE_REGION", "cn-beijing")
    assert dashscope_base_url() == "https://ws9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://proxy/v1")
    assert DashScopeLLM(api_key="k").base_url == "https://proxy/v1"


REPLY = [chunk({"role": "assistant", "content": "I heard"}), chunk({"content": " you."}),
         chunk({}, finish="stop"), usage_chunk(80, 3)]  # fmt: skip


@dataclass
class Host:
    cls: type[OpenAILLM]
    kwargs: dict[str, Any]
    rate: int | None  # sample rate of the WAV sent (None: the input rate)
    data_url: bool = False
    modalities: list[str] | None = None
    audio: dict[str, Any] | None = None
    merged_system: bool = True


HOSTS: dict[str, Host] = {
    "openai-gpt-audio": Host(OpenAILLM, {"model": "gpt-audio-mini"}, None,
                             modalities=["text", "audio"],
                             audio={"voice": "alloy", "format": "pcm16"}, merged_system=False),
    "dashscope": Host(DashScopeLLM, {}, 16_000, data_url=True),
    "dashscope-voice": Host(DashScopeLLM, {"voice": "Tina"}, 16_000, data_url=True,
                            modalities=["text", "audio"], audio={"voice": "Tina", "format": "wav"}),
    "vllm": Host(VllmLLM, {"model": "Qwen/Qwen2.5-Omni-7B"}, 16_000),
    "vllm-omni": Host(VllmOmniLLM, {"model": "Qwen/Qwen3-Omni-30B-A3B-Instruct"}, 16_000,
                      modalities=["text"]),
    "llamacpp": Host(LlamaCppLLM, {"audio_input": True}, 16_000),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(HOSTS))
async def test_request_shape_per_host(name: str) -> None:
    host = HOSTS[name]
    server = FakeServer([SSEStream(REPLY)], models=["served-model"])
    llm = make_llm(server, cls=host.cls, **host.kwargs)
    ctx = ChatContext()
    ctx.add_message("system", "Be brief.")
    ctx.add_message("user", AudioContent(tone(0.5, 24_000)))
    ctx.add_message("system", "Answer in English.")  # a per-response instruction
    chunks = [c async for c in llm.chat(ctx)]
    assert "".join(c.delta for c in chunks) == "I heard you."
    [body] = server.chat_bodies
    assert body["stream"] is True  # DashScope's omni models are only served streamed
    [part] = audio_parts(body)
    assert part["format"] == "wav"
    assert part["data"].startswith("data:;base64,") is host.data_url
    rate, frames = decode_wav(part["data"].removeprefix("data:;base64,"))
    assert rate == (host.rate or 24_000) and abs(frames - rate // 2) <= 2
    assert body.get("modalities") == host.modalities
    assert body.get("audio") == host.audio
    roles = [m["role"] for m in body["messages"]]
    assert roles == (["system", "user"] if host.merged_system else ["system", "user", "system"])


async def test_llamacpp_reports_the_servers_audio_support(caplog: pytest.LogCaptureFixture) -> None:
    props: dict[str, Any] = {"modalities": {"vision": False, "audio": False}}
    seen: list[str] = []

    class PropsServer(FakeServer):
        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/props":
                seen.append(str(request.url))
                return httpx.Response(200, json=props)
            return super().handler(request)

    server = PropsServer([], models=["model.gguf"])
    llm = make_llm(server, cls=LlamaCppLLM, audio_input=True)
    with caplog.at_level(logging.INFO, logger="voice_agent_next"):
        await llm.warmup()
    assert seen == ["http://127.0.0.1:8080/props"]
    assert "does not accept audio" in caplog.text
    props["modalities"]["audio"] = True
    assert await llm.check_audio_support() is True
    props.clear()
    assert await llm.check_audio_support() is None  # an older server: nothing to compare


# ------------------------------------------------------------------- transcribe()
async def test_transcribe_asks_the_model_for_text_only() -> None:
    stream = [chunk({"role": "assistant", "content": "  What's the"}), chunk({"content": " time?"}),
              chunk({}, finish="stop")]  # fmt: skip
    server = FakeServer([SSEStream(stream)])
    llm = make_llm(server, model="gpt-audio", voice="marin")
    metrics = collect_metrics(llm)
    assert await llm.transcribe(tone(0.3, 16_000)) == "What's the time?"
    [body] = server.chat_bodies
    assert body["messages"][0] == {"role": "system", "content": TRANSCRIBE_PROMPT}
    assert len(audio_parts(body)) == 1 and body["temperature"] == 0.0
    assert body["modalities"] == ["text"] and "audio" not in body  # no speech for this
    assert body["stream"] is True and "tools" not in body
    assert metrics == []  # not a reply: kept out of the LLM metrics
    with pytest.raises(ConfigurationError):
        await make_llm(FakeServer([]), model="gpt-4.1-mini").transcribe(tone(0.1, 16_000))


async def test_transcription_prompt_per_model() -> None:
    stream = [chunk({"content": "Hi."}), chunk({}, finish="stop")]
    server = FakeServer([SSEStream(stream)], models=["LFM2.5-Audio-1.5B-Q4_0.gguf"])
    llm = make_llm(server, cls=LlamaCppLLM, audio_input=True)  # the model is discovered
    assert await llm.transcribe(tone(0.2, 16_000)) == "Hi."
    [body] = server.chat_bodies
    # LFM2.5-Audio answers the audio under any other prompt: its own ASR prompt is used
    assert body["messages"][0] == {"role": "system", "content": "Perform ASR."}
    assert body["messages"][1]["content"][1] == {"type": "text", "text": "Transcribe this audio."}
    await llm.transcribe(tone(0.2, 16_000), prompt="Custom.")
    assert server.chat_bodies[-1]["messages"][0]["content"] == "Custom."


# ------------------------------------------------------------- in the cascade
@dataclass
class RoutingServer(FakeServer):
    """Answers transcription requests (``TRANSCRIBE_PROMPT``) and replies separately."""

    transcripts: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    transcribe_delay: float = 0.0
    fail_transcription: bool = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            if body["messages"][0]["content"] == TRANSCRIBE_PROMPT:
                self.requests.append(request)
                if self.fail_transcription:
                    return httpx.Response(500, json={"error": {"message": "boom"}})
                text = self.transcripts.pop(0)
                stream = SSEStream([chunk({"content": text}), chunk({}, finish="stop")],
                                   delay=self.transcribe_delay)  # fmt: skip
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      stream=stream)  # fmt: skip
            answer = self.answers.pop(0)
            self.replies = [SSEStream([chunk({"content": answer}), chunk({}, finish="stop")])]
        return super().handler(request)

    def bodies(self, *, transcription: bool) -> list[dict[str, Any]]:
        return [b for b in self.chat_bodies
                if (b["messages"][0]["content"] == TRANSCRIBE_PROMPT) is transcription]  # fmt: skip


def half_cascade(llm: Any, **options: Any) -> AgentSession:
    return AgentSession(llm=llm, tts=MockTTS(), vad=EnergyVAD(),
                        cascade_options=CascadeOptions(min_endpointing_delay=0.0, **options))  # fmt: skip


def agent_text(rec: Recorder) -> str:
    return "".join(e.delta for e in rec.of("agent_transcript"))


@pytest.mark.parametrize(
    "name", sorted(h for h in HOSTS if HOSTS[h].modalities != ["text", "audio"])
)
async def test_half_cascade_per_host(name: str) -> None:
    """``stt=None``: the user's turn goes to each host as audio; the TTS speaks the reply."""
    host = HOSTS[name]
    server = RoutingServer(answers=["Hello there."], models=["served-model"])
    llm = make_llm(server, cls=host.cls, **host.kwargs)
    session = half_cascade(llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("You are a receptionist."), transport)
    await speak(transport, 0.7, 0.6)
    await wait_for(lambda: "Hello there." in agent_text(rec))
    await session.aclose()
    [body] = server.bodies(transcription=False)
    [part] = audio_parts(body)
    rate, frames = decode_wav(part["data"].removeprefix("data:;base64,"))
    assert rate == 16_000 and frames / rate > 0.6  # the VAD's 16 kHz turn, prefix included
    assert body["messages"][0]["content"].startswith("You are a receptionist.")
    # without a transcriber, the history keeps an empty user turn
    user = [m for m in session.history.items if getattr(m, "role", None) == "user"]
    assert len(user) == 1 and user[0].text == ""


async def test_llm_transcripts_fill_the_history_and_shrink_later_requests() -> None:
    server = RoutingServer(
        transcripts=["What's the weather?", "And tomorrow?"],
        answers=["Sunny.", "Rainy."],
        transcribe_delay=0.05,
    )
    llm = make_llm(server, cls=VllmOmniLLM, model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                   audio_history=1)  # fmt: skip
    session = half_cascade(llm, input_transcriber="llm")
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.7, 0.6)
    await wait_for(lambda: "Sunny." in agent_text(rec)
                   and len([e for e in rec.of("user_transcript") if e.text]) == 1)  # fmt: skip
    await asyncio.sleep(0.3)  # the agent finishes speaking
    await speak(transport, 0.7, 0.6)
    await wait_for(lambda: "Rainy." in agent_text(rec)
                   and len([e for e in rec.of("user_transcript") if e.text]) == 2)  # fmt: skip
    await session.aclose()

    finals = [e for e in rec.of("user_transcript") if e.is_final]
    assert [e.text for e in finals] == ["What's the weather?", "And tomorrow?"]
    users = [m for m in session.history.items if isinstance(m, ChatMessage) and m.role == "user"]
    assert [m.text for m in users] == ["What's the weather?", "And tomorrow?"]
    assert not any(m.metadata.get("transcript_pending") for m in users)
    # the reply did not wait for the transcript; the second request sends turn 1 as text
    first, second = server.bodies(transcription=False)
    assert len(audio_parts(first)) == 1 and len(audio_parts(second)) == 1
    assert {"role": "user", "content": "What's the weather?"} in second["messages"]
    assert len(server.bodies(transcription=True)) == 2


async def test_stt_transcriber_and_failures() -> None:
    class AudioLLM(MockLLM):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.capabilities = LLMCapabilities(audio_input=True)

    stt = MockSTT(transcripts=["book a table"])
    llm = AudioLLM(responses=["For how many?"])
    session = half_cascade(llm, input_transcriber=stt)
    engine = session.engine
    assert isinstance(engine, CascadeEngine) and engine.input_transcriber is stt
    assert stt in engine.components and engine.stt is None  # still a half-cascade
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.7, 0.6)
    await wait_for(lambda: any(e.text == "book a table" for e in rec.of("user_transcript")))
    await wait_for(lambda: "For how many?" in agent_text(rec))
    await session.aclose()
    user_msg = llm.requests[0].last_message("user")
    assert user_msg is not None and isinstance(user_msg.content[0], AudioContent)
    assert user_msg.content[0].transcript == "book a table"  # filled in after the commit

    # a failing transcription resolves the pending user turn with an empty transcript
    server = RoutingServer(answers=["Hi."], fail_transcription=True)
    session = half_cascade(make_llm(server, cls=LlamaCppLLM, audio_input=True),
                           input_transcriber="llm")  # fmt: skip
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.7, 0.6)
    await wait_for(lambda: "Hi." in agent_text(rec) and bool(rec.of("user_transcript")))
    await session.aclose()
    assert [e.text for e in rec.of("user_transcript")] == [""]


def test_input_transcriber_configuration() -> None:
    class AudioLLM(MockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.capabilities = LLMCapabilities(audio_input=True)

    with pytest.raises(ConfigurationError, match="cannot transcribe"):
        CascadeEngine(llm=AudioLLM(), tts=MockTTS(), vad=EnergyVAD(),
                      options=CascadeOptions(input_transcriber="llm"))  # fmt: skip
    engine = CascadeEngine(llm=AudioLLM(), tts=MockTTS(), vad=EnergyVAD(),
                           options=CascadeOptions(input_transcriber="mock"))  # fmt: skip
    assert isinstance(engine.input_transcriber, MockSTT)
    # with an STT the transcriber is not needed (and not created)
    engine = CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), vad=EnergyVAD(),
                           options=CascadeOptions(input_transcriber="mock"))  # fmt: skip
    assert engine.input_transcriber is None and not engine.transcribe_with_llm
