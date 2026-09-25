"""Semantic end-of-turn (#124): fusion of audio and text detectors, the fused detector,
the ``lm_turn`` / ``llm_turn`` text detectors with fakes, and the cascade's fused path."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_endpointing import endpointing
from tests.test_session import Recorder, speak, wait_for
from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions
from voice_agent_next.chat import ChatContext
from voice_agent_next.metrics import EndpointingMetrics, EOTMetrics
from voice_agent_next.models import models_for_spec
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS, MockTurnDetector
from voice_agent_next.registry import create
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.turn import FusedTurnDetector, fuse_end_of_turn, turn_text

# ------------------------------------------------------------------------ fusion


def test_logit_fusion_is_naive_bayes_with_unit_weights() -> None:
    assert fuse_end_of_turn(0.9, 0.5) == pytest.approx(0.9)  # an undecided text: audio only
    assert fuse_end_of_turn(0.9, 0.1) == pytest.approx(0.5)  # opposite opinions cancel
    assert fuse_end_of_turn(0.8, 0.8) == pytest.approx(16 / 17)  # agreeing opinions add up
    # weights scale each opinion's log-odds, the bias shifts the result
    p = fuse_end_of_turn(0.97, 0.2, audio_weight=0.4, text_weight=0.9, bias=0.1)
    x = 0.4 * math.log(0.97 / 0.03) + 0.9 * math.log(0.2 / 0.8) + 0.1
    assert p == pytest.approx(1 / (1 + math.exp(-x)))
    assert fuse_end_of_turn(1.0, 0.0) == pytest.approx(0.5)  # clamped, no infinities


def test_other_fusion_methods() -> None:
    assert fuse_end_of_turn(0.9, 0.5, method="product") == pytest.approx(0.45)
    assert fuse_end_of_turn(0.9, 0.25, method="product", text_weight=0.5) == pytest.approx(0.45)
    assert fuse_end_of_turn(0.9, 0.3, method="min") == 0.3
    assert fuse_end_of_turn(0.9, 0.3, method="mean") == pytest.approx(0.6)
    assert fuse_end_of_turn(0.9, 0.3, method="mean", audio_weight=3.0) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="unknown fusion"):
        fuse_end_of_turn(0.5, 0.5, method="max")  # type: ignore[arg-type]


def test_a_missing_half_leaves_the_other() -> None:
    assert fuse_end_of_turn(None, 0.3) == 0.3
    assert fuse_end_of_turn(0.8, None) == 0.8
    assert fuse_end_of_turn(None, None) is None


def test_turn_text_reads_the_last_user_message_and_the_agent_turn_before_it() -> None:
    ctx = ChatContext()
    assert turn_text(ctx) == ("", "")
    ctx.add_message("system", "Be brief.")
    ctx.add_message("user", "Hi.")
    ctx.add_message("assistant", " How can I help? ")
    ctx.add_function_call("lookup", "{}")
    ctx.add_message("user", " Where is my order ")
    assert turn_text(ctx) == ("How can I help?", "Where is my order")
    assert turn_text(None) == ("", "")


# -------------------------------------------------------------- fused detector


class FakeAudioDetector(MockTurnDetector):
    """An audio detector returning a fixed probability (``None`` audio is never passed)."""

    provider = "fake_audio"
    modality = "audio"

    def __init__(self, probability: float, delay: float = 0.0) -> None:
        super().__init__(model="fake-audio", probability=probability, delay=delay)
        self.audio_seen: list[float] = []

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        assert audio is not None
        self.audio_seen.append(audio.duration)
        return await super()._predict(audio=audio, chat_ctx=chat_ctx)


class FakeTextDetector(MockTurnDetector):
    """A text detector that records what it read; ``incomplete`` words say "not done"."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        super().__init__(model="fake-text", delay=delay)
        self.read: list[tuple[str, str]] = []
        self.fail = fail

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        self.read.append(turn_text(chat_ctx))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return 0.05 if "table" in self.read[-1][1] else 0.8


def user(text: str, agent: str = "") -> ChatContext:
    ctx = ChatContext()
    if agent:
        ctx.add_message("assistant", agent)
    ctx.add_message("user", text)
    return ctx


async def test_fused_detector_combines_both_halves_and_reports_all_metrics() -> None:
    fused = FusedTurnDetector(
        audio=FakeAudioDetector(0.9), text=FakeTextDetector(), audio_weight=1.0, text_weight=1.0,
        bias=0.0,
    )  # fmt: skip
    assert fused.modality == "audio_text" and fused.model == "fake-audio+fake-text"
    seen: list[EOTMetrics] = []
    fused.on("metrics", seen.append)
    audio = AudioFrame.silence(1.0, 16_000)
    p = await fused.predict_end_of_turn(audio=audio, chat_ctx=user("I'd like to book a table"))
    assert p == pytest.approx(fuse_end_of_turn(0.9, 0.05))  # type: ignore[arg-type]
    assert p < 0.5  # the words overrule a confident audio detector
    assert [m.model for m in seen] == ["fake-audio", "fake-text", "fake-audio+fake-text"]
    assert not seen[-1].end_of_turn
    done = await fused.predict_end_of_turn(audio=audio, chat_ctx=user("That's all, thanks."))
    assert done > 0.9
    # nothing to judge on one side: the other decides; on neither: 1.0 (silence decides)
    assert await fused.predict_end_of_turn(audio=audio) == pytest.approx(0.9)
    assert await fused.predict_end_of_turn(chat_ctx=user("book a table")) == pytest.approx(0.05)
    assert await fused.predict_end_of_turn() == 1.0


async def test_fused_detector_budget_and_failures_fall_back_to_audio() -> None:
    slow = FusedTurnDetector(
        audio=FakeAudioDetector(0.7), text=FakeTextDetector(delay=0.5), text_timeout=0.05
    )
    ctx = user("a table")
    assert (
        await slow.predict_end_of_turn(audio=AudioFrame.silence(0.5, 16_000), chat_ctx=ctx) == 0.7
    )
    broken = FusedTurnDetector(audio=FakeAudioDetector(0.7), text=FakeTextDetector(fail=True))
    assert (
        await broken.predict_end_of_turn(audio=AudioFrame.silence(0.5, 16_000), chat_ctx=ctx) == 0.7
    )


def test_fused_detector_from_the_registry_and_its_validation() -> None:
    fused = create(
        "turn",
        {"provider": "fused", "audio": FakeAudioDetector(0.5), "text": {"provider": "mock"},
         "method": "min", "threshold": 0.6},
    )  # fmt: skip
    assert isinstance(fused, FusedTurnDetector)
    assert fused.method == "min" and fused.threshold == 0.6
    with pytest.raises(ValueError, match="audio turn detector"):
        FusedTurnDetector(audio="mock", text="mock")
    with pytest.raises(ValueError, match="text turn detector"):
        FusedTurnDetector(audio=FakeAudioDetector(0.5), text=FakeAudioDetector(0.5))
    with pytest.raises(ValueError, match="unknown fusion"):
        FusedTurnDetector(audio=FakeAudioDetector(0.5), text="mock", method="max")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weights"):
        FusedTurnDetector(audio=FakeAudioDetector(0.5), text="mock", text_weight=-1.0)


def test_models_for_a_fused_detector_list_both_halves() -> None:
    req = models_for_spec("turn", {"provider": "fused", "audio": "smart_turn", "text": "lm_turn"})
    assert [f"{m.provider}/{m.model}" for m in req.models] == [
        "smart_turn/smart-turn-v3.2-cpu",
        "lm_turn/smollm2-135m",
    ]
    assert len(models_for_spec("turn", "fused").models) == 2  # the defaults


# ------------------------------------------------------------------- lm_turn

VOCAB = ["<|im_start|>", "<|im_end|>", "assistant", "user", "hello", "table", "\n"]


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Whitespace tokenizer over ``VOCAB`` (unknown words -> id 4)."""

    @classmethod
    def from_file(cls, path: str) -> FakeTokenizer:
        return cls()

    def no_padding(self) -> None: ...

    def no_truncation(self) -> None: ...

    def token_to_id(self, token: str) -> int | None:
        return VOCAB.index(token) if token in VOCAB else None

    def encode(self, text: str, add_special_tokens: bool = True) -> FakeEncoding:
        words = text.replace("<|im_end|>", " <|im_end|> ").replace("\n", " ").split()
        return FakeEncoding([VOCAB.index(w) if w in VOCAB else 4 for w in words])


class FakeInput:
    def __init__(self, name: str, shape: list[Any], type: str = "tensor(int64)") -> None:
        self.name, self.shape, self.type = name, shape, type


class FakeSession:
    """A causal LM whose next token is ``<|im_end|>`` with p = 0.8 unless the prompt ends
    with "table" (p = 1e-4)."""

    feeds: list[dict[str, Any]] = []

    def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
        self.path = path

    def get_inputs(self) -> list[FakeInput]:
        return [
            FakeInput("input_ids", ["batch", "seq"]),
            FakeInput("attention_mask", ["batch", "seq"]),
            FakeInput("position_ids", ["batch", "seq"]),
            FakeInput("past_key_values.0.key", ["batch", 3, "past", 64], "tensor(float)"),
            FakeInput("past_key_values.0.value", ["batch", 3, "past", 64], "tensor(float)"),
        ]

    def run(self, outputs: list[str], feed: dict[str, Any]) -> list[Any]:
        FakeSession.feeds.append(feed)
        ids = feed["input_ids"][0]
        p_end = 1e-4 if ids[-1] == VOCAB.index("table") else 0.8
        probs = np.full(len(VOCAB), (1.0 - p_end) / (len(VOCAB) - 1))
        probs[VOCAB.index("<|im_end|>")] = p_end
        logits = np.zeros((1, len(ids), len(VOCAB)), dtype=np.float32)
        logits[0, -1] = np.log(probs)
        return [logits]


@pytest.fixture
def fake_lm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    ort = types.ModuleType("onnxruntime")

    class SessionOptions:
        def add_session_config_entry(self, key: str, value: str) -> None: ...

    ort.SessionOptions = SessionOptions  # type: ignore[attr-defined]
    ort.InferenceSession = FakeSession  # type: ignore[attr-defined]
    ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL=0)  # type: ignore[attr-defined]
    ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL=99)  # type: ignore[attr-defined]
    tok = types.ModuleType("tokenizers")
    tok.Tokenizer = FakeTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tok)
    FakeSession.feeds = []
    model, tokenizer = tmp_path / "lm.onnx", tmp_path / "tokenizer.json"
    model.write_bytes(b"onnx")
    tokenizer.write_text("{}")
    return model, tokenizer


def test_lm_turn_prompt_is_the_open_user_message_after_the_agent_turn() -> None:
    from voice_agent_next.providers.lm_turn import format_prompt

    assert format_prompt("How can I help?", " Book a table ") == (
        "<|im_start|>assistant\nHow can I help?<|im_end|>\n<|im_start|>user\nBook a table"
    )
    assert format_prompt("", "Hi") == "<|im_start|>user\nHi"
    long = format_prompt("x" * 500 + " end.", "Hi", max_agent_chars=10)
    assert long.startswith("<|im_start|>assistant\nxxxxx end.<|im_end|>")


async def test_lm_turn_scores_the_end_of_message_probability(fake_lm: tuple[Path, Path]) -> None:
    from voice_agent_next.providers.lm_turn import LMTurnDetector

    model, tokenizer = fake_lm
    raw = LMTurnDetector(model_path=model, tokenizer_path=tokenizer, max_tokens=8)
    assert raw.model == "lm" and raw.calibration is None
    assert raw.end_probability("hello", "hello table") == pytest.approx(1e-4, rel=1e-3)
    assert raw.end_probability("hello", "hello hello") == pytest.approx(0.8, rel=1e-3)
    feed = FakeSession.feeds[-1]
    n = feed["input_ids"].shape[1]
    assert n <= 8  # cut to the last max_tokens tokens
    assert feed["position_ids"].tolist() == [list(range(n))]
    assert feed["past_key_values.0.key"].shape == (1, 3, 0, 64)  # an empty cache
    assert feed["past_key_values.0.key"].dtype == np.float32
    # calibrated: sigmoid(a * ln(p) + b)
    cal = LMTurnDetector(
        model_path=model, tokenizer_path=tokenizer, calibration=(0.2, 1.0), threshold=0.5
    )
    p = await cal.predict_end_of_turn(chat_ctx=user("hello table", agent="hello"))
    assert p == pytest.approx(1 / (1 + math.exp(-(0.2 * math.log(1e-4) + 1.0))), rel=1e-3)
    assert p < 0.5
    assert await cal.predict_end_of_turn(chat_ctx=user("hello")) > 0.5
    assert await cal.predict_end_of_turn(chat_ctx=ChatContext()) == 1.0  # nothing to judge


def test_lm_turn_models_and_validation(tmp_path: Path) -> None:
    from voice_agent_next.errors import ConfigurationError
    from voice_agent_next.providers.lm_turn import MODELS, LMTurnDetector

    d = create("turn", "lm_turn")
    assert isinstance(d, LMTurnDetector) and d.model == "smollm2-135m"
    assert d.calibration == MODELS["smollm2-135m"].calibration and d.modality == "text"
    assert create("turn", "lm_turn/smollm2-360m").model == "smollm2-360m"
    with pytest.raises(ConfigurationError, match="unknown lm_turn model"):
        LMTurnDetector(model="gpt-5")
    with pytest.raises(ConfigurationError, match="tokenizer_path"):
        LMTurnDetector(model_path=tmp_path / "x.onnx")
    with pytest.raises(ConfigurationError, match="not found"):
        LMTurnDetector(model_path=tmp_path / "x.onnx", tokenizer_path=tmp_path / "t")._files()


# ------------------------------------------------------------------ llm_turn


async def test_llm_turn_reads_a_yes_or_no_answer() -> None:
    from voice_agent_next.providers.llm_turn import DEFAULT_PROMPT, LLMTurnDetector

    answers = iter(["Yes.", "no", "Maybe"])
    llm = MockLLM(responses=lambda ctx: next(answers))
    d = LLMTurnDetector(llm=llm, confidence=0.8, fallback=0.6)
    assert d.model == "mock/mock-llm"
    ctx = user("Where is my order?", agent="What can I do for you?")
    assert await d.predict_end_of_turn(chat_ctx=ctx) == pytest.approx(0.8)
    assert await d.predict_end_of_turn(chat_ctx=ctx) == pytest.approx(0.2)
    assert await d.predict_end_of_turn(chat_ctx=ctx) == pytest.approx(0.6)  # neither
    sent = llm.requests[0].messages()
    assert sent[0].role == "system" and sent[0].text == DEFAULT_PROMPT
    assert sent[1].text == "Assistant: What can I do for you?\nUser: Where is my order?"
    assert await d.predict_end_of_turn(chat_ctx=ChatContext()) == 1.0
    await d.aclose()  # a shared LLM instance is left open


async def test_llm_turn_latency_budget() -> None:
    from voice_agent_next.providers.llm_turn import LLMTurnDetector

    d = LLMTurnDetector(llm=MockLLM(responses=lambda ctx: "no", ttft=0.5), timeout=0.05)
    assert await d.predict_end_of_turn(chat_ctx=user("Hi")) == 1.0  # fallback: no objection


def test_yes_probability_from_logprobs() -> None:
    from voice_agent_next.providers.llm_turn import yes_probability

    top = [("Yes", math.log(0.6)), (" no", math.log(0.2)), ("The", math.log(0.2))]
    assert yes_probability(top) == pytest.approx(0.75)
    assert yes_probability([("The", 0.0)]) is None


async def test_llm_turn_uses_logprobs_of_openai_compatible_llms() -> None:
    pytest.importorskip("openai")
    import httpx

    from voice_agent_next.providers.llm_turn import LLMTurnDetector
    from voice_agent_next.providers.ollama import OllamaLLM

    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        top = [
            {"token": "No", "logprob": math.log(0.3), "bytes": None},
            {"token": "Yes", "logprob": math.log(0.6), "bytes": None},
            {"token": "I", "logprob": math.log(0.1), "bytes": None},
        ]
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0, "finish_reason": "length",
                "message": {"role": "assistant", "content": "Yes"},
                "logprobs": {"content": [{"token": "Yes", "logprob": math.log(0.6),
                                          "bytes": None, "top_logprobs": top}]},
            }],
        })  # fmt: skip

    llm = OllamaLLM(
        model="lfm", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    d = LLMTurnDetector(llm=llm)
    p = await d.predict_end_of_turn(chat_ctx=user("Where is my order?"))
    assert p == pytest.approx(0.6 / 0.9)
    [body] = bodies
    assert body["stream"] is False and body["logprobs"] is True and body["top_logprobs"] == 10
    assert body["max_tokens"] == 1 and body["temperature"] == 0.0
    await llm.aclose()


# ------------------------------------------------------------------- cascade


async def fused_turn(words: str, text: FakeTextDetector) -> tuple[Any, EndpointingMetrics]:
    """One user turn through a cascade whose fused detector hears "done" (0.95) and reads
    ``words``; returns the turn's metrics and its endpointing decision."""
    fused = FusedTurnDetector(audio=FakeAudioDetector(0.95), text=text)
    session = AgentSession(
        stt=MockSTT(transcripts=[words], latency=0.01),
        llm=MockLLM(responses=lambda ctx: "Okay."),
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(),
        turn_detector=fused,
        cascade_options=CascadeOptions(max_endpointing_delay=1.2),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.3)
    await wait_for(lambda: bool(rec.turn_metrics()), 6)
    await wait_for(lambda: bool(endpointing(rec)), 3)  # after false_commit_window
    await session.aclose()
    [turn] = rec.turn_metrics()
    [m] = endpointing(rec)
    assert fused.audio.audio_seen  # type: ignore[attr-defined]  # it heard the user's turn
    assert text.read[-1][1] == words  # the text half read the final transcript
    assert m.audio_probability == pytest.approx(0.95)
    assert m.probability == pytest.approx(fused.fuse(m.audio_probability, m.text_probability))
    return turn, m


async def test_cascade_waits_when_text_says_incomplete_and_audio_says_complete() -> None:
    turn, m = await fused_turn("I would like to book a table", FakeTextDetector())
    assert m.text_probability == pytest.approx(0.05)
    assert m.probability is not None and m.probability < 0.5
    assert m.delay == pytest.approx(1.2)  # the ceiling: the user is probably not done
    assert turn.end_of_turn_delay == pytest.approx(1.2, abs=0.2)


async def test_cascade_commits_at_the_floor_when_both_say_complete() -> None:
    turn, m = await fused_turn("That is all, thank you.", FakeTextDetector())
    assert m.text_probability == pytest.approx(0.8)
    assert m.probability is not None and m.probability > 0.5
    assert m.delay == pytest.approx(0.4)
    assert turn.end_of_turn_delay == pytest.approx(0.4, abs=0.15)


async def test_cascade_uses_the_audio_verdict_when_the_text_is_late() -> None:
    text = FakeTextDetector(delay=1.0)
    fused = FusedTurnDetector(audio=FakeAudioDetector(0.95), text=text, text_timeout=0.1)
    session = AgentSession(
        stt=MockSTT(transcripts=["book a table"], latency=0.01),
        llm=MockLLM(responses=lambda ctx: "Okay."),
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(),
        turn_detector=fused,
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.3)
    await wait_for(lambda: bool(endpointing(rec)), 6)
    await session.aclose()
    [m] = endpointing(rec)
    assert m.text_probability is None and m.probability == pytest.approx(0.95)
    assert m.delay == pytest.approx(0.4)


async def test_lm_turn_calibrates_unpunctuated_transcripts_separately(
    fake_lm: tuple[Path, Path],
) -> None:
    from voice_agent_next.providers.lm_turn import LMTurnDetector, is_punctuated, unpunctuated

    assert is_punctuated("Where is my order?") and not is_punctuated("where is my order")
    assert unpunctuated("Where's my ORDER") == "where's my order"
    model, tokenizer = fake_lm
    d = LMTurnDetector(
        model_path=model, tokenizer_path=tokenizer, calibration=(0.2, 1.0),
        calibration_unpunctuated=(0.1, 0.0),
    )  # fmt: skip

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    # an STT without punctuation (sherpa-onnx NeMo): lowercased, its own calibration
    p = await d.predict_end_of_turn(chat_ctx=user("Hello TABLE"))
    assert p == pytest.approx(sigmoid(0.1 * math.log(1e-4)), rel=1e-3)
    p = await d.predict_end_of_turn(chat_ctx=user("hello, table"))
    assert p == pytest.approx(sigmoid(0.2 * math.log(1e-4) + 1.0), rel=1e-3)
    default = LMTurnDetector(model_path=model, tokenizer_path=tokenizer, calibration=(0.2, 1.0))
    assert default.calibration_unpunctuated == (0.2, 1.0)
