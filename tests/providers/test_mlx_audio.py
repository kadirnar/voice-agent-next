"""mlx-audio TTS (``mlx_audio``) against a fake ``mlx_audio`` module (runs on every OS)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from voice_agent_next import create
from voice_agent_next import models as model_catalog
from voice_agent_next.errors import ConfigurationError, MissingDependencyError
from voice_agent_next.metrics import TTSMetrics
from voice_agent_next.providers import _mlx, mlx_audio
from voice_agent_next.providers.mlx_audio import MLXAudioTTS, kokoro_lang_code
from voice_agent_next.registry import get_provider

from .mlx_fakes import MLXFakes


@pytest.fixture
def g2p(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Which of Kokoro's G2P packages are "installed"."""
    present = {"misaki": True, "en_core_web_sm": True, "pip": True}
    monkeypatch.setattr(mlx_audio, "is_installed", lambda module: present.get(module, False))
    return present


def test_registry_and_catalog() -> None:
    spec = get_provider("tts", "mlx-audio")
    assert spec.factory is MLXAudioTTS and spec.default_model == "kokoro"
    assert spec.platforms == ("darwin",) and spec.extra == "mlx" and spec.local
    assert set(spec.models) == {"kokoro", "pocket-tts"}
    kokoro = model_catalog.get_model("mlx-audio/kokoro")
    assert kokoro.files[0].location == "mlx-community/Kokoro-82M-bf16"
    assert "*.safetensors" in kokoro.files[0].patterns and kokoro.license == "Apache-2.0"
    assert model_catalog.get_model("mlx-audio/pocket-tts").size < 300e6


def test_missing_mlx_off_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mlx, "_importable", lambda module: False)
    monkeypatch.setattr(_mlx, "is_apple_silicon", lambda: False)
    with pytest.raises(MissingDependencyError, match="Apple silicon"):
        create("tts", "mlx_audio")


def test_kokoro_language_from_voice() -> None:
    assert kokoro_lang_code("af_heart") == "a"
    assert kokoro_lang_code("bf_emma") == "b"
    assert kokoro_lang_code("jf_alpha") == "j"
    assert kokoro_lang_code("xx_unknown") == "a"


@pytest.mark.usefixtures("mlx_fakes")
def test_option_validation() -> None:
    with pytest.raises(ConfigurationError):
        create("tts", "mlx_audio", speed=0)
    with pytest.raises(ConfigurationError):
        create("tts", "mlx_audio", streaming_interval=0)


@pytest.mark.usefixtures("g2p")
async def test_kokoro_synthesizes(mlx_fakes: MLXFakes) -> None:
    tts = create("tts", "mlx_audio/kokoro", voice="bf_emma", speed=1.1)
    assert (tts.sample_rate, tts.voice) == (24_000, "bf_emma")
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    await tts.warmup()
    assert mlx_fakes.snapshots[0][0] == "mlx-community/Kokoro-82M-bf16"
    assert mlx_fakes.tts_loads == [str(Path("/hf/mlx-community/Kokoro-82M-bf16"))]
    # named after the repository (the architecture for configs without model_type)
    assert mlx_fakes.tts_load_kwargs == [{"model_name_parts": ["kokoro", "82m", "bf16"]}]
    assert mlx_fakes.tts_calls[0]["text"] == "Hello."  # warm-up

    items = [item async for item in tts.synthesize("Good morning! How are you?")]
    audio = [i.frame for i in items if i.frame]
    assert len(audio) == 3 and all(f.sample_rate == 24_000 for f in audio)
    assert sum(f.duration for f in audio) == pytest.approx(0.6)
    assert items[-1].is_final and items[0].text == "Good morning! How are you?"
    call = mlx_fakes.tts_calls[-1]
    assert call["voice"] == "bf_emma" and call["lang_code"] == "b" and call["speed"] == 1.1
    assert call["stream"] is True and call["verbose"] is False
    assert metrics[-1].ttfb is not None and metrics[-1].error is None
    assert mlx_fakes.threads == {"mlx-test_0"}


async def test_pocket_tts_defaults(mlx_fakes: MLXFakes) -> None:
    tts = create("tts", "mlx_audio/pocket-tts", streaming_interval=0.25,
                 generate_options={"temperature": 0.5})  # fmt: skip
    assert tts.voice == "alba"
    await tts.synthesize("Hi there.").collect()
    call = mlx_fakes.tts_calls[-1]
    assert "lang_code" not in call  # Kokoro only
    assert call["streaming_interval"] == 0.25 and call["temperature"] == 0.5
    assert call["voice"] == "alba"
    assert mlx_fakes.tts_loads == [str(Path("/hf/mlx-community/pocket-tts"))]


async def test_unknown_model_is_resampled_to_the_declared_rate(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.tts_sample_rate = 16_000
    tts = create("tts", "mlx_audio/someone/new-tts", sample_rate=24_000, voice="v1")
    assert tts.sample_rate == 24_000
    audio = await tts.synthesize("Resample me, please.").collect()
    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(0.6, abs=0.02)
    assert tts.model_sample_rate == 16_000
    assert mlx_fakes.snapshots[-1][0] == "someone/new-tts"


async def test_nothing_to_say_skips_the_model(mlx_fakes: MLXFakes) -> None:
    tts = create("tts", "mlx_audio/pocket-tts")
    audio = await tts.synthesize(" ... ").collect()
    assert audio.duration == 0 and mlx_fakes.tts_calls == []


async def test_cancellation_closes_the_generator(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.tts_chunks = 50
    mlx_fakes.tts_step_delay = 0.01
    tts = create("tts", "mlx_audio/pocket-tts")
    stream = tts.synthesize("A long answer that the user interrupts.")
    first = await stream.__anext__()
    assert first.frame
    await stream.aclose()
    for _ in range(200):  # the close runs on the MLX thread after the current step
        if mlx_fakes.tts_closed:
            break
        await asyncio.sleep(0.01)
    assert mlx_fakes.tts_closed == [True]
    assert len(mlx_fakes.tts_calls) == 1


async def test_sentence_streaming(mlx_fakes: MLXFakes) -> None:
    tts = create("tts", "mlx_audio/pocket-tts")
    stream = tts.stream()
    stream.push_text("First sentence here. Second")
    stream.push_text(" sentence there.")
    stream.end_input()
    items = [i async for i in stream]
    await stream.aclose()
    assert [c["text"] for c in mlx_fakes.tts_calls] == [
        "First sentence here.",
        "Second sentence there.",
    ]
    assert sum(i.frame.duration for i in items) > 0


async def test_kokoro_without_misaki(mlx_fakes: MLXFakes, g2p: dict[str, bool]) -> None:
    g2p["misaki"] = False
    with pytest.raises(MissingDependencyError, match=r"Python < 3\.13"):
        await create("tts", "mlx_audio/kokoro").warmup()
    assert mlx_fakes.tts_loads == []


async def test_kokoro_spacy_model_without_pip(mlx_fakes: MLXFakes, g2p: dict[str, bool]) -> None:
    g2p["en_core_web_sm"] = False
    g2p["pip"] = False
    with pytest.raises(MissingDependencyError, match=r"uv pip install https://github\.com"):
        await create("tts", "mlx_audio/kokoro").warmup()
    # other languages do not use spaCy; with pip, misaki installs it by itself
    await create("tts", "mlx_audio/kokoro", voice="ff_siwis").warmup()
    g2p["pip"] = True
    await create("tts", "mlx_audio/kokoro").warmup()
    assert len(mlx_fakes.tts_loads) == 2


@pytest.mark.usefixtures("g2p")
async def test_kokoro_uses_the_snapshot_voices(
    mlx_fakes: MLXFakes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "af_heart.safetensors").write_bytes(b"")
    hub = __import__("sys").modules["huggingface_hub"]
    monkeypatch.setattr(hub, "snapshot_download", lambda repo, **kw: str(tmp_path))
    tts = create("tts", "mlx_audio/kokoro")
    await tts.synthesize("Hi.").collect()
    call = mlx_fakes.tts_calls[-1]
    assert call["voice"] == str(voices / "af_heart.safetensors") and call["lang_code"] == "a"
    await tts.synthesize("Hi.", voice="bf_emma").collect()  # not in the snapshot: by name
    assert mlx_fakes.tts_calls[-1]["voice"] == "bf_emma"


@pytest.mark.usefixtures("g2p")
async def test_kokoro_language_option(mlx_fakes: MLXFakes) -> None:
    tts = create("tts", "mlx_audio/kokoro", voice="af_heart", language="en-GB")
    await tts.synthesize("Hello.").collect()
    assert mlx_fakes.tts_calls[-1]["lang_code"] == "b"  # a language tag, mapped
    tts = create("tts", "mlx_audio/kokoro", voice="af_heart", language="j")
    await tts.synthesize("Hello.").collect()
    assert mlx_fakes.tts_calls[-1]["lang_code"] == "j"  # a Kokoro code, as is
    with pytest.warns(DeprecationWarning, match=r"MLXAudioTTS\(lang_code=\.\.\.\)"):
        old = create("tts", "mlx_audio/kokoro", voice="af_heart", lang_code="e")
    assert old.language == old.lang_code == "e"
