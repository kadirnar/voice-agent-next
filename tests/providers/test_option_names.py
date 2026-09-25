"""Constructor option names are consistent across the registered providers, and renamed
options keep working with a DeprecationWarning."""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from voice_agent_next.errors import ConfigurationError
from voice_agent_next.providers._options import deprecated, onnx_providers, renamed
from voice_agent_next.registry import ProviderSpec, list_providers

# deprecated name -> standard name
RENAMED = {
    "lang": "language",
    "lang_code": "language",
    "force_cpu": "device",
    "execution_provider": "device",
    "providers": "device",
    "url": "base_url",
    "request_timeout": "timeout",
    "extra_setup": "extra_config",
    "extra_headers": "headers",
}

SPECS = list_providers()


def _parameters(factory: Any) -> dict[str, inspect.Parameter]:
    """The constructor parameters, following ``**kwargs`` into the base classes."""
    params: dict[str, inspect.Parameter] = {}
    classes = factory.__mro__ if inspect.isclass(factory) else (factory,)
    for cls in classes:
        init = cls.__init__ if inspect.isclass(cls) else cls
        if init is object.__init__:
            break
        sig = inspect.signature(init)
        for name, param in sig.parameters.items():
            if name != "self" and param.kind is not param.VAR_KEYWORD:
                params.setdefault(name, param)
        if not any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            break
    return params


def _id(spec: ProviderSpec) -> str:
    return f"{spec.kind}:{spec.name}"


def test_the_registry_is_populated() -> None:
    kinds = {s.kind for s in SPECS}
    assert {"stt", "tts", "llm", "vad", "turn", "engine"} <= kinds
    assert len(SPECS) > 60


@pytest.mark.parametrize("spec", SPECS, ids=_id)
def test_constructors_accept_model(spec: ProviderSpec) -> None:
    params = _parameters(spec.factory)
    assert "model" in params, f"{_id(spec)} does not accept model="
    param = params["model"]
    assert param.kind is param.KEYWORD_ONLY
    assert param.default is None or isinstance(param.default, str)


@pytest.mark.parametrize("spec", SPECS, ids=_id)
def test_constructors_use_the_standard_option_names(spec: ProviderSpec) -> None:
    params = _parameters(spec.factory)
    for old, new in RENAMED.items():
        if old in params:
            # a deprecated alias only: the standard name exists and the alias defaults to
            # None so that using it can be detected
            assert new in params, f"{_id(spec)}: {old}= without {new}="
            assert params[old].default is None, f"{_id(spec)}: {old}= is not a pure alias"


# ----------------------------------------------------------------------- helpers
def test_renamed_and_deprecated() -> None:
    assert renamed("X", "language", "en", "lang", None) == "en"
    with pytest.warns(DeprecationWarning, match=r"X\(lang=\.\.\.\) is deprecated, use language"):
        assert renamed("X", "language", None, "lang", "fr") == "fr"
    with (
        pytest.warns(DeprecationWarning, match="use language"),
        pytest.raises(ConfigurationError, match="not both"),
    ):
        renamed("X", "language", "en", "lang", "fr")
    with pytest.warns(DeprecationWarning, match="use timeout"):
        assert deprecated("X", "timeout", "request_timeout", 5.0) == 5.0


def test_onnx_providers() -> None:
    assert onnx_providers(None) is None and onnx_providers("auto") is None
    assert onnx_providers("cpu") == ["CPUExecutionProvider"]
    assert onnx_providers("CUDA:0") == ["CUDAExecutionProvider"]
    assert onnx_providers("coreml") == ["CoreMLExecutionProvider"]
    assert onnx_providers("DmlExecutionProvider") == ["DmlExecutionProvider"]
    cuda = ("CUDAExecutionProvider", {"device_id": 1})
    assert onnx_providers([cuda, "CPUExecutionProvider"]) == [cuda, "CPUExecutionProvider"]
    with pytest.raises(ConfigurationError, match="unknown device"):
        onnx_providers("tpu")


# -------------------------------------------------------------- provider aliases
def test_moshi_url_alias() -> None:
    from voice_agent_next.providers.moshi import MoshiEngine
    from voice_agent_next.providers.personaplex import PersonaPlexEngine

    assert MoshiEngine(base_url="ws://h:1").base_url == "ws://h:1"
    with pytest.warns(DeprecationWarning, match=r"MoshiEngine\(url=\.\.\.\)"):
        engine = MoshiEngine(url="ws://h:2")
    assert engine.base_url == engine.url == "ws://h:2"
    with pytest.warns(DeprecationWarning, match=r"PersonaPlexEngine\(url=\.\.\.\)"):
        plex = PersonaPlexEngine(url="wss://h:3")
    assert plex.base_url == "wss://h:3"
    assert PersonaPlexEngine().base_url == "wss://localhost:8998"


def test_timeout_aliases() -> None:
    from voice_agent_next.providers.assemblyai import AssemblyAISTT
    from voice_agent_next.providers.deepgram import DeepgramTTS
    from voice_agent_next.providers.elevenlabs import ElevenLabsSTT

    for cls in (AssemblyAISTT, DeepgramTTS, ElevenLabsSTT):
        assert cls(api_key="k", timeout=7.0).timeout == 7.0
        with pytest.warns(DeprecationWarning, match="use timeout"):
            obj = cls(api_key="k", request_timeout=9.0)
        assert obj.timeout == obj.request_timeout == 9.0
        assert cls(api_key="k").timeout > 0  # the default is kept


def test_gemini_live_extra_config_alias() -> None:
    from voice_agent_next.providers.google.live import GeminiLiveEngine

    extra = {"realtimeInputConfig": {"turnCoverage": "TURN_INCLUDES_ALL_INPUT"}}
    assert GeminiLiveEngine(api_key="k", extra_config=extra).extra_config == extra
    with pytest.warns(DeprecationWarning, match="use extra_config"):
        engine = GeminiLiveEngine(api_key="k", extra_setup=extra)
    assert engine.extra_config == engine.extra_setup == extra


def test_anthropic_extra_and_headers_aliases() -> None:
    pytest.importorskip("anthropic")
    from voice_agent_next.providers.anthropic import AnthropicLLM

    llm = AnthropicLLM(api_key="k", extra={"top_k": 5}, headers={"a": "1"})
    assert (llm.extra, llm.headers) == ({"top_k": 5}, {"a": "1"})
    with pytest.warns(DeprecationWarning, match="is deprecated") as record:
        old = AnthropicLLM(api_key="k", extra_params={"top_k": 5}, extra_headers={"a": "1"})
    assert {str(w.message).split("(")[1].split("=")[0] for w in record} == {
        "extra_params",
        "extra_headers",
    }
    assert (old.extra_params, old.extra_headers) == ({"top_k": 5}, {"a": "1"})


def test_smart_turn_device_alias() -> None:
    from voice_agent_next.providers.smart_turn import SmartTurnDetector

    assert SmartTurnDetector().providers == ("CPUExecutionProvider",)
    assert SmartTurnDetector(device="auto").providers == ()
    assert SmartTurnDetector(device="cuda").providers == ("CUDAExecutionProvider",)
    with pytest.warns(DeprecationWarning, match="use device"):
        old = SmartTurnDetector(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert old.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_silero_device_alias() -> None:
    pytest.importorskip("onnxruntime")
    from voice_agent_next.providers.silero import SileroVAD

    assert SileroVAD().force_cpu and SileroVAD().device == "cpu"
    assert not SileroVAD(device="auto").force_cpu
    with pytest.warns(DeprecationWarning, match="use device"):
        assert not SileroVAD(force_cpu=False).force_cpu
    with pytest.warns(DeprecationWarning, match="use device"):
        assert SileroVAD(force_cpu=True).device == "cpu"
