"""Presets: named, ready-to-run agent configurations, with readiness checks.

A preset is an :class:`~voice_agent_next.config.AppConfig` fragment (the components of an
engine or a cascade) plus what it needs to run: extras, API keys, platform, accelerator,
local model servers. Use one from the command line, a config file or Python::

    van presets                          # which presets are ready here, and what is missing
    van run --preset local-cpu           # validates first, prints the fixes if not ready
    van run                              # picks the best ready preset, and says which

    # agent.yaml
    extends: local-cpu
    llm: ollama/qwen3.5:4b

    from voice_agent_next.presets import load_preset
    from voice_agent_next.app import build_agent, build_session
    cfg = load_preset("cloud-fast", agent={"instructions": "You are a travel agent."})
    session, agent = build_session(cfg), build_agent(cfg)

The defaults come from our own measurements (``docs/benchmarks/results.md``,
``docs/providers/sherpa-onnx.md``, ``docs/hardware.md``) and from the recommended stacks of
research note 03 (§9.2); each preset's ``rationale`` says which. See ``docs/presets.md``.

Readiness (:func:`check_preset`) is computed from the provider registry (dependencies,
credentials, platforms) plus what the registry cannot know: whether the machine has the
GPU a preset is built for, whether the Ollama server runs and has the model, whether the
mlx-lm server runs, and whether the local audio transport is installed. Failover lists are
pruned to their ready members, so a cloud preset still runs with only one of its LLM
vendors' keys set.
"""

from __future__ import annotations

import copy
import difflib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .config import AppConfig, merge_config
from .errors import ConfigurationError, MissingDependencyError, ProviderNotFoundError
from .registry import ComponentKind, get_provider, parse_spec
from .utils.deps import is_installed

if TYPE_CHECKING:
    from .hardware import AppleSilicon, Backend
    from .session import Agent, AgentSession

__all__ = [
    "AUTO_ORDER",
    "LOCAL_TURN_TAKING",
    "PRESETS",
    "Environment",
    "Preset",
    "Problem",
    "Readiness",
    "check_config",
    "check_preset",
    "current_environment",
    "get_preset",
    "list_presets",
    "load_preset",
    "pick_preset",
    "session_from_preset",
]

Where = Literal["local", "cloud", "hybrid"]
Accelerator = Literal["cuda", "apple"]

_KINDS: dict[str, ComponentKind] = {
    "engine": "engine",
    "stt": "stt",
    "llm": "llm",
    "tts": "tts",
    "vad": "vad",
    "turn_detector": "turn",
}
_PLATFORM_NAMES = {"linux": "Linux", "darwin": "macOS", "win32": "Windows"}
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434/v1"
MLX_LM_DEFAULT_URL = "http://127.0.0.1:8080/v1"


@dataclass(frozen=True)
class Preset:
    """A named configuration fragment and what it needs to run."""

    name: str
    summary: str
    """One line for ``van presets``."""
    config: Mapping[str, Any]
    """Raw :class:`AppConfig` mapping (components, and optionally cascade/session options)."""
    rationale: str = ""
    """Why these components (the measurements or research behind the choice)."""
    where: Where = "local"
    platforms: tuple[str, ...] = ("linux", "darwin", "win32")
    accelerator: Accelerator | None = None
    """Hardware the preset is built for: an NVIDIA GPU usable by CUDA, or Apple silicon."""

    def app_config(self) -> AppConfig:
        """The preset as an :class:`AppConfig` (no readiness check: see :func:`load_preset`)."""
        return AppConfig.model_validate({**copy.deepcopy(dict(self.config)), "extends": self.name})

    def components(self) -> list[tuple[str, Any]]:
        """``(config key, provider spec)`` of every component, failover members and the
        halves of a ``fused`` turn detector included."""
        return [
            (key, part)
            for key in _KINDS
            if self.config.get(key) is not None
            for member in _members(self.config[key])
            for part in (member, *_fused_parts(key, member))
        ]

    def stack(self) -> str:
        """The components on one line: ``"engine"`` or ``"stt > llm > tts"``."""
        parts = []
        for key in ("engine", "stt", "llm", "tts"):
            spec = self.config.get(key)
            if spec is None:
                continue
            members = _members(spec)
            more = f" (+{len(members) - 1} failover)" if len(members) > 1 else ""
            parts.append(_describe(members[0]) + more)
        return " > ".join(parts)

    @property
    def extras(self) -> tuple[str, ...]:
        """``voice-agent-next`` extras the components need (all failover members)."""
        found: list[str] = []
        for key, member in self.components():
            spec = _lookup(_KINDS[key], member)
            if spec is not None and spec.extra and spec.extra not in found:
                found.append(spec.extra)
        return tuple(found)

    @property
    def env_vars(self) -> tuple[tuple[str, ...], ...]:
        """Credential variables, one group per provider (any variable of a group suffices)."""
        groups: list[tuple[str, ...]] = []
        for key, member in self.components():
            spec = _lookup(_KINDS[key], member)
            if spec is not None and spec.env and spec.env not in groups:
                groups.append(spec.env)
        return tuple(groups)


# --------------------------------------------------------------------------- presets
_TURN_TAKING = (
    "Silero VAD and Smart Turn v3.2 end the user's turn (Smart Turn runs concurrently with the "
    "STT flush"
)
LOCAL_TURN_TAKING: Mapping[str, Mapping[str, Any]] = {
    "cascade": {
        "min_endpointing_delay": 0.5,
        "max_endpointing_delay": 1.5,
        "preemptive_generation": True,
        "preemptive_tts": True,
    },
    "session": {"max_backchannel_duration": 1.0},
}
"""Turn-taking settings of the local cascade presets (issue #113, measured with
``van bench turns`` on eot-bench and ``van bench turn-taking`` / ``latency``):

* ``max_endpointing_delay`` 1.5 s: Smart Turn says "not done" at 25 % of real turn ends
  (eot-bench English); waiting 2.5 s there was the battery's dead air. 1.5 s keeps every
  such reply under 2 s.
* ``min_endpointing_delay`` 0.5 s: with the lower ceiling, the same false-cutoff rate on
  eot-bench as the old 0.4 / 2.5 s (10.8 %) at 181 ms less mean latency.
* preemptive generation + TTS: the reply is synthesized during the endpointing silence, so
  the extra 0.1 s costs no voice-to-voice latency (T1 p50 637 -> 600 ms on CPU).
* ``max_backchannel_duration`` 1.0 s: small streaming ASR models turn "uh-huh" into words
  ("but high"); short utterances without an interruption word are backchannels.
"""
_TURN_TAKING_NOTE = (
    "Turn-taking defaults from the T4 battery (#113): 0.5-1.5 s endpointing with the reply "
    "prepared during the silence, and short utterances over the agent treated as "
    "backchannels unless they contain an interruption word."
)


FUSED_TURN_DETECTOR: Mapping[str, Any] = {
    "provider": "fused",
    "audio": "smart_turn",
    "text": "lm_turn",
}
"""Smart Turn v3.2 fused with the ``lm_turn`` text model (SmolLM2-135M int8, 137 MB,
extra ``text-turn``), the turn detector of ``local-cpu`` (issue #155). With a streaming
STT the transcript is already there at the pause, so the text half costs ~30 ms of
end-of-turn delay; behind a batch STT (faster-whisper in ``local-gpu``) it waits for the
final transcript and cost 100-230 ms, with no fewer premature replies: see
``docs/benchmarks/results.md``."""


def _local_turn_taking(*, preemptive: bool = True) -> dict[str, Any]:
    cascade = dict(LOCAL_TURN_TAKING["cascade"])
    if not preemptive:  # speculative calls to a paid LLM cost money when discarded
        cascade = {k: v for k, v in cascade.items() if not k.startswith("preemptive")}
    return {"cascade": cascade, "session": dict(LOCAL_TURN_TAKING["session"])}


_PRESETS: tuple[Preset, ...] = (
    Preset(
        name="local-cpu",
        summary="Fully offline on any laptop CPU: sherpa-onnx streaming STT, Ollama, Kokoro",
        config={
            "stt": "sherpa-onnx/zipformer-en-kroko",
            "llm": "ollama/LiquidAI/lfm2.5-1.2b-instruct",
            "tts": "kokoro/v1.0-fp16",
            "vad": "silero",
            "turn_detector": dict(FUSED_TURN_DETECTOR),
            **_local_turn_taking(),
        },
        rationale=(
            "Measured on a Ryzen 5 5600 without GPU (docs/providers/sherpa-onnx.md): the "
            "streaming Kroko Zipformer has the final transcript ~100 ms after the end of "
            "speech, which cuts the end-of-turn delay from 633 to 400 ms against "
            "faster-whisper base and gave the best voice-to-voice latency (p50 1,126 ms, p90 "
            "1,620 ms, 4 % dead air), with cased, punctuated transcripts (57 MB, English). "
            "LFM2.5 1.2B is the local LLM of every T1 measurement (~12 ms warm TTFT on CPU), "
            "so the CPU is left to STT and TTS, the contention research note 03 warns about; "
            "for better tool calling use `--llm ollama/qwen3.5:4b` (note 03 §7.5). Kokoro-82M "
            "is the best open TTS that runs in real time on a CPU (first clause ~400 ms). "
            f"{_TURN_TAKING}, off the critical path). {_TURN_TAKING_NOTE}"
        ),
        where="local",
    ),
    Preset(
        name="local-gpu",
        summary="Local on an NVIDIA GPU: faster-whisper large-v3-turbo on CUDA, Ollama 9B, Kokoro",
        config={
            "stt": "faster-whisper/large-v3-turbo",
            "llm": "ollama/qwen3.5:9b",
            "tts": "kokoro/v1.0-fp16",
            "vad": "silero",
            "turn_detector": "smart_turn",
            **_local_turn_taking(),
        },
        rationale=(
            "On an RTX 5070 Ti faster-whisper's final transcription drops from 391 to 51 ms "
            "(CUDA float16) and the end-of-turn delay reaches its 400 ms floor; v2v p50 went "
            "from 1,523 to 969 ms (docs/hardware.md). large-v3-turbo keeps 99 languages at "
            "GPU speed (note 03 §9.2 names it the multilingual fallback of the 16 GB "
            "profile). Qwen3.5-9B is note 03's LLM for a 16 GB GPU (tool calling: vendor "
            "BFCL-V4 66.1, TAU2 79.1); Ollama puts it on the GPU. Kokoro stays the TTS (it "
            "runs on CUDA too with onnxruntime-gpu, see docs/hardware.md). "
            f"{_TURN_TAKING}). {_TURN_TAKING_NOTE}"
        ),
        where="local",
        platforms=("linux", "win32"),
        accelerator="cuda",
    ),
    Preset(
        name="apple",
        summary="Apple silicon on MLX: Parakeet streaming STT, mlx-lm (or Ollama), Kokoro",
        config={
            "stt": "mlx/parakeet-tdt-0.6b-v3",
            "llm": ["mlx_lm/mlx-community/Qwen3.5-4B-4bit", "ollama/qwen3.5:4b"],
            "tts": "mlx_audio/kokoro",
            "vad": "silero",
            "turn_detector": "smart_turn",
            **_local_turn_taking(),
        },
        rationale=(
            "Note 03 §9.2's Apple silicon stack, all on the GPU through MLX: parakeet-mlx "
            "streams Parakeet TDT 0.6B v3 (25 European languages), idle once the user stops "
            "talking, so the final transcript is one pass (on a GitHub M1 runner the 110M "
            "Parakeet finalized a 3 s turn 127 ms after the flush); mlx_lm.server runs "
            "Qwen3.5-4B at 4-bit (Qwen3-1.7B 4-bit there: 211 ms TTFT and a correct tool "
            "call; note 03 recommends Qwen3.5 / Gemma 4 E4B at 4-bit; use "
            "`--llm mlx_lm/mlx-community/Qwen3.5-9B-4bit` with 16 GB or more), with Ollama "
            "as the failover when no mlx-lm server runs; Kokoro-82M through mlx-audio "
            "(`--tts mlx_audio/pocket-tts` streams audio: 140 ms to first audio against "
            "Kokoro's 542 ms per segment; docs/providers/mlx.md). "
            f"{_TURN_TAKING}). {_TURN_TAKING_NOTE}"
        ),
        where="local",
        platforms=("darwin",),
        accelerator="apple",
    ),
    Preset(
        name="hybrid",
        summary="Local STT/TTS with a cloud LLM (Claude Haiku 4.5 > Groq), local Ollama fallback",
        config={
            "stt": "sherpa-onnx/zipformer-en-kroko",
            "llm": [
                "anthropic/claude-haiku-4-5",
                "groq/openai/gpt-oss-120b",
                "ollama/LiquidAI/lfm2.5-1.2b-instruct",
            ],
            "tts": "kokoro/v1.0-fp16",
            "vad": "silero",
            "turn_detector": "smart_turn",
            **_local_turn_taking(preemptive=False),
        },
        rationale=(
            "The audio stays on the machine (local-cpu's measured STT/TTS) and only text "
            "goes to the cloud: note 03 §9.2 offers a cloud LLM in hybrid mode for machines "
            "too small for a good local LLM. Claude Haiku 4.5 passes 98.0 % of Pipecat's "
            "30-turn tool-use benchmark at 637 ms TTFAT; Groq's gpt-oss-120b answers in 98 ms "
            "(86.3 %); the local LFM2.5 keeps the agent talking offline. Only the members "
            "that are ready are used."
        ),
        where="hybrid",
    ),
    Preset(
        name="cloud-fast",
        summary="Lowest cloud latency: Deepgram Flux, Groq / Cerebras gpt-oss-120b, Cartesia",
        config={
            "stt": "deepgram/flux-general-en",
            "llm": ["groq/openai/gpt-oss-120b", "cerebras/gpt-oss-120b"],
            "tts": "cartesia/sonic-3.6",
        },
        rationale=(
            "Note 03 §9.2 'cloud, low latency': Deepgram Flux ends turns itself (EndOfTurn "
            "commits at once, no VAD or endpointing delay; English); gpt-oss-120b on Groq has "
            "a 98 / 217 ms p50 / p95 TTFAT in Pipecat's benchmark, Cerebras is the fastest "
            "host of the same model (AA: 1,734 tok/s) and the failover; Cartesia Sonic 3.6 "
            "tops the AA TTS arena (1273) and streams text with continuations. For stronger "
            "tool calling see cloud-quality. Needs one key of each stage."
        ),
        where="cloud",
    ),
    Preset(
        name="cloud-quality",
        summary="Best cloud quality: AssemblyAI U3.5 Pro, Claude Haiku 4.5 > GPT-4.1, Cartesia",
        config={
            "stt": "assemblyai/universal-3-5-pro",
            "llm": ["anthropic/claude-haiku-4-5", "openai/gpt-4.1"],
            "tts": ["cartesia/sonic-3.6", "elevenlabs/eleven_flash_v2_5"],
        },
        rationale=(
            "AssemblyAI Universal-3.5 Pro streaming with neural end of turn (formatted "
            "finals, owns the turn: no VAD needed); Claude Haiku 4.5 has the best pass rate "
            "inside 700 ms (98.0 % at 637 ms) and GPT-4.1 (96.3 %) is the failover; Cartesia "
            "Sonic 3.6 with ElevenLabs Flash v2.5 as failover (note 03 §7.1, §9.2)."
        ),
        where="cloud",
    ),
    Preset(
        name="openai-realtime",
        summary="OpenAI Realtime native speech-to-speech (gpt-realtime-2.1)",
        config={"engine": "openai/gpt-realtime-2.1"},
        rationale=(
            "One model hears and speaks: the most natural prosody and the simplest setup "
            "(OPENAI_API_KEY only). No local models, no extras."
        ),
        where="cloud",
    ),
    Preset(
        name="gemini-live",
        summary="Gemini Live native speech-to-speech (gemini-3.8-live)",
        config={"engine": "gemini/gemini-3.8-live"},
        rationale=(
            "Google's native audio model over the Live API WebSocket (GOOGLE_API_KEY or "
            "GEMINI_API_KEY). No local models, no extras."
        ),
        where="cloud",
    ),
)
PRESETS: dict[str, Preset] = {p.name: p for p in _PRESETS}
"""Built-in presets by name."""

AUTO_ORDER: tuple[str, ...] = (
    "local-gpu",
    "apple",
    "local-cpu",
    "hybrid",
    "cloud-fast",
    "cloud-quality",
    "openai-realtime",
    "gemini-live",
)
"""Preference order of ``van run`` without a preset: local first (private, free, and the
hardware-specific stack before the generic one), then hybrid, then cloud."""


def list_presets() -> list[Preset]:
    """All built-in presets, in :data:`AUTO_ORDER`."""
    return [PRESETS[name] for name in AUTO_ORDER]


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def get_preset(name: str) -> Preset:
    """A preset by name (``-``/``_`` and case do not matter)."""
    key = _normalize(name)
    if key in PRESETS:
        return PRESETS[key]
    close = difflib.get_close_matches(key, PRESETS, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise ConfigurationError(
        f"unknown preset {name!r}.{hint} Presets: {', '.join(AUTO_ORDER)} (`van presets`)."
    )


# ----------------------------------------------------------------------- environment
def _nvidia_gpus() -> tuple[str, ...]:
    from .hardware import detect_nvidia

    return tuple(str(gpu) for gpu in detect_nvidia().gpus)


def _apple_silicon() -> AppleSilicon | None:
    from .hardware import detect_apple_silicon

    return detect_apple_silicon()


def _cuda_backend() -> Backend | None:
    """Where faster-whisper's ``device="auto"`` runs (``None``: CTranslate2 not installed)."""
    if not is_installed("ctranslate2"):
        return None
    from .hardware import select_ctranslate2_backend

    try:
        return select_ctranslate2_backend()
    except Exception:  # a broken CTranslate2 install: the model itself will report it
        return None


def _ollama_models(base_url: str) -> list[str] | None:
    """Model names the Ollama server has pulled, or ``None`` when it does not answer.

    Uses ``urllib`` rather than httpx so that a readiness check logs nothing at INFO.
    """
    import json
    import urllib.request

    root = base_url.rstrip("/").removesuffix("/v1")
    if not root.startswith(("http://", "https://")):
        return None
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=1.5) as response:
            models = json.loads(response.read()).get("models", [])
        return [str(m.get("name") or m.get("model")) for m in models]
    except (OSError, ValueError, AttributeError):
        return None


def _openai_models(base_url: str) -> list[str] | None:
    """Model ids of an OpenAI-compatible server (``GET /models``), ``None`` if it does not
    answer (the mlx-lm server's list is the MLX models in the Hugging Face cache)."""
    import json
    import urllib.request

    root = base_url.rstrip("/")
    if not root.startswith(("http://", "https://")):
        return None
    try:
        with urllib.request.urlopen(f"{root}/models", timeout=1.5) as response:
            models = json.loads(response.read()).get("data", [])
        return [str(m.get("id")) for m in models]
    except (OSError, ValueError, AttributeError):
        return None


@dataclass
class Environment:
    """What readiness checks look at; every probe can be replaced (tests fake them)."""

    platform: str = field(default_factory=lambda: sys.platform)
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    installed: Callable[[str], bool] = is_installed
    """Whether a module is importable."""
    nvidia_gpus: Callable[[], Sequence[str]] = _nvidia_gpus
    apple_silicon: Callable[[], AppleSilicon | None] = _apple_silicon
    cuda_backend: Callable[[], Backend | None] = _cuda_backend
    ollama_models: Callable[[str], Sequence[str] | None] = _ollama_models
    """Models of the Ollama server at a ``/v1`` base URL, ``None`` if unreachable."""
    _ollama_cache: dict[str, Sequence[str] | None] = field(default_factory=dict, repr=False)
    mlx_lm_models: Callable[[str], Sequence[str] | None] = _openai_models
    """Models of the mlx-lm server at a ``/v1`` base URL, ``None`` if unreachable."""
    _mlx_lm_cache: dict[str, Sequence[str] | None] = field(default_factory=dict, repr=False)

    def ollama(self, base_url: str) -> Sequence[str] | None:
        if base_url not in self._ollama_cache:
            self._ollama_cache[base_url] = self.ollama_models(base_url)
        return self._ollama_cache[base_url]

    def ollama_url(self, options: Mapping[str, Any]) -> str:
        """The Ollama ``/v1`` URL :class:`~voice_agent_next.providers.ollama.OllamaLLM` uses."""
        from .providers.ollama import ollama_base_url

        if options.get("base_url"):
            return str(options["base_url"])
        if self.environ.get("OLLAMA_BASE_URL"):
            return self.environ["OLLAMA_BASE_URL"]
        host = self.environ.get("OLLAMA_HOST", "").strip()
        return ollama_base_url(host) if host else OLLAMA_DEFAULT_URL

    def mlx_lm(self, base_url: str) -> Sequence[str] | None:
        if base_url not in self._mlx_lm_cache:
            self._mlx_lm_cache[base_url] = self.mlx_lm_models(base_url)
        return self._mlx_lm_cache[base_url]

    def mlx_lm_url(self, options: Mapping[str, Any]) -> str:
        """The ``/v1`` URL :class:`~voice_agent_next.providers.mlx_lm.MLXLMServerLLM` uses."""
        if options.get("base_url"):
            return str(options["base_url"])
        return self.environ.get("MLX_LM_BASE_URL") or MLX_LM_DEFAULT_URL


def current_environment() -> Environment:
    """The real machine (the CLI calls this; tests replace it)."""
    return Environment()


# ------------------------------------------------------------------------- readiness
@dataclass(frozen=True)
class Problem:
    """Something that keeps a configuration from running here, and how to fix it."""

    component: str
    """``stt``, ``llm``..., or ``platform`` / ``gpu`` / ``audio``."""
    message: str
    fix: str | None = None
    """A command or step; ``None`` when nothing on this machine fixes it."""
    extra: str | None = None
    """The missing extra, when installing one is the fix (merged into one command)."""


@dataclass(frozen=True)
class Readiness:
    """Result of :func:`check_config` / :func:`check_preset`."""

    name: str
    problems: tuple[Problem, ...]
    notes: tuple[str, ...]
    """Non-blocking remarks (skipped failover members, where models will run...)."""
    config: dict[str, Any]
    """The raw config to run: failover lists pruned to their ready members."""

    @property
    def ready(self) -> bool:
        return not self.problems

    @property
    def extras(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.extra for p in self.problems if p.extra))

    def fixes(self) -> list[str]:
        """Deduplicated fixes, all missing extras in one ``pip install`` first."""
        out: list[str] = []
        if self.extras:
            out.append(f"pip install 'voice-agent-next[{','.join(self.extras)}]'")
        for p in self.problems:
            if p.fix and not p.extra and p.fix not in out:
                out.append(p.fix)
        return out

    def summary(self) -> str:
        """``"ready"`` or the first problem, for one table cell."""
        if self.ready:
            return "ready"
        more = len(self.problems) - 1
        return self.problems[0].message + (f" (+{more} more)" if more else "")

    def explain(self) -> str:
        """Multi-line report: problems, then numbered fixes."""
        if self.ready:
            return f"preset {self.name!r} is ready"
        lines = [f"{self.name} is not ready on this machine:"]
        lines += [f"  - {p.component}: {p.message}" for p in self.problems]
        fixes = self.fixes()
        if fixes:
            lines.append("to fix:")
            lines += [f"  {i}. {fix}" for i, fix in enumerate(fixes, 1)]
        return "\n".join(lines)


def _members(spec: Any) -> list[Any]:
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, Mapping) and "fallback" in spec:
        return list(spec["fallback"])
    return [spec]


def _split(spec: Any) -> tuple[str, str | None, dict[str, Any]]:
    """``(provider, model, options)`` of a string or mapping spec."""
    if isinstance(spec, Mapping):
        opts = dict(spec)
        target = str(opts.pop("provider", None) or opts.pop("use", None) or "")
        name, model = parse_spec(target)
        return name, opts.pop("model", None) or model, opts
    name, model = parse_spec(str(spec))
    return name, model, {}


def _lookup(kind: ComponentKind, spec: Any) -> Any:
    if not isinstance(spec, (str, Mapping)):
        return None
    try:
        return get_provider(kind, _split(spec)[0])
    except (ProviderNotFoundError, MissingDependencyError, ConfigurationError):
        return None


def _fused_parts(key: str, spec: Any) -> list[Any]:
    """The audio and text detectors of a ``fused`` turn detector (``[]`` for anything else).

    They are created by :class:`~voice_agent_next.turn.FusedTurnDetector` itself, so the
    registry entry of ``fused`` knows nothing of their extras: the checks look at them.
    """
    if key != "turn_detector" or not isinstance(spec, (str, Mapping)):
        return []
    name, _, options = _split(spec)
    if name != "fused":
        return []
    return [options.get("audio", "smart_turn"), options.get("text", "lm_turn")]


def _describe(spec: Any) -> str:
    if isinstance(spec, Mapping):
        return str(spec.get("provider") or spec.get("use") or spec)
    return str(spec)


def _normalize_ollama(model: str) -> str:
    model = model.strip().lower()
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def _member_problems(key: str, spec: Any, env: Environment) -> list[Problem]:
    kind = _KINDS[key]
    if not isinstance(spec, (str, Mapping)):
        return []  # an already-built component instance
    try:
        name, model, options = _split(spec)
        provider = get_provider(kind, name)
    except (ProviderNotFoundError, MissingDependencyError, ConfigurationError) as exc:
        return [Problem(key, str(exc))]
    label = f"{_describe(spec)}"
    if not any(env.platform.startswith(p) for p in provider.platforms):
        return [Problem(key, f"{label} does not run on {env.platform}")]
    problems: list[Problem] = []
    missing = [m for m in provider.requires if not env.installed(m)]
    if missing:
        if provider.extra:
            problems.append(
                Problem(
                    key,
                    f"{label} needs the '{provider.extra}' extra",
                    f"pip install 'voice-agent-next[{provider.extra}]'",
                    provider.extra,
                )
            )
        else:
            problems.append(
                Problem(key, f"{label} needs {', '.join(missing)}", f"pip install {missing[0]}")
            )
    if (
        provider.env
        and not options.get("api_key")
        and not any(env.environ.get(v) for v in provider.env)
    ):
        problems.append(
            Problem(
                key,
                f"{label} needs {' or '.join(provider.env)}",
                f"set {provider.env[0]} (e.g. `export {provider.env[0]}=...`)",
            )
        )
    for part in _fused_parts(key, spec):
        problems += _member_problems(key, part, env)
    if provider.name == "ollama":
        problems += _ollama_problems(key, model or provider.default_model or "", options, env)
    if provider.name == "mlx_lm":
        problems += _mlx_lm_problems(key, model, options, env)
    return problems


def _ollama_problems(
    key: str, model: str, options: Mapping[str, Any], env: Environment
) -> list[Problem]:
    url = env.ollama_url(options)
    models = env.ollama(url)
    if models is None:
        return [
            Problem(
                key,
                f"no Ollama server at {url}",
                "install Ollama (https://ollama.com/download) and start it: `ollama serve`",
            ),
            Problem(key, f"Ollama model {model} may not be pulled yet", f"ollama pull {model}"),
        ]
    if _normalize_ollama(model) not in {_normalize_ollama(m) for m in models}:
        return [Problem(key, f"Ollama has no model {model}", f"ollama pull {model}")]
    return []


def _mlx_lm_problems(
    key: str, model: str | None, options: Mapping[str, Any], env: Environment
) -> list[Problem]:
    """The server must run; it downloads and loads the requested model by itself."""
    url = env.mlx_lm_url(options)
    if env.mlx_lm(url) is not None:
        return []
    name = model if model and model != "default_model" else "<mlx-community model>"
    return [
        Problem(
            key,
            f"no mlx-lm server at {url}",
            f"start it (Apple silicon, extra `mlx`): python -m mlx_lm.server --model {name}",
        )
    ]


def _check_components(
    cfg: Mapping[str, Any], env: Environment
) -> tuple[list[Problem], list[str], dict[str, Any]]:
    problems: list[Problem] = []
    notes: list[str] = []
    resolved = copy.deepcopy(dict(cfg))
    for key in _KINDS:
        spec = cfg.get(key)
        if spec is None:
            continue
        members = _members(spec)
        results = [(m, _member_problems(key, m, env)) for m in members]
        if len(members) == 1:
            problems += results[0][1]
            continue
        ready = [m for m, p in results if not p]
        if not ready:
            problems += results[0][1]
            notes.append(
                f"{key}: no failover member is ready; fixing any one of "
                f"{', '.join(_describe(m) for m in members)} is enough"
            )
            continue
        for member, member_problems in results:
            if member_problems:
                notes.append(f"{key}: skipping {_describe(member)} ({member_problems[0].message})")
        if len(ready) == 1:
            resolved[key] = ready[0]
        elif isinstance(spec, Mapping):
            resolved[key] = {**spec, "fallback": ready}
        else:
            resolved[key] = ready
    return problems, notes, resolved


def _hardware_problems(
    cfg: Mapping[str, Any],
    platforms: Sequence[str],
    accelerator: Accelerator | None,
    env: Environment,
) -> tuple[list[Problem], list[str]]:
    if not any(env.platform.startswith(p) for p in platforms):
        wanted = " or ".join(_PLATFORM_NAMES.get(p, p) for p in platforms)
        return [Problem("platform", f"needs {wanted} (this is {env.platform})")], []
    problems: list[Problem] = []
    notes: list[str] = []
    if accelerator == "cuda":
        gpus = env.nvidia_gpus()
        if not gpus:
            problems.append(
                Problem("gpu", "no NVIDIA GPU found", "use a CPU preset: `--preset local-cpu`")
            )
        else:
            notes.append(f"GPU: {gpus[0]}")
            stt = [m for m in _members(cfg.get("stt")) if isinstance(m, (str, Mapping))]
            uses_ct2 = any(_split(m)[0] == "faster_whisper" for m in stt)
            backend = env.cuda_backend() if uses_ct2 else None
            if backend is not None and backend.device != "cuda":
                problems.append(
                    Problem(
                        "gpu",
                        f"faster-whisper would run on the CPU: {backend.reason}",
                        backend.fix or "run `van doctor` (see docs/hardware.md)",
                    )
                )
    elif accelerator == "apple":
        chip = env.apple_silicon()
        if chip is None:
            problems.append(Problem("platform", "needs an Apple silicon Mac"))
        elif chip.rosetta:
            problems.append(
                Problem(
                    "platform",
                    "Python runs under Rosetta 2 (x86_64): CoreML and Metal need an arm64 Python",
                    "install an arm64 Python (e.g. `uv python install` on the Mac itself)",
                )
            )
        else:
            notes.append(f"Apple silicon: {chip}")
    return problems, notes


def check_config(
    config: AppConfig | Mapping[str, Any],
    *,
    name: str = "config",
    platforms: Sequence[str] = ("linux", "darwin", "win32"),
    accelerator: Accelerator | None = None,
    transport: str | None = None,
    env: Environment | None = None,
) -> Readiness:
    """Whether a configuration can run on this machine, and the fixes if not.

    Args:
        transport: the transport it will run with; ``"local"`` also checks the audio extra.
        env: the machine to check (default: :func:`current_environment`).
    """
    env = env or current_environment()
    raw = config.model_dump(exclude_none=True) if isinstance(config, AppConfig) else dict(config)
    problems, notes = _hardware_problems(raw, platforms, accelerator, env)
    if problems:  # wrong platform: component checks would only add noise
        return Readiness(name, tuple(problems), tuple(notes), copy.deepcopy(raw))
    component_problems, component_notes, resolved = _check_components(raw, env)
    problems += component_problems
    notes += component_notes
    if transport == "local" and not env.installed("sounddevice"):
        problems.append(
            Problem(
                "audio",
                "the microphone/speaker transport needs the 'audio' extra",
                "pip install 'voice-agent-next[audio]'",
                "audio",
            )
        )
    return Readiness(name, tuple(problems), tuple(notes), resolved)


def check_preset(
    preset: str | Preset,
    *,
    overrides: Mapping[str, Any] | None = None,
    transport: str | None = "local",
    env: Environment | None = None,
) -> Readiness:
    """:func:`check_config` for a preset (with optional config ``overrides`` merged in)."""
    p = get_preset(preset) if isinstance(preset, str) else preset
    raw = dict(p.config)
    if overrides:
        raw = merge_config(raw, dict(overrides))
    return check_config(
        raw,
        name=p.name,
        platforms=p.platforms,
        accelerator=p.accelerator,
        transport=transport,
        env=env,
    )


def pick_preset(
    *, transport: str | None = "local", env: Environment | None = None
) -> tuple[Readiness | None, list[Readiness]]:
    """The first ready preset in :data:`AUTO_ORDER` (``None`` if none), and every check."""
    env = env or current_environment()
    checked: list[Readiness] = []
    for name in AUTO_ORDER:
        result = check_preset(name, transport=transport, env=env)
        checked.append(result)
        if result.ready:
            return result, checked
    return None, checked


# ---------------------------------------------------------------------------- Python
def load_preset(
    name: str,
    *,
    check: bool = True,
    env: Environment | None = None,
    **overrides: Any,
) -> AppConfig:
    """A preset as an :class:`AppConfig`, with config ``overrides`` merged on top.

    ``load_preset("local-cpu", llm="ollama/qwen3.5:4b", agent={"instructions": "..."})``.
    With ``check`` (the default), raises :class:`ConfigurationError` listing the fixes when
    the preset cannot run here, and prunes failover lists to their ready members.
    """
    preset = get_preset(name)
    raw = merge_config(dict(preset.config), overrides)
    if check:
        result = check_config(
            raw,
            name=preset.name,
            platforms=preset.platforms,
            accelerator=preset.accelerator,
            env=env,
        )
        if not result.ready:
            raise ConfigurationError(result.explain())
        raw = result.config
    cfg = AppConfig.model_validate({**raw, "extends": preset.name})
    cfg.validate_components()
    return cfg


def session_from_preset(name: str, **kwargs: Any) -> tuple[AgentSession, Agent]:
    """``(session, agent)`` built from :func:`load_preset` (same arguments)."""
    from .app import build_agent, build_session

    cfg = load_preset(name, **kwargs)
    return build_session(cfg), build_agent(cfg)
