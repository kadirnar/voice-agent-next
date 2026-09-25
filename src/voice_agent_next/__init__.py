"""voice-agent-next: real-time speech-to-speech voice agents.

Native speech-to-speech models and streaming cascades (VAD -> STT -> LLM -> TTS)
behind one engine interface; local and cloud providers; Linux, macOS and Windows.

Quick start::

    import asyncio
    from voice_agent_next import Agent, AgentSession
    from voice_agent_next.transports import LoopbackTransport

    async def main() -> None:
        session = AgentSession("mock")  # or "openai/gpt-realtime", or stt=/llm=/tts=
        await session.run(Agent("You are a helpful assistant."), LoopbackTransport())

    asyncio.run(main())
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import events
from .audio import AudioBuffer, AudioFormat, AudioFrame, Resampler, read_wav, resample, write_wav
from .chat import (
    AudioContent,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from .engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from .engines.cascade import CascadeEngine, CascadeOptions
from .errors import (
    ConfigurationError,
    MissingDependencyError,
    ProviderError,
    ProviderNotFoundError,
    VoiceAgentError,
)
from .fallback import FallbackLLM, FallbackSTT, FallbackTTS
from .llm import LLM, ChatChunk, LLMCapabilities, LLMStream
from .registry import create, list_providers, register_provider
from .session import (
    Agent,
    AgentSession,
    AgentState,
    Flow,
    FlowNode,
    Handoff,
    SessionOptions,
    Transition,
    UserState,
)
from .stt import STT, StreamAdapter, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript
from .tools import FunctionTool, ToolContext, ToolScheduling, function_tool
from .tts import TTS, ChunkedStream, SynthesizedAudio, SynthesizeStream, TTSCapabilities
from .turn import TurnDetector
from .vad import VAD, VADEvent, VADEventType, VADOptions

try:
    __version__ = version("voice-agent-next")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "LLM",
    "STT",
    "TTS",
    "VAD",
    "Agent",
    "AgentSession",
    "AgentState",
    "AudioBuffer",
    "AudioContent",
    "AudioFormat",
    "AudioFrame",
    "CascadeEngine",
    "CascadeOptions",
    "ChatChunk",
    "ChatContext",
    "ChatMessage",
    "ChunkedStream",
    "ConfigurationError",
    "EngineCapabilities",
    "EngineConnection",
    "EngineOptions",
    "FallbackLLM",
    "FallbackSTT",
    "FallbackTTS",
    "Flow",
    "FlowNode",
    "FunctionCall",
    "FunctionCallOutput",
    "FunctionTool",
    "Handoff",
    "ImageContent",
    "LLMCapabilities",
    "LLMStream",
    "MissingDependencyError",
    "ProviderError",
    "ProviderNotFoundError",
    "Resampler",
    "S2SEngine",
    "STTCapabilities",
    "STTEvent",
    "STTEventType",
    "STTStream",
    "SessionOptions",
    "StreamAdapter",
    "SynthesizeStream",
    "SynthesizedAudio",
    "TTSCapabilities",
    "ToolContext",
    "ToolScheduling",
    "Transcript",
    "Transition",
    "TurnDetector",
    "UserState",
    "VADEvent",
    "VADEventType",
    "VADOptions",
    "VoiceAgentError",
    "__version__",
    "create",
    "events",
    "function_tool",
    "list_providers",
    "read_wav",
    "register_provider",
    "resample",
    "write_wav",
]
