"""Large language model (LLM) component interface.

Implementing a provider: override :meth:`LLM._chat` to return an :class:`LLMStream`
subclass whose :meth:`LLMStream._run` calls ``self._push(ChatChunk(...))``.

Tool calls must be emitted *complete* (name + full JSON arguments) — providers
accumulate streamed argument deltas internally.

Audio-output ("omni") models (``LLMCapabilities.audio_output``) also stream their own
speech: :attr:`ChatChunk.audio` carries s16le audio deltas next to the text deltas, which
are the transcript of that speech. The cascade plays that audio instead of running a TTS
(see ``docs/concepts/omni-models.md``).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias

from .audio.frame import AudioFrame
from .chat import ChatContext, FunctionCall
from .metrics import LLMMetrics
from .tools import FunctionTool
from .utils.aio import Chan, ChanClosed, cancel_and_wait
from .utils.clock import now
from .utils.emitter import EventEmitter
from .utils.ids import new_id
from .utils.log import logger

__all__ = [
    "LLM",
    "ChatChunk",
    "CompletionUsage",
    "LLMCapabilities",
    "LLMResult",
    "LLMStream",
    "ToolChoice",
]

ToolChoice: TypeAlias = Literal["auto", "required", "none"] | str
"""``"auto"``, ``"required"``, ``"none"`` or the name of a specific function."""


@dataclass(slots=True)
class CompletionUsage:
    prompt_tokens: int = 0
    """The whole prompt, including cache reads and writes."""
    completion_tokens: int = 0
    cached_tokens: int = 0
    """Prompt tokens read from the provider's prompt cache (discounted)."""
    cache_creation_tokens: int = 0
    """Prompt tokens written to the prompt cache (Anthropic: billed at a premium)."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ChatChunk:
    request_id: str
    delta: str = ""
    """Text delta."""
    tool_calls: list[FunctionCall] = field(default_factory=list)
    """Complete tool calls (emitted once each)."""
    usage: CompletionUsage | None = None
    finish_reason: str | None = None
    audio: AudioFrame | None = None
    """Audio delta of an audio-output model's speech (s16le, at
    ``LLMCapabilities.audio_sample_rate``); ``delta`` is its transcript."""
    audio_offset: float | None = None
    """Where ``delta`` starts being spoken, in seconds of the response's audio — for
    models that report how their text and audio interleave. ``None``: unknown (the text
    may run ahead of the audio), and truncation estimates what was heard."""


@dataclass(slots=True)
class LLMResult:
    text: str
    tool_calls: list[FunctionCall]
    usage: CompletionUsage | None


@dataclass(frozen=True, slots=True)
class LLMCapabilities:
    tool_calling: bool = True
    parallel_tool_calls: bool = True
    audio_input: bool = False
    """Accepts :class:`~voice_agent_next.chat.AudioContent` in user messages."""
    image_input: bool = False
    audio_output: bool = False
    """Streams its own speech in :attr:`ChatChunk.audio` (omni models: LFM2.5-Audio,
    gpt-audio, Qwen-Omni...): the cascade can run without a TTS."""
    audio_sample_rate: int = 24_000
    """Sample rate of :attr:`ChatChunk.audio` frames (``audio_output`` models)."""


class LLM(ABC, EventEmitter):
    """Base class for chat LLMs. Emits ``"metrics"`` (:class:`LLMMetrics`)."""

    provider: ClassVar[str] = "unknown"

    def __init__(
        self,
        *,
        model: str,
        capabilities: LLMCapabilities | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        EventEmitter.__init__(self)
        self.model = model
        self.capabilities = capabilities or LLMCapabilities()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        ctx: ChatContext,
        *,
        tools: Sequence[FunctionTool] = (),
        tool_choice: ToolChoice | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMStream:
        """Start a streamed completion over ``ctx``."""
        return self._chat(
            ctx,
            tools=list(tools),
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            extra=dict(extra or {}),
        )

    @abstractmethod
    def _chat(
        self,
        ctx: ChatContext,
        *,
        tools: list[FunctionTool],
        tool_choice: ToolChoice | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> LLMStream:
        """Return an :class:`LLMStream` for the request."""

    async def warmup(self) -> None:
        """Open connections / load the model ahead of the first request (optional)."""

    async def aclose(self) -> None:
        """Release resources."""

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class LLMStream(ABC):
    """A streamed completion. Async-iterate to receive :class:`ChatChunk` objects."""

    def __init__(
        self,
        llm: LLM,
        ctx: ChatContext,
        *,
        tools: list[FunctionTool],
        tool_choice: ToolChoice | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> None:
        self._llm = llm
        self.ctx = ctx
        self.tools = tools
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = extra
        self.request_id = new_id("llm_")
        self._events: Chan[ChatChunk] = Chan()
        self._error: BaseException | None = None
        self._start = now()
        self._first_token: float | None = None
        self._first_audio: float | None = None
        self._usage: CompletionUsage | None = None
        self._cancelled = False
        self._task = asyncio.create_task(self._main(), name=f"{type(self).__name__}._main")

    @abstractmethod
    async def _run(self) -> None:
        """Call the model and push chunks with :meth:`_push`."""

    def _push(self, chunk: ChatChunk) -> None:
        if self._first_token is None and (chunk.delta or chunk.tool_calls or chunk.audio):
            self._first_token = now()
        if self._first_audio is None and chunk.audio:
            self._first_audio = now()
        if chunk.usage is not None:
            self._usage = chunk.usage
        if not self._events.closed:
            self._events.send_nowait(chunk)

    async def _main(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            self._cancelled = True
            raise
        except Exception as exc:
            logger.exception("%s failed", type(self).__name__)
            self._error = exc
        finally:
            self._events.close()
            self._emit_metrics()

    def _emit_metrics(self) -> None:
        end = now()
        usage = self._usage or CompletionUsage()
        gen_time = end - (self._first_token or self._start)
        self._llm.emit(
            "metrics",
            LLMMetrics(
                provider=self._llm.provider,
                model=self._llm.model,
                request_id=self.request_id,
                ttft=None if self._first_token is None else self._first_token - self._start,
                ttfb=None if self._first_audio is None else self._first_audio - self._start,
                duration=end - self._start,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                cache_creation_tokens=usage.cache_creation_tokens,
                tokens_per_second=(usage.completion_tokens / gen_time) if gen_time > 0 else 0.0,
                cancelled=self._cancelled,
                error=None if self._error is None else repr(self._error),
            ),
        )

    async def collect(self) -> LLMResult:
        """Consume the stream and return the full text + tool calls."""
        text: list[str] = []
        calls: list[FunctionCall] = []
        async for chunk in self:
            text.append(chunk.delta)
            calls.extend(chunk.tool_calls)
        return LLMResult("".join(text), calls, self._usage)

    async def aclose(self) -> None:
        await cancel_and_wait(self._task)
        self._events.close()

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __anext__(self) -> ChatChunk:
        try:
            return await self._events.recv()
        except ChanClosed:
            if self._error is not None:
                err, self._error = self._error, None
                raise err from None
            raise StopAsyncIteration from None
