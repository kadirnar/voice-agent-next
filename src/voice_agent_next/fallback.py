"""Provider failover chains: :class:`FallbackSTT`, :class:`FallbackLLM`, :class:`FallbackTTS`.

A fallback wrapper *is* an STT/LLM/TTS, so it drops into a cascade anywhere a single
provider does::

    llm = FallbackLLM([create("llm", "groq/llama-3.3-70b-versatile"),
                       create("llm", "openai/gpt-4.1-mini")])
    # or simply
    llm = create("llm", ["groq/llama-3.3-70b-versatile", "openai/gpt-4.1-mini"])

Providers are tried in order. When one fails in a way another provider may not (network
error, timeout, rate limit, bad credentials, a stream that dies or stalls), the request
moves to the next one, the failed provider sits out a *cooldown* and is then tried again
(optionally earlier, when a background ``warmup()`` probe succeeds).

Switching is only done while it is invisible to the user:

* **LLM** — until the first token. After that the listener may already hear the start of
  the answer, and a second model would continue it with different words (or start over),
  so the error is surfaced instead and the cascade ends the turn.
* **TTS** — per segment, while none of the segment's audio was produced. Text that was
  not synthesized yet is replayed to the next provider.
* **STT** — at any time: the next provider's stream is started and the recent audio of
  the current utterance (a ring buffer) is replayed, so the utterance is not lost.

Each wrapper emits ``"metrics"`` (the providers' own metrics, so every request is
attributed to the provider that served it), ``"provider_failover"``
(:class:`ProviderFailover`) and ``"provider_availability_changed"``
(:class:`ProviderAvailabilityChanged`), and keeps counters in :attr:`stats`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, TypeVar

from .audio.frame import AudioFrame
from .audio.resample import StreamResampler
from .chat import ChatContext
from .errors import (
    AuthenticationError,
    ConfigurationError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    VoiceAgentError,
)
from .llm import LLM, ChatChunk, LLMCapabilities, LLMStream, ToolChoice
from .metrics import TTSMetrics
from .stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from .tools import FunctionTool
from .tts import (
    TTS,
    ChunkedStream,
    SentenceStreamAdapter,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
    _AudioEmitter,
)
from .utils.aio import BackgroundTasks, ChanClosed, cancel_and_wait, closed_outside
from .utils.clock import now
from .utils.emitter import EventEmitter
from .utils.log import logger

__all__ = [
    "FailoverStats",
    "FallbackLLM",
    "FallbackSTT",
    "FallbackTTS",
    "ProviderAvailabilityChanged",
    "ProviderFailover",
    "ProviderHealth",
    "is_failover_error",
]

Kind = Literal["stt", "llm", "tts"]
C = TypeVar("C", STT, LLM, TTS)
FailoverPredicate = Callable[[BaseException], bool]

# Status codes of requests that another provider may well accept.
_SWITCHABLE_STATUS = frozenset({401, 402, 403, 404, 408, 409, 425, 429})


def is_failover_error(exc: BaseException) -> bool:
    """Default policy: may the next provider succeed where this one failed?

    Yes for everything provider-specific — connection errors, timeouts, rate limits,
    bad credentials, 5xx, missing SDKs, and exceptions a provider did not map (usually
    raw network errors). No for a request the provider rejected as invalid (a
    non-retryable 4xx such as 400/422, which every provider would reject) and for the
    library's own configuration/usage errors.
    """
    if isinstance(
        exc,
        (
            AuthenticationError,
            RateLimitError,
            ProviderConnectionError,
            ProviderTimeoutError,
            MissingDependencyError,
        ),
    ):
        return True
    if isinstance(exc, ProviderError):
        code = exc.status_code
        return exc.retryable or code is None or code >= 500 or code in _SWITCHABLE_STATUS
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, VoiceAgentError):
        return False
    return not isinstance(exc, (NotImplementedError, TypeError, AssertionError))


def _reason(exc: BaseException) -> str:
    if isinstance(exc, (ProviderTimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, _StreamEnded):
        return "stream_ended"
    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, AuthenticationError):
        return "auth"
    if isinstance(exc, (ProviderConnectionError, ConnectionError, OSError)):
        return "connection"
    return "error"


class _StreamEnded(ProviderConnectionError):
    """A provider stream stopped before its input ended (e.g. a silent disconnect)."""


def _label(component: STT | LLM | TTS) -> str:
    return f"{component.provider}/{component.model}"


# ------------------------------------------------------------------------ events / stats
@dataclass(slots=True, kw_only=True)
class ProviderFailover:
    """Emitted as ``"provider_failover"`` when a request moves to the next provider."""

    kind: Kind
    from_provider: str
    """``"provider/model"`` that failed."""
    to_provider: str
    reason: str
    """``"timeout"``, ``"connection"``, ``"stream_ended"``, ``"rate_limit"``, ``"auth"``
    or ``"error"``."""
    error: BaseException
    request_id: str
    timestamp: float = field(default_factory=now)


@dataclass(slots=True, kw_only=True)
class ProviderAvailabilityChanged:
    """Emitted as ``"provider_availability_changed"`` when a provider goes down or recovers."""

    kind: Kind
    provider: str
    available: bool
    error: BaseException | None = None
    timestamp: float = field(default_factory=now)


@dataclass(slots=True)
class ProviderHealth:
    """Live health record of one provider in a chain."""

    provider: str
    available: bool = True
    """False from a failure until the provider succeeds again (a request or a probe)."""
    retry_at: float = 0.0
    """While unavailable: when the cooldown ends and requests may try it again."""
    served: int = 0
    """Requests this provider served."""
    failures: int = 0
    last_error: str | None = None


@dataclass(slots=True)
class FailoverStats:
    """Counters of a fallback wrapper (a snapshot, see :attr:`FallbackLLM.stats`)."""

    served: Counter[str] = field(default_factory=Counter)
    """Requests served, by ``"provider/model"``."""
    failures: Counter[str] = field(default_factory=Counter)
    """Failed attempts, by ``"provider/model"``."""
    failovers: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    """Failovers by reason."""


# ------------------------------------------------------------------------------ chain
class _Chain(Generic[C]):
    """Ordering, health, cooldown, probing and bookkeeping shared by the wrappers."""

    def __init__(
        self,
        owner: EventEmitter,
        kind: Kind,
        providers: Sequence[C],
        *,
        cooldown: float,
        probe_interval: float | None,
        probe_timeout: float,
        failover_on: FailoverPredicate | None,
    ) -> None:
        if not providers:
            raise ConfigurationError(f"a fallback {kind} needs at least one provider")
        self.owner = owner
        self.kind = kind
        self.providers: list[C] = list(providers)
        self.labels = [_label(p) for p in self.providers]
        self.health = [ProviderHealth(label) for label in self.labels]
        self.cooldown = cooldown
        self.probe_interval = probe_interval
        self.probe_timeout = probe_timeout
        self.should_failover = failover_on or is_failover_error
        self.failovers = 0
        self.reasons: Counter[str] = Counter()
        self._tasks = BackgroundTasks(f"fallback-{kind}")
        self._probe: asyncio.Task[None] | None = None
        for p in self.providers:
            p.on("metrics", self._forward_metrics)

    def _forward_metrics(self, m: Any) -> None:
        self.owner.emit("metrics", m)

    # ------------------------------------------------------------------ selection
    def _usable(self, i: int) -> bool:
        h = self.health[i]
        return h.available or now() >= h.retry_at

    def next_candidate(self, tried: set[int]) -> int | None:
        """The first usable provider not tried yet; else the one whose cooldown ends first.

        Providers in cooldown are still tried as a last resort: failing over to a
        provider that is probably down beats failing the request outright.
        """
        rest = [i for i in range(len(self.providers)) if i not in tried]
        if not rest:
            return None
        usable = [i for i in rest if self._usable(i)]
        if usable:
            return usable[0]
        return min(rest, key=lambda i: self.health[i].retry_at)

    # ------------------------------------------------------------------ bookkeeping
    def succeeded(self, i: int, *, served: bool = True) -> None:
        h = self.health[i]
        if served:
            h.served += 1
        if not h.available:
            h.available = True
            self.owner.emit(
                "provider_availability_changed",
                ProviderAvailabilityChanged(kind=self.kind, provider=h.provider, available=True),
            )

    def failed(self, i: int, exc: BaseException) -> None:
        h = self.health[i]
        h.failures += 1
        h.last_error = repr(exc)
        h.retry_at = now() + self.cooldown
        logger.warning("fallback %s: %s failed: %r", self.kind, h.provider, exc)
        if h.available:
            h.available = False
            self.owner.emit(
                "provider_availability_changed",
                ProviderAvailabilityChanged(
                    kind=self.kind, provider=h.provider, available=False, error=exc
                ),
            )
        self._start_probe()

    def failover(self, src: int, dst: int, exc: BaseException, request_id: str) -> None:
        reason = _reason(exc)
        self.failovers += 1
        self.reasons[reason] += 1
        self.owner.emit(
            "provider_failover",
            ProviderFailover(
                kind=self.kind,
                from_provider=self.labels[src],
                to_provider=self.labels[dst],
                reason=reason,
                error=exc,
                request_id=request_id,
            ),
        )

    def stats(self) -> FailoverStats:
        return FailoverStats(
            served=Counter({h.provider: h.served for h in self.health if h.served}),
            failures=Counter({h.provider: h.failures for h in self.health if h.failures}),
            failovers=self.failovers,
            reasons=Counter(self.reasons),
        )

    # ------------------------------------------------------------------ probing
    def _start_probe(self) -> None:
        if self.probe_interval is None or (self._probe is not None and not self._probe.done()):
            return
        try:
            self._probe = self._tasks.spawn(self._probe_loop(), name=f"fallback-{self.kind}-probe")
        except RuntimeError:  # no running loop (failure reported from sync code)
            self._probe = None

    async def _probe_loop(self) -> None:
        assert self.probe_interval is not None
        while True:
            await asyncio.sleep(self.probe_interval)
            down = [i for i, h in enumerate(self.health) if not h.available]
            if not down:
                return
            for i in down:
                try:
                    await asyncio.wait_for(self.providers[i].warmup(), self.probe_timeout)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.health[i].retry_at = now() + self.cooldown
                    self.health[i].last_error = repr(exc)
                    continue
                self.succeeded(i, served=False)

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Warm every provider; a failure only marks that provider down (unless all fail)."""
        results = await asyncio.gather(
            *(p.warmup() for p in self.providers), return_exceptions=True
        )
        errors = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
        for i, err in errors:
            self.failed(i, err)
        if len(errors) == len(self.providers):
            raise errors[0][1]

    async def aclose(self) -> None:
        await self._tasks.cancel_all()
        await asyncio.gather(*(p.aclose() for p in self.providers), return_exceptions=True)


def _all_caps(cls: type[Any], caps: Sequence[Any]) -> Any:
    """The capabilities every provider has: the intersection of boolean flags; other fields
    (e.g. an audio sample rate) keep the value all providers share, else the default."""
    default = cls()

    def merge(name: str) -> Any:
        values = [getattr(c, name) for c in caps]
        if all(isinstance(v, bool) for v in values):
            return all(values)
        return (
            values[0] if values and all(v == values[0] for v in values) else getattr(default, name)
        )

    return cls(**{f.name: merge(f.name) for f in dataclasses.fields(cls)})


def _no_providers(kind: str) -> ProviderError:
    return ProviderError(f"fallback {kind}: no provider left to try", provider="fallback")


async def _next_with_timeout(it: AsyncIterator[Any], timeout: float | None) -> Any:
    if timeout is None:
        return await it.__anext__()
    return await asyncio.wait_for(it.__anext__(), max(0.0, timeout))


# ------------------------------------------------------------------------------ LLM
class FallbackLLM(LLM):
    """An :class:`LLM` that fails over between providers, in order, before the first token.

    Args:
        providers: LLMs in priority order.
        first_token_timeout: seconds to wait for the first token (text or tool call)
            before trying the next provider; catches silent stalls. ``None`` disables it.
        cooldown: seconds a failed provider is skipped before being tried again.
        probe_interval: when set, unavailable providers are probed with ``warmup()``
            this often and come back as soon as a probe succeeds.
        probe_timeout: timeout of one probe.
        failover_on: predicate deciding which errors fail over
            (default :func:`is_failover_error`).

    Errors after the first token are not failed over (the user may already be hearing
    the answer); they propagate, and the provider is still put into cooldown so the
    next request starts on a healthy one.
    """

    provider: ClassVar[str] = "fallback"

    def __init__(
        self,
        providers: Sequence[LLM],
        *,
        first_token_timeout: float | None = 10.0,
        cooldown: float = 30.0,
        probe_interval: float | None = None,
        probe_timeout: float = 10.0,
        failover_on: FailoverPredicate | None = None,
    ) -> None:
        super().__init__(model="|".join(_label(p) for p in providers))
        self._chain: _Chain[LLM] = _Chain(
            self,
            "llm",
            providers,
            cooldown=cooldown,
            probe_interval=probe_interval,
            probe_timeout=probe_timeout,
            failover_on=failover_on,
        )
        self.capabilities = _all_caps(LLMCapabilities, [p.capabilities for p in providers])
        self.first_token_timeout = first_token_timeout

    @property
    def providers(self) -> list[LLM]:
        return self._chain.providers

    @property
    def health(self) -> list[ProviderHealth]:
        return self._chain.health

    @property
    def stats(self) -> FailoverStats:
        return self._chain.stats()

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
        return _FallbackLLMStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    async def warmup(self) -> None:
        await self._chain.warmup()

    async def aclose(self) -> None:
        await self._chain.aclose()


class _FallbackLLMStream(LLMStream):
    served_by: str | None = None
    """``"provider/model"`` that produced the answer."""

    async def _run(self) -> None:
        fb: FallbackLLM = self._llm  # type: ignore[assignment]
        chain = fb._chain
        tried: set[int] = set()
        prev: int | None = None
        last: BaseException | None = None
        while True:
            i = chain.next_candidate(tried)
            if i is None:
                raise last or _no_providers("llm")
            if prev is not None and last is not None:
                chain.failover(prev, i, last, self.request_id)
            tried.add(i)
            committed = [False]
            try:
                await self._attempt(i, fb, committed)
                return
            except Exception as exc:
                if not chain.should_failover(exc):
                    raise
                chain.failed(i, exc)
                if committed[0]:
                    raise  # tokens were already streamed: never switch mid-answer
                prev, last = i, exc

    async def _attempt(self, i: int, fb: FallbackLLM, committed: list[bool]) -> None:
        chain = fb._chain
        stream = chain.providers[i].chat(
            self.ctx,
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra=self.extra,
        )
        timeout = fb.first_token_timeout
        deadline = None if timeout is None else now() + timeout
        held: list[ChatChunk] = []  # content-less chunks seen before the first token

        def commit() -> None:
            committed[0] = True
            chain.succeeded(i)
            self.served_by = chain.labels[i]
            for chunk in held:
                self._push(dataclasses.replace(chunk, request_id=self.request_id))
            held.clear()

        try:
            it = stream.__aiter__()
            while True:
                try:
                    if committed[0] or deadline is None:
                        chunk = await it.__anext__()
                    else:
                        chunk = await _next_with_timeout(it, deadline - now())
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise ProviderTimeoutError(
                        f"no first token within {timeout:g}s", provider=chain.labels[i]
                    ) from None
                if not committed[0]:
                    if not (chunk.delta or chunk.tool_calls or chunk.audio):
                        held.append(chunk)
                        continue
                    commit()
                self._push(dataclasses.replace(chunk, request_id=self.request_id))
            if not committed[0]:
                commit()  # a legitimately empty answer
        finally:
            if not closed_outside(self._task):  # being garbage-collected: can't await
                await stream.aclose()

    def _emit_metrics(self) -> None:
        """The providers' own metrics are forwarded instead (they name who served)."""


# ------------------------------------------------------------------------------ TTS
class FallbackTTS(TTS):
    """A :class:`TTS` that fails over between providers while nothing has been heard.

    Output is resampled to one rate (``sample_rate``, default: the first provider's).
    ``capabilities`` are the intersection of the providers' capabilities: with
    ``streaming=False`` the wrapper streams sentence by sentence and every sentence
    fails over on its own; with native streaming on all providers, the text of the
    segments not synthesized yet is replayed to the next provider.

    A segment whose audio has started is never switched mid-way (the listener would
    hear a different voice restart the sentence); the error propagates.

    Voices are provider-specific: configure each provider's ``voice`` rather than
    passing one to the wrapper (a voice passed explicitly is forwarded to all).

    Args:
        providers: TTS engines in priority order.
        first_audio_timeout: seconds from a request (or segment flush) to its first
            audio before trying the next provider. ``None`` disables it.
        cooldown, probe_interval, probe_timeout, failover_on: see :class:`FallbackLLM`.
    """

    provider: ClassVar[str] = "fallback"

    def __init__(
        self,
        providers: Sequence[TTS],
        *,
        sample_rate: int | None = None,
        first_audio_timeout: float | None = 5.0,
        cooldown: float = 30.0,
        probe_interval: float | None = None,
        probe_timeout: float = 10.0,
        failover_on: FailoverPredicate | None = None,
    ) -> None:
        if not providers:
            raise ConfigurationError("a fallback tts needs at least one provider")
        first = providers[0]
        super().__init__(
            model="|".join(_label(p) for p in providers),
            sample_rate=sample_rate or first.sample_rate,
            channels=first.channels,
            capabilities=_all_caps(TTSCapabilities, [p.capabilities for p in providers]),
            clean_text=False,  # each provider cleans with its own settings
            trim_silence=all(p.trim_silence for p in providers),
        )
        self._chain: _Chain[TTS] = _Chain(
            self,
            "tts",
            providers,
            cooldown=cooldown,
            probe_interval=probe_interval,
            probe_timeout=probe_timeout,
            failover_on=failover_on,
        )
        self.first_audio_timeout = first_audio_timeout

    @property
    def providers(self) -> list[TTS]:
        return self._chain.providers

    @property
    def health(self) -> list[ProviderHealth]:
        return self._chain.health

    @property
    def stats(self) -> FailoverStats:
        return self._chain.stats()

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _FallbackChunkedStream(self, text, voice=voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        return _FallbackSynthesizeStream(self, voice=voice)

    def stream(self, *, voice: str | None = None) -> SynthesizeStream:
        if self.capabilities.streaming:
            return super().stream(voice=voice)
        # sentence by sentence, with the usage attributed to the provider of each sentence
        return _FallbackSentenceStream(self, voice=voice or self.voice)

    async def warmup(self) -> None:
        await self._chain.warmup()

    async def aclose(self) -> None:
        await self._chain.aclose()


def _shift(words: list[WordTiming] | None, offset: float) -> list[WordTiming] | None:
    if not words or not offset:
        return words or None
    return [WordTiming(w.word, w.start + offset, w.end + offset, w.confidence) for w in words]


def _forward_audio(
    s: _AudioEmitter, rs: StreamResampler, frame: AudioFrame, words: list[WordTiming] | None
) -> None:
    """Forward a provider's audio (and word timings) onto the wrapper stream, resampled."""
    out = rs.push(frame) if frame else frame
    if out:
        s._push_audio(out, words=words)
    elif words:
        empty = AudioFrame.empty(s._tts.sample_rate, s._tts.channels)
        s._send(SynthesizedAudio(empty, s._request_id, s._segment_id, words=words))


class _FallbackChunkedStream(ChunkedStream):
    served_by: str | None = None
    served_index: int | None = None
    """Position in the chain of the provider that served the request."""

    async def _run(self) -> None:
        fb: FallbackTTS = self._tts  # type: ignore[assignment]
        chain = fb._chain
        tried: set[int] = set()
        prev: int | None = None
        last: BaseException | None = None
        while True:
            i = chain.next_candidate(tried)
            if i is None:
                raise last or _no_providers("tts")
            if prev is not None and last is not None:
                chain.failover(prev, i, last, self._request_id)
            tried.add(i)
            heard = [False]
            try:
                await self._attempt(i, fb, heard)
                return
            except Exception as exc:
                if not chain.should_failover(exc):
                    raise
                chain.failed(i, exc)
                if heard[0]:
                    raise
                prev, last = i, exc

    async def _attempt(self, i: int, fb: FallbackTTS, heard: list[bool]) -> None:
        chain = fb._chain
        inner = chain.providers[i].synthesize(self.text, voice=self.voice)
        # quiet when a wrapper (the sentence adapter) reports this request's usage itself
        inner._metrics_enabled = self._metrics_enabled
        rs = StreamResampler(fb.sample_rate, fb.channels)
        timeout = fb.first_audio_timeout
        deadline = None if timeout is None else now() + timeout
        try:
            it = inner.__aiter__()
            while True:
                try:
                    if heard[0] or deadline is None:
                        item = await it.__anext__()
                    else:
                        item = await _next_with_timeout(it, deadline - now())
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise ProviderTimeoutError(
                        f"no audio within {timeout:g}s", provider=chain.labels[i]
                    ) from None
                if item.frame and not heard[0]:
                    heard[0] = True
                    chain.succeeded(i)
                    self.served_by, self.served_index = chain.labels[i], i
                _forward_audio(self, rs, item.frame, item.words)
            tail = rs.flush()
            if tail:
                self._push_audio(tail)
            if not heard[0]:
                chain.succeeded(i)  # e.g. empty text: nothing to say
                self.served_by, self.served_index = chain.labels[i], i
        finally:
            if not closed_outside(self._task):  # being garbage-collected: can't await
                await inner.aclose()

    def _emit_metrics(self) -> None:
        """The providers' own metrics are forwarded instead."""


@dataclass
class _Served:
    """Usage of one provider within a sentence-by-sentence request."""

    characters: int = 0
    audio_duration: float = 0.0


class _FallbackSentenceStream(SentenceStreamAdapter):
    """The sentence adapter over a :class:`FallbackTTS`: sentences may be served by
    different providers, so the request's usage is reported per serving provider (one
    :class:`TTSMetrics` each, in the order they first served) instead of under
    ``"fallback"``. The characters that belong to no sentence (separators, text of
    sentences no provider could serve) go to the first provider, so the total is still
    the text pushed, counted once."""

    def __init__(self, tts: FallbackTTS, *, voice: str | None) -> None:
        self._served: dict[int, _Served] = {}
        super().__init__(tts, voice=voice)

    async def _play_sentence(self, stream: ChunkedStream, on_chunk: Callable[[], None]) -> None:
        start = self._audio_duration
        try:
            await super()._play_sentence(stream, on_chunk)
        finally:
            i = getattr(stream, "served_index", None)
            if i is not None:
                served = self._served.setdefault(i, _Served())
                served.characters += len(stream.text)
                served.audio_duration += self._audio_duration - start

    def _emit_metrics(self) -> None:
        if not self._metrics_enabled or not self._served:
            super()._emit_metrics()  # nothing was served: reported as the fallback chain
            return
        fb: FallbackTTS = self._tts  # type: ignore[assignment]
        ttfb = None
        if self._first_audio_time is not None:
            ttfb = self._first_audio_time - (self._first_text_time or self._start_time)
        served = list(self._served.items())
        unattributed = self._characters - sum(u.characters for _, u in served)
        audio_left = self._audio_duration - sum(u.audio_duration for _, u in served)
        duration = now() - self._start_time
        for k, (i, u) in enumerate(served):
            first, last = k == 0, k == len(served) - 1
            provider = fb.providers[i]
            fb.emit(
                "metrics",
                TTSMetrics(
                    provider=provider.provider,
                    model=provider.model,
                    request_id=self._request_id,
                    ttfb=ttfb if first else None,
                    duration=duration,
                    audio_duration=u.audio_duration + (audio_left if first else 0.0),
                    characters=u.characters + (max(0, unattributed) if first else 0),
                    streamed=True,
                    cancelled=self._cancelled and last,
                    error=None if self._error is None or not last else repr(self._error),
                ),
            )


class _FallbackSynthesizeStream(SynthesizeStream):
    """Native streaming failover with replay of the segments not synthesized yet."""

    served_by: str | None = None

    def __init__(self, tts: FallbackTTS, *, voice: str | None) -> None:
        # text and flush markers not yet fully synthesized, oldest first
        self._pending: list[Any] = []
        self._input_done = False
        self._seg_heard = False
        self._stall: asyncio.Future[None] | None = None
        self._timer: asyncio.TimerHandle | None = None
        super().__init__(tts, voice=voice)

    # ------------------------------------------------------------------ watchdog
    def _arm(self) -> None:
        fb: FallbackTTS = self._tts  # type: ignore[assignment]
        if fb.first_audio_timeout is None or self._timer is not None or self._stall is None:
            return
        stall = self._stall

        def fire() -> None:
            if not stall.done():
                stall.set_result(None)

        self._timer = asyncio.get_running_loop().call_later(fb.first_audio_timeout, fire)

    def _disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _flushes_pending(self) -> bool:
        return any(self.is_flush(x) for x in self._pending)

    # ------------------------------------------------------------------ main
    async def _run(self) -> None:
        fb: FallbackTTS = self._tts  # type: ignore[assignment]
        chain = fb._chain
        tried: set[int] = set()
        prev: int | None = None
        last: BaseException | None = None
        while True:
            i = chain.next_candidate(tried)
            if i is None:
                raise last or _no_providers("tts")
            if prev is not None and last is not None:
                chain.failover(prev, i, last, self._request_id)
            tried.add(i)
            try:
                await self._attempt(i, fb)
                return
            except Exception as exc:
                if not chain.should_failover(exc):
                    raise
                chain.failed(i, exc)
                if self._seg_heard:
                    raise  # part of this segment was already heard
                prev, last = i, exc

    async def _attempt(self, i: int, fb: FallbackTTS) -> None:
        chain = fb._chain
        inner = chain.providers[i].stream(voice=self.voice)
        self._stall = asyncio.get_running_loop().create_future()
        for item in self._pending:  # replay what the failed provider did not deliver
            if self.is_flush(item):
                inner.flush()
            else:
                inner.push_text(item)
        if self._flushes_pending():
            self._arm()
        if self._input_done:
            inner._input.close()
        succeeded = [False]

        async def pump() -> None:
            while not self._input_done:
                try:
                    item = await self._input.recv()
                except ChanClosed:
                    self._input_done = True
                    # end_input() already queued (and we forwarded) the final flush
                    inner._input.close()
                    return
                self._pending.append(item)
                if self.is_flush(item):
                    inner.flush()
                    self._arm()
                else:
                    assert isinstance(item, str)
                    inner.push_text(item)

        async def forward() -> None:
            offset = self._audio_duration  # where this provider's audio starts on our stream
            rs = StreamResampler(fb.sample_rate, fb.channels)
            async for a in inner:
                if a.text and self._segment_text is None:
                    self._segment_text = a.text
                if a.frame and not self._seg_heard:
                    self._seg_heard = True
                    self._disarm()
                    if not succeeded[0]:
                        succeeded[0] = True
                        chain.succeeded(i)
                        self.served_by = chain.labels[i]
                _forward_audio(self, rs, a.frame, _shift(a.words, offset))
                if a.is_final:
                    # the provider's stream goes on: keep the filter history so the next
                    # segment continues seamlessly and the sample count never drifts
                    tail = rs.drain()
                    if tail:
                        self._push_audio(tail)
                    self._end_segment()
                    self._seg_heard = False
                    self._disarm()
                    flush_at = next(
                        (k for k, x in enumerate(self._pending) if self.is_flush(x)), None
                    )
                    if flush_at is not None:
                        del self._pending[: flush_at + 1]
                    if self._flushes_pending() or (self._input_done and self._pending):
                        self._arm()
            if not self._input_done or self._pending:
                raise _StreamEnded(
                    "TTS stream ended before synthesizing all its text", provider=chain.labels[i]
                )
            if not succeeded[0]:
                chain.succeeded(i)
                self.served_by = chain.labels[i]

        pump_task = asyncio.create_task(pump(), name="fallback-tts-pump")
        fwd_task = asyncio.create_task(forward(), name="fallback-tts-forward")
        try:
            waiting: set[asyncio.Future[Any]] = {fwd_task, self._stall, pump_task}
            while True:
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if fwd_task in done:
                    fwd_task.result()
                    return
                if self._stall in done:
                    raise ProviderTimeoutError(
                        f"no audio within {fb.first_audio_timeout:g}s", provider=chain.labels[i]
                    )
                pump_task.result()  # input finished: keep waiting for the audio
                waiting.discard(pump_task)
        finally:
            self._disarm()
            if not closed_outside(self._task):  # being garbage-collected: can't await
                await cancel_and_wait(pump_task, fwd_task)
                await inner.aclose()

    def _emit_metrics(self) -> None:
        """The providers' own metrics are forwarded instead."""


# ------------------------------------------------------------------------------ STT
class FallbackSTT(STT):
    """An :class:`STT` that fails over between providers, replaying recent audio.

    Streaming: when the current provider's stream fails (an error, a stream that ends
    on its own, or no final transcript within ``final_timeout`` of a flush), the next
    provider's stream is started and the audio of the current utterance — kept in a
    ring buffer of at most ``replay_seconds`` — is pushed into it, so the utterance is
    still transcribed. Audio already covered by a final transcript is not replayed.

    ``capabilities`` are the intersection of the providers'. When some provider is
    batch-only, the wrapper is batch-only too (the cascade then streams it through a
    VAD with :class:`~voice_agent_next.stt.StreamAdapter`), and each utterance fails
    over on its own.

    Args:
        providers: recognizers in priority order.
        replay_seconds: ring-buffer length.
        final_timeout: seconds from a flush to the final transcript (and for a batch
            recognition) before failing over. ``None`` (default) disables it: not every
            provider sends a final for a flush without speech.
        cooldown, probe_interval, probe_timeout, failover_on: see :class:`FallbackLLM`.
    """

    provider: ClassVar[str] = "fallback"

    def __init__(
        self,
        providers: Sequence[STT],
        *,
        sample_rate: int | None = None,
        language: str | None = None,
        replay_seconds: float = 5.0,
        final_timeout: float | None = None,
        cooldown: float = 30.0,
        probe_interval: float | None = None,
        probe_timeout: float = 10.0,
        failover_on: FailoverPredicate | None = None,
    ) -> None:
        if not providers:
            raise ConfigurationError("a fallback stt needs at least one provider")
        super().__init__(
            model="|".join(_label(p) for p in providers),
            capabilities=_all_caps(STTCapabilities, [p.capabilities for p in providers]),
            sample_rate=sample_rate or providers[0].sample_rate,
            language=language,
        )
        self._chain: _Chain[STT] = _Chain(
            self,
            "stt",
            providers,
            cooldown=cooldown,
            probe_interval=probe_interval,
            probe_timeout=probe_timeout,
            failover_on=failover_on,
        )
        self.replay_seconds = replay_seconds
        self.final_timeout = final_timeout

    @property
    def providers(self) -> list[STT]:
        return self._chain.providers

    @property
    def health(self) -> list[ProviderHealth]:
        return self._chain.health

    @property
    def stats(self) -> FailoverStats:
        return self._chain.stats()

    async def transcribe(
        self, audio: AudioFrame | Sequence[AudioFrame], *, language: str | None = None
    ) -> Transcript:
        # no resampling / metrics here: each provider does both itself
        frame = audio if isinstance(audio, AudioFrame) else AudioFrame.concat(audio)
        return await self._recognize(frame, language=language or self.language)

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        chain = self._chain
        tried: set[int] = set()
        prev: int | None = None
        last: BaseException | None = None
        while True:
            i = chain.next_candidate(tried)
            if i is None:
                raise last or _no_providers("stt")
            if prev is not None and last is not None:
                chain.failover(prev, i, last, "")
            tried.add(i)
            try:
                coro = chain.providers[i].transcribe(audio, language=language)
                if self.final_timeout is None:
                    result = await coro
                else:
                    try:
                        result = await asyncio.wait_for(coro, self.final_timeout)
                    except TimeoutError:
                        raise ProviderTimeoutError(
                            f"no transcript within {self.final_timeout:g}s",
                            provider=chain.labels[i],
                        ) from None
            except Exception as exc:
                if not chain.should_failover(exc):
                    raise
                chain.failed(i, exc)
                prev, last = i, exc
                continue
            chain.succeeded(i)
            return result

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _FallbackSTTStream(self, language=language)

    async def warmup(self) -> None:
        await self._chain.warmup()

    async def aclose(self) -> None:
        await self._chain.aclose()


class _FallbackSTTStream(STTStream):
    served_by: str | None = None
    """``"provider/model"`` currently transcribing."""

    def __init__(self, stt: FallbackSTT, *, language: str | None) -> None:
        # (sequence number, frame) of recent audio not yet covered by a final transcript
        self._ring: deque[tuple[int, AudioFrame]] = deque()
        self._ring_duration = 0.0
        self._seq = 0
        self._flush_seq: int | None = None
        """Last frame before a flush whose final transcript has not arrived yet."""
        self._input_done = False
        self._in_speech = False
        super().__init__(stt, language=language)

    def _remember(self, frame: AudioFrame, max_seconds: float) -> None:
        self._seq += 1
        self._ring.append((self._seq, frame))
        self._ring_duration += frame.duration
        while self._ring and self._ring_duration > max_seconds:
            _, old = self._ring.popleft()
            self._ring_duration -= old.duration

    def _forget_through(self, seq: int) -> None:
        while self._ring and self._ring[0][0] <= seq:
            _, old = self._ring.popleft()
            self._ring_duration -= old.duration

    def _emit(self, event: STTEvent) -> None:
        # the providers emit their own metrics; only forward the event
        if not self._events.closed:
            self._events.send_nowait(event)

    async def _run(self) -> None:
        fb: FallbackSTT = self._stt  # type: ignore[assignment]
        chain = fb._chain
        tried: set[int] = set()
        prev: int | None = None
        last: BaseException | None = None
        while True:
            i = chain.next_candidate(tried)
            if i is None:
                raise last or _no_providers("stt")
            if prev is not None and last is not None:
                chain.failover(prev, i, last, self._request_id)
            tried.add(i)
            try:
                await self._attempt(i, fb, tried)
                return
            except Exception as exc:
                if not chain.should_failover(exc):
                    raise
                chain.failed(i, exc)
                prev, last = i, exc

    async def _attempt(self, i: int, fb: FallbackSTT, tried: set[int]) -> None:
        chain = fb._chain
        inner = chain.providers[i].stream(language=self._language)
        self.served_by = chain.labels[i]
        stall: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        timer: list[asyncio.TimerHandle | None] = [None]

        def arm() -> None:
            if fb.final_timeout is None or timer[0] is not None:
                return

            def fire() -> None:
                if not stall.done():
                    stall.set_result(None)

            timer[0] = asyncio.get_running_loop().call_later(fb.final_timeout, fire)

        def disarm() -> None:
            if timer[0] is not None:
                timer[0].cancel()
                timer[0] = None

        # replay the current utterance (and a flush that is still waiting for its final)
        for seq, frame in self._ring:
            inner.push_audio(frame)
            if seq == self._flush_seq:
                inner.flush()
                arm()
        if self._input_done:
            inner.end_input()

        async def pump() -> None:
            while not self._input_done:
                try:
                    item = await self._input.recv()
                except ChanClosed:
                    self._input_done = True
                    inner.end_input()
                    return
                if self.is_flush(item):
                    if self._ring:
                        self._flush_seq = self._seq
                        arm()
                    inner.flush()
                else:
                    assert isinstance(item, AudioFrame)
                    self._remember(item, fb.replay_seconds)
                    inner.push_audio(item)

        async def forward() -> None:
            succeeded = False
            async for ev in inner:
                if not succeeded:
                    succeeded = True
                    chain.succeeded(i, served=False)
                if ev.type == STTEventType.START_OF_SPEECH:
                    if self._in_speech:
                        continue  # the replayed utterance was already announced
                    self._in_speech = True
                elif ev.type == STTEventType.END_OF_SPEECH:
                    self._in_speech = False
                elif ev.type == STTEventType.FINAL_TRANSCRIPT:
                    disarm()
                    # this audio is transcribed: never replay it
                    self._forget_through(
                        self._flush_seq if self._flush_seq is not None else self._seq
                    )
                    self._flush_seq = None
                    chain.health[i].served += 1
                    tried.clear()  # the provider works: later failures start a new round
                    tried.add(i)
                self._emit(ev)
            if not self._input_done:
                raise _StreamEnded("STT stream ended before its input", provider=chain.labels[i])
            if not succeeded:
                chain.succeeded(i, served=False)

        pump_task = asyncio.create_task(pump(), name="fallback-stt-pump")
        fwd_task = asyncio.create_task(forward(), name="fallback-stt-forward")
        try:
            waiting: set[asyncio.Future[Any]] = {fwd_task, stall, pump_task}
            while True:
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if fwd_task in done:
                    fwd_task.result()
                    return
                if stall in done:
                    raise ProviderTimeoutError(
                        f"no final transcript within {fb.final_timeout:g}s of a flush",
                        provider=chain.labels[i],
                    )
                pump_task.result()
                waiting.discard(pump_task)
        finally:
            disarm()
            if not closed_outside(self._task):  # being garbage-collected: can't await
                await cancel_and_wait(pump_task, fwd_task)
                with contextlib.suppress(Exception):
                    await inner.aclose()
