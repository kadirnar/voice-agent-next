"""Cascaded speech-to-speech engine: VAD -> STT -> (turn detector) -> LLM -> TTS.

Exposes the same :class:`~voice_agent_next.engine.EngineConnection` interface as native
speech-to-speech models, so the session treats both identically.

Pipeline per user turn:

1. user audio is streamed continuously into the STT stream and the VAD;
2. VAD ``START_OF_SPEECH`` -> ``InputSpeechStarted`` (the session may barge in);
3. VAD ``END_OF_SPEECH`` -> the STT is flushed, the (optional) turn detector scores the
   utterance, and the turn is committed after the endpointing delay unless speech
   resumes: ``min_endpointing_delay`` (confident) or ``max_endpointing_delay`` (user
   probably not done), or — ``endpointing="dynamic"`` / ``dictation`` — a delay chosen
   from the detector's confidence and the user's learned pauses (``endpointing.py``);
4. the LLM streams text; complete sentences are cleaned and pushed into a TTS stream,
   whose audio is emitted as ``ResponseAudio`` with text aligned per sentence.

Preemptive generation (``CascadeOptions.preemptive_generation``): once the turn has
*probably* ended — step 3 knows the final transcript and only the endpointing delay is
left, or the STT sends ``EAGER_END_OF_TURN`` — step 4 starts speculatively. Its output is
held back and released when the turn is committed with the same transcript and context,
or discarded when the user resumes (see ``docs/concepts/preemptive-generation.md``).

Half-cascade: with ``stt=None`` and an LLM whose ``capabilities.audio_input`` is True
(Ultravox, Qwen-Omni, Gemini, gpt-4o-audio...), the user's audio is passed to the LLM
directly as :class:`~voice_agent_next.chat.AudioContent`. ``CascadeOptions.input_transcriber``
adds the user's words to the history (and ``AudioContent.transcript``) after the commit,
from the audio LLM itself or from a batch STT, without delaying the reply.

Omni models: an LLM whose ``capabilities.audio_output`` is True (LFM2.5-Audio, gpt-audio,
Qwen-Omni...) speaks for itself. Without a TTS (or with ``CascadeOptions.use_llm_audio``)
its audio deltas are played as ``ResponseAudio`` and its text becomes the ``ResponseText``
transcript; with ``stt=None`` too, VAD + turn detection is all that is left of the
cascade (see ``docs/concepts/omni-models.md``).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..audio.buffer import AudioBuffer
from ..audio.frame import AudioFrame
from ..chat import (
    AudioContent,
    ChatContext,
    ChatItem,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
)
from ..engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ..errors import ConfigurationError
from ..events import (
    EngineErrorEvent,
    EngineEvent,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseStatus,
    ResponseText,
    ResponseToolCall,
)
from ..llm import LLM, ChatChunk, CompletionUsage, LLMStream
from ..metrics import (
    EndpointingMetrics,
    EngineMetrics,
    Metrics,
    SpeculationMetrics,
    SpeculationReason,
)
from ..registry import create
from ..stt import STT, StreamAdapter, STTEventType, WordTiming
from ..text.filters import tts_clean
from ..text.sentences import SentenceSegmenter
from ..tools import FunctionTool
from ..tts import TTS
from ..turn import FusedTurnDetector, TurnDetector
from ..utils.aio import BackgroundTasks, Chan, ChanClosed, cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger
from ..vad import VAD, VADEventType
from .endpointing import Endpointer, EndpointingDecision, EndpointingMode

__all__ = ["CascadeConnection", "CascadeEngine", "CascadeOptions"]


@dataclass(slots=True)
class CascadeOptions:
    min_endpointing_delay: float | None = None
    """Silence (from the end of speech) before committing when the user seems done.
    ``None`` = 0.4 s with a turn detector, 0.6 s with VAD only. With
    ``endpointing="dynamic"``: the lower bound (``None`` = 0.25 s / 0.3 s)."""
    max_endpointing_delay: float = 2.5
    """Silence (from the end of speech) before committing when the turn detector says the
    user is probably not done (the upper bound of the dynamic policy)."""
    endpointing: EndpointingMode = "fixed"
    """``"fixed"``: ``min_endpointing_delay`` or ``max_endpointing_delay``. ``"dynamic"``:
    the delay follows the turn detector's confidence (short when it is sure the user is
    done, long when it is sure they are not) and the user's own mid-turn pauses, learned
    during the session. See ``docs/concepts/endpointing.md``."""
    pause_deviations: float = 2.0
    """Dynamic policy: the undecided-pause delay is the user's mean mid-turn pause plus
    this many mean deviations."""
    pause_alpha: float = 0.25
    """Dynamic policy: weight of each new pause in the running mean and deviation."""
    false_commit_window: float = 1.0
    """A commit counts as false (the user was cut off) when they speak again within this
    many seconds; the pause is then learned as a mid-turn pause."""
    dictation: bool = False
    """Dictation mode (numbers, addresses, notes): only a confident turn detector commits
    before ``dictation_max_delay``, never before ``dictation_min_delay``. Switch at runtime
    with ``AgentSession.update_endpointing(dictation=...)``."""
    dictation_min_delay: float = 1.0
    """Dictation: silence before committing when the turn detector says the user is done."""
    dictation_max_delay: float = 5.0
    """Dictation: silence before committing otherwise (or without a turn detector)."""
    dictation_threshold: float | None = None
    """Dictation: turn-detector probability needed to commit at ``dictation_min_delay``
    (``None`` = the detector's own ``threshold``)."""
    final_transcript_timeout: float = 1.0
    """Max wait for the STT's final transcript after flushing."""
    text_filter: Callable[[str], str] | None = tts_clean
    """Applied to each sentence before TTS (markdown/emoji removal by default)."""
    first_sentence_min_chars: int = 4
    """Minimum length of the first spoken chunk (smaller = faster first audio)."""
    first_sentence_max_chars: int | None = 40
    """Split a longer first sentence at its first clause boundary: TTS engines that render
    a whole sentence before emitting audio (Kokoro, most local models) start sooner."""
    max_history_items: int | None = None
    """Truncate the LLM context to this many items (system prompt always kept)."""
    turn_audio_prefix: float = 0.5
    """Seconds of audio kept before speech start for the turn detector / audio LLM."""
    preemptive_generation: bool = False
    """Start the reply speculatively once the turn has *probably* ended: the final
    transcript of a pause is known and only the endpointing delay is left (with a turn
    detector: its probability is at least ``preemptive_threshold``), or the STT sends
    ``EAGER_END_OF_TURN``. Nothing speculative reaches the TTS (unless ``preemptive_tts``),
    the speaker, the session or the chat context: the reply is released when the turn is
    committed with the same transcript and context, and discarded when the user resumes.
    Hides up to the LLM's time to first token; costs an LLM call per discarded attempt.
    Needs an STT (the transcript is what is compared)."""
    preemptive_tts: bool = False
    """Also synthesize the speculative reply before the commit: hides the TTS time to first
    audio too, but the synthesis is wasted whenever the speculation is discarded."""
    preemptive_threshold: float | None = None
    """Minimum turn-detector probability to speculate at a pause. ``None`` = the detector's
    own ``threshold`` (only pauses that end the turn after ``min_endpointing_delay``);
    lower it to also speculate on pauses that wait ``max_endpointing_delay``."""
    preemptive_max_speech: float = 10.0
    """No speculation on user turns longer than this (seconds)."""
    preemptive_max_attempts: int = 3
    """Maximum speculative LLM calls per user turn."""
    use_llm_audio: bool | None = None
    """Play the LLM's own speech (audio-output "omni" models,
    ``LLMCapabilities.audio_output``) instead of synthesizing its text with the TTS.
    ``None``: yes when the LLM outputs audio and no TTS is configured. With a TTS as well,
    the TTS still speaks verbatim text (``say()``, greetings). An audio LLM generates its
    speech with the reply, so in this mode preemptive generation only runs with
    ``preemptive_tts`` (speculative speech is allowed)."""
    input_transcriber: Any = None
    """Half-cascade (``stt=None``): where the user's words for the history come from.
    ``None``: nowhere (the user turn is audio only). ``"llm"``: the audio LLM transcribes
    each turn in a second request (``llm.transcribe()``: the OpenAI-compatible hosts).
    Otherwise an STT (instance or registry spec, e.g. ``"faster_whisper/tiny"``) run on the
    turn's audio. Runs alongside the reply; the transcript fills ``AudioContent.transcript``
    (older turns can then be sent as text: the LLM's ``audio_history``) and reaches the
    session as the turn's final ``user_transcript``. Ignored with an STT."""
    speech_rate: float = 14.0
    """Initial estimate of an audio LLM's speaking rate (characters per second, refined from
    its completed replies): truncation uses it to place a barge-in in a reply whose text
    runs ahead of its audio. 14 is LFM2.5-Audio's measured rate."""


@dataclass
class _Spoken:
    segments: list[tuple[str, float]] = field(default_factory=list)
    """(text, start offset in seconds within the item's audio)."""
    words: list[WordTiming] = field(default_factory=list)
    """Word timings reported by the TTS (relative to the start of the item's audio)."""
    audio_duration: float = 0.0
    text: list[str] = field(default_factory=list)
    llm_audio: bool = False
    """Spoken by an audio-output LLM: ``text`` holds its raw text deltas and ``segments``
    the ones whose audio position the model reported."""
    complete: bool = True
    """All of the reply's text and audio has been generated."""


@dataclass(eq=False)
class _Pause:
    """An endpointing decision for one pause of the user turn ``item_id``."""

    decision: EndpointingDecision
    item_id: str
    speech_end: float
    """When the user stopped speaking (``now()`` clock)."""
    timer: asyncio.TimerHandle | None = None


class _Output:
    """Where a response's events go: straight to the session, or — while the response is
    speculative — held back until the user's turn is committed."""

    def __init__(self, conn: CascadeConnection, *, held: bool = False) -> None:
        self._conn = conn
        self._held: list[EngineEvent] | None = [] if held else None
        self._on_release: list[Callable[[], None]] = []
        self._released = asyncio.Event()
        self.released_at = now()
        """When the response was triggered (the commit, for a speculative one)."""
        self.first_audio: float | None = None
        """When its first audio was delivered."""
        if not held:
            self._released.set()

    @property
    def released(self) -> bool:
        return self._held is None

    def emit(self, event: EngineEvent) -> None:
        if self._held is not None:
            self._held.append(event)
        else:
            self._send(event)

    def on_release(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` when the response is released (at once if it already is)."""
        if self._held is None:
            callback()
        else:
            self._on_release.append(callback)

    def release(self) -> None:
        """The turn was committed: deliver what was held back, stamped now, and stop holding."""
        held, self._held = self._held, None
        if held is None:
            return
        self.released_at = t = now()
        for callback in self._on_release:
            callback()
        self._on_release.clear()
        for event in held:
            event.timestamp = t
            self._send(event)
        self._released.set()

    async def wait_released(self) -> None:
        await self._released.wait()

    def _send(self, event: EngineEvent) -> None:
        if isinstance(event, ResponseAudio):
            if self.first_audio is None:
                self.first_audio = now()
            self._conn._on_agent_audio(event.frame.duration)
        self._conn._emit(event)


class _Prefetch:
    """Reads a speculative LLM stream as it arrives, so its output can be measured and
    replayed once the turn is committed. Iterates like the :class:`LLMStream` it wraps."""

    def __init__(self, stream: LLMStream) -> None:
        self.stream = stream
        self._chunks: Chan[ChatChunk] = Chan()
        self._streamed = 0
        self._usage: CompletionUsage | None = None
        self._error: BaseException | None = None
        self._task = asyncio.create_task(self._pump(), name=f"cascade-prefetch-{stream.request_id}")

    async def _pump(self) -> None:
        try:
            async for chunk in self.stream:
                if chunk.delta or chunk.tool_calls:
                    self._streamed += 1
                if chunk.usage is not None:
                    self._usage = chunk.usage
                self._chunks.send_nowait(chunk)
        except Exception as exc:
            self._error = exc
        finally:
            self._chunks.close()

    @property
    def failed(self) -> bool:
        return self._error is not None

    @property
    def output_tokens(self) -> int:
        """Tokens generated so far (usage once reported, else the streamed chunks)."""
        return self._usage.completion_tokens if self._usage is not None else self._streamed

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __anext__(self) -> ChatChunk:
        try:
            return await self._chunks.recv()
        except ChanClosed:
            if self._error is not None:
                err, self._error = self._error, None
                raise err from None
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        await cancel_and_wait(self._task)
        await self.stream.aclose()
        self._chunks.close()


@dataclass(eq=False)
class _Speculation:
    """A reply generated for the pending user turn before the turn is committed."""

    item_id: str
    """The user item it answers."""
    text: str
    """The transcript it answers."""
    key: tuple[Any, ...]
    """Everything else it depends on (see :meth:`CascadeConnection._context_key`)."""
    response_id: str
    reply: _Prefetch
    output: _Output
    task: asyncio.Task[None]
    started: float = field(default_factory=now)

    def mismatch(self, item_id: str, text: str, key: tuple[Any, ...]) -> SpeculationReason | None:
        """Why this reply cannot answer the turn ``item_id``/``text`` (``None``: it can)."""
        if item_id != self.item_id or _normalize(text) != _normalize(self.text):
            return "transcript"
        if key != self.key:
            return "context"
        if self.reply.failed or self.task.done():
            return "failed"
        return None

    async def aclose(self) -> None:
        try:
            await cancel_and_wait(self.task)
        finally:  # (a task cancelled before it ever ran never closed its reply)
            await self.reply.aclose()


class CascadeEngine(S2SEngine):
    """Speech-to-speech engine assembled from STT, LLM, TTS, VAD and turn detection.

    Components can be instances or registry specs (``"deepgram/nova-3"``).
    """

    provider = "cascade"

    def __init__(
        self,
        *,
        stt: Any = None,
        llm: Any,
        tts: Any = None,
        vad: Any = None,
        turn_detector: Any = None,
        options: CascadeOptions | None = None,
    ) -> None:
        self.llm: LLM = create("llm", llm)
        self.tts: TTS | None = create("tts", tts) if tts is not None else None
        self.options = options or CascadeOptions()
        llm_caps = self.llm.capabilities
        use_llm_audio = self.options.use_llm_audio
        if use_llm_audio is None:
            use_llm_audio = llm_caps.audio_output and self.tts is None
        if use_llm_audio and not llm_caps.audio_output:
            raise ConfigurationError(
                "use_llm_audio needs an LLM that outputs audio (LLMCapabilities.audio_output)"
            )
        if not use_llm_audio and self.tts is None:
            raise ConfigurationError("cascade needs tts=... unless the LLM outputs audio")
        self.llm_audio: bool = use_llm_audio
        """The LLM's own speech is played (no TTS for its replies)."""
        self.vad: VAD | None = create("vad", vad) if vad is not None else None
        self.turn_detector: TurnDetector | None = (
            create("turn", turn_detector) if turn_detector is not None else None
        )
        stt_obj: STT | None = create("stt", stt) if stt is not None else None
        if stt_obj is None and not self.llm.capabilities.audio_input:
            raise ConfigurationError("cascade needs stt=... unless the LLM accepts audio input")
        if stt_obj is None and self.vad is None:
            raise ConfigurationError("an audio-input LLM cascade needs vad=... to segment turns")
        self.input_transcriber: STT | None = None
        """The STT of ``CascadeOptions.input_transcriber`` (half-cascade only)."""
        self.transcribe_with_llm = False
        """``CascadeOptions.input_transcriber == "llm"`` (half-cascade only)."""
        transcriber = self.options.input_transcriber
        if transcriber is not None and stt_obj is None:
            if transcriber == "llm":
                if not callable(getattr(self.llm, "transcribe", None)):
                    raise ConfigurationError(
                        f"input_transcriber='llm': {type(self.llm).__name__} cannot transcribe "
                        "audio; use an STT (input_transcriber='faster_whisper/tiny'...)"
                    )
                self.transcribe_with_llm = True
            else:
                self.input_transcriber = create("stt", transcriber)
        if stt_obj is not None and not stt_obj.capabilities.streaming:
            if self.vad is None:
                raise ConfigurationError(
                    f"{type(stt_obj).__name__} is batch-only; add vad=... so it can be streamed"
                )
            stt_obj = StreamAdapter(stt_obj, self.vad)
        self.stt: STT | None = stt_obj
        input_rate = (
            stt_obj.sample_rate if stt_obj else (self.vad.sample_rate if self.vad else 16_000)
        )
        parts = [c.model for c in (stt_obj, self.llm, self.tts) if c is not None]
        super().__init__(
            model="+".join(parts),
            capabilities=EngineCapabilities(
                native_audio=False,
                server_turn_detection=True,
                tool_calling=self.llm.capabilities.tool_calling,
                input_transcription=stt_obj is not None,
                output_transcription=True,
                truncation=True,
                full_duplex=False,
                text_input=True,
            ),
            input_sample_rate=input_rate,
            output_sample_rate=(
                llm_caps.audio_sample_rate
                if use_llm_audio or self.tts is None
                else self.tts.sample_rate
            ),
        )
        for comp in self.components:
            comp.on("metrics", self._forward_metrics)

    @property
    def components(self) -> list[Any]:
        return [
            c
            for c in (
                self.vad,
                self.stt,
                self.input_transcriber,
                self.turn_detector,
                self.llm,
                self.tts,
            )
            if c is not None
        ]

    def _forward_metrics(self, m: Metrics) -> None:
        self.emit("metrics", m)

    async def connect(self, options: EngineOptions) -> EngineConnection:
        return CascadeConnection(self, options)

    async def warmup(self) -> None:
        await asyncio.gather(*(c.warmup() for c in self.components))

    async def aclose(self) -> None:
        await asyncio.gather(*(c.aclose() for c in self.components), return_exceptions=True)


class CascadeConnection(EngineConnection):
    def __init__(self, engine: CascadeEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._e = engine
        self._opts = engine.options
        self.chat_ctx = ChatContext(options.chat_ctx.items if options.chat_ctx else [])
        self.instructions = options.instructions
        self.tools: list[FunctionTool] = list(options.tools)
        self._tasks = BackgroundTasks("cascade")
        rate = engine.input_sample_rate
        self._vad = engine.vad.stream() if engine.vad else None
        self._stt = engine.stt.stream(language=options.language) if engine.stt else None
        self._stt_task = asyncio.create_task(self._stt_loop()) if self._stt else None
        self._recent: deque[AudioFrame] = deque()
        self._recent_duration = 0.0
        self._turn_audio = AudioBuffer(rate, max_duration=120.0)
        self._user_speaking = False
        self._reset_turn()
        self._final_event = asyncio.Event()
        self._endpoint_task: asyncio.Task[None] | None = None
        self._speech_end_wall: float | None = None
        self._endpointer = Endpointer(self._opts, has_detector=engine.turn_detector is not None)
        self._pause: _Pause | None = None
        """The endpointing decision of the pending pause (until resumed or committed)."""
        self._commit_watch: _Pause | None = None
        """The last committed pause, while a resumption would make it a false commit."""
        # STTs with built-in turn detection (Flux, Ink, AssemblyAI...) own the end of turn:
        # the VAD then only drives speech start/stop (barge-in), never commits.
        self._stt_turns = bool(engine.stt is not None and engine.stt.capabilities.end_of_turn)
        self._last_final_end: float | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._spoken: dict[str, _Spoken] = {}
        self._spec: _Speculation | None = None
        self._agent_audio_end = 0.0
        """When the audio delivered so far (probably) stops playing."""
        self._speech_rate = engine.options.speech_rate
        """Characters per second of the audio LLM's speech (see ``_heard_llm_audio``)."""
        self._commit_gate = asyncio.Event()
        """Clear while automatic commits are deferred (:meth:`defer_commit`)."""
        self._commit_gate.set()
        self._stt_commit_task: asyncio.Task[None] | None = None
        self._vad_input = asyncio.Event()
        """Set whenever the VAD has processed input audio."""

    # ---------------------------------------------------------------- user turn
    def _reset_turn(self) -> None:
        self._turn_item_id = new_id("item_")
        self._turn_finals: list[str] = []
        self._turn_interim = ""
        self._turn_audio.clear()
        self._turn_has_speech = False
        self._spec_attempts = 0

    def _turn_text(self) -> str:
        return " ".join(t.strip() for t in self._turn_finals if t.strip())

    async def _send_audio(self, frame: AudioFrame) -> None:
        if self._stt is not None:
            self._stt.push_audio(frame)
        self._recent.append(frame)
        self._recent_duration += frame.duration
        while (
            self._recent
            and self._recent_duration - self._recent[0].duration > self._opts.turn_audio_prefix
        ):
            self._recent_duration -= self._recent.popleft().duration
        if self._turn_has_speech:  # keep pauses and resumed onsets: audio detectors need them
            self._turn_audio.append(frame)
        if self._vad is None:
            return
        for ev in self._vad.push_audio(frame):
            if ev.type == VADEventType.START_OF_SPEECH:
                self._on_speech_started(self.input_audio_time - ev.speech_duration)
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._on_speech_stopped(self.input_audio_time - ev.silence_duration)
        self._vad_input.set()

    def _on_speech_started(self, audio_time: float | None) -> None:
        onset = self.audio_time_to_wall(audio_time) if audio_time is not None else None
        onset = now() if onset is None else onset
        pause, self._pause = self._pause, None
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()  # the user kept talking: same turn continues
            self._on_resumed(pause, onset)
        elif self._commit_watch is not None:
            self._resolve_commit(self._commit_watch, onset)
        self._discard_speculation("resumed")
        if not self._user_speaking:
            self._user_speaking = True
            if not self._turn_has_speech:
                for f in self._recent:
                    self._turn_audio.append(f)
            self._turn_has_speech = True
        self._emit(InputSpeechStarted(audio_time=audio_time))

    def _on_speech_stopped(self, audio_time: float | None) -> None:
        self._user_speaking = False
        wall = self.audio_time_to_wall(audio_time) if audio_time is not None else None
        self._speech_end_wall = wall if wall is not None else now()
        self._emit(InputSpeechStopped(audio_time=audio_time))
        if self.options.turn_detection and not self._stt_turns:
            self._schedule_endpoint()

    def _schedule_endpoint(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._endpoint_task = self._tasks.spawn(self._endpoint(), name="cascade-endpoint")

    async def _wait_final_transcript(self) -> None:
        if self._stt is None:
            return
        self._final_event.clear()
        self._stt.flush()
        try:
            await asyncio.wait_for(self._final_event.wait(), self._opts.final_transcript_timeout)
        except TimeoutError:
            if self._turn_interim:
                self._turn_finals.append(self._turn_interim)  # fall back to the interim text
                self._turn_interim = ""

    # ------------------------------------------------------------- endpointing
    def update_endpointing(
        self, *, mode: EndpointingMode | None = None, dictation: bool | None = None
    ) -> None:
        """Switch the endpointing policy (``"fixed"``/``"dynamic"``) and/or dictation mode
        from the next pause on. Learned pauses are kept."""
        if mode is not None:
            if mode not in ("fixed", "dynamic"):
                raise ValueError(f"unknown endpointing mode {mode!r}")
            self._endpointer.mode = mode
        if dictation is not None:
            self._endpointer.dictation = dictation

    @property
    def endpointer(self) -> Endpointer:
        """This connection's endpointing state (policy, learned pauses)."""
        return self._endpointer

    def defer_commit(self, deferred: bool) -> None:
        """Hold automatic commits (endpointing, STT end of turn) while ``deferred``.

        The session defers them while its interruption policy judges user speech over the
        agent: a backchannel must be dropped (``clear_input``) before the endpointing delay
        commits it — committing would answer it and cancel the paused reply. Pauses are
        still scored meanwhile; a commit that fell due is made when the deferral ends.
        """
        if deferred:
            self._commit_gate.clear()
        else:
            self._commit_gate.set()

    async def _commit_when_allowed(self) -> None:
        await self._commit_gate.wait()
        await self._commit_turn()

    async def _wait_unconfirmed_speech(self) -> None:
        """Don't commit while the user may have just started speaking again.

        The VAD confirms speech only after ``min_speech_duration`` (+ a window), ~0.15 s
        after its onset; a commit made in that gap answers over a user who resumed. While
        the speech probability is above the activation threshold without a confirmed
        start, wait (at most ``min_speech_duration`` + 0.1 s): ``START_OF_SPEECH`` cancels
        this endpointing (the turn goes on), silence lets it commit."""
        vad = self._vad
        if vad is None or self._e.vad is None or vad.speaking:
            return
        opts = self._e.vad.options
        deadline = now() + opts.min_speech_duration + 0.1
        while vad.probability >= opts.activation_threshold and now() < deadline:
            self._vad_input.clear()  # set by the next input frame the VAD has seen
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._vad_input.wait(), max(0.0, deadline - now()))

    def _on_resumed(self, pause: _Pause | None, onset: float) -> None:
        """The user spoke again before the pending pause was committed."""
        if self._speech_end_wall is None:
            return
        length = max(0.0, onset - self._speech_end_wall)
        if pause is not None and _confident(pause.decision):
            self._endpointer.observe_cutoff(length)
        else:
            self._endpointer.observe_pause(length)
        if pause is not None:
            self._emit_endpointing(pause, committed=False, length=length)

    def _watch_commit(self, pause: _Pause) -> None:
        """``pause`` is being committed: a resumption within ``false_commit_window`` makes
        it a false commit."""
        if self._commit_watch is not None:
            self._resolve_commit(self._commit_watch, None)
        self._commit_watch = pause
        loop = asyncio.get_running_loop()
        window = max(0.0, self._opts.false_commit_window)
        pause.timer = loop.call_later(window, self._resolve_commit, pause, None)

    def _resolve_commit(self, pause: _Pause, onset: float | None) -> None:
        if self._commit_watch is pause:
            self._commit_watch = None
        if pause.timer is not None:
            pause.timer.cancel()
        if self.chat_ctx.get(pause.item_id) is None:
            return  # nothing was committed (no transcript)
        length = None if onset is None else max(0.0, onset - pause.speech_end)
        if length is None:
            self._endpointer.observe_commit()
        elif _confident(pause.decision):
            self._endpointer.observe_cutoff(length)
        else:
            self._endpointer.observe_pause(length)
        self._emit_endpointing(
            pause, committed=True, length=length, false_commit=length is not None
        )

    def _emit_endpointing(
        self, pause: _Pause, *, committed: bool, length: float | None, false_commit: bool = False
    ) -> None:
        d, engine = pause.decision, self._e
        engine.emit(
            "metrics",
            EndpointingMetrics(
                provider=engine.provider,
                model=engine.model,
                item_id=pause.item_id,
                policy=d.policy,
                delay=d.delay,
                probability=d.probability,
                threshold=d.threshold,
                hold=d.hold,
                committed=committed,
                pause=length,
                false_commit=false_commit,
                audio_probability=d.audio_probability,
                text_probability=d.text_probability,
            ),
        )

    def _user_context(self, text: str) -> ChatContext:
        """The conversation plus the pending user ``text`` (what text detectors judge)."""
        ctx = self._llm_context(None)
        ctx.add_message("user", text)
        return ctx

    async def _fused_probability(self, fused: FusedTurnDetector) -> tuple[float, dict[str, Any]]:
        """Run both halves of a fused detector: the audio half concurrently with the STT
        flush, the text half on the transcript — started on the interim text so that it
        overlaps the flush too, and run again only if the final transcript differs."""
        t0 = now()
        audio = self._turn_audio.to_frame() if self._turn_audio else None
        audio_task = asyncio.ensure_future(fused.predict_audio(audio))
        guess = " ".join(p for p in (self._turn_text(), self._turn_interim.strip()) if p)
        text_task: asyncio.Future[float | None] | None = None
        if guess:
            text_task = asyncio.ensure_future(fused.predict_text(self._user_context(guess)))
        try:
            await self._wait_final_transcript()
            final = self._turn_text()
            if _normalize(final) != _normalize(guess):
                if text_task is not None:
                    text_task.cancel()
                text_task = None
                if final:
                    text_task = asyncio.ensure_future(fused.predict_text(self._user_context(final)))
            pa = await audio_task
            if text_task is not None and not text_task.done():
                # the audio verdict alone may already justify a (held) speculative reply:
                # start it now rather than after the text half
                self._speculate(pa)
            pt = await text_task if text_task is not None else None
        finally:
            for task in (audio_task, text_task):
                if task is not None and not task.done():
                    task.cancel()
        p = fused.report(fused.fuse(pa, pt), now() - t0)
        return p, {"audio_probability": pa, "text_probability": pt}

    async def _endpoint(self) -> None:
        t_end = self._speech_end_wall if self._speech_end_wall is not None else now()
        detector = self._e.turn_detector
        if isinstance(detector, FusedTurnDetector):
            fused, parts = await self._fused_probability(detector)
            decision = replace(self._endpointer.decide(fused, detector.threshold), **parts)
            await self._wait_and_commit(decision, t_end, fused)
            return
        early: asyncio.Future[float] | None = None
        if detector is not None and detector.modality == "audio" and self._turn_audio:
            # audio-only detectors don't need the transcript: overlap them with the STT flush
            early = asyncio.ensure_future(
                detector.predict_end_of_turn(audio=self._turn_audio.to_frame())
            )
        try:
            await self._wait_final_transcript()
            prob: float | None = None
            if detector is not None:
                if early is not None:
                    prob = await early
                elif self._turn_text() or self._turn_audio:
                    ctx = self._llm_context(None)
                    ctx.add_message("user", self._turn_text())
                    prob = await detector.predict_end_of_turn(
                        audio=self._turn_audio.to_frame() if self._turn_audio else None,
                        chat_ctx=ctx,
                    )
                else:
                    prob = 1.0
        finally:
            if early is not None and not early.done():
                early.cancel()
        decision = self._endpointer.decide(
            prob, detector.threshold if detector is not None else None
        )
        await self._wait_and_commit(decision, t_end, prob)

    async def _wait_and_commit(
        self, decision: EndpointingDecision, t_end: float, prob: float | None
    ) -> None:
        pause = self._pause = _Pause(decision, self._turn_item_id, t_end)
        remaining = decision.delay - (now() - t_end)
        if remaining > 0:
            # only the silence is left to wait for: the reply can start meanwhile
            self._speculate(prob)
            await asyncio.sleep(remaining)
        await self._commit_gate.wait()  # the session may still drop this turn
        await self._wait_unconfirmed_speech()
        self._pause = None
        self._watch_commit(pause)
        await self._commit_turn()

    async def _stt_loop(self) -> None:
        assert self._stt is not None
        try:
            async for ev in self._stt:
                if ev.type == STTEventType.INTERIM_TRANSCRIPT:
                    self._turn_interim = ev.text
                    partial = " ".join(p for p in (self._turn_text(), ev.text) if p)
                    self._emit(
                        InputTranscript(item_id=self._turn_item_id, text=partial, is_final=False)
                    )
                elif ev.type == STTEventType.FINAL_TRANSCRIPT:
                    if ev.transcript is not None and ev.transcript.end_time is not None:
                        self._last_final_end = ev.transcript.end_time
                    if ev.text.strip():
                        self._turn_finals.append(ev.text)
                        self._emit(
                            InputTranscript(
                                item_id=self._turn_item_id,
                                text=self._turn_text(),
                                is_final=False,
                                language=ev.transcript.language if ev.transcript else None,
                                segment_final=True,
                            )
                        )
                    self._turn_interim = ""
                    self._final_event.set()
                    if self._spec is not None and not self._stt_turns:
                        self._speculate()  # a late final changed the transcript: start over
                elif ev.type == STTEventType.START_OF_SPEECH and self._vad is None:
                    self._on_speech_started(None)
                elif ev.type == STTEventType.END_OF_SPEECH and self._vad is None:
                    # STT audio time = our input stream time (all audio goes to the STT)
                    end = ev.transcript.end_time if ev.transcript else None
                    self._on_speech_stopped(end if end is not None else self._last_final_end)
                elif ev.type == STTEventType.END_OF_TURN and self.options.turn_detection:
                    if self._endpoint_task is not None and not self._endpoint_task.done():
                        self._endpoint_task.cancel()
                    self._stt_commit_task = self._tasks.spawn(self._commit_when_allowed())
                elif ev.type == STTEventType.EAGER_END_OF_TURN:
                    eager = " ".join(p for p in (self._turn_text(), ev.text.strip()) if p)
                    self._speculate(text=eager)
                elif ev.type == STTEventType.TURN_RESUMED:
                    self._discard_speculation("resumed")
        except Exception as exc:
            logger.exception("STT stream failed")
            self._emit(EngineErrorEvent(error=exc, recoverable=False))

    async def _commit_turn(self) -> None:
        text = self._turn_text()
        audio = self._turn_audio.to_frame() if self._turn_audio else None
        use_audio = self._e.stt is None and audio is not None
        if not text and not use_audio:
            self._discard_speculation("transcript")
            self._reset_turn()
            return
        item_id = self._turn_item_id
        spec, self._spec = self._spec, None
        # checked before the user message joins the context the speculation started from
        mismatch = spec.mismatch(item_id, text, self._context_key()) if spec else None
        content: str | AudioContent = AudioContent(audio, None) if use_audio and audio else text
        self.chat_ctx.add_message("user", content, id=item_id)
        self._emit(InputCommitted(item_id=item_id))
        engine = self._e
        if isinstance(content, AudioContent) and (
            engine.input_transcriber is not None or engine.transcribe_with_llm
        ):  # the transcript follows (the session keeps a pending user item meanwhile)
            self._tasks.spawn(self._transcribe_input(item_id, content), name="cascade-transcribe")
        else:
            self._emit(InputTranscript(item_id=item_id, text=text, is_final=True))
        self._reset_turn()
        if spec is not None:
            if mismatch is None:
                await self._start_response(speculation=spec)
                return
            self._drop_speculation(spec, mismatch)
        await self._start_response()

    async def _transcribe_input(self, item_id: str, content: AudioContent) -> None:
        """Half-cascade: transcribe a committed user turn for the history."""
        engine = self._e
        text, language = "", None
        try:
            if engine.input_transcriber is not None:
                result = await engine.input_transcriber.transcribe(content.frame)
                text, language = result.text.strip(), result.language
            else:
                text = await engine.llm.transcribe(content.frame)  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("could not transcribe the user's turn %s", item_id, exc_info=True)
        content.transcript = text or None
        self._emit(InputTranscript(item_id=item_id, text=text, is_final=True, language=language))

    # --------------------------------------------------------------- speculation
    def _speculate(self, probability: float | None = None, *, text: str | None = None) -> None:
        """The pending turn has probably ended: start its reply now, held back until the
        commit. ``probability`` is the turn detector's verdict on this pause, if any."""
        opts, engine = self._opts, self._e
        if not opts.preemptive_generation or engine.stt is None or not self.options.turn_detection:
            return
        if engine.llm_audio and not opts.preemptive_tts:
            return  # an audio LLM's reply *is* speech: speculate only when that is allowed
        text = self._turn_text() if text is None else text.strip()
        if not text:
            return
        detector = engine.turn_detector
        if probability is not None and detector is not None:
            threshold = opts.preemptive_threshold
            if probability < (detector.threshold if threshold is None else threshold):
                return
        item_id, key = self._turn_item_id, self._context_key()
        spec = self._spec
        if spec is not None:
            reason = spec.mismatch(item_id, text, key)
            if reason is None:
                return  # already answering exactly this
            self._discard_speculation(reason)
        if (
            self._spec_attempts >= opts.preemptive_max_attempts
            or self._turn_audio.duration > opts.preemptive_max_speech
            or self._agent_busy()
        ):
            return
        self._spec_attempts += 1
        # the LLM input a reply started at the commit would get: + user message + placeholder
        user = ChatMessage(role="user", content=[text], id=item_id)
        answer = ChatMessage(role="assistant", content=[""], id=new_id("item_"))
        tools = self.tools if engine.llm.capabilities.tool_calling else []
        try:
            stream = engine.llm.chat(self._llm_context(None, (user, answer)), tools=tools)
        except Exception:  # the reply started at the commit will report it
            logger.exception("could not start a speculative reply")
            return
        reply = _Prefetch(stream)
        output = _Output(self, held=True)
        rid = new_id("resp_")
        task = asyncio.create_task(
            self._respond(rid, None, None, output, reply, answer), name=f"cascade-{rid}"
        )
        self._spec = _Speculation(item_id, text, key, rid, reply, output, task)
        logger.debug("speculative reply %s started for %r", rid, text)

    def _agent_busy(self) -> bool:
        """A reply is being generated or (probably) still playing: speech that ends now
        overlapped the agent, and the session's interruption policy decides about it."""
        task = self._response_task
        return (task is not None and not task.done()) or now() < self._agent_audio_end

    def _on_agent_audio(self, duration: float) -> None:
        self._agent_audio_end = max(self._agent_audio_end, now()) + duration

    def _context_key(self) -> tuple[Any, ...]:
        """Everything besides the user's words that the next reply depends on."""
        items = tuple(_item_key(item) for item in self.chat_ctx.items)
        return (self.instructions, tuple(self.tools), items)

    def _check_speculation(self) -> None:
        """Discard the pending speculation if the conversation it started from changed."""
        if self._spec is not None and self._spec.key != self._context_key():
            self._discard_speculation("context")

    def _discard_speculation(self, reason: SpeculationReason) -> None:
        spec, self._spec = self._spec, None
        if spec is not None:
            self._drop_speculation(spec, reason)

    def _drop_speculation(self, spec: _Speculation, reason: SpeculationReason) -> None:
        self._report_speculation(spec, reason)
        spec.task.cancel()
        self._tasks.spawn(spec.aclose(), name=f"cascade-drop-{spec.response_id}")

    def _report_speculation(self, spec: _Speculation, reason: SpeculationReason | None) -> None:
        engine = self._e
        lead = now() - spec.started
        logger.debug(
            "speculative reply %s %s after %.0f ms",
            spec.response_id,
            "kept" if reason is None else f"discarded ({reason})",
            lead * 1000,
        )
        engine.emit(
            "metrics",
            SpeculationMetrics(
                provider=engine.provider,
                model=engine.model,
                request_id=spec.reply.stream.request_id,
                hit=reason is None,
                reason=reason,
                response_id=spec.response_id if reason is None else None,
                lead=lead,
                output_tokens=spec.reply.output_tokens,
            ),
        )

    async def commit_input(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        self._pause = None
        self._user_speaking = False
        if self._vad is not None:
            self._vad.reset()
        await self._wait_final_transcript()
        await self._commit_turn()

    async def clear_input(self) -> None:
        if self._endpoint_task is not None and not self._endpoint_task.done():
            self._endpoint_task.cancel()
        if self._stt_commit_task is not None and not self._stt_commit_task.done():
            self._stt_commit_task.cancel()
        self._pause = None
        self._discard_speculation("cleared")
        self._user_speaking = False
        if self._vad is not None:
            self._vad.reset()
        self._reset_turn()

    # ------------------------------------------------------------------ control
    async def send_text(self, text: str, *, respond: bool = True) -> None:
        self.chat_ctx.add_message("user", text)
        self._check_speculation()
        if respond:
            await self._start_response()

    async def create_response(self, *, instructions: str | None = None) -> None:
        await self._start_response(instructions=instructions)

    async def say(self, text: str) -> None:
        await self._start_response(verbatim=text)

    async def cancel_response(self) -> None:
        self._discard_speculation("cancelled")
        task = self._response_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def truncate(self, item_id: str, audio_end_ms: int) -> str | None:
        self._agent_audio_end = min(self._agent_audio_end, now())  # playback was cut
        msg = self.chat_ctx.get(item_id)
        spoken = self._spoken.get(item_id)
        if not isinstance(msg, ChatMessage) or msg.role != "assistant":
            return None
        end = audio_end_ms / 1000.0
        if spoken is not None and spoken.llm_audio:
            heard = self._heard_llm_audio(spoken, end)
        elif spoken is not None and spoken.words:  # word-exact: TTS reported word timings
            heard = " ".join(w.word for w in spoken.words if w.start < end)
        elif spoken is not None and spoken.segments:
            heard = " ".join(t for t, start in spoken.segments if start < end)
        elif spoken is not None and spoken.audio_duration > 0:
            full = "".join(spoken.text)
            heard = full[: round(len(full) * min(1.0, end / spoken.audio_duration))]
        else:
            heard = ""
        msg.content = [heard.strip()]
        msg.interrupted = True
        self._check_speculation()
        return heard.strip()

    def _heard_llm_audio(self, spoken: _Spoken, end: float) -> str:
        """What was heard of an audio LLM's reply after ``end`` seconds of its audio.

        Exact when the model reported where its text is spoken (``ChatChunk.audio_offset``).
        Otherwise proportional: the text's share of the audio, cut after the word being
        spoken. Omni models generate text ahead of the audio (LFM2.5-Audio: 6 text tokens
        per 0.96 s of audio, so its text is complete when ~40 % of the audio exists), so
        while a reply is still being generated its length is estimated from the speaking
        rate.
        """
        if spoken.segments:
            return "".join(t for t, start in spoken.segments if start < end)
        full = "".join(spoken.text)
        total = spoken.audio_duration
        if not spoken.complete:
            total = max(total, len(full) / self._speech_rate)
        if total <= 0 or not full:
            return ""
        return _cut_after_word(full, round(len(full) * min(1.0, end / total)))

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        self.chat_ctx.append(output)
        self._check_speculation()
        if respond:
            await self._start_response()

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        if instructions is not None:
            self.instructions = instructions
        if tools is not None:
            self.tools = list(tools)
        self._check_speculation()

    async def update_voice(self, voice: str) -> bool:
        self.options.voice = voice  # every reply opens its own TTS stream
        return True

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> bool:
        self.chat_ctx = ChatContext(chat_ctx.items)
        self._check_speculation()
        return True

    async def aclose(self) -> None:
        if self._commit_watch is not None:
            self._resolve_commit(self._commit_watch, None)
        self._discard_speculation("closed")
        await self.cancel_response()
        await self._tasks.cancel_all()
        if self._stt is not None:
            await self._stt.aclose()
        await cancel_and_wait(self._stt_task)
        if self._vad is not None:
            self._vad.close()
        await super().aclose()

    # ----------------------------------------------------------------- response
    def _llm_context(
        self, extra_instructions: str | None, pending: Sequence[ChatItem] = ()
    ) -> ChatContext:
        """The LLM input: instructions + history (+ ``pending`` items, not in the history
        yet: a speculative reply's user turn), truncated to ``max_history_items``."""
        ctx = ChatContext()
        if self.instructions:
            ctx.add_message("system", self.instructions)
        items = ChatContext([*self.chat_ctx.items, *pending])
        if self._opts.max_history_items is not None:
            items.truncate(self._opts.max_history_items)
        ctx.items.extend(items.items)
        if extra_instructions:
            ctx.add_message("system", extra_instructions)
        return ctx

    async def _start_response(
        self,
        *,
        instructions: str | None = None,
        verbatim: str | None = None,
        speculation: _Speculation | None = None,
    ) -> None:
        await self.cancel_response()
        if speculation is not None:  # the committed turn is the one it answers: release it
            self._response_task = speculation.task
            speculation.output.release()
            self._report_speculation(speculation, None)
            return
        rid = new_id("resp_")
        self._response_task = asyncio.create_task(
            self._respond(rid, instructions, verbatim, _Output(self)), name=f"cascade-{rid}"
        )

    async def _respond(
        self,
        rid: str,
        instructions: str | None,
        verbatim: str | None,
        output: _Output,
        reply: _Prefetch | None = None,
        message: ChatMessage | None = None,
    ) -> None:
        """One response. A speculative one (``output`` held, ``reply`` already streaming
        into ``message``) runs the same way, except that it waits for the commit before
        the TTS (unless ``preemptive_tts``) and before it touches the history or reports
        tool calls."""
        engine = self._e
        status: ResponseStatus = "completed"
        error: str | None = None
        stream: LLMStream | _Prefetch | None = None
        if message is None:
            message = ChatMessage(role="assistant", content=[""], id=new_id("item_"))
        msg = message
        item_id = msg.id
        output.emit(ResponseStarted(response_id=rid))
        try:
            output.on_release(lambda: self.chat_ctx.append(msg))
            if verbatim is not None:
                msg.content = [verbatim]
                if engine.tts is not None:
                    await self._speak(rid, item_id, _once(verbatim), output)
                else:
                    logger.warning("say(): no TTS to speak verbatim text; sending the text only")
                    output.emit(ResponseText(response_id=rid, item_id=item_id, delta=verbatim))
            else:
                if reply is not None:
                    stream = reply
                else:
                    ctx = self._llm_context(instructions)
                    tools = self.tools if engine.llm.capabilities.tool_calling else []
                    stream = engine.llm.chat(ctx, tools=tools)
                source = stream
                calls: list[FunctionCall] = []
                text_parts: list[str] = []

                async def text_source() -> AsyncIterator[str]:
                    try:
                        async for chunk in source:
                            if chunk.delta:
                                text_parts.append(chunk.delta)
                                msg.content = ["".join(text_parts)]
                                yield chunk.delta
                            calls.extend(chunk.tool_calls)
                    finally:
                        await source.aclose()

                async def chunk_source() -> AsyncIterator[ChatChunk]:
                    try:
                        async for chunk in source:
                            if chunk.delta:
                                text_parts.append(chunk.delta)
                                msg.content = ["".join(text_parts)]
                            calls.extend(chunk.tool_calls)
                            yield chunk
                    finally:
                        await source.aclose()

                if not self._opts.preemptive_tts:
                    await output.wait_released()  # nothing speculative reaches the TTS
                if engine.llm_audio:
                    await self._relay(rid, item_id, chunk_source(), output)
                else:
                    await self._speak(rid, item_id, text_source(), output)
                await output.wait_released()  # the turn is committed: history and tools
                if not "".join(text_parts).strip():
                    self.chat_ctx.remove(item_id)
                for call in calls:
                    self.chat_ctx.append(call)
                    output.emit(ResponseToolCall(response_id=rid, call=call))
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            logger.exception("cascade response failed")
            status, error = "failed", repr(exc)
            output.emit(EngineErrorEvent(error=exc, recoverable=True))
        finally:
            if stream is not None:
                await stream.aclose()
            output.emit(ResponseDone(response_id=rid, status=status, error=error))
            if output.released:
                t0 = output.released_at
                first_audio = output.first_audio
                engine.emit(
                    "metrics",
                    EngineMetrics(
                        provider=engine.provider,
                        model=engine.model,
                        response_id=rid,
                        ttfb=(first_audio - t0) if first_audio is not None else None,
                        duration=now() - t0,
                        cancelled=status == "cancelled",
                    ),
                )
            else:  # a discarded speculation: as far as anyone knows, it never existed
                self._spoken.pop(item_id, None)

    async def _relay(
        self, rid: str, item_id: str, chunks: AsyncIterator[ChatChunk], output: _Output
    ) -> None:
        """Play an audio-output LLM's own speech: its audio deltas become ``ResponseAudio``
        and its text deltas the ``ResponseText`` transcript (no text filter: the model
        has already said it)."""
        spoken = self._spoken[item_id] = _Spoken(llm_audio=True, complete=False)
        async for chunk in chunks:
            if chunk.delta:
                spoken.text.append(chunk.delta)
                if chunk.audio_offset is not None:
                    spoken.segments.append((chunk.delta, chunk.audio_offset))
                output.emit(ResponseText(response_id=rid, item_id=item_id, delta=chunk.delta))
            if chunk.audio:
                spoken.audio_duration += chunk.audio.duration
                output.emit(ResponseAudio(response_id=rid, item_id=item_id, frame=chunk.audio))
        spoken.complete = True
        chars = len("".join(spoken.text).strip())
        if spoken.audio_duration >= 2.0 and chars >= 20:  # refine the speaking rate
            rate = min(25.0, max(6.0, chars / spoken.audio_duration))
            self._speech_rate = 0.7 * self._speech_rate + 0.3 * rate

    async def _speak(
        self, rid: str, item_id: str, text: AsyncIterator[str], output: _Output
    ) -> None:
        """Stream ``text`` through the TTS, emitting aligned text and audio events."""
        engine = self._e
        assert engine.tts is not None
        tts_stream = engine.tts.stream(voice=self.options.voice)
        aligned = not engine.tts.capabilities.streaming  # sentence adapter reports segment text
        segmenter = SentenceSegmenter(
            min_chars=10,
            first_segment_min_chars=self._opts.first_sentence_min_chars,
            first_segment_max_chars=self._opts.first_sentence_max_chars,
        )
        spoken = self._spoken[item_id] = _Spoken()
        text_filter = self._opts.text_filter

        def push(sentence: str) -> None:
            cleaned = text_filter(sentence) if text_filter else sentence
            if not cleaned.strip():
                return
            spoken.text.append(cleaned + " ")
            if not aligned:
                output.emit(ResponseText(response_id=rid, item_id=item_id, delta=cleaned + " "))
            tts_stream.push_text(cleaned + " ")
            if aligned:
                # sentence-at-a-time TTS: synthesize exactly this segment now (the adapter's
                # own segmenter would otherwise hold a first *clause* until the sentence ends)
                tts_stream.flush()

        async def feed() -> None:
            try:
                async for delta in text:
                    for sentence in segmenter.push(delta):
                        push(sentence)
                for sentence in segmenter.flush():
                    push(sentence)
            finally:
                tts_stream.end_input()

        feeder = asyncio.create_task(feed(), name=f"cascade-feed-{rid}")
        try:
            async for audio in tts_stream:
                if audio.words:
                    spoken.words.extend(audio.words)
                if aligned and audio.text:
                    spoken.segments.append((audio.text, spoken.audio_duration))
                    output.emit(
                        ResponseText(response_id=rid, item_id=item_id, delta=audio.text + " ")
                    )
                if audio.frame:
                    spoken.audio_duration += audio.frame.duration
                    output.emit(ResponseAudio(response_id=rid, item_id=item_id, frame=audio.frame))
            await feeder  # surface LLM errors
        finally:
            await cancel_and_wait(feeder)
            await tts_stream.aclose()


async def _once(text: str) -> AsyncIterator[str]:
    yield text


def _confident(decision: EndpointingDecision) -> bool:
    """The turn detector said the user was done at this pause."""
    p, threshold = decision.probability, decision.threshold
    return p is not None and threshold is not None and p >= threshold


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _cut_after_word(text: str, n: int) -> str:
    """``text[:n]``, extended to the end of the word it cuts (that word was being said)."""
    if n <= 0:
        return ""
    end = min(n, len(text))
    while end < len(text) and not text[end - 1].isspace() and not text[end].isspace():
        end += 1
    return text[:end]


def _item_key(item: ChatItem) -> tuple[Any, ...]:
    """What an LLM sees of a history item, as a snapshot: any later change shows."""
    if isinstance(item, ChatMessage):
        return ("message", item.id, item.role, tuple(item.content), item.interrupted)
    if isinstance(item, FunctionCall):
        return ("call", item.id, item.call_id, item.name, item.arguments)
    return ("output", item.id, item.call_id, item.name, item.output, item.is_error)
