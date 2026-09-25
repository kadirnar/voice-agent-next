"""The simulated caller: streams stimuli in real time, records both sides on one clock.

:class:`CallerEmulator` drives the *user side* of a
:class:`~voice_agent_next.transports.LoopbackTransport` created with
``realtime_playout=True`` (so agent audio is "heard" at real-time speed):

* the user stream is **continuous**, like a microphone: pre-rendered stimuli and the
  silence between them are pushed in fixed chunks (20 ms by default), paced against one
  absolute schedule, so timing errors never accumulate;
* chunk ``i`` covers the acoustic interval ``[t0 + i*chunk, t0 + (i+1)*chunk]``. It is
  pushed when that interval has *ended* (as a real capture device delivers audio) and is
  time-stamped with its capture start, so the engine's own speech-boundary estimates and
  the harness annotations share one clock;
* after each utterance the caller keeps streaming silence until the agent has replied and
  gone quiet (``gap_after_reply``), or until ``reply_timeout`` passes without any agent
  audio (a *missed* turn), then speaks the next utterance. A stimulus with ``barge_in``
  is spoken that many seconds after the agent's reply to the previous turn started, over
  the agent (interruptions, backchannels, coughs);
* agent audio is taken from the transport's playout log (``played_log``), i.e. placed at
  the time it started playing, and truncated at playback clears (barge-in).

The result holds a :class:`~voice_agent_next.bench.recording.DuplexRecording` (user left,
agent right) plus the timing of every turn.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..transports.loopback import LoopbackTransport, PlayedAudio
from ..utils.clock import now
from .recording import DuplexRecording
from .stimuli import Stimulus

__all__ = ["CallResult", "CallerEmulator", "TurnTiming"]


async def _sleep_until(deadline: float) -> None:
    """Sleep until ``now() >= deadline`` (never earlier).

    The event loop's clock is coarse on some platforms (``time.monotonic()`` ticks every
    ~15.6 ms on Windows) and asyncio runs timers up to one clock tick early, so a plain
    ``asyncio.sleep`` can return before the deadline. The rest is slept in a worker thread
    with the (high-resolution) ``time.sleep``.
    """
    delay = deadline - now()
    if delay <= 0:
        await asyncio.sleep(0)
        return
    await asyncio.sleep(delay)
    remaining = deadline - now()
    if remaining > 0:
        await asyncio.to_thread(time.sleep, remaining)


@dataclass(slots=True)
class TurnTiming:
    """When one stimulus was spoken and how the agent reacted (``now()`` clock)."""

    index: int
    stimulus: Stimulus
    start: float
    """Acoustic start of the clip."""
    reply_start: float | None = None
    """Start of the first agent audio played after the user started speaking (after the
    user stopped, for ``barge_in`` turns)."""
    reply_end: float | None = None
    """End of the last agent audio played before the next turn."""
    missed: bool = False
    """No agent audio within ``reply_timeout`` of the end of user speech."""
    timed_out: bool = False
    """The reply did not finish within ``max_reply``."""

    @property
    def speech_start(self) -> float:
        return self.start + self.stimulus.speech_start

    @property
    def reply_after(self) -> float:
        """Agent audio starting from here counts as the reply: the user's onset, or, for a
        turn spoken over the agent (``barge_in``), the end of the user's speech."""
        return self.speech_end if self.stimulus.barge_in is not None else self.speech_start

    @property
    def speech_end(self) -> float:
        """Annotated end of user speech (``t_uoff``)."""
        return self.start + self.stimulus.speech_end

    @property
    def end(self) -> float:
        return self.start + self.stimulus.duration


@dataclass(slots=True)
class CallResult:
    """Everything the caller observed."""

    recording: DuplexRecording
    turns: list[TurnTiming]
    stream_start: float
    stream_end: float
    chunk_duration: float
    chunks_sent: int
    push_lag: list[float] = field(default_factory=list)
    """How late each chunk was delivered relative to its schedule (s)."""
    agent_frames: int = 0
    clear_times: list[float] = field(default_factory=list)
    aborted: bool = False

    @property
    def max_push_lag(self) -> float:
        return max(self.push_lag, default=0.0)


class CallerEmulator:
    """Simulated user for a :class:`LoopbackTransport` (see the module docstring).

    Args:
        transport: the loopback transport the agent session runs on; it must have been
            created with ``realtime_playout=True``.
        chunk: streaming chunk duration (s).
        origin: ``now()`` value used as t = 0 of the recording (default: stream start).
        agent_idle: optional probe (e.g. ``session.agent_state == LISTENING``) that must
            be true before the caller speaks again; guards against talking into pauses
            between reply sentences.
        should_stop: optional probe that aborts the call (e.g. the session closed).
        on_turn: called with each :class:`TurnTiming` once the caller moves on.
        tail: silence streamed after the last turn (s).
        sleep_until: waits until a ``now()`` deadline, never returning early (default: a
            precise per-caller wait). Many concurrent callers can share one timer instead
            (e.g. the overhead track's load test).
    """

    def __init__(
        self,
        transport: LoopbackTransport,
        *,
        chunk: float = 0.02,
        origin: float | None = None,
        agent_idle: Callable[[], bool] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_turn: Callable[[TurnTiming], None] | None = None,
        tail: float = 0.2,
        sleep_until: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if not transport.realtime_playout:
            raise ValueError("CallerEmulator needs LoopbackTransport(realtime_playout=True)")
        if chunk <= 0:
            raise ValueError("chunk must be > 0")
        self.transport = transport
        self.chunk = chunk
        self.origin = origin
        self.agent_idle = agent_idle
        self.should_stop = should_stop
        self.on_turn = on_turn
        self.tail = tail
        self._sleep_until = sleep_until or _sleep_until
        self._rate = transport.input_format.sample_rate
        self._chunk_samples = max(1, round(chunk * self._rate))
        self._chunk_dur = self._chunk_samples / self._rate
        self._t_stream = 0.0
        self._sent = 0
        self._user: list[bytes] = []
        self._lags: list[float] = []
        self._scanned = 0
        self._agent_end = -math.inf
        self._aborted = False
        self._used = False

    # ------------------------------------------------------------------ public
    async def run(
        self,
        stimuli: Sequence[Stimulus],
        *,
        lead_in: float = 0.5,
        reply_timeout: float = 8.0,
        gap_after_reply: float = 0.3,
        max_reply: float = 60.0,
    ) -> CallResult:
        """Speak every stimulus in order; return the recording and per-turn timing.

        A caller places one call: create a new instance (and transport) for the next one.
        """
        if self._used:
            raise RuntimeError("CallerEmulator.run() can only be called once")
        self._used = True
        turns: list[TurnTiming] = []
        self._t_stream = now()
        origin = self._t_stream if self.origin is None else self.origin
        await self._stream_silence(lead_in)
        # e.g. a greeting: never start talking over the agent
        await self._wait_until_quiet(gap_after_reply, max_reply)
        step = self._chunk_samples * 2
        for i, stim in enumerate(stimuli):
            if self._aborted:
                break
            audio = self._prepare(stim.audio)
            turn = TurnTiming(i, stim, start=self._t_stream + self._sent * self._chunk_dur)
            turns.append(turn)
            for k in range(0, len(audio.data), step):
                if not await self._send(audio.data[k : k + step]):
                    break
            nxt = stimuli[i + 1] if i + 1 < len(stimuli) else None
            barge_in = nxt.barge_in if nxt is not None else None
            await self._after_turn(turn, reply_timeout, gap_after_reply, max_reply, barge_in)
            if self.on_turn is not None:
                self.on_turn(turn)
        if not self._aborted:
            await self._stream_silence(self.tail)
        stream_end = self._t_stream + self._sent * self._chunk_dur
        return self._result(turns, origin, stream_end)

    # ----------------------------------------------------------------- streaming
    def _prepare(self, audio: AudioFrame) -> AudioFrame:
        mono = audio.to_mono()
        if mono.sample_rate != self._rate:
            mono = resample(mono, self._rate)
        rem = mono.samples_per_channel % self._chunk_samples
        if rem:
            pad = AudioFrame(bytes((self._chunk_samples - rem) * 2), self._rate, 1)
            mono = AudioFrame.concat([mono, pad])
        return mono

    async def _send(self, pcm: bytes) -> bool:
        """Push one chunk when its acoustic interval has elapsed. False when aborted."""
        if self.should_stop is not None and self.should_stop():
            self._aborted = True
            return False
        if len(pcm) < self._chunk_samples * 2:
            pcm = pcm + bytes(self._chunk_samples * 2 - len(pcm))
        capture_start = self._t_stream + self._sent * self._chunk_dur
        due = capture_start + self._chunk_dur
        await self._sleep_until(due)
        self._lags.append(max(0.0, now() - due))
        self.transport.push_user_audio(AudioFrame(pcm, self._rate, 1, capture_start))
        self._user.append(pcm)
        self._sent += 1
        return True

    async def _stream_silence(self, duration: float) -> None:
        silence = bytes(self._chunk_samples * 2)
        for _ in range(math.ceil(duration / self._chunk_dur - 1e-9)):
            if not await self._send(silence):
                return

    # ------------------------------------------------------------ turn-taking
    def _scan_agent(self, turn: TurnTiming | None) -> None:
        log = self.transport.played_log
        for played in log[self._scanned :]:
            end = played.start_time + played.frame.duration
            self._agent_end = max(self._agent_end, end)
            if turn is not None and played.start_time >= turn.reply_after:
                if turn.reply_start is None:
                    turn.reply_start = played.start_time
                turn.reply_end = end if turn.reply_end is None else max(turn.reply_end, end)
        self._scanned = len(log)

    def _idle(self) -> bool:
        return self.agent_idle() if self.agent_idle is not None else True

    def _quiet(self, gap: float) -> bool:
        return now() - self._agent_end >= gap and self._idle()

    async def _wait_until_quiet(self, gap: float, max_wait: float) -> None:
        deadline = now() + max_wait
        silence = bytes(self._chunk_samples * 2)
        while True:
            self._scan_agent(None)
            if self._quiet(gap) or now() >= deadline:
                return
            if not await self._send(silence):
                return

    async def _after_turn(
        self,
        turn: TurnTiming,
        reply_timeout: float,
        gap: float,
        max_reply: float,
        barge_in: float | None = None,
    ) -> None:
        """Stream silence until the caller may speak again: the agent replied and went
        quiet, or (``barge_in``) its reply has been playing for ``barge_in`` seconds."""
        silence = bytes(self._chunk_samples * 2)
        earliest = turn.end + turn.stimulus.pause
        while True:
            self._scan_agent(turn)
            t = now()
            if t >= earliest:
                if (
                    barge_in is not None
                    and turn.reply_start is not None
                    and t >= turn.reply_start + barge_in
                ):
                    return  # talk over the agent
                if turn.reply_start is None:
                    if not turn.stimulus.expect_reply and self._quiet(gap):
                        return
                    if turn.stimulus.expect_reply and t - turn.speech_end >= reply_timeout:
                        turn.missed = True
                        return
                elif self._quiet(gap):
                    return
                elif t - turn.reply_start >= max_reply:
                    turn.timed_out = True
                    return
            if not await self._send(silence):
                return

    # ------------------------------------------------------------------ result
    def _result(self, turns: list[TurnTiming], origin: float, stream_end: float) -> CallResult:
        recording = DuplexRecording(
            origin, user_rate=self._rate, agent_rate=self.transport.output_format.sample_rate
        )
        if self._user:
            recording.add_user(AudioFrame(b"".join(self._user), self._rate, 1), self._t_stream)
        clears = sorted(self.transport.clear_times)
        played: list[PlayedAudio] = list(self.transport.played_log)
        for p in played:
            start, end = p.start_time, p.start_time + p.frame.duration
            cut = next((c for c in clears if start < c < end), None)
            recording.add_agent(p.frame, start, cut)
        return CallResult(
            recording=recording,
            turns=turns,
            stream_start=self._t_stream,
            stream_end=stream_end,
            chunk_duration=self._chunk_dur,
            chunks_sent=self._sent,
            push_lag=list(self._lags),
            agent_frames=len(played),
            clear_times=clears,
            aborted=self._aborted,
        )
