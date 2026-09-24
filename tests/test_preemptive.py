"""Speculative (preemptive) generation in the cascade (``CascadeOptions.preemptive_generation``).

The reply to a user turn starts once the turn has *probably* ended; it must stay invisible
(no TTS unless ``preemptive_tts``, no events, no history) until the turn is committed, and
be discarded — without a trace — when the user resumes or anything it depends on changes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_session import Recorder, speak, wait_for
from voice_agent_next import (
    Agent,
    AgentSession,
    AgentState,
    AudioFrame,
    CascadeOptions,
    ChatContext,
    ChatMessage,
    FunctionCallOutput,
    LLMCapabilities,
    function_tool,
)
from voice_agent_next.events import (
    InputCommitted,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseText,
    ResponseToolCall,
)
from voice_agent_next.llm import LLMStream
from voice_agent_next.metrics import (
    LLMMetrics,
    SpeculationMetrics,
    TurnMetrics,
    UsageSummary,
)
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockLLM,
    MockSTT,
    MockToolCall,
    MockTTS,
    MockTurnDetector,
    synth_speech,
)
from voice_agent_next.stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils import now

SR = 16_000
RESPONSE_EVENTS = (ResponseStarted, ResponseText, ResponseAudio, ResponseToolCall, ResponseDone)


class EngineLog:
    """Every engine event the session handles, in order."""

    def __init__(self, session: AgentSession) -> None:
        self.events: list[Any] = []
        original = session._handle

        async def spy(ev: Any) -> None:
            self.events.append(ev)
            await original(ev)

        session._handle = spy  # type: ignore[method-assign]

    def of(self, *kinds: type) -> list[Any]:
        return [e for e in self.events if isinstance(e, kinds)]


def speculations(rec: Recorder) -> list[SpeculationMetrics]:
    return [m for m in rec.of("metrics") if isinstance(m, SpeculationMetrics)]


def outcomes(rec: Recorder) -> list[tuple[bool, str | None]]:
    return [(m.hit, m.reason) for m in speculations(rec)]


def cascade(
    *,
    stt: Any = None,
    llm: MockLLM | None = None,
    tts: MockTTS | None = None,
    turn_detector: Any = None,
    **options: Any,
) -> AgentSession:
    options.setdefault("preemptive_generation", True)
    options.setdefault("min_endpointing_delay", 0.8)  # a wide window: robust on slow runners
    return AgentSession(
        stt=stt if stt is not None else MockSTT(transcripts=["book a table"], latency=0.02),
        llm=llm if llm is not None else MockLLM(ttft=0.05),
        tts=tts if tts is not None else MockTTS(chars_per_second=200.0),  # short replies
        vad=EnergyVAD(),  # 0.25 s of silence ends speech
        turn_detector=turn_detector,
        cascade_options=CascadeOptions(**options),
    )


def history(session: AgentSession) -> list[tuple[str, str]]:
    return [(i.role, i.text) for i in session.history.items if isinstance(i, ChatMessage)]


# ------------------------------------------------------------------------------ hits


async def one_turn(**options: Any) -> tuple[TurnMetrics, Recorder]:
    session = AgentSession(
        stt=MockSTT(transcripts=["book a table for two"], latency=0.05),
        llm=MockLLM(responses=lambda ctx: "Sure.", ttft=0.3),
        tts=MockTTS(ttfb=0.1),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.6, **options),
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.6)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    return rec.turn_metrics()[0], rec


async def test_hit_hides_the_llm_time_to_first_token() -> None:
    # speech end -> VAD 0.25 s + STT 0.05 s -> transcript; the commit comes at 0.6 s
    off, _ = await one_turn()
    on, rec = await one_turn(preemptive_generation=True)
    on_tts, rec_tts = await one_turn(preemptive_generation=True, preemptive_tts=True)
    assert off.voice_to_voice is not None and on.voice_to_voice is not None
    assert on_tts.voice_to_voice is not None and on.end_of_turn_delay is not None
    # the turn ends exactly when it did before...
    assert on.end_of_turn_delay == pytest.approx(off.end_of_turn_delay or 0, abs=0.1)
    # ...but the LLM already streamed its reply: the 0.3 s TTFT is gone from the latency
    assert off.voice_to_voice - on.voice_to_voice == pytest.approx(0.3, abs=0.1)
    # pre-synthesis also hides the TTS time to first audio (0.1 s): audio is ready at once
    assert on_tts.voice_to_voice < on.voice_to_voice - 0.04
    assert on_tts.response_ttfb is not None and on_tts.response_ttfb < 0.08
    for r in (rec, rec_tts):
        [m] = speculations(r)
        assert m.hit and m.reason is None and m.response_id and m.lead > 0.2
        assert m.output_tokens > 0  # it had already streamed (the whole reply here)


async def test_hit_is_released_as_one_ordinary_response() -> None:
    llm = MockLLM(ttft=0.05)
    session = cascade(llm=llm)
    log, rec = EngineLog(session), Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)  # speculating...
    await asyncio.sleep(0.2)  # ...and its reply is complete, but held back
    conn: Any = session.connection
    assert conn.chat_ctx.items == [] and session.history.items == []
    assert log.of(InputCommitted, *RESPONSE_EVENTS) == []
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()

    assert outcomes(rec) == [(True, None)]
    assert len(llm.requests) == 1  # the speculative call became the reply
    order = log.of(InputCommitted, InputTranscript, *RESPONSE_EVENTS)
    final = next(i for i, e in enumerate(order) if isinstance(e, InputTranscript) and e.is_final)
    kinds = [type(e) for e in order]
    assert kinds.index(InputCommitted) < final < kinds.index(ResponseStarted)
    assert kinds.count(ResponseStarted) == 1 and kinds.count(ResponseDone) == 1
    [started] = log.of(ResponseStarted)
    assert started.response_id == speculations(rec)[0].response_id
    assert started.timestamp >= log.of(InputCommitted)[0].timestamp  # stamped at the commit
    assert history(session) == [("user", "book a table"), ("assistant", "You said: book a table")]
    assert [(i.role, i.text) for i in conn.chat_ctx.items] == history(session)
    assert session.usage.speculation_hits == 1 and session.usage.speculation_waste_calls == 0


async def test_speculative_reply_gets_exactly_the_llm_input_of_a_normal_one() -> None:
    async def requests(preemptive: bool) -> list[list[tuple[str, str]]]:
        llm = MockLLM(ttft=0.02)
        session = cascade(
            stt=MockSTT(transcripts=["first question", "second question"], latency=0.02),
            llm=llm,
            preemptive_generation=preemptive,
            max_history_items=3,  # the window must cut the history at the same item
            min_endpointing_delay=0.5,
        )
        rec = Recorder(session)
        transport = LoopbackTransport()
        await session.start(Agent("Be brief."), transport)
        for n in (1, 2):
            await speak(transport, 0.5, 0.4)
            await wait_for(lambda n=n: len(rec.turn_metrics()) == n, 5)  # type: ignore[misc]
        await session.aclose()
        assert len(speculations(rec)) == (2 if preemptive else 0)
        return [[(getattr(i, "role", ""), getattr(i, "text", "")) for i in r] for r in llm.requests]

    assert await requests(True) == await requests(False)


# ---------------------------------------------------------------------------- misses


@pytest.mark.parametrize("preemptive_tts", [False, True])
async def test_user_resuming_discards_the_speculation_without_a_trace(
    preemptive_tts: bool,
) -> None:
    llm = MockLLM(ttft=0.05)
    tts = MockTTS(chars_per_second=200.0)
    stt = MockSTT(transcripts=["I would like to", "book a table"], latency=0.02)
    session = cascade(stt=stt, llm=llm, tts=tts, preemptive_tts=preemptive_tts)
    log, rec = EngineLog(session), Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)  # a pause...
    await wait_for(lambda: len(llm.requests) == 1)
    await asyncio.sleep(0.15)  # ...long enough for a speculative reply to stream
    assert bool(tts.requests) == preemptive_tts  # pre-synthesized (but held back) or not
    await speak(transport, 0.5, 0.4)  # ...then the user goes on
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()

    first, second = speculations(rec)
    assert (first.hit, first.reason) == (False, "resumed") and first.output_tokens > 0
    assert (second.hit, second.reason) == (True, None)
    # the session never saw the discarded reply: one turn, one response, one transcript
    assert [e.text for e in log.of(InputTranscript) if e.is_final] == [
        "I would like to book a table"
    ]
    [started] = log.of(ResponseStarted)
    assert started.response_id == second.response_id
    assert {e.response_id for e in log.of(*RESPONSE_EVENTS)} == {second.response_id}
    spoken = "".join(e.delta for e in rec.of("agent_transcript")).strip()
    assert spoken == "You said: I would like to book a table"
    assert history(session) == [
        ("user", "I would like to book a table"),
        ("assistant", "You said: I would like to book a table"),
    ]
    played = sum(p.frame.duration for p in transport.played_log)
    assert played == pytest.approx(sum(e.frame.duration for e in log.of(ResponseAudio)), abs=0.02)
    assert session.usage.speculation_waste_calls == 1 and session.usage.speculation_hits == 1
    assert session.usage.speculation_waste_tokens == first.output_tokens


class LateFinalSTT(STT):
    """Answers a flush with a final transcript, then with a second one a moment later
    (some cloud recognizers finalize a flushed segment in pieces)."""

    provider = "test"

    def __init__(self, first: str, second: str, *, gap: float = 0.15) -> None:
        super().__init__(model="late-final", capabilities=STTCapabilities(streaming=True))
        self.first, self.second, self.gap = first, second, gap

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _LateFinalStream(self, language=language)


class _LateFinalStream(STTStream):
    async def _run(self) -> None:
        stt: LateFinalSTT = self._stt  # type: ignore[assignment]
        heard = False
        async for item in self._input:
            if self.is_flush(item):
                if heard:
                    heard = False
                    self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, Transcript(stt.first)))
                    await asyncio.sleep(stt.gap)
                    self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, Transcript(stt.second)))
            elif isinstance(item, AudioFrame) and item.rms() > 0.01:
                heard = True


async def test_a_changed_transcript_restarts_the_speculation() -> None:
    llm = MockLLM(ttft=0.05)
    session = cascade(stt=LateFinalSTT("book a table", "for two"), llm=llm)
    log, rec = EngineLog(session), Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    # the late final changed the transcript: the reply was restarted before the commit
    assert outcomes(rec) == [(False, "transcript"), (True, None)]
    last_user = [m.last_message("user") for m in llm.requests]
    assert [m.text for m in last_user if m is not None] == [
        "book a table",
        "book a table for two",
    ]
    assert len(log.of(ResponseStarted)) == 1
    assert history(session)[-1] == ("assistant", "You said: book a table for two")


# ------------------------------------------------------------------ STT eager end of turn


class EagerSTT(STT):
    """Streaming STT with its own turn detection and eager end-of-turn events (like
    Deepgram Flux / Cartesia Ink): replays ``script`` = [(audio seconds, event, text)]."""

    provider = "test"

    def __init__(self, script: list[tuple[float, STTEventType, str]]) -> None:
        super().__init__(
            model="eager",
            capabilities=STTCapabilities(streaming=True, interim_results=True, end_of_turn=True),
        )
        self.script = script

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _EagerStream(self, language=language)


class _EagerStream(STTStream):
    async def _run(self) -> None:
        stt: EagerSTT = self._stt  # type: ignore[assignment]
        pending = list(stt.script)
        t = 0.0
        async for item in self._input:
            if not isinstance(item, AudioFrame):
                continue  # flushes: the provider owns the turn
            t += item.duration
            while pending and pending[0][0] <= t + 1e-9:
                at, kind, text = pending.pop(0)
                self._emit(STTEvent(kind, Transcript(text, end_time=at)))


def eager_script(eager: str, final: str, *, resumed: bool = False) -> list[Any]:
    E = STTEventType
    script = [
        (0.05, E.START_OF_SPEECH, ""),
        (0.3, E.INTERIM_TRANSCRIPT, eager),
        (0.6, E.EAGER_END_OF_TURN, eager),
    ]
    if resumed:
        script += [(0.7, E.TURN_RESUMED, eager), (0.8, E.INTERIM_TRANSCRIPT, final)]
    return [*script, (0.9, E.FINAL_TRANSCRIPT, final), (0.9, E.END_OF_TURN, final)]


async def eager_turn(script: list[Any], **options: Any) -> tuple[AgentSession, Recorder, Any]:
    llm = MockLLM(ttft=0.25)
    session = AgentSession(
        stt=EagerSTT(script),
        llm=llm,
        tts=MockTTS(chars_per_second=200.0),  # no VAD: the STT drives the turns
        cascade_options=CascadeOptions(**options),
    )
    log, rec = EngineLog(session), Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    t0 = now()
    await transport.play_user_audio(synth_speech(0.5, SR), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(0.5, SR))  # real time: 0.4 s later...
    await wait_for(lambda: bool(log.of(ResponseAudio)), 5)
    first_audio = log.of(ResponseAudio)[0].timestamp - t0
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    return session, rec, (llm, log, first_audio)


async def test_eager_end_of_turn_starts_the_reply_early() -> None:
    script = eager_script("Book a table for two.", "Book a table for two.")
    _, rec_off, (_, _, off) = await eager_turn(script)
    session, rec, (llm, log, on) = await eager_turn(script, preemptive_generation=True)
    assert outcomes(rec_off) == []
    assert outcomes(rec) == [(True, None)]
    assert len(llm.requests) == 1
    # EagerEndOfTurn came 0.3 s before EndOfTurn: the 0.25 s TTFT was spent meanwhile
    # (one-sided: two separate runs under CPU load can each drift by ~0.1 s)
    assert 0.12 <= off - on <= 0.45
    assert history(session)[-1] == ("assistant", "You said: Book a table for two.")
    assert log.of(InputCommitted)[0].timestamp <= log.of(ResponseStarted)[0].timestamp


@pytest.mark.parametrize("resumed", [True, False])
async def test_eager_speculation_is_dropped_when_the_final_turn_differs(resumed: bool) -> None:
    script = eager_script("Hey can you help", "Hey can you help me?", resumed=resumed)
    session, rec, (llm, log, _) = await eager_turn(script, preemptive_generation=True)
    # TURN_RESUMED cancels at once; without it the final transcript tells at the commit
    assert outcomes(rec) == [(False, "resumed" if resumed else "transcript")]
    assert [r.last_message("user").text for r in llm.requests] == [  # type: ignore[union-attr]
        "Hey can you help",
        "Hey can you help me?",
    ]
    assert len(log.of(ResponseStarted)) == 1
    assert history(session) == [
        ("user", "Hey can you help me?"),
        ("assistant", "You said: Hey can you help me?"),
    ]


class FirstCallFails(MockLLM):
    """The first request fails: when it is made (``"start"``) or while streaming."""

    def __init__(self, mode: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.mode = mode

    def _chat(self, ctx: ChatContext, **kw: Any) -> LLMStream:
        if self.requests:  # later calls work
            return super()._chat(ctx, **kw)
        self.requests.append(ctx.copy())
        if self.mode == "start":
            raise RuntimeError("rejected")
        return _Failing(self, ctx, **kw)


class _Failing(LLMStream):
    async def _run(self) -> None:
        raise RuntimeError("rate limited")


@pytest.mark.parametrize("mode", ["start", "stream"])
async def test_a_failed_speculation_falls_back_to_a_normal_reply(mode: str) -> None:
    llm = FirstCallFails(mode, ttft=0.02)
    session = cascade(llm=llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    # the failure never reaches the session: the committed turn gets a fresh reply
    assert outcomes(rec) == ([] if mode == "start" else [(False, "failed")])
    assert not rec.of("error")
    assert history(session) == [("user", "book a table"), ("assistant", "You said: book a table")]


# ----------------------------------------------------------------------------- tools


@pytest.mark.parametrize("preemptive_tts", [False, True])
async def test_speculative_tool_calls_wait_for_the_commit(preemptive_tts: bool) -> None:
    called: list[float] = []

    @function_tool
    async def get_weather(city: str) -> str:
        """Weather lookup."""
        called.append(now())
        return f"sunny in {city}"

    def script(ctx: ChatContext) -> Any:
        if any(isinstance(i, FunctionCallOutput) for i in ctx.items):
            return "It is sunny."
        return MockToolCall("get_weather", {"city": "Paris"})

    llm = MockLLM(responses=script, ttft=0.02)
    stt = MockSTT(transcripts=["weather in paris"], latency=0.02)
    session = cascade(stt=stt, llm=llm, preemptive_tts=preemptive_tts)
    log, rec = EngineLog(session), Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x", tools=[get_weather]), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)
    await asyncio.sleep(0.2)  # the speculative reply (a tool call) is complete: held back
    assert called == [] and log.of(ResponseToolCall, InputCommitted) == []
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()

    assert outcomes(rec) == [(True, None)]
    [committed] = log.of(InputCommitted)
    assert len(called) == 1 and called[0] >= committed.timestamp  # run once, after the commit
    kinds = [getattr(i, "role", i.type) for i in session.history.items]
    assert kinds == ["user", "function_call", "function_call_output", "assistant"]
    assert rec.turn_metrics()[0].tool_calls == 1


# ----------------------------------------------------------------- context and control


async def test_a_context_change_discards_the_speculation() -> None:
    llm = MockLLM(ttft=0.05)
    session = cascade(llm=llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("Be brief."), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)
    await session.update_instructions("Answer in French.")  # mid-turn: the reply is stale
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert outcomes(rec) == [(False, "context")]
    assert [r.messages()[0].text for r in llm.requests] == ["Be brief.", "Answer in French."]


async def test_truncating_the_history_discards_the_speculation() -> None:
    llm = MockLLM(ttft=0.05)
    session = cascade(llm=llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await session.say("Hello there, how can I help?")
    await wait_for(lambda: AgentState.SPEAKING in rec.states(), 5)
    await wait_for(lambda: session.agent_state == AgentState.LISTENING, 5)  # greeting over
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)
    conn: Any = session.connection
    greeting = conn.chat_ctx.items[0]
    assert await conn.truncate(greeting.id, 0) == ""  # e.g. a late barge-in verdict
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert outcomes(rec) == [(False, "context")]
    seen = [m.text for m in llm.requests[1].messages() if m.role == "assistant"]
    assert seen[0] == ""  # the reply saw the truncated greeting


async def test_clear_input_discards_the_speculation() -> None:
    llm = MockLLM(ttft=0.05)
    session = cascade(llm=llm)
    log, rec = EngineLog(session), Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)
    await session.connection.clear_input()
    await asyncio.sleep(1.2)  # past the endpointing delay: nothing is committed
    await session.aclose()
    assert outcomes(rec) == [(False, "cleared")]
    assert log.of(InputCommitted, *RESPONSE_EVENTS) == [] and session.history.items == []


@pytest.mark.parametrize("action", ["cancel_response", "send_text"])
async def test_cancel_and_new_user_text_discard_the_speculation(action: str) -> None:
    llm = MockLLM(ttft=0.05)
    session = cascade(llm=llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)
    conn = session.connection
    if action == "cancel_response":
        await conn.cancel_response()
    else:
        await conn.send_text("typed meanwhile", respond=False)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)  # the pending turn goes on
    await session.aclose()
    assert outcomes(rec) == [(False, "cancelled" if action == "cancel_response" else "context")]
    assert len(llm.requests) == 2  # the committed turn got a fresh reply
    users = [m.text for m in llm.requests[1].messages() if m.role == "user"]
    typed = ["typed meanwhile"] if action == "send_text" else []
    assert users == [*typed, "book a table"]


# ------------------------------------------------------------------------ gating/budget


async def test_budget_limits_attempts_per_turn() -> None:
    llm = MockLLM(ttft=0.02)
    stt = MockSTT(transcripts=["one", "two", "three", "four"], latency=0.02)
    session = cascade(stt=stt, llm=llm)  # preemptive_max_attempts=3
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    for k in range(4):  # the user pauses four times in one turn
        await speak(transport, 0.3, 0.35)
        if k < 3:
            await wait_for(lambda k=k: len(llm.requests) == k + 1)  # type: ignore[misc]
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert outcomes(rec) == [(False, "resumed")] * 3  # no fourth attempt
    assert len(llm.requests) == 4  # 3 speculative + the reply started at the commit
    assert history(session)[0] == ("user", "one two three four")
    assert session.usage.speculation_waste_calls == 3


async def test_no_speculation_on_long_turns() -> None:
    llm = MockLLM(ttft=0.02)
    session = cascade(llm=llm, preemptive_max_speech=0.8)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 1.0, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert outcomes(rec) == [] and len(llm.requests) == 1


@pytest.mark.parametrize(
    ("probability", "threshold", "speculates"),
    [(0.9, None, True), (0.3, None, False), (0.3, 0.2, True)],
)
async def test_turn_detector_gates_the_speculation(
    probability: float, threshold: float | None, speculates: bool
) -> None:
    llm = MockLLM(ttft=0.02)
    session = cascade(
        llm=llm,
        turn_detector=MockTurnDetector(probability=probability),  # threshold 0.5
        min_endpointing_delay=0.6,
        max_endpointing_delay=0.9,
        preemptive_threshold=threshold,
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert outcomes(rec) == ([(True, None)] if speculates else [])
    assert len(llm.requests) == 1


async def test_no_speculation_while_the_agent_is_talking() -> None:
    llm = MockLLM(ttft=0.02)
    session = cascade(llm=llm, tts=MockTTS(), min_endpointing_delay=0.6)  # ~4.5 s notice
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await session.say("This notice takes a few seconds to play, and the user talks over it.")
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await speak(transport, 0.3, 0.4)  # overlapping speech: the interruption policy decides
    await wait_for(lambda: len(llm.requests) == 1, 5)  # the turn was committed and answered
    await session.aclose()
    assert outcomes(rec) == []


async def test_disabled_by_default_and_without_stt() -> None:
    assert CascadeOptions().preemptive_generation is False
    assert CascadeOptions().preemptive_tts is False
    llm = MockLLM(ttft=0.02)
    session = cascade(llm=llm, preemptive_generation=False)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    assert outcomes(rec) == [] and len(llm.requests) == 1

    class AudioLLM(MockLLM):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.capabilities = LLMCapabilities(audio_input=True)

    audio_llm = AudioLLM(responses=["I heard you."])
    half = AgentSession(
        llm=audio_llm,
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(),
        cascade_options=CascadeOptions(min_endpointing_delay=0.5, preemptive_generation=True),
    )
    rec = Recorder(half)
    transport = LoopbackTransport()
    await half.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await half.aclose()
    assert outcomes(rec) == []  # nothing to compare the committed turn with: no speculation


# --------------------------------------------------------------- barge-in on a speculation


async def test_barge_in_truncates_a_speculative_reply_like_any_other() -> None:
    answer = "This is a very long answer that keeps going and going for quite a while. " * 2
    llm = MockLLM(responses=lambda ctx: answer, ttft=0.05)
    session = cascade(
        llm=llm,
        stt=MockSTT(transcripts=["tell me a story"], latency=0.02),
        tts=MockTTS(realtime_factor=1.0),
        min_endpointing_delay=0.5,
    )
    rec = Recorder(session)
    transport = LoopbackTransport(realtime_playout=True)
    await session.start(Agent("x"), transport)
    await speak(transport, 0.6, 0.5)
    await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
    await asyncio.sleep(1.0)
    await transport.play_user_audio(synth_speech(0.7, SR), realtime=False)  # barge in
    await wait_for(lambda: bool(rec.of("interrupted")), 5)
    await session.aclose()

    assert outcomes(rec)[0] == (True, None)
    ev = rec.of("interrupted")[0]
    conn: Any = session.connection
    msg = conn.chat_ctx.get(ev.item_id)
    assert isinstance(msg, ChatMessage) and msg.interrupted
    assert 0 < len(msg.text) < len(answer.strip()) and answer.startswith(msg.text)
    [said] = [i for i in session.history.items if getattr(i, "role", "") == "assistant"]
    assert said.interrupted and said.text == msg.text  # the heard text, as for any reply


# -------------------------------------------------------------------------- metrics


async def test_speculation_metrics_match_the_llm_calls() -> None:
    llm = MockLLM(ttft=0.05)
    stt = MockSTT(transcripts=["I would like to", "book a table"], latency=0.02)
    session = cascade(stt=stt, llm=llm)
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: len(llm.requests) == 1)
    await asyncio.sleep(0.15)
    await speak(transport, 0.5, 0.4)
    await wait_for(lambda: bool(rec.turn_metrics()), 5)
    await session.aclose()
    llm_calls = {m.request_id: m for m in rec.of("metrics") if isinstance(m, LLMMetrics)}
    wasted, kept = speculations(rec)
    assert {wasted.request_id, kept.request_id} <= set(llm_calls)  # one per LLM request
    assert kept.response_id is not None and wasted.response_id is None
    assert wasted.lead > 0.1 and kept.lead > 0
    assert wasted.output_tokens == 6  # "You said: I would like to", one chunk per word
    assert llm_calls[wasted.request_id].completion_tokens == wasted.output_tokens
    usage = session.usage
    assert (usage.speculation_hits, usage.speculation_waste_calls) == (1, 1)
    assert usage.speculation_waste_tokens == wasted.output_tokens


def test_usage_summary_counts_speculations() -> None:
    usage = UsageSummary()
    kept = SpeculationMetrics(provider="p", model="m", request_id="a", hit=True, output_tokens=7)
    dropped = SpeculationMetrics(
        provider="p", model="m", request_id="b", hit=False, reason="resumed", output_tokens=3
    )
    for m in (kept, dropped, dropped):
        usage.add(m)
    assert usage.speculation_hits == 1
    assert (usage.speculation_waste_calls, usage.speculation_waste_tokens) == (2, 6)
