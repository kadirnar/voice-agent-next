"""A fake Amazon Nova 2 Sonic bidirectional stream for offline tests and demos.

:class:`FakeNovaSonic` is a ``stream_factory`` for
:class:`~voice_agent_next.providers.aws.nova_sonic.NovaSonicSessionEngine`: it replaces the
Bedrock SDK with an in-process :class:`FakeNovaStream` that speaks the Nova Sonic event
protocol (``{"event": {...}}`` JSON events) and behaves like the model in the ways an
engine cares about:

* it **validates the input events** (``sessionStart`` -> ``promptStart`` -> system prompt ->
  chat history -> one audio block; matching ``promptName``/``contentName``; history only
  before audio; tool results only for issued ``toolUseId``s, as JSON objects; the closing
  sequence ``contentEnd`` (audio) -> ``promptEnd`` -> ``sessionEnd``) and records
  violations in :attr:`FakeNovaSonic.errors` (tests assert it stays empty);
* **endpointing** is energy based: after speech followed by ``endpoint_silence`` seconds
  of silence the next scripted :class:`FakeNovaTurn` is answered: the user transcript
  (``USER``/``FINAL`` block), optionally a ``toolUse`` (the reply waits for the
  ``toolResult``), then the reply as a ``SPECULATIVE`` text block, an audio block (paced in
  real time, ``chunk`` seconds per event) and the ``FINAL`` transcript, plus ``usageEvent``;
* **barge-in**: user speech of ``barge_in_after`` seconds while the fake speaks stops the
  audio with ``contentEnd(stopReason="INTERRUPTED")`` (or, with ``legacy_interrupt``, the
  Nova Sonic v1 ``textOutput`` of ``{ "interrupted" : true }``);
* cross-modal text (``interactive`` USER text): ``Say exactly ... "X"`` makes it say X,
  anything else answers the next scripted turn (or "OK.");
* failures on request: ``fail_open`` (raised by the factory), :meth:`FakeNovaStream.drop`
  (the stream ends), :meth:`FakeNovaStream.fail` (``receive`` raises) and
  ``session_limit`` (the stream fails after that many seconds, like the 8-minute limit).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..audio.frame import AudioFrame
from ..errors import ProviderConnectionError
from ..providers.mock import synth_speech
from ..utils.aio import cancel_and_wait

if TYPE_CHECKING:
    from ..providers.aws.nova_sonic import NovaSonicSessionEngine

__all__ = ["FakeNovaSonic", "FakeNovaStream", "FakeNovaTurn", "FakeToolUse"]

_SAY = re.compile(r'Say exactly the following.*?"(.*)"', re.S)


@dataclass
class FakeToolUse:
    """A tool call in a scripted turn; the answer is spoken once the result arrived."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str = "{result}"
    """Spoken after the ``toolResult`` (``{result}`` = its ``result`` field or content)."""


@dataclass
class FakeNovaTurn:
    """One scripted model turn."""

    reply: str
    """Spoken reply ("" for none; with a tool: said while the tool runs)."""
    user: str | None = "hello"
    """User transcript sent before the reply (``None``: no transcript)."""
    tool: FakeToolUse | None = None


class FakeNovaSonic:
    """A ``stream_factory`` producing :class:`FakeNovaStream` sessions (see module docs)."""

    def __init__(
        self,
        turns: Sequence[str | FakeNovaTurn] = (),
        *,
        speech_threshold_db: float = -35.0,
        endpoint_silence: float = 0.4,
        barge_in_after: float = 0.25,
        word_duration: float = 0.25,
        chunk: float = 0.1,
        session_limit: float | None = None,
        fail_open: Exception | None = None,
        legacy_interrupt: bool = False,
        echo_text: bool = False,
    ) -> None:
        self.turns: deque[FakeNovaTurn] = deque(
            t if isinstance(t, FakeNovaTurn) else FakeNovaTurn(t) for t in turns
        )
        self.speech_threshold_db = speech_threshold_db
        self.endpoint_silence = endpoint_silence
        self.barge_in_after = barge_in_after
        self.word_duration = word_duration
        self.chunk = chunk
        self.session_limit = session_limit
        self.fail_open = fail_open
        self.legacy_interrupt = legacy_interrupt
        self.echo_text = echo_text
        self.sessions: list[FakeNovaStream] = []
        self.errors: list[str] = []

    async def __call__(self, engine: NovaSonicSessionEngine) -> FakeNovaStream:
        if self.fail_open is not None:
            raise self.fail_open
        stream = FakeNovaStream(self, engine.model, engine.output_sample_rate)
        self.sessions.append(stream)
        return stream

    def next_turn(self) -> FakeNovaTurn | None:
        return self.turns.popleft() if self.turns else None

    async def aclose(self) -> None:
        for session in self.sessions:
            await session.close()


class FakeNovaStream:
    """One fake bidirectional stream (a Nova Sonic ``NovaStream``)."""

    def __init__(self, fake: FakeNovaSonic, model: str, output_rate: int) -> None:
        self.fake = fake
        self.model = model
        self.output_rate = output_rate
        self.session_id = f"sess_{len(fake.sessions) + 1:03d}"
        self.events: list[dict[str, Any]] = []
        """Every input event received (``{name: body}``)."""
        self.session_start: dict[str, Any] = {}
        self.prompt_start: dict[str, Any] = {}
        self.system = ""
        self.history: list[tuple[str, str]] = []
        self.text_messages: list[str] = []
        self.tool_results: dict[str, str] = {}
        self.audio_config: dict[str, Any] = {}
        self.audio_received = 0.0
        self.graceful = False
        """The client sent the full closing sequence."""
        self.closed = False
        self._out: asyncio.Queue[dict[str, Any] | Exception | None] = asyncio.Queue()
        self._prompt_name: str | None = None
        self._blocks: dict[str, dict[str, Any]] = {}
        self._audio_block: str | None = None
        self._audio_done = False
        self._prompt_ended = False
        self._completion = str(uuid.uuid4())
        self._completion_started = False
        self._tools: dict[str, asyncio.Future[str]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._speech_run = 0.0
        self._silence_run = 0.0
        self._heard = False
        self._speaking = False
        self._barged = False
        self._busy = False
        self._usage = {"input": [0, 0], "output": [0, 0]}  # [speech, text]
        if fake.session_limit is not None:
            self._spawn(self._limit(fake.session_limit))

    # ------------------------------------------------------------------ interface
    async def send(self, event: Mapping[str, Any]) -> None:
        if self.closed:
            self._error("event sent after the stream was closed")
            raise ProviderConnectionError("fake nova: the stream is closed")
        body_map = event.get("event")
        if not isinstance(body_map, Mapping) or len(body_map) != 1:
            self._error(f"malformed event {event!r}")
            return
        ((name, body),) = body_map.items()
        self.events.append({name: body})
        handler = getattr(self, f"_in_{name}", None)
        if handler is None:
            self._error(f"unknown input event {name}")
            return
        if name not in ("sessionStart", "sessionEnd"):
            if not self.prompt_start and name != "promptStart":
                self._error(f"{name} before promptStart")
            elif name != "promptStart" and body.get("promptName") != self._prompt_name:
                self._error(f"{name}: unknown promptName {body.get('promptName')!r}")
        handler(body)

    async def receive(self) -> dict[str, Any] | None:
        item = await self._out.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        await cancel_and_wait(*self._tasks)
        self._out.put_nowait(None)

    # ------------------------------------------------------------------ test hooks
    def drop(self) -> None:
        """End the output stream abruptly (as a network drop would)."""
        self._out.put_nowait(None)

    def fail(self, error: Exception) -> None:
        """Make ``receive`` raise ``error`` (a modeled error in the stream)."""
        self._out.put_nowait(error)

    def of(self, name: str) -> list[dict[str, Any]]:
        return [e[name] for e in self.events if name in e]

    # --------------------------------------------------------------- input events
    def _error(self, message: str) -> None:
        self.fake.errors.append(f"{self.session_id}: {message}")

    def _in_sessionStart(self, body: dict[str, Any]) -> None:
        if len(self.events) != 1:
            self._error("sessionStart is not the first event")
        self.session_start = body

    def _in_promptStart(self, body: dict[str, Any]) -> None:
        if not self.session_start or self.prompt_start:
            self._error("promptStart out of order")
        self.prompt_start = body
        self._prompt_name = body.get("promptName")
        audio = body.get("audioOutputConfiguration") or {}
        if audio.get("sampleRateHertz") != self.output_rate:
            self._error(f"unexpected output rate {audio.get('sampleRateHertz')}")
        tools = (body.get("toolConfiguration") or {}).get("tools") or []
        for tool in tools:
            schema = tool.get("toolSpec", {}).get("inputSchema", {}).get("json")
            if not isinstance(schema, str):
                self._error("toolSpec.inputSchema.json must be a JSON string")

    def _in_contentStart(self, body: dict[str, Any]) -> None:
        name = body.get("contentName")
        if not name or name in self._blocks:
            self._error(f"contentStart with a missing/duplicate contentName {name!r}")
            return
        kind, role = body.get("type"), body.get("role")
        interactive = body.get("interactive")
        block: dict[str, Any] = {"type": kind, "role": role, "interactive": interactive, "text": []}
        if kind == "AUDIO":
            if self._audio_block is not None or self._audio_done:
                self._error("a second audio block")
            self._audio_block = name
            self.audio_config = body.get("audioInputConfiguration") or {}
        elif kind == "TEXT":
            late = self._audio_block is not None
            if late and (role == "SYSTEM" or not body.get("interactive")):
                self._error(f"{role} text (system prompt/history) after audio started")
        elif kind == "TOOL":
            config = body.get("toolResultInputConfiguration") or {}
            use_id = config.get("toolUseId")
            if use_id not in self._tools:
                self._error(f"tool result for an unknown toolUseId {use_id!r}")
            block["tool_use_id"] = use_id
        else:
            self._error(f"unknown content type {kind!r}")
        self._blocks[name] = block

    def _block(self, body: dict[str, Any], kind: str) -> dict[str, Any] | None:
        block = self._blocks.get(body.get("contentName", ""))
        if block is None or block["type"] != kind:
            self._error(f"{kind} event for an unknown/closed block {body.get('contentName')!r}")
            return None
        return block

    def _in_textInput(self, body: dict[str, Any]) -> None:
        block = self._block(body, "TEXT")
        content = body.get("content")
        if block is None or not isinstance(content, str):
            return
        if len(content.encode()) > 50_000:
            self._error("textInput larger than 50 KB")
        block["text"].append(content)

    def _in_audioInput(self, body: dict[str, Any]) -> None:
        if body.get("contentName") != self._audio_block:
            self._error("audioInput outside the audio block")
            return
        data = base64.b64decode(body.get("content") or "")
        rate = int(self.audio_config.get("sampleRateHertz") or 16_000)
        self._on_audio(AudioFrame(data, rate))

    def _in_toolResult(self, body: dict[str, Any]) -> None:
        block = self._block(body, "TOOL")
        content = body.get("content")
        if block is None or not isinstance(content, str):
            return
        try:
            parsed = json.loads(content)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            self._error("toolResult content is not a JSON object")
        use_id = str(block.get("tool_use_id"))
        self.tool_results[use_id] = content
        fut = self._tools.get(use_id)
        if fut is not None and not fut.done():
            fut.set_result(content)

    def _in_contentEnd(self, body: dict[str, Any]) -> None:
        name = body.get("contentName", "")
        block = self._blocks.pop(name, None)
        if block is None:
            self._error(f"contentEnd for an unknown block {name!r}")
            return
        if block["type"] == "AUDIO":
            self._audio_block = None
            self._audio_done = True
        elif block["type"] == "TEXT":
            text = "".join(block["text"])
            if block["role"] == "SYSTEM":
                self.system = text
            elif block["interactive"] and block["role"] == "USER":
                self.text_messages.append(text)
                self._on_text_message(text)
            else:
                self.history.append((str(block["role"]), text))

    def _in_promptEnd(self, body: dict[str, Any]) -> None:
        if self._audio_block is not None:
            self._error("promptEnd while the audio block is open")
        self._prompt_ended = True

    def _in_sessionEnd(self, body: dict[str, Any]) -> None:
        if not self._prompt_ended:
            self._error("sessionEnd before promptEnd")
        self.graceful = True

    # --------------------------------------------------------------------- model
    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _limit(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.fail(ProviderConnectionError("aws: the connection reached its time limit"))

    def _on_audio(self, frame: AudioFrame) -> None:
        self.audio_received += frame.duration
        self._usage["input"][0] += 1
        fake = self.fake
        if frame.dbfs() >= fake.speech_threshold_db:
            self._speech_run += frame.duration
            self._silence_run = 0.0
            if self._speaking:
                if self._speech_run >= fake.barge_in_after and not self._barged:
                    self._barged = True
                    self._heard = True
            elif self._speech_run >= 0.1:
                self._heard = True
        else:
            self._silence_run += frame.duration
            if self._silence_run > 0.15:
                self._speech_run = 0.0
        if self._heard and self._silence_run >= fake.endpoint_silence and not self._busy:
            self._heard = False
            self._start_turn(self.fake.next_turn() or FakeNovaTurn("OK.", user=None))

    def _on_text_message(self, text: str) -> None:
        if self.fake.echo_text:
            self._spawn(self._user_block(text))
        match = _SAY.search(text)
        if match:
            turn = FakeNovaTurn(match.group(1), user=None)
        else:
            turn = self.fake.next_turn() or FakeNovaTurn("OK.", user=None)
            turn = FakeNovaTurn(turn.reply, user=None, tool=turn.tool)
        self._start_turn(turn)

    def _start_turn(self, turn: FakeNovaTurn) -> None:
        self._busy = True
        self._spawn(self._run_turn(turn))

    def _put(self, name: str, **body: Any) -> None:
        body = {
            "sessionId": self.session_id,
            "promptName": self._prompt_name,
            "completionId": self._completion,
            **body,
        }
        self._out.put_nowait({"event": {name: body}})

    def _stage(self, stage: str) -> str:
        return json.dumps({"generationStage": stage})

    async def _user_block(self, text: str) -> None:
        cid = str(uuid.uuid4())
        self._put("contentStart", contentId=cid, type="TEXT", role="USER",
                  additionalModelFields=self._stage("FINAL"),
                  textOutputConfiguration={"mediaType": "text/plain"})  # fmt: skip
        self._put("textOutput", contentId=cid, content=text, role="USER")
        self._put("contentEnd", contentId=cid, type="TEXT", stopReason="PARTIAL_TURN")

    async def _run_turn(self, turn: FakeNovaTurn) -> None:
        try:
            if not self._completion_started:
                self._completion_started = True
                self._put("completionStart")
            if turn.user:
                await self._user_block(turn.user)
            if turn.tool is None:
                await self._say(turn.reply)
                return
            tool = turn.tool
            use_id = f"tooluse_{uuid.uuid4().hex[:8]}"
            self._tools[use_id] = asyncio.get_running_loop().create_future()
            cid = str(uuid.uuid4())
            self._put("contentStart", contentId=cid, type="TOOL", role="TOOL",
                      toolUseOutputConfiguration={"mediaType": "application/json"})  # fmt: skip
            self._put("toolUse", contentId=cid, toolName=tool.name, toolUseId=use_id,
                      content=json.dumps(tool.arguments))  # fmt: skip
            self._put("contentEnd", contentId=cid, type="TOOL", stopReason="TOOL_USE")
            if turn.reply:
                await self._say(turn.reply)
            self._busy = False  # the user may talk while the tool runs
            content = await asyncio.wait_for(self._tools[use_id], 30)
            self._busy = True
            try:
                data = json.loads(content)
            except ValueError:
                data = content
            result = data.get("result", content) if isinstance(data, dict) else content
            await self._say(tool.answer.format(result=result))
        finally:
            self._busy = False

    async def _say(self, text: str) -> bool:
        """Speak ``text`` sentence by sentence; ``False`` if the user barged in."""
        sentences = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
        self._speaking, self._barged = True, False
        try:
            for i, sentence in enumerate(sentences):
                if not await self._say_sentence(sentence, last=i == len(sentences) - 1):
                    return False
        finally:
            self._speaking = False
        return True

    async def _say_sentence(self, text: str, *, last: bool) -> bool:
        """One sentence: speculative text, audio (paced in real time), final text."""
        fake = self.fake
        spec = str(uuid.uuid4())
        self._put("contentStart", contentId=spec, type="TEXT", role="ASSISTANT",
                  additionalModelFields=self._stage("SPECULATIVE"),
                  textOutputConfiguration={"mediaType": "text/plain"})  # fmt: skip
        self._put("textOutput", contentId=spec, content=text, role="ASSISTANT")
        self._put("contentEnd", contentId=spec, type="TEXT", stopReason="PARTIAL_TURN")
        audio = str(uuid.uuid4())
        self._put("contentStart", contentId=audio, type="AUDIO", role="ASSISTANT",
                  audioOutputConfiguration={"mediaType": "audio/lpcm",
                                            "sampleRateHertz": self.output_rate,
                                            "sampleSizeBits": 16, "encoding": "base64",
                                            "channelCount": 1})  # fmt: skip
        words = text.split()
        steps = max(1, round(max(fake.chunk, len(words) * fake.word_duration) / fake.chunk))
        spoken = steps
        for i in range(steps):
            if self._barged:
                spoken = i
                break
            pcm = synth_speech(fake.chunk, self.output_rate, frequency=180.0,
                               offset=i * round(fake.chunk * self.output_rate))  # fmt: skip
            self._put("audioOutput", contentId=audio, content=pcm.to_base64())
            self._usage["output"][0] += 1
            await asyncio.sleep(fake.chunk)
        interrupted = spoken < steps
        self._usage["output"][1] += len(words)
        self._put_usage()
        end = "END_TURN" if last else "PARTIAL_TURN"
        if not interrupted:
            self._put("contentEnd", contentId=audio, type="AUDIO", stopReason=end)
        elif fake.legacy_interrupt:
            marker = str(uuid.uuid4())
            self._put("contentStart", contentId=marker, type="TEXT", role="ASSISTANT",
                      additionalModelFields=self._stage("SPECULATIVE"))  # fmt: skip
            self._put("textOutput", contentId=marker, content='{ "interrupted" : true }')
            self._put("contentEnd", contentId=marker, type="TEXT", stopReason="PARTIAL_TURN")
            self._put("contentEnd", contentId=audio, type="AUDIO", stopReason="END_TURN")
        else:
            self._put("contentEnd", contentId=audio, type="AUDIO", stopReason="INTERRUPTED")
        heard = " ".join(words[: round(len(words) * spoken / steps)])
        final = str(uuid.uuid4())
        self._put("contentStart", contentId=final, type="TEXT", role="ASSISTANT",
                  additionalModelFields=self._stage("FINAL"),
                  textOutputConfiguration={"mediaType": "text/plain"})  # fmt: skip
        if heard:
            self._put("textOutput", contentId=final, content=heard, role="ASSISTANT")
        if interrupted and not fake.legacy_interrupt:
            end = "INTERRUPTED"
        self._put("contentEnd", contentId=final, type="TEXT", stopReason=end)
        return not interrupted

    def _put_usage(self) -> None:
        total = {
            side: {"speechTokens": counts[0], "textTokens": counts[1]}
            for side, counts in self._usage.items()
        }
        totals = [sum(c) for c in self._usage.values()]
        with contextlib.suppress(Exception):
            self._put(
                "usageEvent",
                details={"total": total},
                totalInputTokens=totals[0],
                totalOutputTokens=totals[1],
                totalTokens=sum(totals),
            )
