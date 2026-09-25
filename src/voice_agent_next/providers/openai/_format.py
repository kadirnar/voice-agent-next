"""Conversion from voice-agent-next chat types to the OpenAI Chat Completions format.

Pure functions without any dependency on the ``openai`` SDK, shared by
:class:`~voice_agent_next.providers.openai.llm.OpenAILLM` and every OpenAI-compatible
host.

:func:`to_chat_messages` turns the item-based :class:`~voice_agent_next.chat.ChatContext`
into Chat Completions messages:

* ``system``/``developer`` messages keep their role (``developer`` can be mapped to
  ``system`` for servers whose chat templates only know ``system``) and, by default, their
  position; ``system_messages`` moves them for chat templates that only accept one
  leading system message;
* ``user`` content becomes a plain string, or a list of ``text``/``image_url``/
  ``input_audio`` parts when it carries media. How audio is encoded (WAV or raw PCM,
  sample rate, plain base64 or a ``data:`` URL) depends on the host
  (:class:`AudioInputFormat`); older audio turns can be replaced by their transcripts
  (``audio_history``);
* consecutive assistant text and :class:`~voice_agent_next.chat.FunctionCall` items
  are grouped into one assistant message with ``tool_calls``, and every call's
  :class:`~voice_agent_next.chat.FunctionCallOutput` is placed right after it as a
  ``tool`` message (the order the API requires);
* interrupted assistant messages are sent as heard (their content is already
  truncated by the session); messages left empty are skipped.

The API rejects a request whose tool calls lack outputs (or whose tool outputs lack a
call), which happens when a new turn starts while a tool is still running or after
history truncation. Such unmatched items are dropped instead of failing the request.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ...audio.frame import AudioFrame
from ...audio.resample import resample
from ...audio.wav import wav_bytes
from ...chat import (
    AudioContent,
    ChatContext,
    ChatItem,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from ...llm import ToolChoice
from ...tools import FunctionTool
from ...utils.log import logger

__all__ = [
    "AudioInputFormat",
    "SystemMessagePolicy",
    "audio_content_part",
    "to_chat_messages",
    "to_chat_tools",
    "to_tool_choice",
]

SystemMessagePolicy: TypeAlias = Literal["keep", "merge", "as_user"]
"""Where ``system``/``developer`` messages go (see :func:`to_chat_messages`)."""


def to_chat_tools(tools: Sequence[FunctionTool]) -> list[dict[str, Any]]:
    """``FunctionTool`` list -> Chat Completions ``tools`` (``{"type": "function", ...}``)."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        function: dict[str, Any] = {"name": tool.name}
        if tool.description:
            function["description"] = tool.description
        function["parameters"] = tool.parameters or {"type": "object", "properties": {}}
        if tool.strict:
            function["strict"] = True
        out.append({"type": "function", "function": function})
    return out


def to_tool_choice(choice: ToolChoice | None) -> str | dict[str, Any] | None:
    """``"auto"``/``"required"``/``"none"`` pass through; any other string names a function."""
    if choice is None:
        return None
    if choice in ("auto", "required", "none"):
        return choice
    return {"type": "function", "function": {"name": choice}}


@dataclass(frozen=True, slots=True)
class AudioInputFormat:
    """How user audio is encoded in ``input_audio`` parts (it differs per host).

    * OpenAI (``gpt-audio``, ``gpt-4o-audio``), vLLM / vLLM-Omni and llama.cpp take
      base64 WAV (``format="wav"``) and resample it themselves;
    * DashScope (Qwen-Omni) wants the base64 as a ``data:;base64,...`` URL
      (``data_url=True``);
    * ``format="pcm16"`` sends raw little-endian 16-bit PCM without a header, for servers
      that take it (the sample rate is then implicit: set ``sample_rate`` to the one the
      server expects).
    """

    format: Literal["wav", "pcm16"] = "wav"
    """Container of the base64 payload (and the part's ``format`` field)."""
    sample_rate: int | None = None
    """Resample to this rate first (``None``: keep the input rate). Audio encoders of
    open models (Whisper-style: Qwen-Omni, Ultravox, Voxtral, Gemma) run at 16 kHz."""
    data_url: bool = False
    """Send ``data:;base64,<payload>`` instead of the bare base64 string."""

    def encode(self, frame: AudioFrame) -> dict[str, Any]:
        """The ``input_audio`` object for one clip."""
        if frame.channels > 1:
            frame = frame.to_mono()
        if self.sample_rate and frame.sample_rate != self.sample_rate:
            frame = resample(frame, self.sample_rate)
        payload = wav_bytes(frame) if self.format == "wav" else frame.data
        data = base64.b64encode(payload).decode("ascii")
        if self.data_url:
            data = f"data:;base64,{data}"
        return {"data": data, "format": self.format}


def audio_content_part(
    frame: AudioFrame, audio_format: AudioInputFormat | None = None
) -> dict[str, Any]:
    """An ``input_audio`` content part for audio-input chat models (base64 WAV by default)."""
    return {
        "type": "input_audio",
        "input_audio": (audio_format or AudioInputFormat()).encode(frame),
    }


@dataclass
class _AssistantTurn:
    text: str = ""
    calls: list[FunctionCall] = field(default_factory=list)


def to_chat_messages(
    ctx: ChatContext | Iterable[ChatItem],
    *,
    developer_role: Literal["developer", "system"] = "developer",
    audio_input: bool = False,
    system_messages: SystemMessagePolicy = "keep",
    audio_format: AudioInputFormat | None = None,
    audio_history: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a chat context to Chat Completions ``messages``.

    Args:
        ctx: the conversation (a :class:`ChatContext` or any iterable of items).
        developer_role: role used for ``developer`` messages. OpenAI accepts
            ``"developer"``; most OpenAI-compatible servers only know ``"system"``.
        audio_input: send :class:`AudioContent` as ``input_audio`` parts. When False
            (text-only models), the audio's transcript is sent instead, if it has one.
        system_messages: ``"keep"`` leaves system/developer messages where they are.
            Many chat templates (Qwen, Gemma, … served by llama.cpp, vLLM or LM Studio)
            reject a system message after the first message, which the cascade sends for
            per-response instructions. ``"merge"`` joins them all into one leading system
            message; ``"as_user"`` merges only the leading ones and sends later ones as
            user messages (adjacent user messages are then joined, for templates that
            require alternating roles).
        audio_format: encoding of ``input_audio`` parts (default: base64 WAV at the
            input rate).
        audio_history: with ``audio_input``, send only the most recent ``audio_history``
            audio clips as audio; older clips that have a transcript are sent as text
            (a clip without one stays audio, so the model never loses a turn). Re-sending
            every earlier turn's audio makes each request slower and dearer as the
            conversation grows. ``None``: all clips as audio.
    """
    items = list(ctx.items if isinstance(ctx, ChatContext) else ctx)
    as_text: set[int] = set()  # ids of AudioContent sent as their transcript
    if audio_input and audio_history is not None:
        clips = [
            c
            for item in items
            if isinstance(item, ChatMessage) and item.role == "user"
            for c in item.content
            if isinstance(c, AudioContent) and c.frame
        ]
        older = clips[: max(0, len(clips) - max(0, audio_history))]
        as_text = {id(c) for c in older if c.transcript}
    audio = _AudioOptions(audio_input, audio_format, as_text)
    outputs: dict[str, FunctionCallOutput] = {}
    for item in items:
        if isinstance(item, FunctionCallOutput):
            outputs.setdefault(item.call_id, item)

    messages: list[dict[str, Any]] = []
    answered: set[str] = set()
    turn: _AssistantTurn | None = None

    def flush() -> None:
        nonlocal turn
        if turn is None:
            return
        calls: list[FunctionCall] = []
        for call in turn.calls:
            if call.call_id in answered or call.call_id not in outputs:
                logger.debug("dropping tool call %s (%s) without output", call.call_id, call.name)
                continue
            answered.add(call.call_id)
            calls.append(call)
        message: dict[str, Any] = {"role": "assistant"}
        if turn.text:
            message["content"] = turn.text
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments or "{}"},
                }
                for call in calls
            ]
        if turn.text or calls:
            messages.append(message)
            for call in calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": outputs[call.call_id].output,
                    }
                )
        turn = None

    for item in items:
        if isinstance(item, FunctionCall):
            if turn is None:
                turn = _AssistantTurn()
            turn.calls.append(item)
        elif isinstance(item, FunctionCallOutput):
            flush()  # outputs are emitted right after the call they answer
        elif item.role == "assistant":
            text = item.text
            if not text.strip():
                continue
            if turn is not None and turn.calls and not turn.text:
                turn.text = text  # text streamed alongside the tool calls of one response
            else:
                flush()
                turn = _AssistantTurn(text=text)
        else:
            flush()
            message = _to_message(item, developer_role=developer_role, audio=audio)
            if message is not None:
                messages.append(message)
    flush()

    orphans = [cid for cid in outputs if cid not in answered]
    if orphans:
        logger.debug("dropping tool outputs without a matching call: %s", orphans)
    if system_messages == "keep":
        return messages
    return _place_system_messages(messages, as_user=system_messages == "as_user")


def _place_system_messages(
    messages: list[dict[str, Any]], *, as_user: bool
) -> list[dict[str, Any]]:
    system: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] not in ("system", "developer"):
            rest.append(message)
        elif not as_user or not rest:  # merged into the leading system message
            system.append(message)
        else:
            rest.append({"role": "user", "content": message["content"]})
    out: list[dict[str, Any]] = []
    if system:
        text = "\n\n".join(m["content"] for m in system)
        out.append({"role": system[0]["role"], "content": text})
    for message in rest:
        prev = out[-1] if out else None
        if as_user and prev is not None and prev["role"] == message["role"] == "user":
            prev["content"] = _join_user_content(prev["content"], message["content"])
        else:
            out.append(message)
    return out


def _join_user_content(a: str | list[dict[str, Any]], b: str | list[dict[str, Any]]) -> Any:
    if isinstance(a, str) and isinstance(b, str):
        return f"{a}\n\n{b}"

    def parts(c: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"type": "text", "text": c}] if isinstance(c, str) else list(c)

    return parts(a) + parts(b)


@dataclass(slots=True)
class _AudioOptions:
    enabled: bool
    format: AudioInputFormat | None
    as_text: set[int]


def _to_message(
    msg: ChatMessage, *, developer_role: str, audio: _AudioOptions
) -> dict[str, Any] | None:
    if msg.role in ("system", "developer"):
        text = msg.text
        if not text.strip():
            return None
        return {"role": developer_role if msg.role == "developer" else "system", "content": text}
    content = _user_content(msg, audio=audio)
    if content is None:
        logger.debug("skipping empty %s message %s", msg.role, msg.id)
        return None
    return {"role": "user", "content": content}


def _user_content(msg: ChatMessage, *, audio: _AudioOptions) -> str | list[dict[str, Any]] | None:
    parts: list[dict[str, Any]] = []
    media = False
    for c in msg.content:
        if isinstance(c, str):
            if c:
                parts.append({"type": "text", "text": c})
        elif isinstance(c, AudioContent):
            if audio.enabled and c.frame and id(c) not in audio.as_text:
                parts.append(audio_content_part(c.frame, audio.format))
                media = True
            elif c.transcript:
                parts.append({"type": "text", "text": c.transcript})
        elif isinstance(c, ImageContent):
            parts.append({"type": "image_url", "image_url": {"url": c.url}})
            media = True
    if not media:
        text = "".join(p["text"] for p in parts)
        return text if text.strip() else None
    return parts
