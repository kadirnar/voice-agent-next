"""Conversation state shared by LLMs, engines and the session.

A :class:`ChatContext` is an ordered list of *items*:

* :class:`ChatMessage` – a system/developer/user/assistant message (text and/or audio),
* :class:`FunctionCall` – a tool call requested by the model,
* :class:`FunctionCallOutput` – the result of executing a tool call.

This item-based model maps cleanly onto both OpenAI Chat Completions (tool calls
grouped into an assistant message) and item-based APIs (OpenAI Responses/Realtime,
Gemini, Anthropic content blocks). Provider modules own the conversion.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from .audio.frame import AudioFrame
from .utils.ids import new_id

__all__ = [
    "AudioContent",
    "ChatContext",
    "ChatItem",
    "ChatMessage",
    "ChatRole",
    "FunctionCall",
    "FunctionCallOutput",
    "ImageContent",
]

ChatRole = Literal["system", "developer", "user", "assistant"]


@dataclass(slots=True)
class AudioContent:
    """Audio attached to a message (for audio-input LLMs / half-cascade engines)."""

    frame: AudioFrame
    transcript: str | None = None


@dataclass(slots=True)
class ImageContent:
    """An image attached to a message. ``url`` may be an https:// or data: URL."""

    url: str
    mime_type: str | None = None


MessageContent: TypeAlias = str | AudioContent | ImageContent


@dataclass(slots=True)
class ChatMessage:
    role: ChatRole
    content: list[MessageContent] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("msg_"))
    interrupted: bool = False
    """True if an assistant message was cut off by the user (content = what was actually heard)."""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    type: Literal["message"] = "message"

    @property
    def text(self) -> str:
        """Concatenated text content (audio transcripts included when there is no text)."""
        parts: list[str] = []
        for c in self.content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, AudioContent) and c.transcript:
                parts.append(c.transcript)
        return "".join(parts)

    @property
    def audio(self) -> list[AudioContent]:
        return [c for c in self.content if isinstance(c, AudioContent)]


@dataclass(slots=True)
class FunctionCall:
    """A tool call emitted by a model. ``arguments`` is the raw JSON string."""

    name: str
    arguments: str = "{}"
    call_id: str = field(default_factory=lambda: new_id("call_"))
    id: str = field(default_factory=lambda: new_id("fc_"))
    created_at: float = field(default_factory=time.time)
    type: Literal["function_call"] = "function_call"

    def parsed_arguments(self) -> dict[str, Any]:
        """Parse ``arguments`` (empty string -> ``{}``). Raises ``ValueError`` on bad JSON."""
        if not self.arguments or not self.arguments.strip():
            return {}
        value = json.loads(self.arguments)
        if not isinstance(value, dict):
            raise ValueError(f"tool arguments must be a JSON object, got {type(value).__name__}")
        return value


@dataclass(slots=True)
class FunctionCallOutput:
    """The result of a tool call, sent back to the model."""

    call_id: str
    output: str
    name: str = ""
    is_error: bool = False
    id: str = field(default_factory=lambda: new_id("fco_"))
    created_at: float = field(default_factory=time.time)
    type: Literal["function_call_output"] = "function_call_output"


ChatItem: TypeAlias = ChatMessage | FunctionCall | FunctionCallOutput


class ChatContext:
    """An ordered, mutable list of conversation items."""

    def __init__(self, items: Iterable[ChatItem] | None = None) -> None:
        self.items: list[ChatItem] = list(items or [])

    # ---------------------------------------------------------------- construction
    @classmethod
    def empty(cls) -> ChatContext:
        return cls()

    def add_message(
        self,
        role: ChatRole,
        content: MessageContent | list[MessageContent],
        *,
        interrupted: bool = False,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            role=role,
            content=list(content) if isinstance(content, list) else [content],
            interrupted=interrupted,
            metadata=dict(metadata or {}),
        )
        if id is not None:
            msg.id = id
        self.items.append(msg)
        return msg

    def add_function_call(
        self, name: str, arguments: str, call_id: str | None = None
    ) -> FunctionCall:
        call = FunctionCall(name=name, arguments=arguments)
        if call_id is not None:
            call.call_id = call_id
        self.items.append(call)
        return call

    def add_function_output(
        self, call_id: str, output: str, *, name: str = "", is_error: bool = False
    ) -> FunctionCallOutput:
        out = FunctionCallOutput(call_id=call_id, output=output, name=name, is_error=is_error)
        self.items.append(out)
        return out

    def append(self, item: ChatItem) -> None:
        self.items.append(item)

    # --------------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[ChatItem]:
        return iter(self.items)

    def messages(self) -> list[ChatMessage]:
        return [i for i in self.items if isinstance(i, ChatMessage)]

    def get(self, item_id: str) -> ChatItem | None:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def index_of(self, item_id: str) -> int | None:
        for i, item in enumerate(self.items):
            if item.id == item_id:
                return i
        return None

    def last_message(self, role: ChatRole | None = None) -> ChatMessage | None:
        for item in reversed(self.items):
            if isinstance(item, ChatMessage) and (role is None or item.role == role):
                return item
        return None

    # ------------------------------------------------------------------ mutations
    def copy(self) -> ChatContext:
        """Shallow copy of the item list (items themselves are shared)."""
        return ChatContext(self.items)

    def remove(self, item_id: str) -> ChatItem | None:
        idx = self.index_of(item_id)
        return None if idx is None else self.items.pop(idx)

    def truncate(self, max_items: int) -> None:
        """Keep the leading system/developer messages plus the last ``max_items`` other items.

        Never starts the kept tail with an orphan :class:`FunctionCallOutput`.
        """
        head = [
            i
            for i in self.items
            if isinstance(i, ChatMessage) and i.role in ("system", "developer")
        ]
        rest = [i for i in self.items if i not in head]
        tail = rest[-max_items:] if max_items > 0 else []
        while tail and isinstance(tail[0], FunctionCallOutput):
            tail.pop(0)
        self.items = head + tail

    # ------------------------------------------------------------- serialization
    def to_dict(self, *, include_audio: bool = False) -> dict[str, Any]:
        """JSON-serializable representation (audio omitted unless ``include_audio``)."""
        out: list[dict[str, Any]] = []
        for item in self.items:
            if isinstance(item, ChatMessage):
                content: list[Any] = []
                for c in item.content:
                    if isinstance(c, str):
                        content.append(c)
                    elif isinstance(c, AudioContent):
                        entry: dict[str, Any] = {"type": "audio", "transcript": c.transcript}
                        if include_audio:
                            entry.update(
                                data=c.frame.to_base64(),
                                sample_rate=c.frame.sample_rate,
                                channels=c.frame.channels,
                            )
                        content.append(entry)
                    else:
                        content.append({"type": "image", "url": c.url, "mime_type": c.mime_type})
                out.append(
                    {
                        "type": "message",
                        "id": item.id,
                        "role": item.role,
                        "content": content,
                        "interrupted": item.interrupted,
                        "created_at": item.created_at,
                        "metadata": item.metadata,
                    }
                )
            elif isinstance(item, FunctionCall):
                out.append(
                    {
                        "type": "function_call",
                        "id": item.id,
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                        "created_at": item.created_at,
                    }
                )
            else:
                out.append(
                    {
                        "type": "function_call_output",
                        "id": item.id,
                        "call_id": item.call_id,
                        "name": item.name,
                        "output": item.output,
                        "is_error": item.is_error,
                        "created_at": item.created_at,
                    }
                )
        return {"items": out}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatContext:
        ctx = cls()
        for raw in data.get("items", []):
            kind = raw.get("type", "message")
            if kind == "message":
                content: list[MessageContent] = []
                for c in raw.get("content", []):
                    if isinstance(c, str):
                        content.append(c)
                    elif c.get("type") == "audio" and "data" in c:
                        frame = AudioFrame.from_base64(
                            c["data"], c["sample_rate"], c.get("channels", 1)
                        )
                        content.append(AudioContent(frame, c.get("transcript")))
                    elif c.get("type") == "audio":
                        if c.get("transcript"):
                            content.append(c["transcript"])
                    elif c.get("type") == "image":
                        content.append(ImageContent(c["url"], c.get("mime_type")))
                msg = ctx.add_message(
                    raw["role"],
                    content,
                    interrupted=raw.get("interrupted", False),
                    metadata=raw.get("metadata"),
                )
                msg.id = raw.get("id", msg.id)
                msg.created_at = raw.get("created_at", msg.created_at)
            elif kind == "function_call":
                call = ctx.add_function_call(
                    raw["name"], raw.get("arguments", "{}"), raw.get("call_id")
                )
                call.id = raw.get("id", call.id)
            elif kind == "function_call_output":
                out = ctx.add_function_output(
                    raw["call_id"],
                    raw.get("output", ""),
                    name=raw.get("name", ""),
                    is_error=raw.get("is_error", False),
                )
                out.id = raw.get("id", out.id)
        return ctx

    def __repr__(self) -> str:
        return f"ChatContext({len(self.items)} items)"
