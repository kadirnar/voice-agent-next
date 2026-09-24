"""OpenAI Realtime protocol pieces used by the server: wire audio formats, the session
configuration (``session.update`` in the GA and beta dialects), conversation items and
client errors.

The server keeps everything in the GA shape; beta-dialect input is mapped onto it and
beta-dialect output is rendered from it (see :func:`event_name`).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias

from ..audio.codecs import alaw_decode, alaw_encode, mulaw_decode, mulaw_encode
from ..tools import FunctionTool

__all__ = [
    "BETA_EVENT_NAMES",
    "DEFAULT_TURN_DETECTION",
    "ClientError",
    "Dialect",
    "Item",
    "SessionConfig",
    "WireFormat",
    "apply_session_update",
    "event_name",
    "parse_item",
    "parse_response_modalities",
]

Dialect: TypeAlias = Literal["ga", "beta"]
"""``ga``: the current Realtime API; ``beta``: the ``OpenAI-Beta: realtime=v1`` protocol."""

BETA_EVENT_NAMES: Final[Mapping[str, str]] = {
    "conversation.item.added": "conversation.item.created",
    "response.output_audio.delta": "response.audio.delta",
    "response.output_audio.done": "response.audio.done",
    "response.output_audio_transcript.delta": "response.audio_transcript.delta",
    "response.output_audio_transcript.done": "response.audio_transcript.done",
    "response.output_text.delta": "response.text.delta",
    "response.output_text.done": "response.text.done",
}
"""GA server event names -> their beta equivalents."""

GA_ONLY_EVENTS: Final = frozenset({"conversation.item.done"})
"""Server events the beta protocol does not have (not sent to beta clients)."""

DEFAULT_TURN_DETECTION: Final[Mapping[str, Any]] = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "idle_timeout_ms": None,
    "create_response": True,
    "interrupt_response": True,
}
_SEMANTIC_VAD_DEFAULTS: Final[Mapping[str, Any]] = {
    "type": "semantic_vad",
    "eagerness": "auto",
    "create_response": True,
    "interrupt_response": True,
}
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")
_MIN_RATE, _MAX_RATE = 8_000, 48_000


def event_name(etype: str, dialect: Dialect) -> str | None:
    """Server event name for ``dialect`` (``None``: not part of that dialect)."""
    if dialect == "beta":
        if etype in GA_ONLY_EVENTS:
            return None
        return BETA_EVENT_NAMES.get(etype, etype)
    return etype


class ClientError(Exception):
    """A client event was rejected; it is answered with an ``error`` event."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_value",
        param: str | None = None,
        type: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.type = type

    def to_json(self, event_id: str | None) -> dict[str, Any]:
        return {
            "type": self.type,
            "code": self.code,
            "message": self.message,
            "param": self.param,
            "event_id": event_id,
        }


# ------------------------------------------------------------------------ audio formats
@dataclass(frozen=True, slots=True)
class WireFormat:
    """Audio format on the wire: ``audio/pcm`` (s16le mono), ``audio/pcmu`` or
    ``audio/pcma`` (G.711 μ-law / A-law at 8 kHz)."""

    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: int = 24_000

    @property
    def bytes_per_sample(self) -> int:
        return 2 if self.type == "audio/pcm" else 1

    def to_json(self) -> dict[str, Any]:
        if self.type == "audio/pcm":
            return {"type": "audio/pcm", "rate": self.rate}
        return {"type": self.type}

    def beta_name(self) -> str:
        return {"audio/pcm": "pcm16", "audio/pcmu": "g711_ulaw", "audio/pcma": "g711_alaw"}[
            self.type
        ]

    def decode(self, data: bytes) -> bytes:
        """Wire bytes -> s16le PCM (``data`` must hold whole samples)."""
        if self.type == "audio/pcmu":
            return mulaw_decode(data)
        if self.type == "audio/pcma":
            return alaw_decode(data)
        return data

    def encode(self, pcm: bytes) -> bytes:
        """s16le PCM -> wire bytes."""
        if self.type == "audio/pcmu":
            return mulaw_encode(pcm)
        if self.type == "audio/pcma":
            return alaw_encode(pcm)
        return pcm


def _parse_format(value: Any, param: str) -> WireFormat:
    if not isinstance(value, Mapping):
        raise ClientError(f"{param} must be an object", param=param)
    kind = value.get("type", "audio/pcm")
    if kind == "audio/pcm":
        rate = value.get("rate", 24_000)
        if isinstance(rate, float) and rate.is_integer():
            rate = int(rate)
        valid = isinstance(rate, int) and not isinstance(rate, bool)
        if not valid or not _MIN_RATE <= rate <= _MAX_RATE:
            raise ClientError(
                f"{param}.rate must be an integer sample rate in [{_MIN_RATE}, {_MAX_RATE}] "
                "(24000 is the standard rate)",
                param=f"{param}.rate",
            )
        return WireFormat("audio/pcm", rate)
    if kind in ("audio/pcmu", "audio/pcma"):
        rate = value.get("rate", 8_000)
        if rate != 8_000:
            raise ClientError(f"{kind} is 8000 Hz audio", param=f"{param}.rate")
        return WireFormat(kind, 8_000)
    raise ClientError(
        f"Invalid value: {kind!r}. Supported values are: 'audio/pcm', 'audio/pcmu' and "
        "'audio/pcma'.",
        param=f"{param}.type",
    )


_BETA_FORMATS: Final[Mapping[str, WireFormat]] = {
    "pcm16": WireFormat("audio/pcm", 24_000),
    "g711_ulaw": WireFormat("audio/pcmu", 8_000),
    "g711_alaw": WireFormat("audio/pcma", 8_000),
}


def _parse_beta_format(value: Any, param: str) -> WireFormat:
    if isinstance(value, str) and value in _BETA_FORMATS:
        return _BETA_FORMATS[value]
    if isinstance(value, Mapping):  # some beta clients already send GA format objects
        return _parse_format(value, param)
    raise ClientError(
        f"Invalid value: {value!r}. Supported values are: 'pcm16', 'g711_ulaw' and 'g711_alaw'.",
        param=param,
    )


# ------------------------------------------------------------------ session configuration
@dataclass
class SessionConfig:
    """The session configuration of one connection (GA shape).

    Only part of it drives the engine (instructions, tools, voice, turn detection,
    transcription language, audio formats, output modalities); the rest is accepted and
    echoed back so clients see the configuration they sent.
    """

    instructions: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = "auto"
    output_modalities: list[str] = field(default_factory=lambda: ["audio"])
    voice: str | None = None
    speed: float = 1.0
    input_format: WireFormat = field(default_factory=WireFormat)
    output_format: WireFormat = field(default_factory=WireFormat)
    transcription: dict[str, Any] | None = None
    turn_detection: dict[str, Any] | None = field(
        default_factory=lambda: dict(DEFAULT_TURN_DETECTION)
    )
    noise_reduction: dict[str, Any] | None = None
    max_output_tokens: int | str = "inf"
    temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Other GA/beta fields (``tracing``, ``truncation``, ``prompt``, ...), echoed only."""

    def copy(self) -> SessionConfig:
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ derived
    @property
    def text_only(self) -> bool:
        return "audio" not in self.output_modalities

    @property
    def vad(self) -> bool:
        """The engine detects user turns (``turn_detection`` is not null)."""
        return self.turn_detection is not None

    @property
    def create_response(self) -> bool:
        td = self.turn_detection
        return td is not None and td.get("create_response", True) is not False

    @property
    def interrupt_response(self) -> bool:
        td = self.turn_detection
        return td is not None and td.get("interrupt_response", True) is not False

    @property
    def transcribe(self) -> bool:
        """Input audio transcription events are sent to the client."""
        return self.transcription is not None

    @property
    def language(self) -> str | None:
        cfg = self.transcription or {}
        language = cfg.get("language")
        return language if isinstance(language, str) and language.strip() else None

    def function_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool(
                name=str(tool["name"]),
                description=str(tool.get("description") or ""),
                parameters=dict(tool.get("parameters") or {"type": "object", "properties": {}}),
            )
            for tool in self.tools
        ]

    # ---------------------------------------------------------------- rendering
    def render(
        self, dialect: Dialect, *, session_id: str, model: str, expires_at: int
    ) -> dict[str, Any]:
        """The session object of ``session.created`` / ``session.updated``."""
        if dialect == "beta":
            return {
                "id": session_id,
                "object": "realtime.session",
                "model": model,
                "expires_at": expires_at,
                "modalities": ["text"] if self.text_only else ["text", "audio"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": self.input_format.beta_name(),
                "output_audio_format": self.output_format.beta_name(),
                "input_audio_transcription": copy.deepcopy(self.transcription),
                "turn_detection": copy.deepcopy(self.turn_detection),
                "input_audio_noise_reduction": copy.deepcopy(self.noise_reduction),
                "tools": copy.deepcopy(self.tools),
                "tool_choice": copy.deepcopy(self.tool_choice),
                "temperature": 0.8 if self.temperature is None else self.temperature,
                "max_response_output_tokens": self.max_output_tokens,
                "speed": self.speed,
                **copy.deepcopy(self.extra),
            }
        return {
            "type": "realtime",
            "object": "realtime.session",
            "id": session_id,
            "model": model,
            "output_modalities": list(self.output_modalities),
            "instructions": self.instructions,
            "tools": copy.deepcopy(self.tools),
            "tool_choice": copy.deepcopy(self.tool_choice),
            "max_output_tokens": self.max_output_tokens,
            "tracing": None,
            "truncation": "auto",
            "prompt": None,
            "include": None,
            "expires_at": expires_at,
            "audio": {
                "input": {
                    "format": self.input_format.to_json(),
                    "transcription": copy.deepcopy(self.transcription),
                    "noise_reduction": copy.deepcopy(self.noise_reduction),
                    "turn_detection": copy.deepcopy(self.turn_detection),
                },
                "output": {
                    "format": self.output_format.to_json(),
                    "voice": self.voice,
                    "speed": self.speed,
                },
            },
            **copy.deepcopy(self.extra),
        }


_BETA_KEYS: Final = frozenset(
    {
        "modalities",
        "input_audio_format",
        "output_audio_format",
        "input_audio_transcription",
        "input_audio_noise_reduction",
        "max_response_output_tokens",
        "temperature",
    }
)
_GA_KEYS: Final = frozenset({"type", "audio", "output_modalities", "max_output_tokens"})


def detect_dialect(session: Mapping[str, Any]) -> Dialect | None:
    """Dialect of a ``session.update`` payload (``None``: nothing dialect-specific in it)."""
    if _GA_KEYS & session.keys():
        return "ga"
    if _BETA_KEYS & session.keys():
        return "beta"
    return None


def apply_session_update(
    config: SessionConfig, session: Any, dialect: Dialect
) -> tuple[SessionConfig, Dialect]:
    """Validate a ``session.update`` payload and return the updated configuration.

    Returns a new :class:`SessionConfig` (``config`` is left untouched when the update is
    rejected with :class:`ClientError`) and the dialect of the payload.
    """
    if not isinstance(session, Mapping):
        raise ClientError(
            "Missing required parameter: 'session'.",
            code="missing_required_parameter",
            param="session",
        )
    dialect = detect_dialect(session) or dialect
    new = config.copy()
    if dialect == "beta":
        _apply_beta(new, session)
    else:
        _apply_ga(new, session)
    return new, dialect


def _apply_common(cfg: SessionConfig, s: Mapping[str, Any], p: str) -> set[str]:
    """Fields shared by both dialects; returns the keys it consumed."""
    if "instructions" in s:
        if not isinstance(s["instructions"], str):
            raise ClientError("instructions must be a string", param=f"{p}.instructions")
        cfg.instructions = s["instructions"]
    if "tools" in s:
        cfg.tools = _parse_tools(s["tools"], f"{p}.tools")
    if "tool_choice" in s:
        cfg.tool_choice = copy.deepcopy(s["tool_choice"])
    if "voice" in s:  # beta, xAI-style GA
        cfg.voice = _parse_voice(s["voice"], f"{p}.voice")
    if "turn_detection" in s:  # beta, xAI-style GA
        cfg.turn_detection = _parse_turn_detection(s["turn_detection"], f"{p}.turn_detection")
    if "speed" in s:
        cfg.speed = _parse_speed(s["speed"], f"{p}.speed")
    return {"instructions", "tools", "tool_choice", "voice", "turn_detection", "speed"}


def _apply_ga(cfg: SessionConfig, s: Mapping[str, Any]) -> None:
    p = "session"
    kind = s.get("type", "realtime")
    if kind != "realtime":
        raise ClientError(
            f"Invalid value: {kind!r}. Only 'realtime' sessions are supported by this server "
            "(transcription sessions are not).",
            param=f"{p}.type",
        )
    used = _apply_common(cfg, s, p) | {"type", "model", "object", "id", "expires_at"}
    if "output_modalities" in s:
        cfg.output_modalities = _parse_modalities(s["output_modalities"], f"{p}.output_modalities")
    if "max_output_tokens" in s:
        cfg.max_output_tokens = _parse_max_tokens(s["max_output_tokens"], f"{p}.max_output_tokens")
    audio = s.get("audio")
    if audio is not None:
        if not isinstance(audio, Mapping):
            raise ClientError("audio must be an object", param=f"{p}.audio")
        inp = audio.get("input")
        if inp is not None:
            if not isinstance(inp, Mapping):
                raise ClientError("audio.input must be an object", param=f"{p}.audio.input")
            q = f"{p}.audio.input"
            if "format" in inp:
                cfg.input_format = _parse_format(inp["format"], f"{q}.format")
            if "transcription" in inp:
                cfg.transcription = _parse_optional_object(
                    inp["transcription"], f"{q}.transcription"
                )
            if "turn_detection" in inp:
                cfg.turn_detection = _parse_turn_detection(
                    inp["turn_detection"], f"{q}.turn_detection"
                )
            if "noise_reduction" in inp:
                cfg.noise_reduction = _parse_optional_object(
                    inp["noise_reduction"], f"{q}.noise_reduction"
                )
        out = audio.get("output")
        if out is not None:
            if not isinstance(out, Mapping):
                raise ClientError("audio.output must be an object", param=f"{p}.audio.output")
            q = f"{p}.audio.output"
            if "format" in out:
                cfg.output_format = _parse_format(out["format"], f"{q}.format")
            if "voice" in out:
                cfg.voice = _parse_voice(out["voice"], f"{q}.voice")
            if "speed" in out:
                cfg.speed = _parse_speed(out["speed"], f"{q}.speed")
    used |= {"output_modalities", "max_output_tokens", "audio"}
    for key, value in s.items():
        if key not in used:  # tracing, truncation, prompt, include, reasoning, ...
            cfg.extra[key] = copy.deepcopy(value)


def _apply_beta(cfg: SessionConfig, s: Mapping[str, Any]) -> None:
    p = "session"
    used = _apply_common(cfg, s, p) | {"model", "object", "id", "expires_at"}
    if "modalities" in s:
        cfg.output_modalities = _parse_modalities(s["modalities"], f"{p}.modalities")
    if "input_audio_format" in s:
        cfg.input_format = _parse_beta_format(s["input_audio_format"], f"{p}.input_audio_format")
    if "output_audio_format" in s:
        cfg.output_format = _parse_beta_format(s["output_audio_format"], f"{p}.output_audio_format")
    if "input_audio_transcription" in s:
        cfg.transcription = _parse_optional_object(
            s["input_audio_transcription"], f"{p}.input_audio_transcription"
        )
    if "input_audio_noise_reduction" in s:
        cfg.noise_reduction = _parse_optional_object(
            s["input_audio_noise_reduction"], f"{p}.input_audio_noise_reduction"
        )
    if "max_response_output_tokens" in s:
        cfg.max_output_tokens = _parse_max_tokens(
            s["max_response_output_tokens"], f"{p}.max_response_output_tokens"
        )
    if "temperature" in s:
        value = s["temperature"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ClientError("temperature must be a number", param=f"{p}.temperature")
        cfg.temperature = float(value)
    used |= {
        "modalities", "input_audio_format", "output_audio_format", "input_audio_transcription",
        "input_audio_noise_reduction", "max_response_output_tokens", "temperature",
    }  # fmt: skip
    for key, value in s.items():
        if key not in used:
            cfg.extra[key] = copy.deepcopy(value)


def _parse_tools(value: Any, param: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ClientError("tools must be an array", param=param)
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for i, raw in enumerate(value):
        q = f"{param}[{i}]"
        if not isinstance(raw, Mapping):
            raise ClientError("each tool must be an object", param=q)
        tool = dict(raw)
        if isinstance(tool.get("function"), Mapping):  # Chat-Completions style (nested)
            tool = {"type": "function", **dict(tool["function"])}
        kind = tool.get("type", "function")
        if kind != "function":
            raise ClientError(
                f"Unsupported tool type {kind!r}: this server only runs function tools, which "
                "the client executes (MCP tools are not supported).",
                code="unsupported_tool_type",
                param=f"{q}.type",
            )
        name = tool.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.match(name):
            raise ClientError("tool name must match ^[A-Za-z0-9_.-]{1,128}$", param=f"{q}.name")
        if name in names:
            raise ClientError(f"duplicate tool name {name!r}", param=f"{q}.name")
        names.add(name)
        description = tool.get("description")
        if description is not None and not isinstance(description, str):
            raise ClientError("tool description must be a string", param=f"{q}.description")
        parameters = tool.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ClientError("tool parameters must be a JSON schema object",
                              param=f"{q}.parameters")  # fmt: skip
        entry: dict[str, Any] = {"type": "function", "name": name}
        if description is not None:
            entry["description"] = description
        if parameters is not None:
            entry["parameters"] = copy.deepcopy(dict(parameters))
        tools.append(entry)
    return tools


def _parse_voice(value: Any, param: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping) and isinstance(value.get("id"), str):
        return str(value["id"])  # GA custom voice object
    if isinstance(value, str) and value.strip():
        return value
    raise ClientError("voice must be a voice name", param=param)


def _parse_speed(value: Any, param: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.25 <= value <= 4:
        raise ClientError("speed must be a number in [0.25, 4.0]", param=param)
    return float(value)


def _parse_modalities(value: Any, param: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(v in ("text", "audio") for v in value):
        raise ClientError("modalities must be a non-empty subset of ['text', 'audio']", param=param)
    return ["audio"] if "audio" in value else ["text"]


def parse_response_modalities(body: Mapping[str, Any]) -> list[str] | None:
    """``output_modalities`` (GA) / ``modalities`` (beta) of a ``response.create``."""
    for key in ("output_modalities", "modalities"):
        if key in body and body[key] is not None:
            return _parse_modalities(body[key], f"response.{key}")
    return None


def _parse_max_tokens(value: Any, param: str) -> int | str:
    if value == "inf":
        return "inf"
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4096:
        raise ClientError("max output tokens must be an integer in [1, 4096] or 'inf'", param=param)
    return value


def _parse_optional_object(value: Any, param: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ClientError(f"{param.rsplit('.', 1)[-1]} must be an object or null", param=param)
    return copy.deepcopy(dict(value))


def _parse_turn_detection(value: Any, param: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ClientError("turn_detection must be an object or null", param=param)
    kind = value.get("type")
    if kind == "server_vad":
        td = {**DEFAULT_TURN_DETECTION, **value}
    elif kind == "semantic_vad":
        td = {**_SEMANTIC_VAD_DEFAULTS, **value}
    else:
        raise ClientError(
            f"Invalid value: {kind!r}. Supported values are: 'server_vad' and 'semantic_vad'.",
            param=f"{param}.type",
        )
    for flag in ("create_response", "interrupt_response"):
        if not isinstance(td.get(flag), bool):
            raise ClientError(f"{flag} must be a boolean", param=f"{param}.{flag}")
    threshold = td.get("threshold")
    if threshold is not None and (
        isinstance(threshold, bool) or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):  # fmt: skip
        raise ClientError("threshold must be a number in [0, 1]", param=f"{param}.threshold")
    for key in ("prefix_padding_ms", "silence_duration_ms", "idle_timeout_ms"):
        ms = td.get(key)
        if ms is not None and (isinstance(ms, bool) or not isinstance(ms, int) or ms < 0):
            raise ClientError(f"{key} must be a non-negative integer", param=f"{param}.{key}")
    eagerness = td.get("eagerness")
    if eagerness is not None and eagerness not in ("low", "medium", "high", "auto"):
        raise ClientError(
            "eagerness must be one of 'low', 'medium', 'high', 'auto'", param=f"{param}.eagerness"
        )
    return td


# ------------------------------------------------------------------------ conversation
ItemType: TypeAlias = Literal["message", "function_call", "function_call_output"]
Role: TypeAlias = Literal["user", "assistant", "system"]
ContentType: TypeAlias = Literal["input_audio", "input_text", "output_audio", "output_text"]


@dataclass
class Item:
    """A conversation item as the client sees it."""

    id: str
    type: ItemType
    role: Role | None = None
    content: ContentType | None = None
    text: str | None = None
    """Text, or the transcript of audio content (``None``: not transcribed yet)."""
    status: str = "completed"
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""
    output: str = ""
    engine_id: str | None = None
    """Id of the matching item inside the engine connection, when known."""
    audio_ms: float = 0.0
    """Assistant audio sent to the client (what ``conversation.item.truncate`` may cut)."""
    input_seconds: float = 0.0
    """User audio committed into this item."""

    def render(self, dialect: Dialect) -> dict[str, Any]:
        base: dict[str, Any] = {"id": self.id, "object": "realtime.item", "type": self.type}
        if self.type == "function_call":
            return {
                **base,
                "status": self.status,
                "name": self.name,
                "call_id": self.call_id,
                "arguments": self.arguments,
            }
        if self.type == "function_call_output":
            return {**base, "status": self.status, "call_id": self.call_id, "output": self.output}
        part: dict[str, Any]
        if self.content == "input_audio":
            part = {"type": "input_audio", "transcript": self.text}
        elif self.content == "output_audio":
            kind = "audio" if dialect == "beta" else "output_audio"
            part = {"type": kind, "transcript": self.text or ""}
        elif self.content == "output_text":
            part = {"type": "text" if dialect == "beta" else "output_text", "text": self.text or ""}
        else:
            part = {"type": "input_text", "text": self.text or ""}
        return {**base, "status": self.status, "role": self.role, "content": [part]}


def parse_item(raw: Any) -> Item:
    """A client item (``conversation.item.create``) -> :class:`Item` (id may be empty)."""
    p = "item"
    if not isinstance(raw, Mapping):
        raise ClientError("Missing required parameter: 'item'.", code="missing_required_parameter",
                          param=p)  # fmt: skip
    item_id = raw.get("id") or ""
    if not isinstance(item_id, str) or len(item_id) > 64:
        raise ClientError("item.id must be a string of at most 64 characters", param=f"{p}.id")
    kind = raw.get("type", "message")
    if kind == "function_call_output":
        call_id, output = raw.get("call_id"), raw.get("output")
        if not isinstance(call_id, str) or not call_id:
            raise ClientError("Missing required parameter: 'item.call_id'.",
                              code="missing_required_parameter", param=f"{p}.call_id")  # fmt: skip
        if not isinstance(output, str):
            raise ClientError("item.output must be a string", param=f"{p}.output")
        return Item(item_id, "function_call_output", call_id=call_id, output=output)
    if kind == "function_call":
        name, arguments = raw.get("name"), raw.get("arguments", "{}")
        call_id = raw.get("call_id")
        if not isinstance(name, str) or not name:
            raise ClientError("Missing required parameter: 'item.name'.",
                              code="missing_required_parameter", param=f"{p}.name")  # fmt: skip
        if not isinstance(arguments, str):
            raise ClientError("item.arguments must be a JSON string", param=f"{p}.arguments")
        if call_id is not None and not isinstance(call_id, str):
            raise ClientError("item.call_id must be a string", param=f"{p}.call_id")
        return Item(item_id, "function_call", call_id=call_id, name=name, arguments=arguments)
    if kind != "message":
        raise ClientError(
            f"Unsupported item type {kind!r} (supported: message, function_call, "
            "function_call_output).",
            param=f"{p}.type",
        )
    role = raw.get("role")
    if role == "developer":
        role = "system"
    if role not in ("user", "assistant", "system"):
        raise ClientError(
            f"Invalid value: {role!r}. Supported values are: 'user', 'assistant' and 'system'.",
            param=f"{p}.role",
        )
    content = raw.get("content")
    if not isinstance(content, list):
        raise ClientError("item.content must be an array", param=f"{p}.content")
    texts: list[str] = []
    for i, part in enumerate(content):
        q = f"{p}.content[{i}]"
        if not isinstance(part, Mapping):
            raise ClientError("content parts must be objects", param=q)
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text"):
            text = part.get("text")
            if not isinstance(text, str):
                raise ClientError("text content needs a 'text' string", param=f"{q}.text")
            texts.append(text)
        elif ptype in ("input_audio", "output_audio", "audio"):
            if part.get("audio"):
                raise ClientError(
                    "Audio content in conversation.item.create is not supported by this "
                    "server: stream user audio with input_audio_buffer.append instead (a "
                    "'transcript' without 'audio' is accepted).",
                    code="unsupported_content_type",
                    param=f"{q}.audio",
                )
            transcript = part.get("transcript")
            if isinstance(transcript, str):
                texts.append(transcript)
        else:
            raise ClientError(f"Unsupported content type {ptype!r}.",
                              code="unsupported_content_type", param=f"{q}.type")  # fmt: skip
    ctype: ContentType = "output_text" if role == "assistant" else "input_text"
    return Item(item_id, "message", role=role, content=ctype, text="".join(texts))
