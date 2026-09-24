"""Conversion from voice-agent-next chat types to the Gemini ``generateContent`` format.

Pure functions without any ``google-genai`` import: they return the snake_case dict form
that the SDK accepts for ``contents`` and ``config`` (it validates and serializes them).

How a :class:`~voice_agent_next.chat.ChatContext` maps onto ``generateContent``
(:func:`to_gemini_contents`):

* system/developer messages become the ``system_instruction`` (joined with blank lines);
  instructions given before the conversation come first, later ones (per-response
  instructions) are appended;
* ``user`` -> ``user`` and ``assistant`` -> ``model`` contents; consecutive items of the same
  role merge into one content. A conversation that starts with the model gets a short
  placeholder user turn in front, one that ends with the model gets one at the end;
* :class:`~voice_agent_next.chat.FunctionCall` items become ``function_call`` parts of the
  model content; their :class:`~voice_agent_next.chat.FunctionCallOutput` s become
  ``function_response`` parts (``{"output": ...}`` or ``{"error": ...}``) of the user
  content that immediately follows. Calls without an output and outputs without a call are
  dropped, because the API requires one response per call;
* images become ``inline_data`` (``data:`` URLs) or ``file_data`` (``https://`` / ``gs://``);
  user audio becomes ``inline_data`` WAV (models with audio input), or its transcript;
* an interrupted assistant message contributes exactly the text the user heard.

Thought signatures: Gemini 3 models return an opaque ``thought_signature`` on the first
``function_call`` part of each step and validate it when the call is sent back within the
same turn. The provider remembers signatures per ``call_id`` (:class:`CallMeta`) and puts
them back on the right parts; calls it has no signature for (another provider's history,
an injected call) get :data:`SKIP_THOUGHT_SIGNATURE` where the API would require one.
"""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
import urllib.parse
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ...audio.wav import wav_bytes
from ...chat import (
    AudioContent,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from ...errors import ConfigurationError
from ...llm import ToolChoice
from ...tools import FunctionTool
from ...utils.log import logger

__all__ = [
    "CONTINUE_PLACEHOLDER",
    "SKIP_THOUGHT_SIGNATURE",
    "START_PLACEHOLDER",
    "CallMeta",
    "GeminiPrompt",
    "default_thinking_config",
    "gemini_generation",
    "to_gemini_contents",
    "to_gemini_tool_config",
    "to_gemini_tools",
]

START_PLACEHOLDER = "(start of the conversation)"
"""User turn inserted when the conversation is empty or starts with the model."""
CONTINUE_PLACEHOLDER = "(continue)"
"""User turn appended when the conversation ends with a model turn."""
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
"""Documented placeholder ``thoughtSignature`` for function calls the model did not produce
in this conversation (e.g. history from another provider); it skips the validation."""

MINIMAL_THINKING_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
)
"""Gemini 3 models that accept ``thinking_level="minimal"`` (the others start at ``low``)."""

_MODEL_VERSION = re.compile(r"^gemini-(\d+)(?:\.(\d+))?(?:-|$)")


@dataclass(slots=True)
class CallMeta:
    """What the provider remembers about a function call the model produced."""

    signature: bytes | None = None
    """The ``thought_signature`` of the call's part (Gemini 3), if it carried one."""
    api_id: bool = False
    """The ``call_id`` came from the API (``function_call.id``) and is echoed back."""


@dataclass(slots=True)
class GeminiPrompt:
    """A :class:`ChatContext` converted to ``system_instruction`` and ``contents``."""

    system_instruction: str | None = None
    contents: list[dict[str, Any]] = field(default_factory=list)
    tool_names: set[str] = field(default_factory=set)
    """Names of the functions called in ``contents``."""


# ------------------------------------------------------------------------------ models
def _model_name(model: str) -> str:
    """``"models/gemini-3.8-flash"`` / ``"publishers/google/models/..."`` -> bare id."""
    return model.strip().rsplit("/", 1)[-1].lower()


def gemini_generation(model: str) -> tuple[int, int] | None:
    """``(major, minor)`` of a ``gemini-X.Y-...`` model id, ``None`` for other models."""
    match = _MODEL_VERSION.match(_model_name(model))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def default_thinking_config(model: str) -> dict[str, Any] | None:
    """The lowest-latency thinking setting for ``model`` (``None``: leave the default).

    Gemini 3+: ``thinking_level="minimal"`` where supported, else ``"low"`` (thinking cannot
    be turned off). Gemini 2.5 Flash / Flash-Lite: ``thinking_budget=0`` (off); 2.5 Pro:
    ``128``, its minimum. Other models (Gemma, tuned models...): not set.
    """
    version = gemini_generation(model)
    if version is None:
        return None
    name = _model_name(model)
    if version[0] >= 3:
        level = "minimal" if _is_one_of(name, MINIMAL_THINKING_MODELS) else "low"
        return {"thinking_level": level}
    if version == (2, 5):
        if name.startswith("gemini-2.5-flash"):
            return {"thinking_budget": 0}
        if name.startswith("gemini-2.5-pro"):
            return {"thinking_budget": 128}
    return None


def _is_one_of(name: str, models: Collection[str]) -> bool:
    """``name`` is one of ``models`` or a dated/versioned variant (``-001``, ``-preview-...``),
    but not a different model that shares the prefix (``gemini-3.6-flash-lite``)."""
    for model in models:
        if name == model:
            return True
        if name.startswith(model + "-") and not name[len(model) + 1 :].startswith("lite"):
            return True
    return False


# ------------------------------------------------------------------------------- tools
def to_gemini_tools(tools: Sequence[FunctionTool]) -> list[dict[str, Any]]:
    """``FunctionTool`` list -> one ``Tool`` with ``function_declarations`` (JSON schemas)."""
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        decl: dict[str, Any] = {"name": tool.name}
        if tool.description:
            decl["description"] = tool.description
        schema = dict(tool.parameters or {})
        if schema.get("properties"):
            schema.setdefault("type", "object")
            decl["parameters_json_schema"] = schema
        declarations.append(decl)
    return [{"function_declarations": declarations}] if declarations else []


def to_gemini_tool_config(
    choice: ToolChoice | None, *, tool_names: Collection[str] = ()
) -> dict[str, Any] | None:
    """Map a :data:`~voice_agent_next.llm.ToolChoice` to ``tool_config``.

    ``"auto"`` (or ``None``) -> omitted (``AUTO``), ``"required"``/``"any"`` -> ``ANY``,
    ``"none"`` -> ``NONE``, a tool name -> ``ANY`` restricted to that function.
    """
    if choice is None or choice == "auto":
        return None
    if choice == "none":
        mode: dict[str, Any] = {"mode": "NONE"}
    elif choice in ("required", "any"):
        mode = {"mode": "ANY"}
    else:
        if tool_names and choice not in tool_names:
            raise ConfigurationError(
                f"tool_choice {choice!r} is not one of the provided tools: {sorted(tool_names)}"
            )
        mode = {"mode": "ANY", "allowed_function_names": [choice]}
    return {"function_calling_config": mode}


# ----------------------------------------------------------------------------- contents
def to_gemini_contents(
    ctx: ChatContext,
    *,
    audio_input: bool = True,
    calls: Mapping[str, CallMeta] | None = None,
    fallback_signature: bytes | None = None,
) -> GeminiPrompt:
    """Convert ``ctx`` to ``system_instruction`` + ``contents`` (see the module docstring).

    Args:
        ctx: the conversation.
        audio_input: send :class:`~voice_agent_next.chat.AudioContent` as audio; when False
            its transcript is sent instead.
        calls: remembered thought signatures / API ids by ``call_id``.
        fallback_signature: signature for function calls of the current turn that have
            none (``None``: send them without one).

    Raises:
        ConfigurationError: audio without a transcript while ``audio_input`` is False, or
            an image with an unsupported URL.
    """
    known = calls or {}
    items = list(ctx.items)
    call_names = {item.call_id: item.name for item in items if isinstance(item, FunctionCall)}
    outputs: dict[str, FunctionCallOutput] = {}
    for item in items:
        if isinstance(item, FunctionCallOutput) and item.call_id in call_names:
            outputs.setdefault(item.call_id, item)

    prompt = GeminiPrompt()
    contents = prompt.contents
    stable: list[str] = []  # system text before the conversation
    late: list[str] = []  # system text after it started (per-response instructions)
    pending: list[dict[str, Any]] = []  # function responses answering the open model turn
    sent: set[str] = set()
    started = False

    def parts_of(role: str) -> list[dict[str, Any]]:
        if not contents or contents[-1]["role"] != role:
            contents.append({"role": role, "parts": []})
        return cast(list[dict[str, Any]], contents[-1]["parts"])

    def flush_responses() -> None:
        # the responses must directly follow the model turn holding their calls
        if pending:
            parts_of("user").extend(pending)
            pending.clear()

    for item in items:
        if isinstance(item, ChatMessage):
            if item.role in ("system", "developer"):
                text = _plain_text(item)
                if text:
                    (late if started else stable).append(text)
                continue
            started = True
            flush_responses()
            if item.role == "user":
                parts = _user_parts(item, audio_input=audio_input)
            else:
                text = _plain_text(item)
                parts = [{"text": text}] if text else []
            if parts:
                parts_of("user" if item.role == "user" else "model").extend(parts)
        elif isinstance(item, FunctionCall):
            started = True
            output = outputs.get(item.call_id)
            if output is None or item.call_id in sent:
                logger.debug("gemini: skipping tool call %s (no output/duplicate)", item.call_id)
                continue
            sent.add(item.call_id)
            meta = known.get(item.call_id)
            parts_of("model").append(_function_call_part(item, meta))
            pending.append(_function_response_part(item, output, meta))
            prompt.tool_names.add(item.name)
        else:
            started = True
            if item.call_id not in call_names:
                logger.debug("gemini: skipping tool output %s without a call", item.call_id)
            flush_responses()
    flush_responses()

    if not contents or contents[0]["role"] != "user":
        contents.insert(0, {"role": "user", "parts": [{"text": START_PLACEHOLDER}]})
    if contents[-1]["role"] == "model":
        contents.append({"role": "user", "parts": [{"text": CONTINUE_PLACEHOLDER}]})
    if fallback_signature is not None:
        _fill_missing_signatures(contents, fallback_signature)
    system = [*stable, *late]
    prompt.system_instruction = "\n\n".join(system) if system else None
    return prompt


def _fill_missing_signatures(contents: list[dict[str, Any]], signature: bytes) -> None:
    """Give each model step of the current turn a signature on its first function call.

    The current turn starts after the last user content that is not only function
    responses; Gemini 3 rejects its function calls without a signature (older turns are
    not validated).
    """
    start = 0
    for index, content in enumerate(contents):
        if content["role"] == "user" and any(
            "function_response" not in part for part in content["parts"]
        ):
            start = index + 1
    for content in contents[start:]:
        if content["role"] != "model":
            continue
        call_parts = [part for part in content["parts"] if "function_call" in part]
        if call_parts and not any(part.get("thought_signature") for part in call_parts):
            call_parts[0]["thought_signature"] = signature


def _function_call_part(call: FunctionCall, meta: CallMeta | None) -> dict[str, Any]:
    try:
        args = call.parsed_arguments()
    except ValueError:
        logger.warning("gemini: tool call %s has invalid JSON arguments; sent as {}", call.call_id)
        args = {}
    function_call: dict[str, Any] = {"name": call.name, "args": args}
    if meta is not None and meta.api_id:
        function_call["id"] = call.call_id
    part: dict[str, Any] = {"function_call": function_call}
    if meta is not None and meta.signature:
        part["thought_signature"] = meta.signature
    return part


def _function_response_part(
    call: FunctionCall, output: FunctionCallOutput, meta: CallMeta | None
) -> dict[str, Any]:
    value = _tool_value(output.output)
    if output.is_error:
        response = {"error": value if value != "" else "The tool call failed."}
    else:
        response = {"output": value}
    function_response: dict[str, Any] = {"name": call.name, "response": response}
    if meta is not None and meta.api_id:
        function_response["id"] = call.call_id
    return {"function_response": function_response}


def _tool_value(text: str) -> Any:
    """Tool output as structured JSON when it is a JSON object/array, else the string."""
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(stripped)
        except ValueError:
            pass
    return text


def _plain_text(msg: ChatMessage) -> str:
    """Text of a system/assistant message (audio transcripts when it has no text)."""
    parts = [c for c in msg.content if isinstance(c, str)]
    if not any(p.strip() for p in parts):
        parts = [c.transcript for c in msg.content if isinstance(c, AudioContent) and c.transcript]
    if any(isinstance(c, ImageContent) for c in msg.content):
        logger.warning("gemini: images are only sent in user messages; dropped one")
    text = "".join(parts)
    return text if text.strip() else ""


def _user_parts(msg: ChatMessage, *, audio_input: bool) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    text: list[str] = []

    def flush_text() -> None:
        joined = "".join(text)
        text.clear()
        if joined.strip():
            parts.append({"text": joined})

    for content in msg.content:
        if isinstance(content, str):
            text.append(content)
        elif isinstance(content, AudioContent):
            if audio_input and content.frame:
                flush_text()
                parts.append(
                    {"inline_data": {"mime_type": "audio/wav", "data": wav_bytes(content.frame)}}
                )
            elif content.transcript and content.transcript.strip():
                text.append(content.transcript)
            elif content.frame:
                raise ConfigurationError(
                    f"user message {msg.id} contains audio without a transcript, but audio "
                    "input is disabled (audio_input=False): add an STT stage or enable it"
                )
        else:
            flush_text()
            parts.append(_image_part(content))
    flush_text()
    return parts


def _image_part(image: ImageContent) -> dict[str, Any]:
    url = image.url.strip()
    if url.startswith("data:"):
        header, sep, payload = url[5:].partition(",")
        if not sep:
            raise ConfigurationError("malformed data: URL in ImageContent")
        media_type, *params = header.split(";")
        media_type = (media_type.strip() or image.mime_type or "").lower()
        if not media_type:
            raise ConfigurationError(
                "image data: URL has no media type; set ImageContent.mime_type"
            )
        if any(p.strip().lower() == "base64" for p in params):
            compact = "".join(payload.split())
            compact += "=" * (-len(compact) % 4)
            try:
                if "-" in compact or "_" in compact:
                    data = base64.urlsafe_b64decode(compact)
                else:
                    data = base64.b64decode(compact)
            except (binascii.Error, ValueError) as exc:
                raise ConfigurationError(f"invalid base64 in image data: URL: {exc}") from exc
        else:
            data = urllib.parse.unquote_to_bytes(payload)
        return {"inline_data": {"mime_type": media_type, "data": data}}
    if url.startswith(("https://", "http://", "gs://")):
        path = urllib.parse.urlsplit(url).path
        mime = image.mime_type or mimetypes.guess_type(path)[0] or "image/jpeg"
        return {"file_data": {"file_uri": url, "mime_type": mime}}
    raise ConfigurationError(
        f"unsupported image URL for Gemini (expected https://, gs:// or data:): {url[:40]!r}"
    )
