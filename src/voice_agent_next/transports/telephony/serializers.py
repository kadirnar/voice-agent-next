"""Frame serializers for telephony media streams (Twilio, Telnyx, Vonage, Plivo).

A serializer translates one provider's WebSocket dialect to and from a small common model:

* :meth:`TelephonySerializer.parse` turns an inbound WebSocket message into
  :class:`TelephonyEvent` objects — the stream start (call metadata and audio format),
  caller audio (decoded to s16le PCM), DTMF digits, playback marks coming back, audio
  cleared, stream stopped, provider errors;
* :meth:`~TelephonySerializer.encode_audio`, :meth:`~TelephonySerializer.encode_clear`
  and :meth:`~TelephonySerializer.encode_mark` build the outbound messages (agent audio in
  the stream's codec, "drop queued audio", "tell me when playback reaches this point");
* :meth:`~TelephonySerializer.hangup_request` builds the REST call that ends the call.

Everything in a start message comes from the network. Call identifiers are checked against
the provider's documented format (:attr:`TelephonySerializer.call_id_pattern`) before the
stream is accepted, and they are percent-encoded again in :meth:`hangup_request`. Account
identifiers used in REST URLs come from the configuration only, never from the wire.

Serializers are stateful (one instance per call): the stream identifiers and the audio
format are only known once the provider's start message has been parsed.

Protocol references (checked 2026-09-24):

* Twilio Media Streams: https://www.twilio.com/docs/voice/media-streams/websocket-messages
* Telnyx media streaming:
  https://developers.telnyx.com/docs/voice/programmable-voice/media-streaming
* Vonage WebSockets: https://developer.vonage.com/en/voice/voice-api/concepts/websockets
* Plivo audio streaming:
  https://www.plivo.com/docs/voice-agents/audio-streaming/concepts/audio-streaming-reference
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import httpx
import numpy as np

from ...audio.codecs import alaw_decode, alaw_encode, mulaw_decode, mulaw_encode
from ...errors import ConfigurationError, TransportError

__all__ = [
    "SERIALIZERS",
    "AudioCleared",
    "AudioCodec",
    "AudioReceived",
    "CallInfo",
    "DtmfReceived",
    "MarkReached",
    "PlivoSerializer",
    "ProviderError",
    "StreamStarted",
    "StreamStopped",
    "TelephonyEvent",
    "TelephonyProtocolError",
    "TelephonySerializer",
    "TelnyxSerializer",
    "TwilioSerializer",
    "VonageSerializer",
    "create_serializer",
]

CodecName = Literal["mulaw", "alaw", "l16"]
ByteOrder = Literal["little", "big"]


class TelephonyProtocolError(TransportError):
    """A provider message could not be understood (malformed, unsupported codec...)."""


# ------------------------------------------------------------------------------ codecs
@dataclass(frozen=True, slots=True)
class AudioCodec:
    """A telephony wire codec: G.711 μ-law / A-law (8 bit) or 16-bit linear PCM."""

    name: CodecName
    sample_rate: int
    byteorder: ByteOrder = "little"
    """Byte order of ``l16`` samples on the wire (RTP's L16 is big-endian)."""

    def __post_init__(self) -> None:
        if self.name not in ("mulaw", "alaw", "l16"):
            raise ConfigurationError(f"unsupported telephony codec {self.name!r}")
        if not 8_000 <= self.sample_rate <= 48_000:
            raise ConfigurationError(f"unsupported telephony sample rate {self.sample_rate}")

    @property
    def bytes_per_sample(self) -> int:
        return 2 if self.name == "l16" else 1

    def decode(self, data: bytes) -> bytes:
        """Wire bytes -> s16le PCM (a trailing odd byte of L16 is dropped)."""
        if self.name == "mulaw":
            return mulaw_decode(data)
        if self.name == "alaw":
            return alaw_decode(data)
        data = data[: len(data) - len(data) % 2]
        return data if self.byteorder == "little" else _byteswap16(data)

    def encode(self, pcm: bytes) -> bytes:
        """s16le PCM -> wire bytes."""
        if self.name == "mulaw":
            return mulaw_encode(pcm)
        if self.name == "alaw":
            return alaw_encode(pcm)
        return pcm if self.byteorder == "little" else _byteswap16(pcm)


def _byteswap16(data: bytes) -> bytes:
    return np.frombuffer(data, dtype="<i2").astype(">i2").tobytes()


# ------------------------------------------------------------------------------ events
@dataclass(slots=True)
class CallInfo:
    """What the provider told us about the call when the stream started."""

    provider: str
    call_id: str | None = None
    """Twilio ``callSid``, Telnyx ``call_control_id``, Plivo ``callId`` (Vonage: none)."""
    stream_id: str | None = None
    account_id: str | None = None
    from_number: str | None = None
    to_number: str | None = None
    custom_parameters: dict[str, Any] = field(default_factory=dict)
    """Twilio ``<Parameter>``s, Telnyx ``custom_parameters``/``client_state``, Vonage
    ``headers`` of the NCCO endpoint, Plivo ``extraHeaders``."""
    encoding: str = ""
    """The provider's name for the inbound encoding (``audio/x-mulaw``, ``PCMU``...)."""
    sample_rate: int = 8_000
    raw: dict[str, Any] = field(default_factory=dict)
    """The provider's start message, verbatim."""


@dataclass(slots=True)
class StreamStarted:
    call: CallInfo


@dataclass(slots=True)
class AudioReceived:
    pcm: bytes
    """Caller audio, s16le mono at the input codec's rate."""


@dataclass(slots=True)
class DtmfReceived:
    digit: str
    duration_ms: int | None = None


@dataclass(slots=True)
class MarkReached:
    """Playback reached a mark/checkpoint/notify sent earlier (or it was flushed by clear)."""

    name: str


@dataclass(slots=True)
class AudioCleared:
    """The provider acknowledged a clear (Vonage ``websocket:cleared``, Plivo ``clearedAudio``)."""


@dataclass(slots=True)
class StreamStopped:
    """The media stream ended (the caller hung up or the call was redirected)."""


@dataclass(slots=True)
class ProviderError:
    message: str
    raw: dict[str, Any] = field(default_factory=dict)


TelephonyEvent = (
    StreamStarted
    | AudioReceived
    | DtmfReceived
    | MarkReached
    | AudioCleared
    | StreamStopped
    | ProviderError
)


# ------------------------------------------------------------------------------ base
class TelephonySerializer(ABC):
    """Translates one provider's media-stream WebSocket dialect (one instance per call).

    Args:
        api_base: REST API root used by :meth:`hangup_request` (tests, regional edges,
            compatible self-hosted backends).
    """

    provider: ClassVar[str]
    default_sample_rate: ClassVar[int] = 8_000
    fixed_frames: ClassVar[bool] = False
    """Outbound audio must be sent in whole ``frame_duration`` messages (pad the tail)."""
    json_audio: ClassVar[bool] = True
    """Audio travels base64-encoded inside JSON text messages (else: binary frames)."""
    default_api_base: ClassVar[str] = ""
    call_id_pattern: ClassVar[re.Pattern[str] | None] = None
    """The provider's call identifier format: start messages with another call ID are
    rejected (the ID ends up in the REST hang-up URL)."""

    def __init__(self, *, api_base: str | None = None) -> None:
        self.api_base = (api_base or self.default_api_base).rstrip("/")
        self.call: CallInfo | None = None
        self.input_codec = AudioCodec("mulaw", self.default_sample_rate)
        self.output_codec = self.input_codec

    @property
    def started(self) -> bool:
        return self.call is not None

    @abstractmethod
    def parse(self, message: str | bytes) -> list[TelephonyEvent]:
        """Inbound WebSocket message -> events (``[]`` for messages we do not need).

        Raises :class:`TelephonyProtocolError` for malformed or unsupported messages.
        """

    @classmethod
    def valid_call_id(cls, call_id: str | None) -> bool:
        """``call_id`` has the provider's documented format (any non-empty ID when the
        provider has no REST API here)."""
        if not call_id:
            return False
        return cls.call_id_pattern is None or cls.call_id_pattern.fullmatch(call_id) is not None

    @abstractmethod
    def encode_audio(self, pcm: bytes) -> str | bytes:
        """s16le PCM at ``output_codec.sample_rate`` -> one outbound media message."""

    @abstractmethod
    def encode_clear(self) -> str | bytes:
        """The message that drops audio queued at the provider (barge-in)."""

    @abstractmethod
    def encode_mark(self, name: str) -> str | bytes:
        """The message that makes the provider echo ``name`` once playback reaches it."""

    def hangup_request(self) -> httpx.Request | None:
        """The REST request that ends the call, or ``None`` (no credentials / unsupported)."""
        return None

    # ------------------------------------------------------------------ helpers
    def _start(self, call: CallInfo, input_codec: AudioCodec, output_codec: AudioCodec) -> None:
        if call.call_id is not None and not self.valid_call_id(call.call_id):
            raise TelephonyProtocolError(f"invalid {self.provider} call ID {call.call_id[:80]!r}")
        self.call = call
        self.input_codec = input_codec
        self.output_codec = output_codec


def _load(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        try:
            message = message.decode()
        except UnicodeDecodeError:
            raise TelephonyProtocolError("unexpected binary message") from None
    try:
        data = json.loads(message)
    except ValueError:
        raise TelephonyProtocolError("message is not valid JSON") from None
    if not isinstance(data, dict):
        raise TelephonyProtocolError("message is not a JSON object")
    return data


def _dumps(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"))


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _unb64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise TelephonyProtocolError("media payload must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise TelephonyProtocolError("media payload is not valid base64") from None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def _rate(value: Any, default: int) -> int:
    try:
        rate = int(value)
    except (TypeError, ValueError):
        return default
    return rate if 8_000 <= rate <= 48_000 else default


def _digit(value: Any) -> DtmfReceived | None:
    digit = _str(value)
    return DtmfReceived(digit) if digit else None


def _inbound(track: Any) -> bool:
    """Media of the caller's track (``both_tracks`` streams also carry our own audio)."""
    return track is None or not str(track).lower().startswith("outbound")


def _credential(value: str | None, env: str) -> str | None:
    return value or os.environ.get(env) or None


def _segment(value: str) -> str:
    """One URL path segment: ``/``, ``..``, ``?``, ``#`` and ``%`` cannot escape it."""
    return quote(value, safe="")


# Documented identifier formats (see the protocol references above and the docs page).
_TWILIO_CALL_SID = re.compile(r"CA[0-9a-f]{32}")
_TWILIO_ACCOUNT_SID = re.compile(r"AC[0-9a-f]{32}")
_TELNYX_CALL_CONTROL_ID = re.compile(r"v[0-9]{1,2}:[A-Za-z0-9_-]{1,256}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


_CODEC_ALIASES: dict[str, CodecName] = {
    "audio/x-mulaw": "mulaw",
    "audio/mulaw": "mulaw",
    "audio/pcmu": "mulaw",
    "pcmu": "mulaw",
    "mulaw": "mulaw",
    "ulaw": "mulaw",
    "audio/x-alaw": "alaw",
    "audio/alaw": "alaw",
    "audio/pcma": "alaw",
    "pcma": "alaw",
    "alaw": "alaw",
    "audio/x-l16": "l16",
    "audio/l16": "l16",
    "l16": "l16",
    "linear16": "l16",
}


def _codec_name(encoding: Any) -> CodecName:
    key = str(encoding or "").split(";")[0].strip().lower()
    name = _CODEC_ALIASES.get(key)
    if name is None:
        raise TelephonyProtocolError(
            f"unsupported stream encoding {encoding!r}; use μ-law, A-law or L16"
        )
    return name


# ------------------------------------------------------------------------------ Twilio
class TwilioSerializer(TelephonySerializer):
    """Twilio Media Streams (``<Connect><Stream>``): JSON, base64 μ-law 8 kHz both ways.

    Args:
        account_sid / auth_token: REST credentials for :meth:`hangup_request` (default:
            ``TWILIO_ACCOUNT_SID`` / ``TWILIO_AUTH_TOKEN``). The account SID is never taken
            from the ``start`` message; a stream whose ``accountSid`` is not the configured
            one is rejected.
    """

    provider = "twilio"
    default_api_base = "https://api.twilio.com"
    call_id_pattern = _TWILIO_CALL_SID

    def __init__(
        self,
        *,
        account_sid: str | None = None,
        auth_token: str | None = None,
        api_base: str | None = None,
    ) -> None:
        super().__init__(api_base=api_base)
        self.account_sid = _credential(account_sid, "TWILIO_ACCOUNT_SID")
        self.auth_token = _credential(auth_token, "TWILIO_AUTH_TOKEN")

    def parse(self, message: str | bytes) -> list[TelephonyEvent]:
        data = _load(message)
        event = data.get("event")
        if event == "media":
            media = _dict(data.get("media"))
            if not _inbound(media.get("track")):
                return []
            return [AudioReceived(self.input_codec.decode(_unb64(media.get("payload"))))]
        if event == "mark":
            name = _str(_dict(data.get("mark")).get("name"))
            return [MarkReached(name)] if name is not None else []
        if event == "dtmf":
            dtmf = _digit(_dict(data.get("dtmf")).get("digit"))
            return [dtmf] if dtmf else []
        if event == "start":
            return [self._on_start(data)]
        if event == "stop":
            return [StreamStopped()]
        return []  # "connected", future events

    def _on_start(self, data: dict[str, Any]) -> StreamStarted:
        start = _dict(data.get("start"))
        fmt = _dict(start.get("mediaFormat"))
        encoding = str(fmt.get("encoding") or "audio/x-mulaw")
        codec = AudioCodec(_codec_name(encoding), _rate(fmt.get("sampleRate"), 8_000))
        account = _str(start.get("accountSid"))
        if self.account_sid and account is not None and account != self.account_sid:
            raise TelephonyProtocolError(
                f"stream of account {account[:40]!r}, not the configured TWILIO_ACCOUNT_SID"
            )
        call = CallInfo(
            provider=self.provider,
            call_id=_str(start.get("callSid")),
            stream_id=_str(start.get("streamSid") or data.get("streamSid")),
            account_id=account,
            custom_parameters=dict(_dict(start.get("customParameters"))),
            encoding=encoding,
            sample_rate=codec.sample_rate,
            raw=data,
        )
        # Twilio only accepts μ-law 8 kHz back, whatever it streams to us
        self._start(call, codec, AudioCodec("mulaw", 8_000))
        return StreamStarted(call)

    @property
    def _sid(self) -> str | None:
        return self.call.stream_id if self.call else None

    def encode_audio(self, pcm: bytes) -> str:
        payload = _b64(self.output_codec.encode(pcm))
        return _dumps({"event": "media", "streamSid": self._sid, "media": {"payload": payload}})

    def encode_clear(self) -> str:
        return _dumps({"event": "clear", "streamSid": self._sid})

    def encode_mark(self, name: str) -> str:
        return _dumps({"event": "mark", "streamSid": self._sid, "mark": {"name": name}})

    def hangup_request(self) -> httpx.Request | None:
        call = self.call
        account = self.account_sid  # configuration only: never the wire's accountSid
        if call is None or not account or not self.auth_token:
            return None
        if not self.valid_call_id(call.call_id) or not _TWILIO_ACCOUNT_SID.fullmatch(account):
            return None
        assert call.call_id is not None
        path = f"/2010-04-01/Accounts/{_segment(account)}/Calls/{_segment(call.call_id)}.json"
        return httpx.Request(
            "POST",
            self.api_base + path,
            data={"Status": "completed"},
            headers={"Authorization": _basic_auth(account, self.auth_token)},
        )


# ------------------------------------------------------------------------------ Telnyx
_TELNYX_CODECS: dict[CodecName, str] = {"mulaw": "PCMU", "alaw": "PCMA", "l16": "L16"}


class TelnyxSerializer(TelephonySerializer):
    """Telnyx media streaming with ``stream_bidirectional_mode: "rtp"``.

    JSON messages with base64 RTP payloads (no RTP headers). The inbound codec is read from
    the ``start`` message; the outbound codec is the ``stream_bidirectional_codec`` /
    ``bidirectionalCodec`` you configured on the call, which Telnyx does not echo back.

    Args:
        api_key: REST key for :meth:`hangup_request` (default: ``TELNYX_API_KEY``).
        outbound_encoding: ``PCMU``/``PCMA``/``L16`` sent to Telnyx (default: the inbound
            encoding).
        outbound_sample_rate: rate of the outbound audio (default: the inbound rate for
            G.711, 16 kHz for L16).
        l16_byteorder: byte order of L16 samples. RTP's L16 is network order
            (``"big"``, RFC 3551), which is the default.
    """

    provider = "telnyx"
    default_api_base = "https://api.telnyx.com"
    call_id_pattern = _TELNYX_CALL_CONTROL_ID

    def __init__(
        self,
        *,
        api_key: str | None = None,
        outbound_encoding: str | None = None,
        outbound_sample_rate: int | None = None,
        l16_byteorder: ByteOrder = "big",
        api_base: str | None = None,
    ) -> None:
        super().__init__(api_base=api_base)
        self.api_key = _credential(api_key, "TELNYX_API_KEY")
        self.outbound_encoding = outbound_encoding
        self.outbound_sample_rate = outbound_sample_rate
        self.l16_byteorder: ByteOrder = l16_byteorder
        if outbound_encoding is not None:
            _codec_name(outbound_encoding)  # fail early on typos

    def parse(self, message: str | bytes) -> list[TelephonyEvent]:
        data = _load(message)
        event = data.get("event")
        if event == "media":
            media = _dict(data.get("media"))
            if not _inbound(media.get("track")):
                return []
            return [AudioReceived(self.input_codec.decode(_unb64(media.get("payload"))))]
        if event == "mark":
            name = _str(_dict(data.get("mark")).get("name"))
            return [MarkReached(name)] if name is not None else []
        if event == "dtmf":
            dtmf = _digit(_dict(data.get("dtmf")).get("digit"))
            return [dtmf] if dtmf else []
        if event == "start":
            return [self._on_start(data)]
        if event == "stop":
            return [StreamStopped()]
        if event == "error":
            payload = _dict(data.get("payload"))
            text = (
                f"{payload.get('title', 'error')} ({payload.get('code')}): {payload.get('detail')}"
            )
            return [ProviderError(text, data)]
        return []

    def _on_start(self, data: dict[str, Any]) -> StreamStarted:
        start = _dict(data.get("start"))
        fmt = _dict(start.get("media_format"))
        encoding = str(fmt.get("encoding") or "PCMU")
        name = _codec_name(encoding)
        in_codec = AudioCodec(name, _rate(fmt.get("sample_rate"), 8_000), self.l16_byteorder)
        out_name = _codec_name(self.outbound_encoding) if self.outbound_encoding else name
        default_rate = 16_000 if out_name == "l16" else 8_000
        if self.outbound_encoding is None and out_name == name:
            default_rate = in_codec.sample_rate
        out_rate = self.outbound_sample_rate or default_rate
        out_codec = AudioCodec(out_name, out_rate, self.l16_byteorder)
        params: dict[str, Any] = dict(_dict(start.get("custom_parameters")))
        if start.get("client_state") is not None:
            params.setdefault("client_state", start["client_state"])
        call = CallInfo(
            provider=self.provider,
            call_id=_str(start.get("call_control_id")),
            stream_id=_str(data.get("stream_id") or start.get("stream_id")),
            account_id=_str(start.get("user_id")),
            from_number=_str(start.get("from")),
            to_number=_str(start.get("to")),
            custom_parameters=params,
            encoding=encoding,
            sample_rate=in_codec.sample_rate,
            raw=data,
        )
        self._start(call, in_codec, out_codec)
        return StreamStarted(call)

    def encode_audio(self, pcm: bytes) -> str:
        return _dumps({"event": "media", "media": {"payload": _b64(self.output_codec.encode(pcm))}})

    def encode_clear(self) -> str:
        return _dumps({"event": "clear"})

    def encode_mark(self, name: str) -> str:
        return _dumps({"event": "mark", "mark": {"name": name}})

    @property
    def outbound_codec_name(self) -> str:
        """Telnyx's name for the outbound codec (``stream_bidirectional_codec``)."""
        return _TELNYX_CODECS[self.output_codec.name]

    def hangup_request(self) -> httpx.Request | None:
        call = self.call
        if call is None or not self.api_key or not self.valid_call_id(call.call_id):
            return None
        assert call.call_id is not None
        return httpx.Request(
            "POST",
            f"{self.api_base}/v2/calls/{_segment(call.call_id)}/actions/hangup",
            json={},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )


# ------------------------------------------------------------------------------ Vonage
_RATE_RE = re.compile(r"rate\s*=\s*(\d+)", re.IGNORECASE)


class VonageSerializer(TelephonySerializer):
    """Vonage Voice API WebSockets: binary 16-bit little-endian PCM, 20 ms frames.

    The rate comes from the endpoint's ``content-type`` (``audio/l16;rate=16000``),
    announced again in the first ``websocket:connected`` text message. Control messages
    are JSON text: ``{"action": "clear"}`` and ``{"action": "notify", "payload": ...}``
    (echoed back as ``websocket:notify`` once the audio sent before it has played).

    REST hang-up needs a Vonage application JWT (RS256): not implemented, close the
    WebSocket to end the leg instead.
    """

    provider = "vonage"
    default_sample_rate = 16_000
    fixed_frames = True
    json_audio = False

    def __init__(self, *, sample_rate: int = 16_000, api_base: str | None = None) -> None:
        super().__init__(api_base=api_base)
        self.input_codec = self.output_codec = AudioCodec("l16", sample_rate)

    def parse(self, message: str | bytes) -> list[TelephonyEvent]:
        if isinstance(message, bytes | bytearray | memoryview):
            return [AudioReceived(self.input_codec.decode(bytes(message)))]
        data = _load(message)
        event = data.get("event")
        if event == "websocket:notify":
            name = _str(_dict(data.get("payload")).get("name"))
            return [MarkReached(name)] if name is not None else []
        if event == "websocket:dtmf":
            digit = data.get("digit", _dict(data.get("dtmf")).get("digit"))
            dtmf = _digit(digit)
            if dtmf is not None and isinstance(data.get("duration"), int | float):
                dtmf.duration_ms = int(data["duration"])
            return [dtmf] if dtmf else []
        if event == "websocket:cleared":
            return [AudioCleared()]
        if event == "websocket:connected":
            return [self._on_connected(data)]
        if event == "websocket:error":
            return [ProviderError(str(data.get("message") or data), data)]
        return []

    def _on_connected(self, data: dict[str, Any]) -> StreamStarted:
        content_type = str(data.get("content-type") or "audio/l16;rate=16000")
        if _codec_name(content_type) != "l16":
            raise TelephonyProtocolError(f"unsupported Vonage content-type {content_type!r}")
        match = _RATE_RE.search(content_type)
        rate = _rate(match.group(1) if match else None, self.input_codec.sample_rate)
        codec = AudioCodec("l16", rate)
        headers = {k: v for k, v in data.items() if k not in ("event", "content-type")}
        call = CallInfo(
            provider=self.provider,
            call_id=_str(headers.get("uuid") or headers.get("call_uuid")),
            custom_parameters=headers,
            encoding=content_type,
            sample_rate=rate,
            raw=data,
        )
        self._start(call, codec, codec)
        return StreamStarted(call)

    def encode_audio(self, pcm: bytes) -> bytes:
        return self.output_codec.encode(pcm)

    def encode_clear(self) -> str:
        return _dumps({"action": "clear"})

    def encode_mark(self, name: str) -> str:
        return _dumps({"action": "notify", "payload": {"name": name}})


# ------------------------------------------------------------------------------ Plivo
class PlivoSerializer(TelephonySerializer):
    """Plivo audio streams (``<Stream bidirectional="true">``): JSON, base64 payloads.

    ``playAudio`` must use the stream's ``contentType`` (``audio/x-mulaw;rate=8000``,
    ``audio/x-l16;rate=8000`` or ``audio/x-l16;rate=16000``); playback position comes
    from ``checkpoint`` -> ``playedStream``.

    Args:
        auth_id / auth_token: REST credentials for :meth:`hangup_request` (default:
            ``PLIVO_AUTH_ID`` / ``PLIVO_AUTH_TOKEN``). A stream whose ``accountId`` is not
            the configured Auth ID is rejected.
        l16_byteorder: byte order of ``audio/x-l16`` samples (default little-endian).
    """

    provider = "plivo"
    default_api_base = "https://api.plivo.com"
    call_id_pattern = _UUID

    def __init__(
        self,
        *,
        auth_id: str | None = None,
        auth_token: str | None = None,
        l16_byteorder: ByteOrder = "little",
        api_base: str | None = None,
    ) -> None:
        super().__init__(api_base=api_base)
        self.auth_id = _credential(auth_id, "PLIVO_AUTH_ID")
        self.auth_token = _credential(auth_token, "PLIVO_AUTH_TOKEN")
        self.l16_byteorder: ByteOrder = l16_byteorder
        self._content_type = "audio/x-mulaw"

    def parse(self, message: str | bytes) -> list[TelephonyEvent]:
        data = _load(message)
        event = data.get("event")
        if event == "media":
            media = _dict(data.get("media"))
            if not _inbound(media.get("track")):
                return []
            return [AudioReceived(self.input_codec.decode(_unb64(media.get("payload"))))]
        if event == "playedStream":
            name = _str(data.get("name"))
            return [MarkReached(name)] if name is not None else []
        if event == "dtmf":
            dtmf = _digit(_dict(data.get("dtmf")).get("digit"))
            return [dtmf] if dtmf else []
        if event == "clearedAudio":
            return [AudioCleared()]
        if event == "start":
            return [self._on_start(data)]
        if event == "stop":
            return [StreamStopped()]
        return []

    def _on_start(self, data: dict[str, Any]) -> StreamStarted:
        start = _dict(data.get("start"))
        fmt = start.get("mediaFormat")
        if isinstance(fmt, dict):
            encoding = str(fmt.get("encoding") or "audio/x-mulaw")
            rate = _rate(fmt.get("sampleRate"), 8_000)
        else:  # "audio/x-l16;rate=16000"
            encoding = str(fmt or "audio/x-mulaw")
            match = _RATE_RE.search(encoding)
            rate = _rate(match.group(1) if match else None, 8_000)
        name = _codec_name(encoding)
        codec = AudioCodec(name, rate, self.l16_byteorder)
        account = _str(start.get("accountId"))
        if self.auth_id and account is not None and account != self.auth_id:
            raise TelephonyProtocolError(
                f"stream of account {account[:40]!r}, not the configured PLIVO_AUTH_ID"
            )
        self._content_type = "audio/x-l16" if name == "l16" else f"audio/x-{name}"
        call = CallInfo(
            provider=self.provider,
            call_id=_str(start.get("callId")),
            stream_id=_str(start.get("streamId") or data.get("streamId")),
            account_id=account,
            custom_parameters=_parse_extra_headers(
                data.get("extra_headers", start.get("extra_headers"))
            ),
            encoding=encoding,
            sample_rate=rate,
            raw=data,
        )
        self._start(call, codec, codec)
        return StreamStarted(call)

    @property
    def _sid(self) -> str | None:
        return self.call.stream_id if self.call else None

    def encode_audio(self, pcm: bytes) -> str:
        media = {
            "contentType": self._content_type,
            "sampleRate": self.output_codec.sample_rate,
            "payload": _b64(self.output_codec.encode(pcm)),
        }
        return _dumps({"event": "playAudio", "media": media})

    def encode_clear(self) -> str:
        return _dumps({"event": "clearAudio", "streamId": self._sid})

    def encode_mark(self, name: str) -> str:
        return _dumps({"event": "checkpoint", "streamId": self._sid, "name": name})

    def hangup_request(self) -> httpx.Request | None:
        call = self.call
        if call is None or not self.auth_id or not self.auth_token:
            return None
        if not self.valid_call_id(call.call_id):
            return None
        assert call.call_id is not None
        return httpx.Request(
            "DELETE",
            f"{self.api_base}/v1/Account/{_segment(self.auth_id)}/Call/{_segment(call.call_id)}/",
            headers={"Authorization": _basic_auth(self.auth_id, self.auth_token)},
        )


def _parse_extra_headers(value: Any) -> dict[str, Any]:
    """Plivo ``extraHeaders``: ``"key1=value1;key2=value2"`` (the start message) or
    ``"key1=value1,key2=value2"`` (the XML attribute), or already a dict."""
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    out: dict[str, Any] = {}
    for part in re.split(r"[;,]", value):
        key, sep, val = part.partition("=")
        if key.strip():
            out[key.strip()] = val.strip() if sep else ""
    return out


def _basic_auth(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


# ------------------------------------------------------------------------------ registry
SERIALIZERS: dict[str, type[TelephonySerializer]] = {
    "twilio": TwilioSerializer,
    "telnyx": TelnyxSerializer,
    "vonage": VonageSerializer,
    "plivo": PlivoSerializer,
}


def create_serializer(provider: str, **options: Any) -> TelephonySerializer:
    """``create_serializer("twilio", auth_token=...)`` -> a fresh serializer for one call."""
    cls = SERIALIZERS.get(provider.lower())
    if cls is None:
        raise ConfigurationError(
            f"unknown telephony provider {provider!r}; known: {', '.join(sorted(SERIALIZERS))}"
        )
    return cls(**options)
