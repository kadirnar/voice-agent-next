"""Telephony media streams: fake Twilio/Telnyx/Vonage/Plivo carriers replay each provider's
documented messages over a real WebSocket and simulate playback (jitter buffer + marks)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from voice_agent_next import Agent, AgentSession, AudioFrame
from voice_agent_next.audio.codecs import alaw_encode, mulaw_decode, mulaw_encode
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.session import Interrupted, SessionClosed
from voice_agent_next.transports import create_transport
from voice_agent_next.transports.telephony import (
    AudioCodec,
    PlivoSerializer,
    TelephonyServer,
    TelephonyTransport,
    TelnyxSerializer,
    TwilioSerializer,
    TwilioTransport,
    VonageSerializer,
    VonageTransport,
    create_serializer,
    plivo_stream_xml,
    telnyx_stream_texml,
    twilio_stream_twiml,
    vonage_ncco,
)
from voice_agent_next.transports.telephony.serializers import (
    AudioCleared,
    AudioReceived,
    DtmfReceived,
    MarkReached,
    ProviderError,
    StreamStarted,
    StreamStopped,
    TelephonyProtocolError,
)
from voice_agent_next.utils import cancel_and_wait

PROVIDERS = ["twilio", "telnyx", "vonage", "plivo"]
STREAM_SID = "MZ18ad3ab5a668481ce02b83e7395059f0"
CALL_SID = "CA5a1ebd8dcc0ff4c2a2ea3fcbbdf3a1c4"
TELNYX_STREAM = "32DE0DEA-53CB-4B21-89A4-9E1819C043BC"
TELNYX_CALL = "v2:T02llQxIyaRkhfRKxgAP8nY511EhFLizdvdUKJiSw8d6A9BborherQ"
PLIVO_STREAM = "87654321-4321-4321-4321-cba987654321"
PLIVO_CALL = "12345678-1234-1234-1234-123456789abc"


async def wait_for(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ------------------------------------------------------------------ documented messages
def twilio_start(custom: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "accountSid": "AC00000000000000000000000000000000",
            "streamSid": STREAM_SID,
            "callSid": CALL_SID,
            "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            "customParameters": custom or {},
        },
        "streamSid": STREAM_SID,
    }


def telnyx_start(encoding: str = "PCMU", rate: int = 8000) -> dict[str, Any]:
    return {
        "event": "start",
        "sequence_number": "1",
        "stream_id": TELNYX_STREAM,
        "start": {
            "user_id": "3E6F995F-85F7-4705-9741-53B116D28237",
            "call_control_id": TELNYX_CALL,
            "call_session_id": "ff55a038-6f5d-11ef-9692-02420aeffb1f",
            "from": "+13122010094",
            "to": "+13122123456",
            "tags": ["TAG1", "TAG2"],
            "client_state": "aGF2ZSBhIG5pY2UgZGF5ID1d",
            "media_format": {"encoding": encoding, "sample_rate": rate, "channels": 1},
        },
    }


def plivo_start(encoding: str = "audio/x-mulaw", rate: int = 8000) -> dict[str, Any]:
    return {
        "event": "start",
        "sequenceNumber": 1,
        "start": {
            "callId": PLIVO_CALL,
            "streamId": PLIVO_STREAM,
            "accountId": "MAXXXXXXXXXXXXXXXXXX",
            "tracks": ["inbound"],
            "mediaFormat": {"encoding": encoding, "sampleRate": rate},
        },
        "extra_headers": "userId=12345;sessionId=abc-xyz",
    }


def vonage_connected(rate: int = 16_000) -> dict[str, Any]:
    return {"event": "websocket:connected", "content-type": f"audio/l16;rate={rate}", "user": "ada"}


# --------------------------------------------------------------------------- codecs
def test_codecs_roundtrip_and_byte_order() -> None:
    pcm = np.array([0, 1, -2, 1000, -32768, 32767], dtype="<i2").tobytes()
    little, big = AudioCodec("l16", 16_000), AudioCodec("l16", 16_000, "big")
    assert little.encode(pcm) == pcm and little.decode(pcm + b"\x01") == pcm
    assert big.encode(pcm) == np.frombuffer(pcm, "<i2").astype(">i2").tobytes()
    assert big.decode(big.encode(pcm)) == pcm
    mulaw = AudioCodec("mulaw", 8_000)
    assert mulaw.encode(pcm) == mulaw_encode(pcm) and mulaw.decode(b"\xff\x7f") == mulaw_decode(
        b"\xff\x7f"
    )
    assert AudioCodec("alaw", 8_000).encode(pcm) == alaw_encode(pcm)
    assert (mulaw.bytes_per_sample, little.bytes_per_sample) == (1, 2)


# ----------------------------------------------------------------------- serializers
def test_twilio_serializer_parses_and_encodes_documented_messages() -> None:
    s = TwilioSerializer()
    assert s.parse(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})) == []
    (started,) = s.parse(json.dumps(twilio_start({"lang": "en"})))
    assert isinstance(started, StreamStarted)
    call = started.call
    assert (call.call_id, call.stream_id, call.custom_parameters) == (
        CALL_SID,
        STREAM_SID,
        {"lang": "en"},
    )
    assert (s.input_codec.name, s.input_codec.sample_rate, s.output_codec.name) == (
        "mulaw",
        8000,
        "mulaw",
    )
    media = {
        "event": "media",
        "sequenceNumber": "3",
        "media": {
            "track": "inbound",
            "chunk": "1",
            "timestamp": "5",
            "payload": b64(b"\xff" * 160),
        },
        "streamSid": STREAM_SID,
    }
    (audio,) = s.parse(json.dumps(media))
    assert isinstance(audio, AudioReceived) and audio.pcm == mulaw_decode(b"\xff" * 160)
    dtmf = {
        "event": "dtmf",
        "streamSid": STREAM_SID,
        "sequenceNumber": "5",
        "dtmf": {"track": "inbound_track", "digit": "1"},
    }
    assert s.parse(json.dumps(dtmf)) == [DtmfReceived("1")]
    mark = {"event": "mark", "sequenceNumber": "4", "streamSid": STREAM_SID, "mark": {"name": "x"}}
    assert s.parse(json.dumps(mark)) == [MarkReached("x")]
    stop = {
        "event": "stop",
        "sequenceNumber": "5",
        "stop": {"accountSid": "AC0", "callSid": CALL_SID},
        "streamSid": STREAM_SID,
    }
    assert s.parse(json.dumps(stop)) == [StreamStopped()]

    pcm = synth_speech(0.02, 8000).data
    assert json.loads(s.encode_audio(pcm)) == {
        "event": "media",
        "streamSid": STREAM_SID,
        "media": {"payload": b64(mulaw_encode(pcm))},
    }
    assert json.loads(s.encode_clear()) == {"event": "clear", "streamSid": STREAM_SID}
    assert json.loads(s.encode_mark("m1")) == {
        "event": "mark",
        "streamSid": STREAM_SID,
        "mark": {"name": "m1"},
    }
    for bad in ("not json", "[1]", json.dumps({"event": "media", "media": {"payload": "***"}})):
        with pytest.raises(TelephonyProtocolError):
            s.parse(bad)


def test_telnyx_serializer_codecs_and_documented_messages() -> None:
    s = TelnyxSerializer()
    assert s.parse(json.dumps({"event": "connected", "version": "1.0.0"})) == []
    (started,) = s.parse(json.dumps(telnyx_start()))
    assert isinstance(started, StreamStarted)
    assert (started.call.call_id, started.call.stream_id) == (TELNYX_CALL, TELNYX_STREAM)
    assert started.call.from_number == "+13122010094"
    assert started.call.custom_parameters["client_state"] == "aGF2ZSBhIG5pY2UgZGF5ID1d"
    assert (s.output_codec.name, s.output_codec.sample_rate, s.outbound_codec_name) == (
        "mulaw",
        8000,
        "PCMU",
    )
    outbound = {
        "event": "media",
        "sequence_number": "4",
        "stream_id": TELNYX_STREAM,
        "media": {"track": "outbound", "chunk": "2", "timestamp": "5", "payload": b64(b"\0")},
    }
    assert s.parse(json.dumps(outbound)) == []  # both_tracks: our own audio is ignored
    inbound = {**outbound, "media": {**outbound["media"], "track": "inbound"}}
    assert s.parse(json.dumps(inbound)) == [AudioReceived(mulaw_decode(b"\0"))]
    dtmf = {
        "event": "dtmf",
        "stream_id": TELNYX_STREAM,
        "sequence_number": "5",
        "occurred_at": "2025-06-05T08:54:19.698408Z",
        "dtmf": {"digit": "#"},
    }
    assert s.parse(json.dumps(dtmf)) == [DtmfReceived("#")]
    error = {
        "event": "error",
        "stream_id": TELNYX_STREAM,
        "payload": {"code": 100002, "title": "malformed_frame", "detail": "bad"},
    }
    (err,) = s.parse(json.dumps(error))
    assert isinstance(err, ProviderError) and "malformed_frame" in err.message
    assert s.parse(json.dumps({"event": "stop", "stream_id": TELNYX_STREAM})) == [StreamStopped()]
    assert json.loads(s.encode_clear()) == {"event": "clear"}
    assert json.loads(s.encode_mark("m")) == {"event": "mark", "mark": {"name": "m"}}
    assert json.loads(s.encode_audio(b"\0\0"))["media"]["payload"] == b64(mulaw_encode(b"\0\0"))

    # L16 16 kHz, network byte order (RTP), with a different outbound codec
    s = TelnyxSerializer(outbound_encoding="PCMA")
    s.parse(json.dumps(telnyx_start("L16", 16_000)))
    assert (s.input_codec.name, s.input_codec.sample_rate, s.input_codec.byteorder) == (
        "l16",
        16_000,
        "big",
    )
    assert (s.output_codec.name, s.output_codec.sample_rate) == ("alaw", 8000)
    wire = np.array([1000], dtype=">i2").tobytes()
    media = {"event": "media", "media": {"track": "inbound", "payload": b64(wire)}}
    assert s.parse(json.dumps(media)) == [AudioReceived(np.array([1000], "<i2").tobytes())]
    with pytest.raises(TelephonyProtocolError, match="unsupported"):
        TelnyxSerializer().parse(json.dumps(telnyx_start("OPUS", 16_000)))


def test_vonage_serializer_binary_audio_and_json_control() -> None:
    s = VonageSerializer()
    (started,) = s.parse(json.dumps(vonage_connected(8000)))
    assert isinstance(started, StreamStarted)
    assert started.call.custom_parameters == {"user": "ada"}
    assert s.input_codec == s.output_codec == AudioCodec("l16", 8000)
    frame = synth_speech(0.02, 8000).data
    assert s.parse(frame) == [AudioReceived(frame)]
    assert s.encode_audio(frame) == frame  # binary, little-endian
    dtmf = {"event": "websocket:dtmf", "digit": "5", "duration": 260}
    assert s.parse(json.dumps(dtmf)) == [DtmfReceived("5", 260)]
    assert s.parse(json.dumps({"event": "websocket:cleared"})) == [AudioCleared()]
    notify = {"event": "websocket:notify", "payload": {"name": "m7"}}
    assert s.parse(json.dumps(notify)) == [MarkReached("m7")]
    assert json.loads(s.encode_clear()) == {"action": "clear"}
    assert json.loads(s.encode_mark("m7")) == {"action": "notify", "payload": {"name": "m7"}}
    assert s.hangup_request() is None  # needs a Vonage JWT: not supported


def test_plivo_serializer_documented_messages() -> None:
    s = PlivoSerializer(l16_byteorder="little")
    (started,) = s.parse(json.dumps(plivo_start("audio/x-l16", 16_000)))
    assert isinstance(started, StreamStarted)
    assert (started.call.call_id, started.call.stream_id) == (PLIVO_CALL, PLIVO_STREAM)
    assert started.call.custom_parameters == {"userId": "12345", "sessionId": "abc-xyz"}
    assert s.output_codec == AudioCodec("l16", 16_000)
    media = {
        "event": "media",
        "sequenceNumber": 42,
        "streamId": PLIVO_STREAM,
        "media": {
            "track": "inbound",
            "timestamp": "1705312200000",
            "chunk": 41,
            "payload": b64(b"\1\0"),
        },
        "extra_headers": "userId=12345;sessionId=abc-xyz",
    }
    assert s.parse(json.dumps(media)) == [AudioReceived(b"\1\0")]
    dtmf = {
        "event": "dtmf",
        "sequenceNumber": 50,
        "streamId": PLIVO_STREAM,
        "dtmf": {"track": "inbound", "digit": "5", "timestamp": "1705312250000"},
    }
    assert s.parse(json.dumps(dtmf)) == [DtmfReceived("5")]
    played = {"event": "playedStream", "sequenceNumber": 75, "streamId": PLIVO_STREAM, "name": "c1"}
    assert s.parse(json.dumps(played)) == [MarkReached("c1")]
    cleared = {"event": "clearedAudio", "sequenceNumber": 80, "streamId": PLIVO_STREAM}
    assert s.parse(json.dumps(cleared)) == [AudioCleared()]
    assert json.loads(s.encode_audio(b"\1\0")) == {
        "event": "playAudio",
        "media": {"contentType": "audio/x-l16", "sampleRate": 16000, "payload": b64(b"\1\0")},
    }
    assert json.loads(s.encode_mark("c1")) == {
        "event": "checkpoint",
        "streamId": PLIVO_STREAM,
        "name": "c1",
    }
    assert json.loads(s.encode_clear()) == {"event": "clearAudio", "streamId": PLIVO_STREAM}

    mulaw = PlivoSerializer()
    mulaw.parse(json.dumps(plivo_start()))
    assert json.loads(mulaw.encode_audio(b"\0\0"))["media"]["contentType"] == "audio/x-mulaw"


def test_hangup_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    twilio = TwilioSerializer()
    twilio.parse(json.dumps(twilio_start()))
    assert twilio.hangup_request() is None  # no credentials
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    twilio = TwilioSerializer(api_base="https://twilio.test")
    assert twilio.hangup_request() is None  # stream not started
    twilio.parse(json.dumps(twilio_start()))
    req = twilio.hangup_request()
    assert req is not None and req.method == "POST"
    assert str(req.url) == (
        f"https://twilio.test/2010-04-01/Accounts/AC00000000000000000000000000000000/Calls/"
        f"{CALL_SID}.json"
    )
    assert req.read() == b"Status=completed"
    assert req.headers["authorization"].startswith("Basic ")

    telnyx = TelnyxSerializer(api_key="KEY")
    telnyx.parse(json.dumps(telnyx_start()))
    req = telnyx.hangup_request()
    assert req is not None and req.method == "POST"
    assert str(req.url) == f"https://api.telnyx.com/v2/calls/{TELNYX_CALL}/actions/hangup"
    assert req.headers["authorization"] == "Bearer KEY"

    plivo = PlivoSerializer(auth_id="MA1", auth_token="tok")
    plivo.parse(json.dumps(plivo_start()))
    req = plivo.hangup_request()
    assert req is not None and req.method == "DELETE"
    assert str(req.url) == f"https://api.plivo.com/v1/Account/MA1/Call/{PLIVO_CALL}/"


def test_markup_helpers() -> None:
    twiml = twilio_stream_twiml("wss://example.com/twilio?x=1&y=2", {"lang": "en"})
    assert twiml.endswith(
        '<Response><Connect><Stream url="wss://example.com/twilio?x=1&amp;y=2">'
        '<Parameter name="lang" value="en"/></Stream></Connect></Response>'
    )
    texml = telnyx_stream_texml("wss://example.com/telnyx", codec="L16", sample_rate=16000)
    assert 'bidirectionalMode="rtp" bidirectionalCodec="L16"' in texml
    assert 'bidirectionalSamplingRate="16000"' in texml and "<Pause" in texml
    assert vonage_ncco("wss://example.com/vonage", headers={"user": "ada"}) == [
        {
            "action": "connect",
            "endpoint": [
                {
                    "type": "websocket",
                    "uri": "wss://example.com/vonage",
                    "content-type": "audio/l16;rate=16000",
                    "headers": {"user": "ada"},
                }
            ],
        }
    ]
    xml = plivo_stream_xml("wss://example.com/plivo", extra_headers={"a": 1})
    assert '<Stream bidirectional="true" keepCallAlive="true"' in xml
    assert (
        'contentType="audio/x-mulaw;rate=8000" extraHeaders="a=1">wss://example.com/plivo<' in xml
    )


def test_create_transport_and_serializer_registry() -> None:
    for provider in PROVIDERS:
        transport = create_transport({"type": provider, "port": 0})
        assert isinstance(transport, TelephonyTransport) and transport.provider == provider
        assert transport.capabilities.playback_position and transport.capabilities.dtmf
        assert not transport.capabilities.messages
    assert create_transport("vonage").input_format.sample_rate == 16_000
    generic = create_transport({"type": "telephony", "provider": "plivo", "auth_id": "x"})
    assert isinstance(generic, TelephonyTransport) and generic.provider == "plivo"
    assert isinstance(create_serializer("TWILIO"), TwilioSerializer)
    with pytest.raises(Exception, match="unknown telephony provider"):
        create_serializer("skype")
    with pytest.raises(TypeError):
        TelephonyTransport(provider=VonageSerializer(), auth_token="x")


# --------------------------------------------------------------------- fake carriers
class Carrier:
    """A fake provider: sends the provider's messages and plays our audio in real time.

    Playback model: every media chunk plays after ``latency`` (a jitter buffer) and after
    the previous chunk; a mark comes back when the audio before it has played, pending
    marks are flushed (Twilio, Telnyx) or dropped (Vonage, Plivo) by a clear.
    """

    def __init__(self, provider: str, url: str, *, latency: float = 0.0) -> None:
        self.provider = provider
        self.url = url
        self.latency = latency
        self.rate = 16_000 if provider == "vonage" else 8_000
        self.ws: ClientConnection | None = None
        self.log: list[tuple[str, Any]] = []  # ("audio", pcm) | ("mark", name) | ("clear", t)
        self.chunks: list[list[float]] = []  # [start, duration]
        self.last_end = 0.0
        self.pending: dict[str, asyncio.TimerHandle] = {}
        self.echoed: list[str] = []
        self.played_at_clear: list[float] = []
        self.seq = 1
        self.tasks: set[asyncio.Task[Any]] = set()
        self.marks: list[str] = []
        self.auto_echo = True
        self.clears = 0
        self._reader: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Carrier:
        self.ws = await connect(self.url)
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        assert self.ws is not None
        for handle in self.pending.values():
            handle.cancel()
        await self.ws.close()
        await cancel_and_wait(self._reader)

    async def wait_closed(self, timeout: float = 5.0) -> None:
        assert self._reader is not None
        await asyncio.wait_for(asyncio.shield(self._reader), timeout)

    # -------------------------------------------------------------- caller -> agent
    async def send(self, message: dict[str, Any] | bytes) -> None:
        assert self.ws is not None
        await self.ws.send(message if isinstance(message, bytes) else json.dumps(message))

    async def start(self) -> None:
        if self.provider == "twilio":
            await self.send({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            await self.send(twilio_start({"caller": "+15550100"}))
        elif self.provider == "telnyx":
            await self.send({"event": "connected", "version": "1.0.0"})
            await self.send(telnyx_start())
        elif self.provider == "plivo":
            await self.send(plivo_start())
        else:
            await self.send(vonage_connected(self.rate))

    async def speak(self, seconds: float = 0.8, silence: float = 0.6) -> None:
        audio = AudioFrame.concat(
            [synth_speech(seconds, self.rate), AudioFrame.silence(silence, self.rate)]
        )
        step = round(0.02 * self.rate) * 2
        for i in range(0, len(audio.data), step):
            await self.send(self.media(audio.data[i : i + step]))
            await asyncio.sleep(0)

    def media(self, pcm: bytes) -> dict[str, Any] | bytes:
        self.seq += 1
        payload = b64(mulaw_encode(pcm))
        if self.provider == "twilio":
            return {
                "event": "media",
                "sequenceNumber": str(self.seq),
                "media": {"track": "inbound", "chunk": str(self.seq), "timestamp": "0",
                          "payload": payload},
                "streamSid": STREAM_SID,
            }  # fmt: skip
        if self.provider == "telnyx":
            return {
                "event": "media",
                "sequence_number": str(self.seq),
                "stream_id": TELNYX_STREAM,
                "media": {"track": "inbound", "chunk": str(self.seq), "timestamp": "0",
                          "payload": payload},
            }  # fmt: skip
        if self.provider == "plivo":
            return {
                "event": "media",
                "sequenceNumber": self.seq,
                "streamId": PLIVO_STREAM,
                "media": {"track": "inbound", "timestamp": "0", "chunk": self.seq,
                          "payload": payload},
                "extra_headers": "",
            }  # fmt: skip
        return pcm  # vonage: raw binary L16

    async def press(self, digit: str) -> None:
        if self.provider == "twilio":
            await self.send(
                {
                    "event": "dtmf",
                    "streamSid": STREAM_SID,
                    "sequenceNumber": "9",
                    "dtmf": {"track": "inbound_track", "digit": digit},
                }
            )
        elif self.provider == "telnyx":
            await self.send(
                {
                    "event": "dtmf",
                    "stream_id": TELNYX_STREAM,
                    "sequence_number": "9",
                    "occurred_at": "2025-06-05T08:54:19.698408Z",
                    "dtmf": {"digit": digit},
                }
            )
        elif self.provider == "plivo":
            await self.send(
                {
                    "event": "dtmf",
                    "sequenceNumber": 9,
                    "streamId": PLIVO_STREAM,
                    "dtmf": {"track": "inbound", "digit": digit, "timestamp": "0"},
                    "extra_headers": "",
                }
            )
        else:
            await self.send({"event": "websocket:dtmf", "digit": digit, "duration": 260})

    async def hang_up(self) -> None:
        """The caller hangs up: Twilio/Telnyx send ``stop``; Vonage/Plivo close the socket."""
        if self.provider == "twilio":
            await self.send(
                {
                    "event": "stop",
                    "sequenceNumber": "99",
                    "stop": {"accountSid": "AC0", "callSid": CALL_SID},
                    "streamSid": STREAM_SID,
                }
            )
        elif self.provider == "telnyx":
            await self.send(
                {
                    "event": "stop",
                    "sequence_number": "99",
                    "stream_id": TELNYX_STREAM,
                    "stop": {"user_id": "u", "call_control_id": TELNYX_CALL},
                }
            )
        else:
            await self.close()

    # -------------------------------------------------------------- agent -> caller
    async def _read(self) -> None:
        assert self.ws is not None
        try:
            async for message in self.ws:
                self._on_message(message)
        except ConnectionClosed:
            pass

    def _on_message(self, message: str | bytes) -> None:
        if isinstance(message, bytes):
            assert self.provider == "vonage"
            assert len(message) == 640, len(message)  # whole 20 ms frames
            self._play(message)
            return
        data = json.loads(message)
        kind = data.get("event") or data.get("action")
        if self.provider in ("twilio", "telnyx"):
            if self.provider == "twilio":
                assert data["streamSid"] == STREAM_SID
            if kind == "media":
                self._play(mulaw_decode(base64.b64decode(data["media"]["payload"])))
            elif kind == "mark":
                self._mark(data["mark"]["name"])
            elif kind == "clear":
                self._clear()
        elif self.provider == "plivo":
            if kind == "playAudio":
                media = data["media"]
                assert (media["contentType"], media["sampleRate"]) == ("audio/x-mulaw", 8000)
                self._play(mulaw_decode(base64.b64decode(media["payload"])))
            elif kind == "checkpoint":
                assert data["streamId"] == PLIVO_STREAM
                self._mark(data["name"])
            elif kind == "clearAudio":
                assert data["streamId"] == PLIVO_STREAM
                self._clear()
        else:
            if kind == "notify":
                self._mark(data["payload"]["name"])
            elif kind == "clear":
                self._clear()
        if kind not in ("media", "playAudio"):
            self.log.append((str(kind), data))

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    def _play(self, pcm: bytes) -> None:
        t = self._now()
        duration = len(pcm) / 2 / self.rate
        start = max(t + self.latency, self.last_end)
        self.chunks.append([start, duration])
        self.last_end = start + duration
        self.log.append(("audio", pcm))

    def played(self, t: float | None = None) -> float:
        t = self._now() if t is None else t
        return sum(min(max(t - s, 0.0), d) for s, d in self.chunks)

    def _mark(self, name: str) -> None:
        self.marks.append(name)
        if not self.auto_echo:
            return
        delay = max(0.0, self.last_end - self._now())
        loop = asyncio.get_running_loop()
        self.pending[name] = loop.call_later(delay, lambda: self._echo(name))

    def _echo(self, name: str) -> None:
        self.pending.pop(name, None)
        self.echoed.append(name)
        if self.provider == "twilio":
            msg: dict[str, Any] = {
                "event": "mark",
                "sequenceNumber": "4",
                "streamSid": STREAM_SID,
                "mark": {"name": name},
            }
        elif self.provider == "telnyx":
            msg = {
                "event": "mark",
                "stream_id": TELNYX_STREAM,
                "sequence_number": "5",
                "mark": {"name": name},
            }
        elif self.provider == "plivo":
            msg = {"event": "playedStream", "sequenceNumber": 75, "streamId": PLIVO_STREAM,
                   "name": name}  # fmt: skip
        else:
            msg = {"event": "websocket:notify", "payload": {"name": name}}
        self._spawn(self.send(msg))

    def _clear(self) -> None:
        self.clears += 1
        t = self._now()
        self.played_at_clear.append(self.played(t))
        kept = []
        for chunk in self.chunks:
            start, duration = chunk
            if start < t:
                chunk[1] = min(duration, t - start)
                kept.append(chunk)
        self.chunks = kept
        self.last_end = t
        pending, self.pending = self.pending, {}
        for name, handle in pending.items():
            handle.cancel()
            if self.provider in ("twilio", "telnyx"):
                self._echo(name)  # "sends back mark messages matching any remaining marks"
        if self.provider == "vonage":
            self._spawn(self.send({"event": "websocket:cleared"}))
        elif self.provider == "plivo":
            cleared = {"event": "clearedAudio", "sequenceNumber": 80, "streamId": PLIVO_STREAM}
            self._spawn(self.send(cleared))

    def _spawn(self, coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def audio(self) -> bytes:
        return b"".join(m for k, m in self.log if k == "audio")

    def of(self, kind: str) -> list[Any]:
        return [m for k, m in self.log if k == kind]


class Calls:
    """A TelephonyServer with the mock engine; records transports, sessions and events."""

    def __init__(
        self,
        provider: str,
        responses: list[str] | None = None,
        *,
        transcripts: list[str] | None = None,
        greeting: str | None = None,
        serializer_options: dict[str, Any] | None = None,
        transport_options: dict[str, Any] | None = None,
    ) -> None:
        self.transports: list[TelephonyTransport] = []
        self.sessions: list[AgentSession] = []
        self.interruptions: list[Interrupted] = []
        self.closed: list[SessionClosed] = []
        self.digits: list[str] = []

        def session_factory(transport: TelephonyTransport) -> AgentSession:
            session = AgentSession(MockEngine(responses=responses, transcripts=transcripts))
            session.on("interrupted", self.interruptions.append)
            session.on("close", self.closed.append)
            transport.on("dtmf", self.digits.append)
            self.transports.append(transport)
            self.sessions.append(session)
            return session

        self.server = TelephonyServer(
            session_factory,
            lambda: Agent("be brief", greeting=greeting),
            provider=provider,
            port=0,
            serializer_options=serializer_options,
            transport_options=transport_options,
        )

    async def __aenter__(self) -> Calls:
        await self.server.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.server.aclose()

    @property
    def url(self) -> str:
        return self.server.url + "stream"


# ------------------------------------------------------------------- full sessions
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_phone_call_audio_both_ways_and_dtmf(provider: str) -> None:
    answer = "Hi! Nice to meet you."
    async with (
        Calls(provider, [answer], transcripts=["hello"], greeting="Welcome.") as calls,
        Carrier(provider, calls.url) as carrier,
    ):
        await carrier.start()
        await wait_for(lambda: calls.sessions)
        transport = calls.transports[0]
        assert transport.call is not None and transport.call.provider == provider
        rate = carrier.rate
        assert transport.input_format.sample_rate == transport.output_format.sample_rate == rate

        # the greeting plays, marks come back as playback progresses
        greeting_bytes = len("Welcome.") / 15 * rate * 2
        await wait_for(lambda: len(carrier.audio()) >= greeting_bytes * 0.85)
        await wait_for(lambda: transport.buffered_duration() == 0.0 and carrier.echoed)

        await carrier.press("7")
        await carrier.press("#")
        await wait_for(lambda: calls.digits == ["7", "#"])

        await carrier.speak()
        heard_before = len(carrier.audio())
        total = heard_before + len(answer) / 15 * rate * 2
        await wait_for(lambda: len(carrier.audio()) >= total * 0.9, timeout=8)
        await asyncio.wait_for(transport.wait_for_playout(), 5)
        answer_audio = (len(carrier.audio()) - heard_before) / 2 / rate
        assert answer_audio == pytest.approx(len(answer) / 15, abs=0.1)

        session = calls.sessions[0]
        texts = [(m.role, m.text) for m in session.history.messages()]
        assert texts == [("assistant", "Welcome."), ("user", "hello"), ("assistant", answer)]
        assert carrier.clears == 0 and not calls.interruptions
        assert len(carrier.echoed) >= 10  # a mark about every 100 ms of audio


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_barge_in_clears_and_truncates_to_what_the_caller_heard(provider: str) -> None:
    long_answer = "This answer keeps going and going for quite a while, on and on. " * 3
    latency = 0.5  # the carrier's jitter buffer: the caller hears everything 0.5 s late
    async with (
        Calls(provider, [long_answer.strip()], transcripts=["tell me a story"]) as calls,
        Carrier(provider, calls.url, latency=latency) as carrier,
    ):
        await carrier.start()
        await wait_for(lambda: calls.sessions)
        await carrier.speak(0.6, 0.5)
        await wait_for(lambda: carrier.audio())
        await asyncio.sleep(1.2)  # the caller hears ~0.7 s of the answer
        await carrier.speak(0.4, 0.0)  # barge in
        await wait_for(lambda: carrier.clears and calls.interruptions)

        heard = carrier.played_at_clear[0]
        played = calls.interruptions[0].played
        assert 0.3 < heard < 2.5
        # marks make the truncation exact despite the jitter buffer (a wall-clock
        # estimate would be `latency` too long)
        assert played == pytest.approx(heard, abs=0.2), (played, heard)
        conn = calls.sessions[0].connection
        assert conn.truncations and conn.truncations[-1][1] == round(played * 1000)
        assistant = calls.sessions[0].history.messages()[-1]
        assert assistant.interrupted and 0 < len(assistant.text) < len(long_answer.strip())

        # nothing of the interrupted answer is sent after the clear
        await asyncio.sleep(0.3)
        kinds = [k for k, _ in carrier.log]
        clear = {"plivo": "clearAudio"}.get(provider, "clear")
        assert "audio" not in kinds[kinds.index(clear) :] and carrier.clears == 1
        assert calls.transports[0].buffered_duration() == 0.0


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_caller_hang_up_ends_the_session_without_rest_call(provider: str) -> None:
    requests: list[httpx.Request] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(200))
    )
    options = {
        "twilio": {"account_sid": "AC1", "auth_token": "t"},
        "telnyx": {"api_key": "k"},
        "plivo": {"auth_id": "MA1", "auth_token": "t"},
        "vonage": {},
    }[provider]
    async with (
        Calls(provider, serializer_options=options, transport_options={"http_client": client})
        as calls,
        Carrier(provider, calls.url) as carrier,
    ):  # fmt: skip
        await carrier.start()
        await wait_for(lambda: calls.sessions)
        await carrier.hang_up()
        await wait_for(lambda: calls.closed)
        assert calls.closed[0].reason == "user_disconnected"
        await wait_for(lambda: not calls.server.sessions)
        if provider in ("twilio", "telnyx"):
            assert calls.transports[0].stopped
            await carrier.wait_closed()
    assert requests == []
    await client.aclose()


@pytest.mark.parametrize("provider", ["twilio", "telnyx", "plivo"])
async def test_agent_hang_up_calls_the_rest_api(provider: str) -> None:
    requests: list[httpx.Request] = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(204))
    )
    options = {
        "twilio": {"account_sid": "AC1", "auth_token": "t"},
        "telnyx": {"api_key": "k"},
        "plivo": {"auth_id": "MA1", "auth_token": "t"},
    }[provider]
    async with (
        Calls(provider, serializer_options=options, transport_options={"http_client": client})
        as calls,
        Carrier(provider, calls.url) as carrier,
    ):  # fmt: skip
        await carrier.start()
        await wait_for(lambda: calls.sessions)
        await calls.sessions[0].aclose()  # e.g. the agent said goodbye
        await carrier.wait_closed()
        assert len(requests) == 1
        assert requests[0].method == ("DELETE" if provider == "plivo" else "POST")
        call_id = {"twilio": CALL_SID, "telnyx": TELNYX_CALL, "plivo": PLIVO_CALL}[provider]
        assert call_id in str(requests[0].url)
        assert not await calls.transports[0].hangup()  # only once
    await client.aclose()


# -------------------------------------------------------------- transport details
async def test_marks_drive_buffered_duration_and_fall_back_to_the_wall_clock() -> None:
    transports: asyncio.Queue[TelephonyTransport] = asyncio.Queue()

    async def handler(ws: Any) -> None:
        transport = TwilioTransport(ws, mark_interval=0.1, mark_grace=0.5)
        await transport.start()
        await transports.put(transport)
        await transport.wait_disconnected()
        await transport.aclose()

    from websockets.asyncio.server import serve

    async with serve(handler, "127.0.0.1", 0) as ws_server:
        port = ws_server.sockets[0].getsockname()[1]
        carrier = Carrier("twilio", f"ws://127.0.0.1:{port}/")
        carrier.auto_echo = False
        async with carrier:
            await carrier.start()
            transport = await transports.get()
            assert transport.buffered_duration() == 0.0
            await transport.write_audio(AudioFrame.silence(1.0, 8000))
            await wait_for(lambda: len(carrier.marks) == 1)
            assert [len(m) for m in carrier.of("audio")] == [320] * 50
            assert carrier.marks == ["van-1"]
            await asyncio.sleep(0.5)  # no echo: real-time extrapolation
            assert transport.buffered_duration() == pytest.approx(0.5, abs=0.2)
            await asyncio.sleep(1.2)
            assert transport.buffered_duration() == 0.0

            # the caller cannot be past a mark that has not come back yet
            await transport.write_audio(AudioFrame.silence(0.3, 8000))
            await transport.write_audio(AudioFrame.silence(0.3, 8000))
            await wait_for(lambda: len(carrier.marks) == 3)
            assert transport.buffered_duration() == pytest.approx(0.6, abs=0.05)
            await asyncio.sleep(0.45)
            assert transport.buffered_duration() == pytest.approx(0.3, abs=0.03)  # capped
            carrier._echo("van-2")  # playback reached it: re-anchored there
            await asyncio.sleep(0.1)
            assert transport.buffered_duration() == pytest.approx(0.2, abs=0.06)
            carrier._echo("van-3")
            await wait_for(lambda: transport.buffered_duration() == 0.0)
            carrier._echo("van-1")  # stale / unknown marks are ignored
            carrier._echo("bogus")

            # marks that never come back are given up on after `mark_grace`
            await transport.write_audio(AudioFrame.silence(0.3, 8000))
            await transport.write_audio(AudioFrame.silence(0.3, 8000))
            await asyncio.sleep(0.45)
            assert transport.buffered_duration() == pytest.approx(0.3, abs=0.03)
            await asyncio.sleep(0.6)  # past van-4's due time + grace: wall clock only
            assert transport.buffered_duration() == 0.0

            # clear drops queued audio and pending marks
            await transport.write_audio(AudioFrame.silence(0.5, 8000))
            await transport.clear_audio()
            assert transport.buffered_duration() == 0.0
            await wait_for(lambda: carrier.of("clear"))
            carrier._echo("van-6")  # flushed back by the clear
            await asyncio.sleep(0.05)
            assert transport.buffered_duration() == 0.0

            # app messages have no channel on a phone call; raw provider messages do
            await transport.send_message({"type": "transcript"})
            await transport.send_raw({"event": "custom", "streamSid": STREAM_SID})
            await wait_for(lambda: carrier.of("custom"))
            assert not carrier.of("transcript")


async def test_vonage_pads_the_last_frame_and_bad_starts_are_rejected() -> None:
    transports: asyncio.Queue[TelephonyTransport] = asyncio.Queue()
    errors: list[Exception] = []

    async def handler(ws: Any) -> None:
        transport = VonageTransport(ws, start_timeout=0.5)
        try:
            await transport.start()
        except Exception as exc:
            errors.append(exc)
            return
        await transports.put(transport)
        await transport.wait_disconnected()
        await transport.aclose()

    from websockets.asyncio.server import serve

    async with serve(handler, "127.0.0.1", 0) as ws_server:
        url = f"ws://127.0.0.1:{ws_server.sockets[0].getsockname()[1]}/"
        async with Carrier("vonage", url) as carrier:
            await carrier.start()
            transport = await transports.get()
            await transport.write_audio(AudioFrame.silence(0.05, 16_000))  # 2.5 frames
            await wait_for(lambda: len(carrier.of("audio")) == 3)  # 640-byte frames checked
            assert 0 < transport.buffered_duration() <= 0.06
            await wait_for(lambda: carrier.of("notify"))

        # unsupported content type: the connection is closed
        async with Carrier("vonage", url) as bad:
            await bad.send({"event": "websocket:connected", "content-type": "audio/opus"})
            await bad.wait_closed()
        # no start message at all
        async with Carrier("vonage", url) as silent:
            await silent.wait_closed()
        await wait_for(lambda: len(errors) == 2)
        assert "unsupported" in str(errors[0]) and "no start message" in str(errors[1])


async def test_standalone_transport_serves_one_call() -> None:
    transport = create_transport({"type": "plivo", "host": "127.0.0.1", "port": 0})
    assert isinstance(transport, TelephonyTransport)
    await transport.listen()
    session = AgentSession(MockEngine(responses=["Standalone answer."]))
    run = asyncio.create_task(session.run(Agent("x", greeting="Hello there."), transport))
    try:
        async with Carrier("plivo", transport.url) as carrier:
            await carrier.start()
            await wait_for(lambda: len(carrier.audio()) > 0.5 * 8000 * 2)
            assert transport.call is not None
            assert transport.call.custom_parameters["userId"] == "12345"
        await asyncio.wait_for(run, 5)  # the call ended: the session ends
        assert session.closed
    finally:
        await cancel_and_wait(run)
        await transport.aclose()
