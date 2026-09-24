"""Transport interface: moves audio between the user and the agent session.

Implementations: local microphone/speaker, WebSocket, WebRTC, telephony (Twilio,
Telnyx, ...), files, and the in-memory :class:`LoopbackTransport` used by tests and
the benchmark's simulated users.

Events emitted (``transport.on(...)``): ``"connected"``, ``"disconnected"``,
``"message"`` (a ``dict`` from the client's data channel, if any).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..audio.frame import AudioFormat, AudioFrame
from ..utils.emitter import EventEmitter

__all__ = ["Transport", "TransportCapabilities"]


@dataclass(frozen=True, slots=True)
class TransportCapabilities:
    pause: bool = False
    """Playback can be paused/resumed (false-interruption recovery)."""
    playback_position: bool = False
    """``buffered_duration()`` reflects what the listener actually heard (device or marks)."""
    dtmf: bool = False
    """Emits ``"dtmf"`` events with the pressed key."""
    messages: bool = False
    """Has a data channel for JSON messages (``send_message`` / ``"message"`` events)."""


class Transport(ABC, EventEmitter):
    """Bidirectional audio transport.

    Args:
        input_format: format of the user audio this transport produces.
        output_format: format :meth:`write_audio` expects (the session resamples).
    """

    capabilities: TransportCapabilities = TransportCapabilities()

    def __init__(self, *, input_format: AudioFormat, output_format: AudioFormat) -> None:
        EventEmitter.__init__(self)
        self.input_format = input_format
        self.output_format = output_format

    async def start(self) -> None:
        """Open devices / accept the connection. Called by the session."""

    async def aclose(self) -> None:
        """Release devices / close the connection."""

    @abstractmethod
    def audio_input(self) -> AsyncIterator[AudioFrame]:
        """User audio frames (``input_format``). The iterator ends when the user leaves."""

    @abstractmethod
    async def write_audio(self, frame: AudioFrame) -> None:
        """Queue agent audio (``output_format``) for playback."""

    @abstractmethod
    async def clear_audio(self) -> None:
        """Drop all queued agent audio immediately (barge-in)."""

    async def pause_audio(self) -> None:
        """Pause playback, keeping queued audio (requires ``capabilities.pause``)."""
        raise NotImplementedError(f"{type(self).__name__} cannot pause playback")

    async def resume_audio(self) -> None:
        """Resume paused playback (requires ``capabilities.pause``)."""
        raise NotImplementedError(f"{type(self).__name__} cannot pause playback")

    def buffered_duration(self) -> float:
        """Seconds of agent audio accepted by :meth:`write_audio` but not yet played."""
        return 0.0

    async def wait_for_playout(self) -> None:
        """Wait until all queued agent audio has been played."""
        while (remaining := self.buffered_duration()) > 0:  # noqa: ASYNC110 - polling a clock
            await asyncio.sleep(min(remaining, 0.05))

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a JSON-able message to the client (transcripts, state...). Default: no-op."""

    async def __aenter__(self) -> Transport:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
