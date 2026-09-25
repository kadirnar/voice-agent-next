"""Re-framing of PCM16 byte streams that arrive in arbitrarily split chunks."""

from __future__ import annotations

from .frame import SAMPLE_WIDTH

__all__ = ["PCM16Reassembler"]


class PCM16Reassembler:
    """Turns a stream of byte chunks into whole PCM16 sample frames.

    Network APIs (base64 audio deltas over WebSocket/HTTP/2) may split audio at any
    byte. Dropping the odd byte of a chunk shifts every later sample by one byte and
    turns the rest of the stream into noise, so the incomplete trailing bytes are
    carried over and prepended to the next chunk instead.

    Args:
        channels: interleaved channel count; output is aligned to whole frames
            (``2 * channels`` bytes).
    """

    __slots__ = ("_carry", "_frame_bytes")

    def __init__(self, channels: int = 1) -> None:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self._frame_bytes = SAMPLE_WIDTH * channels
        self._carry = b""

    @property
    def pending(self) -> int:
        """Bytes held back until the rest of their sample frame arrives."""
        return len(self._carry)

    def push(self, data: bytes) -> bytes:
        """Add a chunk; return the longest whole-frame prefix of the stream so far."""
        if self._carry:
            data = self._carry + data
        cut = len(data) - len(data) % self._frame_bytes
        self._carry = data[cut:]
        return data[:cut] if cut < len(data) else data

    def reset(self) -> bytes:
        """Start a new stream (e.g. a new response); return the discarded partial frame."""
        carry, self._carry = self._carry, b""
        return carry
