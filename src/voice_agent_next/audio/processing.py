"""Audio processing hooks applied to microphone audio before it reaches the engine.

Typical processors: acoustic echo cancellation (AEC), noise suppression (NS),
automatic gain control (AGC). AEC needs the *far-end* (agent playback) signal as a
reference, which the session feeds through :meth:`AudioProcessor.process_render`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .frame import AudioFrame

__all__ = ["AudioProcessor", "ProcessorChain"]


class AudioProcessor(ABC):
    """Synchronous, frame-by-frame audio processor. Must be fast (runs on the audio path)."""

    @abstractmethod
    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        """Process near-end (microphone) audio and return the cleaned frame."""

    def process_render(self, frame: AudioFrame) -> None:
        """Receive far-end (speaker) audio as echo-cancellation reference. Optional."""

    def reset(self) -> None:
        """Reset internal state (e.g. between sessions)."""

    def close(self) -> None:
        """Release native resources."""


class ProcessorChain(AudioProcessor):
    """Applies processors in order."""

    def __init__(self, processors: Sequence[AudioProcessor]) -> None:
        self.processors = list(processors)

    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        for p in self.processors:
            frame = p.process_capture(frame)
        return frame

    def process_render(self, frame: AudioFrame) -> None:
        for p in self.processors:
            p.process_render(frame)

    def reset(self) -> None:
        for p in self.processors:
            p.reset()

    def close(self) -> None:
        for p in self.processors:
            p.close()
