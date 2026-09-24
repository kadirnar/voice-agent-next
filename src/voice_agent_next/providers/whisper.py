"""``whisper`` is an alias of the ``faster_whisper`` STT provider.

Specs resolve by module name, so ``create("stt", "whisper/small")`` imports this module,
which imports (and thereby registers) :mod:`voice_agent_next.providers.faster_whisper`
together with its ``whisper`` alias.
"""

from __future__ import annotations

from .faster_whisper import FasterWhisperSTT

__all__ = ["FasterWhisperSTT"]
