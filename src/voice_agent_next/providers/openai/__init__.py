"""OpenAI provider package.

Submodules register their components on import:

* ``llm``      — Chat Completions LLM (also the base for every OpenAI-compatible server)
* ``stt``      — realtime transcription / batch transcription
* ``tts``      — speech synthesis
* ``realtime`` — the Realtime speech-to-speech engine (+ compatible backends)
"""

from __future__ import annotations

from . import llm, realtime, stt, tts  # noqa: F401  (import registers the components)
