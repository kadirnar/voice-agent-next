"""Google provider package (Gemini API via ``google-genai``).

Submodules register their components on import:

* ``llm``  — Gemini chat LLM
* ``live`` — Gemini Live speech-to-speech engine
* ``tts``  — Gemini TTS
"""

from __future__ import annotations

from . import live, llm, tts  # noqa: F401  (import registers the components)
