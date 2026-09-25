"""``openai-live`` is GPT-Live, implemented in :mod:`voice_agent_next.providers.openai.live`.

Specs such as ``"openai-live/gpt-live-1"`` import this module first (module name =
provider name), which loads the OpenAI provider package.
"""

from __future__ import annotations

from .openai import live  # noqa: F401  (import registers the components)
