"""``claude`` alias for the Anthropic provider.

The registry resolves ``"claude/claude-haiku-4-5"`` by importing
``voice_agent_next.providers.claude``; importing :mod:`.anthropic` registers the
Anthropic LLM together with its ``claude`` alias, so the alias works in a fresh process.
"""

from __future__ import annotations

from . import anthropic  # noqa: F401  (import registers the provider and its alias)
