"""``gemini`` is an alias of the ``google`` provider package.

Specs such as ``"gemini/gemini-3.8-live"`` import this module first (module name =
provider name), which loads :mod:`voice_agent_next.providers.google` and maps the alias.
"""

from __future__ import annotations

from ..registry import register_alias
from . import google  # noqa: F401  (import registers the components)

register_alias("gemini", "google")
