"""Library logger. Applications configure handlers; the library never does."""

from __future__ import annotations

import logging

__all__ = ["logger"]

logger = logging.getLogger("voice_agent_next")
logger.addHandler(logging.NullHandler())
