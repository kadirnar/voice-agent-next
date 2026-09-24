"""Short random identifiers for requests, items, responses and sessions."""

from __future__ import annotations

import uuid

__all__ = ["new_id"]


def new_id(prefix: str = "") -> str:
    """Return a short random id such as ``"resp_3f9a1c0b7d2e"``."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"
