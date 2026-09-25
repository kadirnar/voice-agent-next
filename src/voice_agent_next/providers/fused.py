"""``fused``: an audio and a text end-of-turn detector combined into one.

``turn_detector: {provider: fused, audio: smart_turn, text: lm_turn}``. The class lives in
:mod:`voice_agent_next.turn` (the cascade runs its halves itself); this module registers
it. See :class:`~voice_agent_next.turn.FusedTurnDetector` and
``docs/providers/lm-turn.md``.
"""

from __future__ import annotations

from ..registry import register_provider
from ..turn import FusedTurnDetector

__all__ = ["FusedTurnDetector"]

register_provider(
    "turn",
    "fused",
    description="Audio + text end-of-turn detectors fused (e.g. Smart Turn + lm_turn)",
    default_model=None,
    env=(),
    local=True,
)(FusedTurnDetector)
