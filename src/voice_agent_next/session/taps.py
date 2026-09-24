"""Low-level observers of an :class:`~voice_agent_next.session.AgentSession`.

The public session events (``user_transcript``, ``metrics``...) describe the conversation;
a :class:`SessionTap` additionally sees the raw audio timeline and every engine event,
which recorders and tracers need. Taps are attached with ``AgentSession(record=...,
trace=...)``; a session without taps pays one empty-list check per hook.

Every hook runs synchronously inside the session's loops: keep it cheap and never block.
Exceptions are logged and swallowed by the session (a broken observer must not break the
call).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audio.frame import AudioFrame
    from ..events import EngineEvent
    from .session import AgentSession

__all__ = ["SessionTap"]


class SessionTap:
    """Base class of session observers; every hook is a no-op by default.

    All times are :func:`~voice_agent_next.utils.now` values.
    """

    def session_started(self, session: AgentSession, t: float) -> None:
        """The transport is open (formats are known); ``t`` is the start of the call."""

    def session_closing(self, session: AgentSession, reason: str, t: float) -> None:
        """The engine and transport are closed; finalize now (before ``wait_closed()``
        returns)."""

    def user_audio(self, frame: AudioFrame, t: float) -> None:
        """A user audio frame (transport input format, before audio processors) arrived at
        ``t``."""

    def agent_audio(self, frame: AudioFrame, start: float) -> None:
        """An agent audio frame (transport output format) was handed to the transport; it
        is scheduled to play from ``start``."""

    def playback_paused(self, t: float) -> None:
        """Playback (including audio already queued in the transport) stopped at ``t``."""

    def playback_shifted(self, paused_at: float, delta: float) -> None:
        """A pause that began at ``paused_at`` ended: audio scheduled after ``paused_at``
        plays ``delta`` seconds later."""

    def playback_resumed(self) -> None:
        """The playback clock runs again (after :meth:`playback_shifted`, or with no shift
        when paused audio is dropped)."""

    def playback_cleared(self, t: float) -> None:
        """Agent audio scheduled at or after ``t`` was dropped (barge-in)."""

    def engine_event(self, event: EngineEvent) -> None:
        """An engine event, before the session handles it."""
