"""What :class:`~voice_agent_next.stt.StreamAdapter` runs for the Whisper providers
(``faster_whisper``, ``mlx_whisper``): VAD-cut utterances with interim transcripts and a
VAD-aware :class:`~voice_agent_next.stt_guard.HallucinationGuard`.

A provider plugs in through :class:`WhisperBackend`: it decodes a list of VAD frames on its
own worker (a thread pool for CTranslate2, the MLX thread for mlx-whisper), and says
whether a final decode can run next to an interim one (a second model replica).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import StreamResampler
from ..metrics import STTMetrics
from ..stt import STTCapabilities, STTEvent, STTEventType, STTStream, Transcript
from ..utils.aio import cancel_and_wait
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger

if TYPE_CHECKING:
    from ..stt import StreamAdapter

__all__ = ["WhisperAdapterStream", "WhisperBackend", "frames_to_samples", "whisper_language"]

SAMPLE_RATE = 16_000
"""Whisper's native input rate."""
PACING = 2.0
"""Queue at least this many times the last interim decode's duration between decodes."""
MIN_INTERIM_AUDIO = 0.3
"""Utterances shorter than this (seconds, VAD prefix included) get no interim decode."""


class WhisperBackend(Protocol):
    """The provider side of :class:`WhisperAdapterStream`."""

    model: str
    capabilities: STTCapabilities
    word_timestamps: bool
    final_from_interim: bool

    @property
    def provider(self) -> str: ...

    @property
    def resolved_interim_interval(self) -> float: ...

    @property
    def parallel_final(self) -> bool:
        """A final decode can run next to an interim decode in flight."""
        ...

    async def decode_frames(
        self,
        frames: Sequence[AudioFrame],
        language: str | None,
        *,
        interim: bool,
        vad_confidence: float | None,
    ) -> Transcript: ...

    def emit(self, event: str, *args: Any) -> None: ...


class WhisperAdapterStream(STTStream):
    """Like the adapter's own stream, the VAD cuts the input into utterances and each one
    gets one final transcript. In addition:

    * **interim transcripts** (``interim_results=True``): while the VAD reports speech, the
      utterance so far is re-decoded in the background whenever enough new speech has
      arrived: ``interim_interval``, or twice the duration of the last decode when that is
      longer (back-off on a slow device), so that the model is busy at most about half of
      the time. No decode starts while the latest VAD window is below the activation
      threshold (pauses, the trailing silence before END_OF_SPEECH): a decode that starts
      on the last voiced window has the VAD's ``min_silence_duration`` to finish before
      the final one is needed. A decode still running when the utterance ends is awaited
      and its result discarded (a model call cannot be interrupted), unless the backend
      can run the final next to it (``parallel_final``, e.g. a second model replica).
    * **final from interim** (``final_from_interim=True``): when the utterance ends and no
      voiced VAD window arrived after the latest interim decode (in flight or done) took
      its audio, that decode has heard all of the speech: its transcript becomes the final
      one and no second decode runs. Not with ``word_timestamps`` (interim decodes do not
      align words).
    * **VAD-aware hallucination guard**: the mean speech probability of the utterance's
      speech windows is passed to the guard.
    """

    def __init__(
        self, adapter: StreamAdapter, whisper: WhisperBackend, *, language: str | None
    ) -> None:
        self._adapter = adapter
        self._whisper = whisper
        self._segment_id = new_id("seg_")
        self._live: str | None = None
        """Segment whose interim results are still wanted (``None`` once it ends)."""
        self._probs: list[float] = []
        """VAD probabilities of the utterance's speech windows."""
        self._voiced = False
        """The latest VAD window is at or above the activation threshold."""
        self._since_step = 0.0
        """Seconds of input since the last interim decode started (or the utterance)."""
        self._step_time = 0.0
        """Duration of the last interim decode (seconds)."""
        self._step: asyncio.Task[None] | None = None
        self._heard_since_step = True
        """A voiced VAD window arrived after the latest interim decode of the utterance
        took its audio (or no interim decode has started in this utterance)."""
        self._last_interim: tuple[str, Transcript] | None = None
        """``(segment id, transcript)`` of the latest successful interim decode."""
        self._partial = ""
        self._detected: str | None = None
        """Language detected by an interim decode (reused by the next ones)."""
        self.interim_decodes = 0
        """Interim decodes run by this stream (diagnostics)."""
        self.final_waits: list[float] = []
        """Seconds each final transcript waited for an interim decode in flight."""
        self.finals_from_interim = 0
        """Final transcripts taken from an interim decode (diagnostics)."""
        super().__init__(adapter, language=language)

    # -------------------------------------------------------------- event loop
    async def _run(self) -> None:
        from ..vad import VADEventType

        vad = self._adapter.vad
        activation = vad.options.activation_threshold
        deactivation = vad.options.effective_deactivation
        vad_stream = vad.stream(emit_inference_events=True)
        try:
            async for item in self._input:
                if self.is_flush(item):
                    if vad_stream.speaking:
                        await self._finish(vad_stream.speech_frames())
                        vad_stream.reset()
                    continue
                assert isinstance(item, AudioFrame)
                for ev in vad_stream.push_audio(item):
                    if ev.type == VADEventType.START_OF_SPEECH:
                        self._live = self._segment_id
                        # the first decode comes after half an interval: the VAD has
                        # already held back min_speech_duration, and the prefix padding
                        # gives the decoder context
                        self._since_step = self._whisper.resolved_interim_interval / 2
                        self._emit(
                            STTEvent(STTEventType.START_OF_SPEECH, segment_id=self._segment_id)
                        )
                    elif ev.type == VADEventType.END_OF_SPEECH:
                        await self._finish(list(ev.frames))
                    elif ev.type == VADEventType.INFERENCE_DONE:
                        self._voiced = ev.speaking and ev.probability >= activation
                        if self._voiced:
                            self._heard_since_step = True
                        if ev.speaking and ev.probability >= deactivation:
                            self._probs.append(ev.probability)
                if vad_stream.speaking:
                    self._since_step += item.duration
                    self._maybe_step(vad_stream.speech_frames)
            if vad_stream.speaking:
                await self._finish(vad_stream.speech_frames())
        finally:
            await cancel_and_wait(self._step)
            vad_stream.close()

    @property
    def _threshold(self) -> float:
        """Seconds of new input before the next interim decode."""
        return max(self._whisper.resolved_interim_interval, PACING * self._step_time)

    @property
    def _vad_confidence(self) -> float | None:
        return sum(self._probs) / len(self._probs) if self._probs else None

    def _maybe_step(self, speech_frames: Callable[[], list[AudioFrame]]) -> None:
        if not self._whisper.capabilities.interim_results or not self._voiced:
            return
        if self._step is not None and not self._step.done():
            return
        if self._live is None or self._since_step < self._threshold:
            return
        frames = speech_frames()
        if sum(f.duration for f in frames) < MIN_INTERIM_AUDIO:
            return
        self._since_step = 0.0
        self._heard_since_step = False
        self._last_interim = None
        self._step = asyncio.create_task(
            self._interim(frames, self._live, self._vad_confidence),
            name=f"{self._whisper.provider}-interim",
        )

    async def _interim(
        self, frames: list[AudioFrame], segment_id: str, vad_confidence: float | None
    ) -> None:
        whisper = self._whisper
        language = whisper_language(self._language) or self._detected
        t0 = now()
        try:
            transcript = await whisper.decode_frames(
                frames, language, interim=True, vad_confidence=vad_confidence
            )
        except Exception as exc:  # interims are best effort; the final reports errors
            logger.warning("%s: interim decode failed: %s", whisper.provider, exc)
            return
        finally:
            self._step_time = now() - t0
            self.interim_decodes += 1
        self._last_interim = (segment_id, transcript)
        if segment_id != self._live:
            return  # the utterance ended meanwhile: its final transcript is coming
        self._detected = self._detected or transcript.language
        text = transcript.text
        if text and text != self._partial:
            self._partial = text
            self._emit(STTEvent(STTEventType.INTERIM_TRANSCRIPT, transcript, segment_id))

    async def _finish(self, frames: list[AudioFrame]) -> None:
        whisper = self._whisper
        segment_id = self._segment_id
        self._live = None  # a decode in flight is now stale (unless it becomes the final)
        reuse = (
            whisper.final_from_interim
            and not whisper.word_timestamps
            and self._step is not None
            and not self._heard_since_step
        )
        step = self._step
        t0 = now()
        if step is not None and not step.done() and (reuse or not whisper.parallel_final):
            # one model replica: the final would queue behind the decode anyway; or the
            # decode in flight has heard the whole utterance and becomes the final
            await asyncio.wait({step})
            self.final_waits.append(now() - t0)
        else:  # idle, or the final runs next to the interim decode
            self.final_waits.append(0.0)
        interim = self._last_interim
        vad_confidence = self._vad_confidence
        self._segment_id = new_id("seg_")
        self._probs = []
        self._voiced = False
        self._since_step = 0.0
        self._heard_since_step = True
        self._last_interim = None
        self._partial = ""
        self._detected = None
        if not frames:
            transcript = None
        elif reuse and interim is not None and interim[0] == segment_id:
            transcript = interim[1]
            self.finals_from_interim += 1
            self._report(frames, now() - t0, None)
        else:
            transcript = await self._final(frames, vad_confidence)
        if transcript is not None and transcript.text.strip():
            self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, segment_id))
        self._emit(STTEvent(STTEventType.END_OF_SPEECH, segment_id=segment_id))

    async def _final(self, frames: list[AudioFrame], vad_confidence: float | None) -> Transcript:
        """Like :meth:`STT.transcribe` (same metrics), with the VAD's confidence."""
        t0 = now()
        error: str | None = None
        try:
            return await self._whisper.decode_frames(
                frames,
                whisper_language(self._language),
                interim=False,
                vad_confidence=vad_confidence,
            )
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            self._report(frames, now() - t0, error)

    def _report(self, frames: Sequence[AudioFrame], duration: float, error: str | None) -> None:
        whisper = self._whisper
        whisper.emit(
            "metrics",
            STTMetrics(
                provider=whisper.provider,
                model=whisper.model,
                request_id=new_id("stt_"),
                audio_duration=sum(f.duration for f in frames),
                duration=duration,
                streamed=False,
                error=error,
            ),
        )


def frames_to_samples(frames: Sequence[AudioFrame]) -> npt.NDArray[np.float32]:
    """VAD frames (any rate) as 16 kHz mono float32 samples."""
    frame = AudioFrame.concat(list(frames))
    if frame.sample_rate != SAMPLE_RATE or frame.channels != 1:
        rs = StreamResampler(SAMPLE_RATE, 1)
        frame = AudioFrame.concat([rs.push(frame), rs.flush()])
    return frame.to_float32()


def whisper_language(language: str | None) -> str | None:
    """``"en-US"``/``"pt_BR"`` -> ``"en"``/``"pt"``; empty, ``"auto"``, ``"multi"`` -> detect."""
    if not language:
        return None
    code = language.strip().replace("_", "-").split("-")[0].lower()
    return None if code in ("", "auto", "multi") else code
