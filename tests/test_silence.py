from __future__ import annotations

import numpy as np
import pytest

from voice_agent_next.audio import AudioFrame
from voice_agent_next.audio.silence import SilenceTrimmer
from voice_agent_next.providers.mock import MockTTS, synth_speech
from voice_agent_next.stt import WordTiming
from voice_agent_next.tts import ChunkedStream

SR = 24_000


def padded(lead: float, speech: float, pause: float, speech2: float, tail: float) -> AudioFrame:
    silence = lambda d: AudioFrame.silence(d, SR)  # noqa: E731
    parts = [
        silence(lead),
        synth_speech(speech, SR),
        silence(pause),
        synth_speech(speech2, SR),
        silence(tail),
    ]
    return AudioFrame.concat([p for p in parts if p])


def run(trimmer: SilenceTrimmer, audio: AudioFrame, chunk: float) -> AudioFrame:
    out = [trimmer.push(audio.slice(t, t + chunk)) for t in np.arange(0, audio.duration, chunk)]
    out.append(trimmer.flush())
    return AudioFrame.concat([f for f in out if f] or [AudioFrame.empty(SR)])


@pytest.mark.parametrize("chunk", [0.005, 0.02, 0.137, 1.0])
def test_trims_padding_but_keeps_inner_pauses(chunk: float) -> None:
    audio = padded(lead=0.12, speech=0.5, pause=0.3, speech2=0.4, tail=0.55)
    trimmer = SilenceTrimmer(SR)
    out = run(trimmer, audio, chunk)
    # 0.5 + 0.3 (kept inner pause) + 0.4 of content, plus <= 20 ms lead and <= 100 ms tail
    assert out.duration == pytest.approx(1.2 + 0.02 + 0.1, abs=0.02)
    assert trimmer.dropped_leading == pytest.approx(0.10, abs=0.011)
    assert trimmer.dropped_trailing == pytest.approx(0.45, abs=0.011)


def test_all_silence_emits_nothing() -> None:
    trimmer = SilenceTrimmer(SR)
    out = run(trimmer, AudioFrame.silence(0.5, SR), 0.02)
    assert out.duration == 0.0
    assert not trimmer.started


class PaddedTTS(MockTTS):
    """Pads every sentence like Kokoro does, and reports word timings per sentence."""

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _PaddedStream(self, text, voice=voice)


class _PaddedStream(ChunkedStream):
    async def _run(self) -> None:
        tts: PaddedTTS = self._tts  # type: ignore[assignment]
        speech = tts.audio_duration_for(self.text)
        audio = AudioFrame.concat(
            [AudioFrame.silence(0.1, SR), synth_speech(speech, SR), AudioFrame.silence(0.5, SR)]
        )
        words = self.text.split()
        per = speech / len(words)
        timings = [WordTiming(w, 0.1 + i * per, 0.1 + (i + 1) * per) for i, w in enumerate(words)]
        first = True
        for t in np.arange(0, audio.duration, 0.05):
            self._push_audio(audio.slice(t, t + 0.05), words=timings if first else None)
            first = False


async def collect(tts: MockTTS, text: str) -> tuple[AudioFrame, list[WordTiming]]:
    stream = tts.stream()
    stream.push_text(text)
    stream.end_input()
    frames, words = [], []
    async for chunk in stream:
        if chunk.frame:
            frames.append(chunk.frame)
        words.extend(chunk.words or [])
    await stream.aclose()
    return AudioFrame.concat(frames), words


async def test_sentence_adapter_trims_padding_and_shifts_word_timings() -> None:
    text = "First sentence here. Second sentence follows now."
    tts = PaddedTTS(chars_per_second=20)
    audio, words = await collect(tts, text)
    s1, s2 = (
        tts.audio_duration_for("First sentence here."),
        tts.audio_duration_for("Second sentence follows now."),
    )
    # untrimmed would be 2 x (0.1 + speech + 0.5); trimmed keeps <= 20 ms lead + 100 ms tail each
    assert audio.duration == pytest.approx(s1 + s2 + 2 * 0.12, abs=0.03)
    # the second sentence's first word starts right after sentence 1 (+ its kept tail/lead)
    second = next(w for w in words if w.word == "Second")
    assert second.start == pytest.approx(0.02 + s1 + 0.1 + 0.02, abs=0.03)
    assert words[0].start == pytest.approx(0.02, abs=0.011)


async def test_trimming_can_be_disabled() -> None:
    tts = PaddedTTS(chars_per_second=20)
    tts.trim_silence = False
    audio, _ = await collect(tts, "Only one sentence here.")
    assert audio.duration == pytest.approx(
        0.1 + tts.audio_duration_for("Only one sentence here.") + 0.5, abs=0.03
    )
