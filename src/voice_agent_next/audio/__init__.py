"""Audio primitives and DSP helpers (formats, buffers, resampling, codecs, WAV I/O)."""

from __future__ import annotations

from .buffer import AudioBuffer, FrameChunker
from .codecs import (
    alaw_decode,
    alaw_encode,
    float32_to_pcm16,
    mulaw_decode,
    mulaw_encode,
    pcm16_to_float32,
)
from .frame import SAMPLE_WIDTH, AudioFormat, AudioFrame
from .pcm import PCM16Reassembler
from .processing import AudioProcessor, ProcessorChain
from .resample import Resampler, StreamResampler, resample
from .wav import WavWriter, read_wav, wav_bytes, write_wav

__all__ = [
    "SAMPLE_WIDTH",
    "AudioBuffer",
    "AudioFormat",
    "AudioFrame",
    "AudioProcessor",
    "FrameChunker",
    "PCM16Reassembler",
    "ProcessorChain",
    "Resampler",
    "StreamResampler",
    "WavWriter",
    "alaw_decode",
    "alaw_encode",
    "float32_to_pcm16",
    "mulaw_decode",
    "mulaw_encode",
    "pcm16_to_float32",
    "read_wav",
    "resample",
    "wav_bytes",
    "write_wav",
]
