from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from tests.conftest import dominant_frequency, tone
from voice_agent_next.audio import (
    AudioBuffer,
    AudioFormat,
    AudioFrame,
    FrameChunker,
    Resampler,
    StreamResampler,
    WavWriter,
    alaw_decode,
    alaw_encode,
    float32_to_pcm16,
    mulaw_decode,
    mulaw_encode,
    pcm16_to_float32,
    read_wav,
    resample,
    wav_bytes,
    write_wav,
)
from voice_agent_next.utils.deps import is_installed

# ------------------------------------------------------------------------ AudioFrame


def test_frame_basic_properties() -> None:
    f = AudioFrame(bytes(3200), 16_000)
    assert f.samples_per_channel == 1600
    assert f.duration == pytest.approx(0.1)
    assert f.duration_ms == pytest.approx(100)
    assert f.format == AudioFormat(16_000, 1)
    assert bool(f)
    assert not AudioFrame.empty(16_000)


def test_frame_rejects_misaligned_data() -> None:
    with pytest.raises(ValueError):
        AudioFrame(b"\x00\x00\x00", 16_000)
    with pytest.raises(ValueError):
        AudioFrame(bytes(6), 16_000, channels=2)  # 6 bytes is not a multiple of 4
    with pytest.raises(ValueError):
        AudioFrame(b"", 0)


def test_frame_numpy_roundtrip_mono_and_stereo() -> None:
    x = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    f = AudioFrame.from_numpy(x, 8000)
    assert np.array_equal(f.to_numpy(), x)
    stereo = np.stack([x, -x.astype(np.int32).clip(-32768, 32767).astype(np.int16)], axis=1)
    fs = AudioFrame.from_numpy(stereo, 8000)
    assert fs.channels == 2
    assert fs.to_numpy().shape == (5, 2)
    assert np.array_equal(fs.to_numpy(), stereo)


def test_frame_from_float_clips() -> None:
    f = AudioFrame.from_numpy(np.array([2.0, -2.0, 0.5], dtype=np.float32), 16_000)
    assert f.to_numpy().tolist() == [32767, -32768, 16384]


def test_frame_to_mono_and_channels() -> None:
    stereo = AudioFrame.from_numpy(np.array([[100, 300], [-100, -300]], dtype=np.int16), 16_000)
    mono = stereo.to_mono()
    assert mono.channels == 1
    assert mono.to_numpy().tolist() == [200, -200]
    back = mono.to_channels(2)
    assert back.to_numpy().tolist() == [[200, 200], [-200, -200]]


def test_frame_slice_concat_silence_base64() -> None:
    f = tone(440, 1.0, 16_000)
    a, b = f.slice(0, 0.25), f.slice(0.25)
    assert a.duration == pytest.approx(0.25)
    assert AudioFrame.concat([a, b]).data == f.data
    s = AudioFrame.silence(0.5, 24_000)
    assert s.duration == pytest.approx(0.5)
    assert s.rms() == 0.0
    assert s.dbfs() == float("-inf")
    assert AudioFrame.from_base64(f.to_base64(), 16_000).data == f.data
    with pytest.raises(ValueError):
        AudioFrame.concat([f, AudioFrame.silence(0.1, 8000)])


def test_frame_levels() -> None:
    f = tone(440, 1.0, 16_000, amplitude=0.5)
    assert f.rms() == pytest.approx(0.5 / np.sqrt(2), rel=0.01)
    assert f.dbfs() == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.1)


# ------------------------------------------------------------------ buffer / chunker


def test_audio_buffer_bounded_keeps_latest() -> None:
    buf = AudioBuffer(16_000, max_duration=0.5)
    for i in range(10):
        buf.append(AudioFrame.from_numpy(np.full(1600, i, dtype=np.int16), 16_000))
    assert buf.duration == pytest.approx(0.5)
    samples = buf.to_frame().to_numpy()
    assert samples[0] == 5 and samples[-1] == 9
    buf.keep_last(0.1)
    assert buf.duration == pytest.approx(0.1)
    assert buf.pop_all().duration == pytest.approx(0.1)
    assert not buf


def test_audio_buffer_rejects_other_format() -> None:
    buf = AudioBuffer(16_000)
    with pytest.raises(ValueError):
        buf.append(AudioFrame.silence(0.1, 8000))


def test_frame_chunker_exact_sizes_and_timestamps() -> None:
    chunker = FrameChunker(16_000, samples_per_frame=512)
    src = AudioFrame(bytes(2 * 1000), 16_000, timestamp=10.0)
    out = chunker.push(src)
    assert [f.samples_per_channel for f in out] == [512]
    assert out[0].timestamp == 10.0
    out2 = chunker.push(AudioFrame(bytes(2 * 100), 16_000, timestamp=10.0625))
    assert [f.samples_per_channel for f in out2] == [512]
    assert out2[0].timestamp == pytest.approx(10.0 + 512 / 16_000)
    rest = chunker.flush(pad=True)
    assert len(rest) == 1 and rest[0].samples_per_channel == 512
    assert chunker.flush() == []


def test_frame_chunker_by_duration() -> None:
    chunker = FrameChunker(8000, frame_duration=0.02)
    assert chunker.samples_per_frame == 160
    with pytest.raises(ValueError):
        FrameChunker(8000)


# ---------------------------------------------------------------------------- codecs

_ALL_PCM = np.arange(-32768, 32768, dtype="<i2").tobytes()


def test_mulaw_bit_exact_with_audioop_reference() -> None:
    # sha256 of audioop.lin2ulaw over the full int16 range (CPython 3.12)
    digest = hashlib.sha256(mulaw_encode(_ALL_PCM)).hexdigest()
    assert digest == "81d633c9e6972a18c74a58720b96cb8ca0bdd096d4060b646dd708c3b846019a"


def test_alaw_bit_exact_with_audioop_reference() -> None:
    digest = hashlib.sha256(alaw_encode(_ALL_PCM)).hexdigest()
    assert digest == "38488f6fd710f4686360edc4d38639f96c491595ef93f8eb8d62d5e07ca6ce7b"


@pytest.mark.parametrize(
    ("value", "ulaw", "alaw"),
    [(0, 0xFF, 0xD5), (-1, 0x7E, 0x55), (1000, 0xCE, 0xFA), (-1000, 0x4E, 0x7A),
     (32767, 0x80, 0xAA), (-32768, 0x00, 0x2A)],
)  # fmt: skip
def test_g711_known_values(value: int, ulaw: int, alaw: int) -> None:
    pcm = np.array([value], dtype="<i2").tobytes()
    assert mulaw_encode(pcm)[0] == ulaw
    assert alaw_encode(pcm)[0] == alaw


@pytest.mark.parametrize(("enc", "dec"), [(mulaw_encode, mulaw_decode), (alaw_encode, alaw_decode)])
def test_g711_roundtrip_error_is_small(enc, dec) -> None:  # type: ignore[no-untyped-def]
    x = tone(300, 0.2, 8000, amplitude=0.8)
    y = AudioFrame(dec(enc(x.data)), 8000)
    err = np.abs(x.to_numpy().astype(np.int32) - y.to_numpy().astype(np.int32))
    # G.711 has ~3% relative quantization error on loud signals
    assert err.max() < 1100
    assert len(enc(x.data)) == x.samples_per_channel


def test_float_pcm_conversion() -> None:
    x = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
    back = pcm16_to_float32(float32_to_pcm16(x))
    assert np.allclose(back, x, atol=1e-4)


# ------------------------------------------------------------------------ resampling

BACKENDS = ["numpy"] + (["soxr"] if is_installed("soxr") else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("src", "dst"),
    [(48_000, 16_000), (16_000, 24_000), (8000, 16_000), (44_100, 16_000), (24_000, 8000)],
)
def test_resample_preserves_frequency_and_length(backend: str, src: int, dst: int) -> None:
    x = tone(440, 1.0, src)
    y = resample(x, dst, backend=backend)  # type: ignore[arg-type]
    assert y.sample_rate == dst
    assert abs(y.samples_per_channel - dst) <= 2
    assert dominant_frequency(y) == pytest.approx(440, abs=3)
    assert y.rms() == pytest.approx(x.rms(), rel=0.05)


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_resampler_matches_duration(backend: str) -> None:
    x = tone(1000, 1.0, 48_000)
    rs = Resampler(48_000, 16_000, backend=backend)  # type: ignore[arg-type]
    chunks = [rs.push(x.slice(t, t + 0.02)) for t in np.arange(0, 1.0, 0.02)]
    chunks.append(rs.flush())
    y = AudioFrame.concat([c for c in chunks if c] or [AudioFrame.empty(16_000)])
    assert abs(y.samples_per_channel - 16_000) <= 64
    assert dominant_frequency(y) == pytest.approx(1000, abs=3)


def test_numpy_resampler_attenuates_aliasing() -> None:
    # 7 kHz at 48 kHz must be filtered out when converting to 8 kHz (Nyquist 4 kHz)
    x = tone(7000, 0.5, 48_000)
    y = resample(x, 8000, backend="numpy")
    assert y.rms() < 0.01 * x.rms() * 10  # at least ~20 dB attenuation


def test_numpy_streaming_equals_one_shot_after_delay() -> None:
    x = tone(500, 0.5, 16_000)
    rs = Resampler(16_000, 24_000, backend="numpy")
    streamed = AudioFrame.concat([rs.push(x.slice(t, t + 0.01)) for t in np.arange(0, 0.5, 0.01)])
    one_shot = resample(x, 24_000, backend="numpy")
    # both contain the same signal (streaming output is delayed by the filter group delay)
    assert dominant_frequency(streamed) == pytest.approx(dominant_frequency(one_shot), abs=5)


def test_resampler_passthrough_and_validation() -> None:
    x = tone(440, 0.1, 16_000)
    rs = Resampler(16_000, 16_000)
    assert rs.push(x) is x
    with pytest.raises(ValueError):
        rs.push(tone(440, 0.1, 8000))


def test_stream_resampler_handles_rate_and_channel_changes() -> None:
    srs = StreamResampler(16_000, 1)
    stereo = AudioFrame.from_numpy(np.zeros((480, 2), dtype=np.int16), 48_000)
    out = srs.push(stereo)
    assert out.channels == 1 and out.sample_rate == 16_000
    same = tone(440, 0.1, 16_000)
    # the 48 kHz filter tail is emitted first (it used to be left in the stale
    # resampler and come out *after* newer audio on the next flush)
    switched = srs.push(same)
    assert switched.data.endswith(same.data)
    assert srs.push(same) is same
    assert not srs.flush()


# --------------------------------------------------------------------------------- WAV


def test_wav_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    x = tone(440, 0.3, 22_050)
    path = tmp_path / "a.wav"
    write_wav(path, x)
    y = read_wav(path)
    assert y.sample_rate == 22_050 and y.data == x.data
    assert read_wav(wav_bytes([x, x])).duration == pytest.approx(0.6)


def test_wav_writer_incremental(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "b.wav"
    with WavWriter(path, 16_000) as w:
        w.write(tone(440, 0.1, 16_000))
        w.write(tone(440, 0.1, 16_000))
        with pytest.raises(ValueError):
            w.write(tone(440, 0.1, 8000))
    assert read_wav(path).duration == pytest.approx(0.2)


@pytest.mark.parametrize("width", [1, 3, 4])
def test_read_wav_other_sample_widths(width: int) -> None:
    import wave

    buf = io.BytesIO()
    x = np.array([0, 16384, -16384], dtype=np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(8000)
        if width == 1:
            w.writeframes(((x >> 8) + 128).astype(np.uint8).tobytes())
        elif width == 3:
            i32 = x.astype(np.int32) << 8
            w.writeframes(b"".join(int(v).to_bytes(3, "little", signed=True) for v in i32))
        else:
            w.writeframes((x.astype(np.int32) << 16).astype("<i4").tobytes())
    f = read_wav(buf.getvalue())
    assert f.to_numpy().tolist() == [0, 16384, -16384]
