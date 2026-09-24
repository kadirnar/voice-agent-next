"""Sample-format conversions and G.711 (μ-law / A-law) codecs.

``audioop`` was removed from the standard library in Python 3.13, so the telephony
codecs are implemented here with vectorized numpy (bit-exact with ITU-T G.711 /
the classic Sun reference implementation).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "alaw_decode",
    "alaw_encode",
    "float32_to_pcm16",
    "mulaw_decode",
    "mulaw_encode",
    "pcm16_to_float32",
]

_I16 = np.dtype("<i2")


def pcm16_to_float32(data: bytes) -> npt.NDArray[np.float32]:
    """s16le bytes -> float32 array in ``[-1, 1)``."""
    return np.frombuffer(data, dtype=_I16).astype(np.float32) / 32768.0


def float32_to_pcm16(samples: npt.ArrayLike) -> bytes:
    """float array in ``[-1, 1]`` -> s16le bytes (clipped)."""
    arr = np.asarray(samples, dtype=np.float32)
    return np.clip(np.round(arr * 32767.0), -32768, 32767).astype(_I16).tobytes()


# --------------------------------------------------------------------------- μ-law
_MULAW_BIAS = 0x84
_MULAW_CLIP_14 = 8159
# segment end points in the 14-bit magnitude domain (CCITT/Sun reference, as in audioop)
_MULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)


def mulaw_encode(data: bytes) -> bytes:
    """Encode s16le PCM to 8-bit G.711 μ-law (bit-exact with ``audioop.lin2ulaw``)."""
    x = np.frombuffer(data, dtype=_I16).astype(np.int32) >> 2  # 14-bit
    mask = np.where(x < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(x), _MULAW_CLIP_14) + (_MULAW_BIAS >> 2)
    seg = np.searchsorted(_MULAW_SEG_END, mag, side="left")  # first seg_end >= mag
    uval = (np.minimum(seg, 7) << 4) | ((mag >> (seg + 1)) & 0x0F)
    uval = np.where(seg >= 8, 0x7F, uval)
    return ((uval ^ mask) & 0xFF).astype(np.uint8).tobytes()


def _build_mulaw_decode_table() -> npt.NDArray[np.int16]:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = (((mantissa << 3) + _MULAW_BIAS) << exponent) - _MULAW_BIAS
    return np.where(sign != 0, -sample, sample).astype(np.int16)


_MULAW_DECODE = _build_mulaw_decode_table()


def mulaw_decode(data: bytes) -> bytes:
    """Decode 8-bit G.711 μ-law to s16le PCM."""
    idx = np.frombuffer(data, dtype=np.uint8)
    return _MULAW_DECODE[idx].astype(_I16).tobytes()


# --------------------------------------------------------------------------- A-law
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def alaw_encode(data: bytes) -> bytes:
    """Encode s16le PCM to 8-bit G.711 A-law (bit-exact with ``audioop.lin2alaw``)."""
    pcm = np.frombuffer(data, dtype=_I16).astype(np.int32) >> 3  # 13-bit magnitude domain
    mask = np.where(pcm >= 0, 0xD5, 0x55)
    pcm = np.where(pcm >= 0, pcm, -pcm - 1)
    seg = np.searchsorted(_ALAW_SEG_END, pcm, side="left")  # first seg_end >= pcm
    shift = np.where(seg < 2, 1, seg)
    aval = (np.minimum(seg, 7) << 4) | ((pcm >> shift) & 0x0F)
    aval = np.where(seg >= 8, 0x7F, aval)
    return ((aval ^ mask) & 0xFF).astype(np.uint8).tobytes()


def _build_alaw_decode_table() -> npt.NDArray[np.int16]:
    a = np.arange(256, dtype=np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, t + 0x108)
    t = np.where(seg > 1, t << np.maximum(seg - 1, 0), t)
    return np.where((a & 0x80) != 0, t, -t).astype(np.int16)


_ALAW_DECODE = _build_alaw_decode_table()


def alaw_decode(data: bytes) -> bytes:
    """Decode 8-bit G.711 A-law to s16le PCM."""
    idx = np.frombuffer(data, dtype=np.uint8)
    return _ALAW_DECODE[idx].astype(_I16).tobytes()
