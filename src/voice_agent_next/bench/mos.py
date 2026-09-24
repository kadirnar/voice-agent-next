"""Reference-free MOS predictors for the TTS track (research note 06, §5 and §8.3).

MOS predictors estimate a listening-test score without a reference. They are
*regression signals*, not rankings: TTSDS2 found they correlate inconsistently with
human ratings across domains, so compare a system with itself over time.

* :class:`DNSMOS` (``dnsmos``) — Microsoft's DNSMOS P.835 (``sig_bak_ovr.onnx``, 1.2 MB,
  CC BY 4.0, from ``microsoft/DNS-Challenge``). It predicts ``SIG`` (speech quality),
  ``BAK`` (background noise) and ``OVRL`` (overall) on a 1-5 scale from 16 kHz audio,
  exactly like the reference ``dnsmos_local.py``: clips are tiled to at least 9.01 s,
  scored on 9.01 s windows with a 1 s hop, the raw outputs mapped through the published
  calibration polynomials and averaged. It needs ``onnxruntime`` (``onnx`` extra); the
  model is downloaded once into the model cache (``VAN_OFFLINE=1`` forbids that).

Other predictors (UTMOSv2, NISQA) plug in through :class:`MOSPredictor`; UTMOSv2 needs
PyTorch and a large checkpoint, so it is not built in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..utils.deps import require
from ..utils.download import download

__all__ = [
    "DNSMOS",
    "DNSMOS_SHA256",
    "DNSMOS_URL",
    "MOS_PREDICTORS",
    "MOSPredictor",
    "make_mos_predictor",
]

MOS_PREDICTORS: tuple[str, ...] = ("none", "dnsmos")

_DNSMOS_COMMIT = "82f1b17e7776a43eee395d0f45bae8abb700ad00"
DNSMOS_URL = (
    f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{_DNSMOS_COMMIT}"
    "/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)
DNSMOS_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"

_RATE = 16_000
_WINDOW = 9.01  # seconds scored per inference (the model's fixed input length)
# calibration polynomials of dnsmos_local.py (non-personalized), highest power first
_P_SIG = (-0.08397278, 1.22083953, 0.0052439)
_P_BAK = (-0.13166888, 1.60915514, -0.39604546)
_P_OVR = (-0.06766283, 1.11546468, 0.04602535)


@runtime_checkable
class MOSPredictor(Protocol):
    """Predicts quality scores of one clip (keys become metric names, e.g. ``dnsmos_ovrl``)."""

    name: str

    async def load(self) -> None:
        """Download / load the model (raises when unavailable)."""
        ...

    async def score(self, audio: AudioFrame) -> dict[str, float]: ...

    def describe(self) -> dict[str, Any]: ...


def dnsmos_windows(x: npt.NDArray[np.float32]) -> list[npt.NDArray[np.float32]]:
    """The 9.01 s windows ``dnsmos_local.py`` scores: the clip is tiled until it is at
    least 9.01 s long, then cut with a 1 s hop."""
    need = int(_WINDOW * _RATE)
    if x.size == 0:
        return []
    while x.size < need:
        x = np.concatenate([x, x])
    hops = int(np.floor(x.size / _RATE) - _WINDOW) + 1
    out = []
    for i in range(hops):
        seg = x[i * _RATE : int((i + _WINDOW) * _RATE)]
        if seg.size >= need:
            out.append(seg)
    return out


def dnsmos_calibrate(sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
    """Raw model outputs -> P.835 MOS (the published polynomial fits)."""
    return (
        float(np.polyval(_P_SIG, sig)),
        float(np.polyval(_P_BAK, bak)),
        float(np.polyval(_P_OVR, ovr)),
    )


class DNSMOS:
    """DNSMOS P.835 (SIG/BAK/OVRL) with onnxruntime on the CPU."""

    name = "dnsmos"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self._session: Any = None

    def _load(self) -> None:
        ort = require("onnxruntime", extra="onnx")
        path = self.model_path or download(
            DNSMOS_URL, filename="sig_bak_ovr.onnx", subdir="dnsmos", sha256=DNSMOS_SHA256
        )
        self.model_path = Path(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    async def load(self) -> None:
        if self._session is None:
            await asyncio.to_thread(self._load)

    def _score(self, audio: AudioFrame) -> dict[str, float]:
        if self._session is None:
            self._load()
        mono = resample(audio.to_mono(), _RATE)
        x = mono.to_float32()
        windows = dnsmos_windows(x)
        if not windows:
            return {}
        input_name = self._session.get_inputs()[0].name
        scores = []
        for seg in windows:
            raw = self._session.run(None, {input_name: seg[np.newaxis, :].astype(np.float32)})
            sig, bak, ovr = (float(v) for v in np.asarray(raw[0]).reshape(-1)[:3])
            scores.append(dnsmos_calibrate(sig, bak, ovr))
        mean = np.mean(np.asarray(scores, dtype=np.float64), axis=0)
        return {
            "dnsmos_sig": float(mean[0]),
            "dnsmos_bak": float(mean[1]),
            "dnsmos_ovrl": float(mean[2]),
        }

    async def score(self, audio: AudioFrame) -> dict[str, float]:
        return await asyncio.to_thread(self._score, audio)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": "DNSMOS P.835 sig_bak_ovr.onnx (non-personalized)",
            "url": DNSMOS_URL,
            "sha256": DNSMOS_SHA256,
            "license": "CC-BY-4.0",
            "sample_rate": _RATE,
            "window_s": _WINDOW,
        }


def make_mos_predictor(spec: str | MOSPredictor | None) -> MOSPredictor | None:
    """``None``/``"none"`` -> no predictor; ``"dnsmos"`` or ``"dnsmos:<model.onnx>"``."""
    if spec is None or not isinstance(spec, str):
        return spec
    name, _, arg = spec.partition(":")
    name = name.strip().lower()
    if name in ("", "none", "off"):
        return None
    if name == "dnsmos":
        return DNSMOS(arg or None)
    raise ValueError(f"unknown MOS predictor {spec!r}; use one of {', '.join(MOS_PREDICTORS)}")
