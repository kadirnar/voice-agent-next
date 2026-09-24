"""sherpa-onnx (k2-fsa): streaming speech recognition, synthesis and VAD on every OS.

``create("stt", "sherpa-onnx/nemo-fastconformer-en-80ms")``, ``create("tts", "sherpa-onnx")``,
``create("vad", "sherpa-onnx/silero")`` (alias ``sherpa``). Requires the ``sherpa-onnx`` extra:
CPU wheels for Linux, macOS and Windows (x64 and arm64) with ONNX Runtime bundled.

Speech recognition (:class:`SherpaOnnxSTT`)
    *Streaming* models (``OnlineRecognizer``: Zipformer / NeMo FastConformer / Nemotron
    transducers, Paraformer, CTC) produce interim transcripts while the user speaks.
    :meth:`~voice_agent_next.stt.STTStream.flush` finalizes in one decoding step or two:
    the unprocessed tail is padded with just enough silence for the model's last chunk,
    decoded, and the utterance is emitted as ``FINAL_TRANSCRIPT`` — typically 20-60 ms on a
    desktop CPU instead of re-transcribing the whole utterance. The next audio goes to a
    fresh recognizer stream. sherpa's own endpoint rules are off by default (the cascade's
    VAD and turn detector decide); with ``endpoint_detection=True`` an endpoint emits
    ``FINAL_TRANSCRIPT`` + ``END_OF_SPEECH``.

    *Offline* models (``OfflineRecognizer``: Moonshine, Parakeet TDT, SenseVoice, Whisper)
    are batch recognizers (``capabilities.streaming=False``); the cascade streams them with
    :class:`~voice_agent_next.stt.StreamAdapter` and its VAD, like faster-whisper.

Speech synthesis (:class:`SherpaOnnxTTS`)
    ``OfflineTts`` with Piper/VITS, Kokoro and Matcha models: speaker ids (or Kokoro voice
    names), speed. Audio is streamed out sentence by sentence as sherpa generates it;
    :meth:`TTS.stream` uses the base :class:`~voice_agent_next.tts.SentenceStreamAdapter`.

Voice activity detection (:class:`SherpaOnnxVAD`)
    Silero VAD or TEN VAD through sherpa's own ONNX Runtime: no ``onnxruntime`` package
    needed. sherpa reports speech/non-speech per window (probability 1.0 or 0.0).

Models come from a pinned catalog (:data:`SHERPA_MODELS`: release archives of
k2-fsa/sherpa-onnx with SHA-256 digests), downloaded once into the shared model cache and
extracted safely; a local model directory, an archive URL (``sha256=``) or explicit
``files=`` work too. Every sherpa-onnx call runs on one worker thread per component, never
on the event loop. See ``docs/providers/sherpa-onnx.md`` for the model table and options.
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

from ..audio.frame import SAMPLE_WIDTH, AudioFrame
from ..audio.resample import StreamResampler
from ..errors import ConfigurationError, MissingDependencyError, ProviderError, VoiceAgentError
from ..registry import register_provider
from ..stt import STT, STTCapabilities, STTEvent, STTEventType, STTStream, Transcript, WordTiming
from ..tts import TTS, ChunkedStream
from ..utils.aio import Chan, ChanClosed
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.download import download, download_archive
from ..utils.ids import new_id
from ..utils.log import logger
from ..vad import VAD, VADOptions

__all__ = [
    "DEFAULT_STT_MODEL",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_VAD_MODEL",
    "SHERPA_MODELS",
    "ModelKind",
    "SherpaModel",
    "SherpaOnnxSTT",
    "SherpaOnnxTTS",
    "SherpaOnnxVAD",
    "split_long_audio",
]

T = TypeVar("T")

_PROVIDER = "sherpa_onnx"
_EXTRA = "sherpa-onnx"
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_CACHE_SUBDIR = "sherpa-onnx"
_SAMPLE_RATE = 16_000
"""Every sherpa-onnx recognizer and VAD in the catalog runs at 16 kHz."""

ModelKind = Literal[
    "online-transducer",
    "online-paraformer",
    "online-zipformer2-ctc",
    "online-nemo-ctc",
    "offline-transducer",
    "offline-nemo-transducer",
    "offline-nemo-ctc",
    "offline-moonshine",
    "offline-moonshine-v2",
    "offline-sense-voice",
    "offline-whisper",
    "tts-vits",
    "tts-kokoro",
    "tts-matcha",
    "vad-silero",
    "vad-ten",
]

_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "online-transducer": ("encoder", "decoder", "joiner", "tokens"),
    "online-paraformer": ("encoder", "decoder", "tokens"),
    "online-zipformer2-ctc": ("model", "tokens"),
    "online-nemo-ctc": ("model", "tokens"),
    "offline-transducer": ("encoder", "decoder", "joiner", "tokens"),
    "offline-nemo-transducer": ("encoder", "decoder", "joiner", "tokens"),
    "offline-nemo-ctc": ("model", "tokens"),
    "offline-moonshine": (
        "preprocessor",
        "encoder",
        "uncached_decoder",
        "cached_decoder",
        "tokens",
    ),
    "offline-moonshine-v2": ("encoder", "decoder", "tokens"),
    "offline-sense-voice": ("model", "tokens"),
    "offline-whisper": ("encoder", "decoder", "tokens"),
    "tts-vits": ("model", "tokens"),
    "tts-kokoro": ("model", "voices", "tokens"),
    "tts-matcha": ("acoustic_model", "vocoder", "tokens"),
    "vad-silero": ("model",),
    "vad-ten": ("model",),
}
"""Files each model kind needs (``data_dir``, ``lexicon`` and ``rule_fsts`` are optional)."""

_OPTIONAL_FILES = ("data_dir", "lexicon", "rule_fsts")

_FILE_GLOBS: dict[str, tuple[str, ...]] = {
    "encoder": ("encoder*.onnx", "encode*.onnx", "*-encoder*.onnx", "encoder*.ort"),
    "decoder": ("decoder*.onnx", "*-decoder*.onnx", "decoder*.ort"),
    "joiner": ("joiner*.onnx",),
    "tokens": ("tokens.txt", "*tokens.txt"),
    "model": ("model*.onnx", "*.onnx"),
    "preprocessor": ("preprocess*.onnx",),
    "uncached_decoder": ("uncached_decode*.onnx",),
    "cached_decoder": ("cached_decode*.onnx",),
    "voices": ("voices*.bin",),
    "acoustic_model": ("model-steps-*.onnx", "model*.onnx"),
    "vocoder": ("vocos*.onnx", "hifigan*.onnx"),
    "data_dir": ("espeak-ng-data",),
}
"""How files are found in a local model directory (``*.int8.*`` variants win)."""

_TTS_DEFAULT_RATES = {"tts-vits": 22_050, "tts-kokoro": 24_000, "tts-matcha": 22_050}
_MAX_DURATION = {
    "offline-moonshine": 8.0,
    "offline-moonshine-v2": 8.0,
    "offline-sense-voice": 28.0,
    "offline-whisper": 29.0,
}
"""Longest audio (s) these offline models decode in one pass (Moonshine returns nothing
beyond ~9 s, Whisper sees 30 s windows); longer input is split at pauses."""


@dataclass(frozen=True, slots=True)
class SherpaModel:
    """A pinned sherpa-onnx release asset and how to load it.

    ``files`` maps each role the model kind needs (``encoder``, ``tokens``, ``data_dir``...)
    to a path inside the extracted archive (``""``: the asset itself, for single files).
    ``extra_assets`` are further single-file downloads, e.g. Matcha's vocoder.
    """

    name: str
    kind: ModelKind
    asset: str
    sha256: str
    size: int
    """Download size in bytes."""
    files: Mapping[str, str]
    languages: str
    license: str
    release: str = "asr-models"
    description: str = ""
    sample_rate: int | None = None
    """TTS output rate (Hz)."""
    speakers: tuple[str, ...] = ()
    """TTS voice names indexed by speaker id (empty: numeric ids only)."""
    default_voice: str | None = None
    timestamps: bool = False
    """The model reports token timestamps (word timings are derived from them)."""
    language_prompt: bool = False
    """The model takes the language as a per-stream prompt (Nemotron 3.5)."""
    language_detection: bool = False
    """The model detects the spoken language by itself."""
    options: Mapping[str, Any] = field(default_factory=dict)
    """Extra keyword arguments for the sherpa-onnx factory (e.g. ``model_type``)."""
    extra_assets: tuple[tuple[str, str, str, str], ...] = ()
    """``(role, release, file name, sha256)`` of additional single-file downloads."""

    @property
    def url(self) -> str:
        return f"{_RELEASES}/{self.release}/{self.asset}"

    @property
    def is_archive(self) -> bool:
        return self.asset.endswith(".tar.bz2")

    @property
    def task(self) -> str:
        return "stt" if self.kind.startswith(("online", "offline")) else self.kind.split("-")[0]

    @property
    def streaming(self) -> bool:
        return self.kind.startswith("online")

    @property
    def stem(self) -> str:
        return self.asset.removesuffix(".tar.bz2")


_TRANSDUCER_INT8 = {
    "encoder": "encoder.int8.onnx",
    "decoder": "decoder.int8.onnx",
    "joiner": "joiner.int8.onnx",
    "tokens": "tokens.txt",
}
_KROKO = {
    "encoder": "encoder.onnx",
    "decoder": "decoder.onnx",
    "joiner": "joiner.onnx",
    "tokens": "tokens.txt",
}
_MOONSHINE_V2 = {
    "encoder": "encoder_model.ort",
    "decoder": "decoder_model_merged.ort",
    "tokens": "tokens.txt",
}
_NEMO_FASTCONFORMER = "NVIDIA FastConformer hybrid large streaming (114M), NeMo ASRSET"
_NEMOTRON_EN = "NVIDIA Nemotron Speech Streaming EN 0.6B, cache-aware"
_NEMOTRON_35 = "NVIDIA Nemotron 3.5 ASR Streaming 0.6B, cache-aware, 40 locales + auto LID"
_KROKO_DESC = "Banafo Kroko community streaming Zipformer (cased, punctuated)"
_KOKORO_V1_VOICES = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole",
    "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
    "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
    "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora",
    "em_alex", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex",
    "pm_santa", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi",
    "zm_yunxia", "zm_yunyang", "em_santa",
)  # fmt: skip
"""Speaker ids of sherpa-onnx's Kokoro v1.0 voice table (scripts/kokoro/v1.0 upstream)."""
_KOKORO_FILES = {
    "model": "model.onnx",
    "voices": "voices.bin",
    "tokens": "tokens.txt",
    "data_dir": "espeak-ng-data",
    "lexicon": "lexicon-us-en.txt,lexicon-zh.txt",
    "rule_fsts": "phone-zh.fst,date-zh.fst,number-zh.fst",
}


def _nemo_fastconformer(ms: int, asset_sha: str, size: int) -> SherpaModel:
    return SherpaModel(
        f"nemo-fastconformer-en-{ms}ms",
        "online-transducer",
        f"sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-{ms}ms-int8.tar.bz2",
        asset_sha,
        size,
        _TRANSDUCER_INT8,
        "en",
        "CC-BY-4.0",
        description=f"{_NEMO_FASTCONFORMER}, {ms} ms look-ahead, int8, lowercase",
        timestamps=True,
    )


def _nemotron(family: str, ms: int, asset_sha: str, size: int) -> SherpaModel:
    if family == "en":
        name, stem = (
            f"nemotron-en-{ms}ms",
            f"nemotron-speech-streaming-en-0.6b-{ms}ms-int8-2026-04-25",
        )
        languages, license_, desc = "en", "NVIDIA Open Model License", _NEMOTRON_EN
    else:
        name, stem = (
            f"nemotron-3.5-{ms}ms",
            f"nemotron-3.5-asr-streaming-0.6b-{ms}ms-int8-2026-06-11",
        )
        languages, license_, desc = "40 locales", "OpenMDW-1.1", _NEMOTRON_35
    return SherpaModel(
        name,
        "online-transducer",
        f"sherpa-onnx-{stem}.tar.bz2",
        asset_sha,
        size,
        _TRANSDUCER_INT8,
        languages,
        license_,
        description=f"{desc}, {ms} ms chunks, int8",
        timestamps=True,
        language_prompt=family != "en",
        language_detection=family != "en",
    )


def _kroko(lang: str, asset_sha: str, size: int) -> SherpaModel:
    return SherpaModel(
        f"zipformer-{lang}-kroko",
        "online-transducer",
        f"sherpa-onnx-streaming-zipformer-{lang}-kroko-2025-08-06.tar.bz2",
        asset_sha,
        size,
        _KROKO,
        lang,
        "CC-BY-SA-4.0",
        description=_KROKO_DESC,
        timestamps=True,
        options={"model_type": "zipformer2"},
    )


def _piper(voice: str, asset_sha: str, size: int, license_: str) -> SherpaModel:
    lang = voice.split("-")[0]
    return SherpaModel(
        f"piper-{voice}",
        "tts-vits",
        f"vits-piper-{voice}-int8.tar.bz2",
        asset_sha,
        size,
        {"model": f"{voice}.onnx", "tokens": "tokens.txt", "data_dir": "espeak-ng-data"},
        lang,
        license_,
        release="tts-models",
        description=f"Piper VITS voice {voice} (int8)",
        sample_rate=22_050,
        default_voice="0",
    )


_CATALOG: tuple[SherpaModel, ...] = (
    # ------------------------------------------------------------- streaming STT
    _nemo_fastconformer(
        80, "7bd33a914e93370a1ba9c2066d9e841bdcad8613fa2a00537c1ae15d851a14d8", 102_813_625
    ),
    _nemo_fastconformer(
        480, "da93061cbf7b708b6b65976f70b29f519be29df750d8cdcabf98c65645930f13", 105_913_204
    ),
    _nemo_fastconformer(
        1040, "821b3d601f73629af1afc7c7f0edb15bed333186b8146d20c35496530058e9f5", 103_918_085
    ),
    _nemotron(
        "en", 80, "caaf92069dbd1ca054f8e17cab179813bc28b4585f5c392540357ece4722333d", 463_945_379
    ),
    _nemotron(
        "en", 160, "0ae73a41cd51599dc7cac9ac083d9d35de53d762ca45923505fde47a3751814b", 463_945_198
    ),
    _nemotron(
        "en", 560, "78e2b79fcf7271553a74402a76b771b09ea40117a39566a79f52235b23db6358", 463_945_051
    ),
    _nemotron(
        "en", 1120, "840c48deed02d4a5975716e7b12dc0a8b1ba620776c6366f7e5677d8907edd73", 463_945_058
    ),
    _nemotron(
        "3.5", 80, "fb170128c496db33a1fb9f5f9f823257f42f911224ee218bb429f3c2eaf90a8d", 475_274_007
    ),
    _nemotron(
        "3.5", 160, "a81909a1780d84cff16d73c15e13e67d9d81d8839faf14870d507d8499f7a61a", 475_273_363
    ),
    _nemotron(
        "3.5", 320, "5f311142337a5c161e92d49f7a3009d8607d3836f39d610bff5307c74d1d2c53", 475_272_949
    ),
    _nemotron(
        "3.5", 560, "c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a", 475_271_763
    ),
    _nemotron(
        "3.5", 1120, "adbdd5e9fef87300c37cebfcfc4f1ebe56845c860c8a760af0a1dd65ce9beed3", 475_276_334
    ),
    _kroko("en", "c8676e5ff9ac2a85296e53ee0fd4d5fb1db6770e7a7647166eeafe349ade6834", 57_267_600),
    _kroko("fr", "e6ffd3dc43725cd6c8137b05c739f15607d0df946b9b90eb141e10059efca024", 57_220_361),
    _kroko("de", "9e27b783c20e67b0d0f13a258c1861fce199917c969d9176a438bee38df64962", 57_565_698),
    _kroko("es", "31b2230a95d23290b308b393da930015a4b2105cb3abb9367aed35f7fcf29cf1", 124_394_665),
    # --------------------------------------------------------------- offline STT
    SherpaModel(
        "moonshine-tiny-en",
        "offline-moonshine-v2",
        "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2",
        "9ec31b342d8fa3240c3b81b8f82e1cf7e3ac467c93ca5a999b741d5887164f8d",
        29_858_559,
        _MOONSHINE_V2,
        "en",
        "MIT",
        description="Moonshine tiny (quantized): edge-class, cased and punctuated",
    ),
    SherpaModel(
        "moonshine-base-en",
        "offline-moonshine-v2",
        "sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2",
        "43232c1d13013d37317163baec3135bd771a186a4356f28c889bab453bb0e891",
        111_266_225,
        _MOONSHINE_V2,
        "en",
        "MIT",
        description="Moonshine base (quantized): cased and punctuated",
    ),
    SherpaModel(
        "parakeet-tdt-0.6b-v2",
        "offline-nemo-transducer",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        "157c157bc51155e03e37d2466522a3a737dd9c72bb25f36eb18912964161e1ad",
        482_468_385,
        _TRANSDUCER_INT8,
        "en",
        "CC-BY-4.0",
        description="NVIDIA Parakeet TDT 0.6B v2 (int8): cased and punctuated",
        timestamps=True,
    ),
    SherpaModel(
        "parakeet-tdt-0.6b-v3",
        "offline-nemo-transducer",
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf",
        487_170_055,
        _TRANSDUCER_INT8,
        "25 European languages",
        "CC-BY-4.0",
        description="NVIDIA Parakeet TDT 0.6B v3 (int8): cased, punctuated, language auto-detected",
        timestamps=True,
        language_detection=True,
    ),
    SherpaModel(
        "sense-voice",
        "offline-sense-voice",
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
        "7305f7905bfcf77fa0b39388a313f3da35c68d971661a65475b56fb2162c8e63",
        165_783_878,
        {"model": "model.int8.onnx", "tokens": "tokens.txt"},
        "zh, en, ja, ko, yue",
        "FunASR model license",
        description="SenseVoice Small (int8): language ID, inverse text normalization",
        timestamps=True,
        language_detection=True,
    ),
    # ----------------------------------------------------------------------- TTS
    _piper(
        "en_US-libritts_r-medium",
        "7e4552e239988f4896872822b56e99e0e9e00958164e3f6bdf5ee14391fbe829",
        23_398_348,
        "MIT (voice), CC-BY-4.0 (LibriTTS-R data)",
    ),
    _piper(
        "en_US-ljspeech-medium",
        "24dc3bd77dd48c291e52c297878d3437c9492f245d823d7f6a06c4bbb67f4b6b",
        21_090_429,
        "MIT (voice), public domain (LJSpeech data)",
    ),
    SherpaModel(
        "kokoro-multi-lang-v1_0",
        "tts-kokoro",
        "kokoro-multi-lang-v1_0.tar.bz2",
        "c5f7e2d2caf082bc1d20fb70334a61d99d20b484500aad32e7cf84c128ea3298",
        349_906_910,
        _KOKORO_FILES,
        "en, zh, es, fr, hi, it, ja, pt",
        "Apache-2.0",
        release="tts-models",
        description="Kokoro-82M v1.0, 54 voices",
        sample_rate=24_000,
        speakers=_KOKORO_V1_VOICES,
        default_voice="af_heart",
    ),
    SherpaModel(
        "kokoro-multi-lang-v1_0-int8",
        "tts-kokoro",
        "kokoro-int8-multi-lang-v1_0.tar.bz2",
        "4c3052abaa60943a341f193888cf6abd68787dae6ab8ae5c925a706caa247e4e",
        132_303_094,
        {**_KOKORO_FILES, "model": "model.int8.onnx"},
        "en, zh, es, fr, hi, it, ja, pt",
        "Apache-2.0",
        release="tts-models",
        description="Kokoro-82M v1.0 (int8), 54 voices",
        sample_rate=24_000,
        speakers=_KOKORO_V1_VOICES,
        default_voice="af_heart",
    ),
    SherpaModel(
        "matcha-en_US-ljspeech",
        "tts-matcha",
        "matcha-icefall-en_US-ljspeech.tar.bz2",
        "ea75702da7456a8b1874728278a835220dc8a26f4e8bd93c83bf53dc27679845",
        76_741_121 + 53_884_024,
        {
            "acoustic_model": "model-steps-3.onnx",
            "tokens": "tokens.txt",
            "data_dir": "espeak-ng-data",
        },
        "en",
        "Apache-2.0 (model), MIT (Vocos vocoder), public domain (LJSpeech data)",
        release="tts-models",
        description="Matcha-TTS (icefall) + Vocos 22 kHz vocoder",
        sample_rate=22_050,
        default_voice="0",
        extra_assets=(
            (
                "vocoder",
                "vocoder-models",
                "vocos-22khz-univ.onnx",
                "0574a135aa1db2de6e181050db2ec528496cacd4a4701fc5d7faf9f9804c0081",
            ),
        ),
    ),
    # ----------------------------------------------------------------------- VAD
    SherpaModel(
        "silero",
        "vad-silero",
        "silero_vad.onnx",
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        643_854,
        {"model": ""},
        "any",
        "MIT",
        description="Silero VAD (sherpa-onnx export), 32 ms windows",
    ),
    SherpaModel(
        "ten-vad",
        "vad-ten",
        "ten-vad.onnx",
        "718cb7eef47e3cf5ddbe7e967a7503f46b8b469c0706872f494dfa921b486206",
        332_211,
        {"model": ""},
        "any",
        "Apache-2.0 with additional conditions (TEN-framework/ten-vad)",
        description="TEN VAD, 16 ms windows",
    ),
)

SHERPA_MODELS: dict[str, SherpaModel] = {m.name: m for m in _CATALOG}
"""The model catalog by short name. The release archive name (without ``.tar.bz2``) works
as a model name too."""
_BY_STEM: dict[str, SherpaModel] = {m.stem: m for m in _CATALOG}

DEFAULT_STT_MODEL = "nemo-fastconformer-en-80ms"
DEFAULT_TTS_MODEL = "piper-en_US-libritts_r-medium"
DEFAULT_VAD_MODEL = "silero"


# --------------------------------------------------------------------- helpers
def _check_installed() -> None:
    """Fail fast (without importing the native module) when the extra is missing."""
    if not is_installed("sherpa_onnx"):
        require("sherpa_onnx", extra=_EXTRA, package="sherpa-onnx")


def _import_sherpa() -> Any:
    _check_installed()
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError as exc:  # e.g. libonnxruntime missing: sherpa-onnx-core not installed
        raise MissingDependencyError(
            f"sherpa-onnx is installed but cannot be loaded ({exc}). Its native libraries "
            "come from the sherpa-onnx-core package of the same version: pip install "
            "'voice-agent-next[sherpa-onnx]'"
        ) from exc


def _task_models(task: str) -> tuple[str, ...]:
    return tuple(m.name for m in _CATALOG if m.task == task)


def _kinds(task: str) -> list[str]:
    prefixes = ("online-", "offline-") if task == "stt" else (f"{task}-",)
    return [k for k in _REQUIRED_FILES if k.startswith(prefixes)]


def _lookup(name: str, task: str) -> SherpaModel | None:
    spec = SHERPA_MODELS.get(name) or _BY_STEM.get(name.removesuffix(".tar.bz2"))
    if spec is not None and spec.task != task:
        raise ConfigurationError(f"sherpa-onnx model {name!r} is a {spec.task} model, not {task}")
    return spec


def _is_url(value: str) -> bool:
    return value.startswith(("https://", "http://"))


def _pick(directory: Path, patterns: Sequence[str]) -> Path | None:
    """First match for ``patterns`` in ``directory``: int8 variants, then shorter names win."""
    for pattern in patterns:
        found = sorted(
            directory.glob(pattern), key=lambda p: (".int8." not in p.name, len(p.name), p.name)
        )
        if found:
            return found[0]
    return None


def _detect_kind(directory: Path, task: str) -> str | None:
    """Guess a model directory's kind from its files and name (sherpa-onnx naming)."""
    names = {p.name for p in directory.iterdir()}
    lower = directory.name.lower()
    streaming = "streaming" in lower or "online" in lower
    if task == "stt":
        if {"encoder_model.ort", "decoder_model_merged.ort"} <= names:
            return "offline-moonshine-v2"
        if any(n.startswith("uncached_decode") for n in names):
            return "offline-moonshine"
        if any(n.startswith("joiner") and n.endswith(".onnx") for n in names):
            if streaming:
                return "online-transducer"
            return "offline-nemo-transducer" if "nemo" in lower else "offline-transducer"
        if "whisper" in lower:
            return "offline-whisper"
        if "sense-voice" in lower or "sense_voice" in lower:
            return "offline-sense-voice"
        if "paraformer" in lower and streaming:
            return "online-paraformer"
        if "nemo" in lower and "ctc" in lower:
            return "online-nemo-ctc" if streaming else "offline-nemo-ctc"
        if "zipformer" in lower and "ctc" in lower and streaming:
            return "online-zipformer2-ctc"
        return None
    if task == "tts":
        if "kitten" in lower:
            return None
        if any(n.startswith("voices") and n.endswith(".bin") for n in names):
            return "tts-kokoro"
        if "matcha" in lower or any(n.startswith("model-steps") for n in names):
            return "tts-matcha"
        if "tokens.txt" in names and any(n.endswith(".onnx") for n in names):
            return "tts-vits"
    return None


@dataclass(slots=True)
class _ModelRef:
    """What a component was configured with; turned into files on the worker thread."""

    task: str
    kind: str
    spec: SherpaModel | None
    source: str
    """Catalog name, local path or URL."""
    sha256: str | None
    files: dict[str, Path]
    """Explicit per-role overrides."""
    root: Path | None = None
    """Local directory holding a catalog model (no download)."""

    @property
    def label(self) -> str:
        return self.spec.name if self.spec is not None else self.source


def _model_ref(
    task: str,
    model: str,
    *,
    kind: str | None,
    files: Mapping[str, str | os.PathLike[str]] | None,
    sha256: str | None,
) -> _ModelRef:
    overrides = {role: Path(p).expanduser() for role, p in (files or {}).items()}
    kinds = _kinds(task)
    if kind is not None and kind not in kinds:
        raise ConfigurationError(f"unknown sherpa-onnx {task} kind {kind!r}; valid: {kinds}")
    spec = _lookup(model, task)
    if spec is not None:
        return _ModelRef(task, kind or spec.kind, spec, model, spec.sha256, overrides)
    if _is_url(model):
        if kind is None:
            raise ConfigurationError(f"sherpa-onnx: pass kind= for model URLs ({model})")
        return _ModelRef(task, kind, None, model, sha256, overrides)
    path = Path(model).expanduser()
    local_spec = _BY_STEM.get(path.name) if path.is_dir() else None
    if local_spec is not None and local_spec.task == task:  # an extracted catalog archive
        kind = kind or local_spec.kind
        return _ModelRef(task, kind, local_spec, str(path), None, overrides, root=path)
    if path.is_dir():
        detected = kind or _detect_kind(path, task)
        if detected is None:
            raise ConfigurationError(
                f"cannot tell which kind of sherpa-onnx {task} model {path} holds; pass kind= "
                f"(one of {kinds})"
            )
        return _ModelRef(task, detected, None, str(path), None, overrides)
    if path.is_file() or (overrides and kind is not None):
        if kind is None:
            raise ConfigurationError(f"sherpa-onnx: pass kind= for the model file {path}")
        if path.is_file():
            overrides.setdefault(_REQUIRED_FILES[kind][0], path)
        return _ModelRef(task, kind, None, str(path), None, overrides)
    raise ConfigurationError(
        f"unknown sherpa-onnx {task} model {model!r}: not in the catalog "
        f"({', '.join(_task_models(task))}) and not a local directory or file"
    )


def _materialize(ref: _ModelRef) -> dict[str, Path]:
    """Download (if needed) and locate every file the model needs (blocking)."""
    spec = ref.spec
    found: dict[str, Path] = {}
    if spec is not None:
        if ref.root is not None or spec.is_archive:
            root = ref.root or download_archive(spec.url, sha256=spec.sha256, subdir=_CACHE_SUBDIR)
            found = {
                role: Path(",".join(str(root / part) for part in rel.split(",")))
                for role, rel in spec.files.items()
                if role not in ref.files
            }
        else:
            path = download(spec.url, subdir=_CACHE_SUBDIR, sha256=spec.sha256)
            found = {role: path for role in spec.files if role not in ref.files}
        for role, release, name, digest in spec.extra_assets:
            if role not in ref.files:
                url = f"{_RELEASES}/{release}/{name}"
                found[role] = download(url, subdir=_CACHE_SUBDIR, sha256=digest)
    elif _is_url(ref.source):
        if ref.source.endswith((".tar.bz2", ".tar.gz", ".tar.xz", ".tar")):
            root = download_archive(ref.source, sha256=ref.sha256, subdir=_CACHE_SUBDIR)
            found = _find_files(root, ref.kind, exclude=ref.files)
        else:
            path = download(ref.source, subdir=_CACHE_SUBDIR, sha256=ref.sha256)
            found = {_REQUIRED_FILES[ref.kind][0]: path}
    else:
        path = Path(ref.source).expanduser()
        if path.is_dir():
            found = _find_files(path, ref.kind, exclude=ref.files)
    found.update(ref.files)
    missing = [
        f"{role} ({found[role]})" if role in found else role
        for role in _REQUIRED_FILES[ref.kind]
        if role not in found or not _exists(found[role])
    ]
    if missing:
        raise ConfigurationError(
            f"sherpa-onnx model {ref.label} ({ref.kind}) is missing: {', '.join(missing)}; "
            "pass files={role: path} to point at them"
        )
    return found


def _exists(path: Path) -> bool:
    # comma-separated lists (lexicons, rule FSTs) are checked element by element
    return all(Path(p).exists() for p in str(path).split(","))


def _find_files(directory: Path, kind: str, *, exclude: Mapping[str, Path]) -> dict[str, Path]:
    roles = [r for r in (*_REQUIRED_FILES[kind], *_OPTIONAL_FILES) if r not in exclude]
    found: dict[str, Path] = {}
    for role in roles:
        if role == "lexicon":
            lexicons = sorted(directory.glob("lexicon*.txt"))
            if lexicons:
                found[role] = Path(",".join(str(p) for p in lexicons))
            continue
        if role == "rule_fsts":
            continue  # only from the catalog or files=
        patterns = _FILE_GLOBS.get(role, ())
        if kind == "tts-vits" and role == "model":
            patterns = ("*.onnx",)
        match = _pick(directory, patterns)
        if match is not None:
            found[role] = match
    return found


def _joined(root_files: Mapping[str, Path], role: str) -> str:
    """A file role as the string sherpa expects (``""`` when absent)."""
    path = root_files.get(role)
    return "" if path is None else str(path)


def _sherpa_error(exc: BaseException, action: str) -> VoiceAgentError:
    if isinstance(exc, VoiceAgentError):
        return exc
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return ConfigurationError(f"sherpa-onnx {action} failed: {exc}")
    return ProviderError(f"sherpa-onnx {action} failed: {exc}", provider=_PROVIDER)


class _Worker:
    """One worker thread per component: sherpa-onnx objects are only touched from it."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self._name)
            executor = self._executor
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def _text_case(text: str, mode: str) -> str:
    if mode == "lower" or (mode == "auto" and text.isupper()):
        return text.lower()
    return text


def _is_cjk(ch: str) -> bool:
    return (
        "\u3040" <= ch <= "\u30ff"  # hiragana, katakana
        or "\u3400" <= ch <= "\u9fff"  # CJK ideographs
        or "\uac00" <= ch <= "\ud7af"  # hangul syllables
    )


_LAST_WORD_DURATION = 0.25
"""Assumed duration of the last word (token timestamps mark where a token starts)."""
_LANG_TAG = re.compile(r"<\|([a-z]{2,3})\|>")


def _words(
    tokens: Sequence[str], starts: Sequence[float], offset: float, *, end: float | None
) -> list[WordTiming]:
    """Group sub-word tokens into words: a leading space (or ``\u2581``) or a CJK character
    starts a new word. A word ends where the next one starts."""
    words: list[tuple[str, float]] = []
    for token, start in zip(tokens, starts, strict=False):
        piece = token.replace("\u2581", " ")  # SentencePiece word marker
        stripped = piece.strip()
        if not stripped:
            continue
        new_word = (
            not words or piece[:1].isspace() or _is_cjk(stripped[0]) or _is_cjk(words[-1][0][-1])
        )
        if new_word:
            words.append((stripped, offset + float(start)))
        else:
            words[-1] = (words[-1][0] + stripped, words[-1][1])
    timings: list[WordTiming] = []
    for i, (word, start) in enumerate(words):
        if i + 1 < len(words):
            stop = words[i + 1][1]
        else:
            stop = start + _LAST_WORD_DURATION
            if end is not None:
                stop = max(start, min(stop, end))
        timings.append(WordTiming(word, start, max(start, stop)))
    return timings


def split_long_audio(
    samples: npt.NDArray[np.float32], max_samples: int, sample_rate: int
) -> list[npt.NDArray[np.float32]]:
    """Split audio longer than ``max_samples`` at the quietest 20 ms frame of the last third
    of each piece (a pause between words, usually)."""
    if max_samples <= 0 or len(samples) <= max_samples:
        return [samples]
    frame = max(1, sample_rate // 50)
    pieces: list[npt.NDArray[np.float32]] = []
    start = 0
    while len(samples) - start > max_samples:
        lo, hi = start + max_samples * 2 // 3, start + max_samples
        count = (hi - lo) // frame
        if count > 0:
            window = samples[lo : lo + count * frame].reshape(count, frame)
            cut = lo + int(np.argmin(np.mean(window * window, axis=1))) * frame + frame // 2
        else:
            cut = hi
        pieces.append(samples[start:cut])
        start = cut
    pieces.append(samples[start:])
    return pieces


# ----------------------------------------------------------------------------- STT
@dataclass(slots=True)
class _Snapshot:
    """A recognizer result, converted on the worker thread."""

    text: str
    start: float | None = None
    end: float | None = None
    words: list[WordTiming] | None = None
    confidence: float | None = None
    language: str | None = None
    endpoint: bool = False


@register_provider(
    "stt",
    "sherpa-onnx",
    description="sherpa-onnx local STT: streaming Zipformer/NeMo/Nemotron, offline Moonshine/"
    "Parakeet/SenseVoice (CPU, every OS)",
    default_model=DEFAULT_STT_MODEL,
    models=_task_models("stt"),
    env=(),
    extra=_EXTRA,
    requires=("sherpa_onnx",),
    local=True,
    aliases=("sherpa",),
)
class SherpaOnnxSTT(STT):
    """Local speech recognition with sherpa-onnx (streaming or offline models).

    Args:
        model: a catalog name (:data:`SHERPA_MODELS`, e.g. ``"nemo-fastconformer-en-80ms"``,
            ``"zipformer-en-kroko"``, ``"moonshine-tiny-en"``, ``"parakeet-tdt-0.6b-v3"``) or
            its release archive name, a local model directory, or an archive URL.
        language: language code. Passed to SenseVoice/Whisper and as the language prompt of
            Nemotron 3.5 (``"auto"`` detects); otherwise only reported in transcripts.
        kind: model kind (:data:`ModelKind`) for directories/URLs that are not in the
            catalog; guessed from the file names of a local directory when omitted.
        files: explicit files by role (``encoder``, ``decoder``, ``joiner``, ``tokens``,
            ``model``...), overriding the catalog or the directory scan.
        sha256: expected digest when ``model`` is a URL.
        num_threads: ONNX Runtime threads for the recognizer.
        execution_provider: ``"cpu"`` (default), ``"cuda"`` or ``"coreml"`` (the PyPI wheels
            are CPU-only and fall back to CPU).
        decoding_method: ``"greedy_search"`` or ``"modified_beam_search"`` (transducers).
        max_active_paths: beam size for ``modified_beam_search``.
        endpoint_detection: use sherpa's endpoint rules (streaming models): an endpoint
            emits ``FINAL_TRANSCRIPT`` + ``END_OF_SPEECH``. Off by default: the cascade's
            VAD and turn detector decide and :meth:`STTStream.flush` finalizes.
        rule1_min_trailing_silence: endpoint after this much silence with nothing decoded.
        rule2_min_trailing_silence: endpoint after this much silence following speech.
        rule3_min_utterance_length: endpoint when an utterance gets this long (s).
        tail_padding: silence (s) appended when finalizing a streaming utterance. Default:
            computed from the model's chunk geometry, just enough to decode the tail.
        interim_results: emit ``INTERIM_TRANSCRIPT`` events (streaming models).
        word_timestamps: fill :attr:`Transcript.words` when the model reports token times.
        text_case: ``"auto"`` lowercases all-caps output (LibriSpeech-style models),
            ``"lower"`` always lowercases, ``"keep"`` leaves the text alone.
        max_segment_duration: offline models: split longer audio at pauses (default: the
            catalog's limit, e.g. 8 s for Moonshine).
        recognizer_options: extra keyword arguments for the sherpa-onnx factory
            (``hotwords_file``, ``modeling_unit``, ``lm``, ``blank_penalty``...).
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        kind: str | None = None,
        files: Mapping[str, str | os.PathLike[str]] | None = None,
        sha256: str | None = None,
        num_threads: int = 2,
        execution_provider: str = "cpu",
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        endpoint_detection: bool = False,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
        tail_padding: float | None = None,
        interim_results: bool = True,
        word_timestamps: bool = True,
        text_case: Literal["auto", "lower", "keep"] = "auto",
        max_segment_duration: float | None = None,
        recognizer_options: Mapping[str, Any] | None = None,
    ) -> None:
        _check_installed()
        if num_threads < 1:
            raise ConfigurationError(f"sherpa-onnx: num_threads must be >= 1, got {num_threads}")
        if decoding_method not in ("greedy_search", "modified_beam_search"):
            # sherpa-onnx terminates the process on an unknown decoding method
            raise ConfigurationError(
                "sherpa-onnx: decoding_method must be 'greedy_search' or 'modified_beam_search', "
                f"got {decoding_method!r}"
            )
        if text_case not in ("auto", "lower", "keep"):
            raise ConfigurationError(f"sherpa-onnx: invalid text_case {text_case!r}")
        if tail_padding is not None and tail_padding < 0:
            raise ConfigurationError("sherpa-onnx: tail_padding must be >= 0")
        ref = _model_ref("stt", model or DEFAULT_STT_MODEL, kind=kind, files=files, sha256=sha256)
        spec = ref.spec
        streaming = ref.kind.startswith("online")
        if decoding_method == "modified_beam_search" and "transducer" not in ref.kind:
            raise ConfigurationError(
                f"sherpa-onnx: modified_beam_search needs a transducer model, not {ref.kind}"
            )
        timestamps = spec.timestamps if spec is not None else "transducer" in ref.kind
        super().__init__(
            model=ref.label,
            capabilities=STTCapabilities(
                streaming=streaming,
                interim_results=streaming and interim_results,
                word_timestamps=word_timestamps and timestamps,
                end_of_turn=False,
                language_detection=ref.kind in ("offline-sense-voice", "offline-whisper")
                or (spec is not None and spec.language_detection),
            ),
            sample_rate=_SAMPLE_RATE,
            language=language,
        )
        self.kind = ref.kind
        self.num_threads = num_threads
        self.execution_provider = execution_provider
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self.endpoint_detection = endpoint_detection
        self.rule1_min_trailing_silence = rule1_min_trailing_silence
        self.rule2_min_trailing_silence = rule2_min_trailing_silence
        self.rule3_min_utterance_length = rule3_min_utterance_length
        self.tail_padding = tail_padding
        self.interim_results = interim_results
        self.word_timestamps = word_timestamps and timestamps
        self.text_case = text_case
        self.max_segment_duration = max_segment_duration or _MAX_DURATION.get(ref.kind)
        self.recognizer_options = dict(recognizer_options or {})
        self._ref = ref
        self._worker = _Worker("sherpa-onnx-stt")
        self._recognizer: Any = None
        self._window = 0
        """Samples a fresh stream needs before its first decoding step."""
        self._shift = 0
        """Samples each further decoding step consumes."""

    @property
    def streaming(self) -> bool:
        return self.kind.startswith("online")

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Download (first run only) and load the model on the worker thread."""
        await self._worker.run(self._ensure_recognizer)

    async def aclose(self) -> None:
        """Stop the worker thread (a decoding step already running finishes first)."""
        self._worker.close()
        self._recognizer = None

    def _ensure_recognizer(self) -> Any:
        """The loaded recognizer (worker thread only)."""
        if self._recognizer is None:
            t0 = now()
            files = _materialize(self._ref)
            so = _import_sherpa()
            try:
                recognizer = self._create(so, files)
                if self.streaming:
                    self._window, self._shift = _chunk_geometry(recognizer)
            except Exception as exc:
                raise _sherpa_error(exc, f"loading {self.model}") from exc
            self._recognizer = recognizer
            logger.info("sherpa-onnx: loaded %s (%s) in %.2f s", self.model, self.kind, now() - t0)
        return self._recognizer

    def _create(self, so: Any, files: Mapping[str, Path]) -> Any:
        kind = self.kind
        spec = self._ref.spec
        options: dict[str, Any] = {
            "num_threads": self.num_threads,
            "provider": self.execution_provider,
        }
        if spec is not None:
            options.update(spec.options)
        endpoint = {
            "enable_endpoint_detection": self.endpoint_detection,
            "rule1_min_trailing_silence": self.rule1_min_trailing_silence,
            "rule2_min_trailing_silence": self.rule2_min_trailing_silence,
            "rule3_min_utterance_length": self.rule3_min_utterance_length,
        }
        f = {role: str(path) for role, path in files.items()}
        lang = (self.language or "").strip()
        if kind == "online-transducer":
            options.update(endpoint, decoding_method=self.decoding_method,
                           max_active_paths=self.max_active_paths)  # fmt: skip
            factory = so.OnlineRecognizer.from_transducer
            args = {k: f[k] for k in ("tokens", "encoder", "decoder", "joiner")}
        elif kind == "online-paraformer":
            options.update(endpoint)
            factory = so.OnlineRecognizer.from_paraformer
            args = {k: f[k] for k in ("tokens", "encoder", "decoder")}
        elif kind in ("online-zipformer2-ctc", "online-nemo-ctc"):
            options.update(endpoint)
            factory = (
                so.OnlineRecognizer.from_zipformer2_ctc
                if kind == "online-zipformer2-ctc"
                else so.OnlineRecognizer.from_nemo_ctc
            )
            args = {"tokens": f["tokens"], "model": f["model"]}
        elif kind in ("offline-transducer", "offline-nemo-transducer"):
            options.update(decoding_method=self.decoding_method,
                           max_active_paths=self.max_active_paths)  # fmt: skip
            if kind == "offline-nemo-transducer":
                options.setdefault("model_type", "nemo_transducer")
            factory = so.OfflineRecognizer.from_transducer
            args = {k: f[k] for k in ("encoder", "decoder", "joiner", "tokens")}
        elif kind == "offline-nemo-ctc":
            factory = so.OfflineRecognizer.from_nemo_ctc
            args = {"model": f["model"], "tokens": f["tokens"]}
        elif kind == "offline-moonshine":
            factory = so.OfflineRecognizer.from_moonshine
            args = {k: f[k] for k in _REQUIRED_FILES[kind]}
        elif kind == "offline-moonshine-v2":
            factory = so.OfflineRecognizer.from_moonshine_v2
            args = {k: f[k] for k in ("encoder", "decoder", "tokens")}
        elif kind == "offline-sense-voice":
            options.update(language=_short_language(lang) or "auto", use_itn=True)
            factory = so.OfflineRecognizer.from_sense_voice
            args = {"model": f["model"], "tokens": f["tokens"]}
        elif kind == "offline-whisper":
            options.update(language=_short_language(lang) or "", task="transcribe")
            factory = so.OfflineRecognizer.from_whisper
            args = {k: f[k] for k in ("encoder", "decoder", "tokens")}
        else:  # pragma: no cover - kinds are validated at construction
            raise ConfigurationError(f"unsupported sherpa-onnx STT kind {kind!r}")
        options.update(self.recognizer_options)
        return factory(**args, **options)

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        return await self._worker.run(fn, *args)

    # ---------------------------------------------------------------- recognition
    def _create_stream(self, *, language: str | None) -> STTStream:
        if not self.streaming:  # pragma: no cover - STT.stream() checks capabilities first
            raise NotImplementedError
        return _SherpaOnlineStream(self, language=language)

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        # STT.transcribe() has already resampled the audio to 16 kHz mono.
        samples = audio.to_float32()
        if self.streaming:
            snap = await self._run(self._decode_online_sync, samples, language)
        else:
            snap = await self._run(self._decode_offline_sync, samples, language)
        return Transcript(
            text=snap.text,
            language=snap.language or language,
            confidence=snap.confidence,
            start_time=snap.start,
            end_time=snap.end,
            words=snap.words if self.word_timestamps else None,
        )

    def _decode_online_sync(
        self, samples: npt.NDArray[np.float32], language: str | None
    ) -> _Snapshot:
        recognizer = self._ensure_recognizer()
        if samples.size == 0:
            return _Snapshot("")
        try:
            stream = self._new_stream(language)
            stream.accept_waveform(_SAMPLE_RATE, samples)
            pad = self._padding(len(samples))
            if pad:
                stream.accept_waveform(_SAMPLE_RATE, np.zeros(pad, dtype=np.float32))
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            return self._snapshot(recognizer.get_result_all(stream), 0.0, len(samples))
        except Exception as exc:
            raise _sherpa_error(exc, "recognition") from exc

    def _decode_offline_sync(
        self, samples: npt.NDArray[np.float32], language: str | None
    ) -> _Snapshot:
        recognizer = self._ensure_recognizer()
        if samples.size == 0:
            return _Snapshot("")
        limit = round((self.max_segment_duration or 0) * _SAMPLE_RATE)
        texts: list[str] = []
        words: list[WordTiming] = []
        probs: list[float] = []
        detected: str | None = None
        offset = 0
        try:
            for piece in split_long_audio(samples, limit, _SAMPLE_RATE):
                stream = recognizer.create_stream()
                stream.accept_waveform(_SAMPLE_RATE, piece)
                recognizer.decode_stream(stream)
                result = stream.result
                text = _clean_text(result.text)
                if text:
                    texts.append(text)
                start = offset / _SAMPLE_RATE
                tokens = list(getattr(result, "tokens", []) or [])
                stamps = list(getattr(result, "timestamps", []) or [])
                if self.word_timestamps and tokens and len(stamps) == len(tokens):
                    words.extend(
                        _words(tokens, stamps, start, end=(offset + len(piece)) / _SAMPLE_RATE)
                    )
                probs.extend(float(p) for p in (getattr(result, "ys_log_probs", []) or []))
                detected = detected or _language_tag(getattr(result, "lang", ""))
                offset += len(piece)
        except Exception as exc:
            raise _sherpa_error(exc, "recognition") from exc
        text = _text_case(" ".join(texts), self.text_case)
        return _Snapshot(
            text=text,
            start=words[0].start if words else None,
            end=words[-1].end if words else None,
            words=words or None,
            confidence=_confidence(probs),
            language=detected,
        )

    # ------------------------------------------------------ streaming internals
    def _new_stream(self, language: str | None) -> Any:
        """A fresh sherpa ``OnlineStream`` (worker thread)."""
        stream = self._ensure_recognizer().create_stream()
        spec = self._ref.spec
        lang = (language or self.language or "").strip()
        if lang and spec is not None and spec.language_prompt:
            stream.set_option("language", lang)
        return stream

    def _padding(self, fed: int) -> int:
        """Zero samples to append so that every fed sample gets decoded.

        A stream decodes its first chunk once ``window`` samples are available and every
        further one ``shift`` samples later. Covering ``fed`` samples takes
        ``ceil(fed / shift)`` chunks; one more chunk lets the transducer emit the last
        token (its final letter or punctuation is often decided one chunk late), so the
        stream needs ``window + chunks * shift`` samples in total, plus two feature frames
        of margin. With upstream's fixed 0.66 s the NeMo 80 ms model decodes up to two
        chunks more than needed (~20 ms each on a desktop CPU).
        """
        if self.tail_padding is not None:
            return round(self.tail_padding * _SAMPLE_RATE)
        window, shift = self._window, self._shift
        if window <= 0 or shift <= 0:
            return round(0.66 * _SAMPLE_RATE)  # upstream's default for unknown geometry
        chunks = max(1, math.ceil(fed / shift))
        return max(0, window + chunks * shift + 2 * _SAMPLE_RATE // 100 - fed)

    def _snapshot(
        self, result: Any, offset: float, fed: int, *, endpoint: bool = False
    ) -> _Snapshot:
        """Convert an ``OnlineRecognizerResult`` (times relative to ``offset``, in seconds)."""
        text = _text_case(_clean_text(result.text), self.text_case)
        base = offset + float(getattr(result, "start_time", 0.0) or 0.0)
        tokens = list(result.tokens or [])
        stamps = list(result.timestamps or [])
        words = (
            _words(tokens, stamps, base, end=offset + fed / _SAMPLE_RATE)
            if tokens and len(stamps) == len(tokens)
            else []
        )
        probs = [float(p) for p in (getattr(result, "ys_probs", None) or [])]
        return _Snapshot(
            text=text,
            start=words[0].start if words else None,
            end=words[-1].end if words else None,
            words=words if self.word_timestamps else None,
            confidence=_confidence(probs),
            endpoint=endpoint,
        )


def _clean_text(text: str) -> str:
    return _LANG_TAG.sub("", text or "").strip()


def _language_tag(value: str | None) -> str | None:
    """``"<|en|>"`` (SenseVoice) or ``"en"`` -> ``"en"``."""
    if not value:
        return None
    match = _LANG_TAG.search(value)
    return match.group(1) if match else value.strip() or None


def _short_language(language: str) -> str:
    code = language.replace("_", "-").split("-")[0].lower()
    return "" if code in ("auto", "multi") else code


def _confidence(log_probs: Sequence[float]) -> float | None:
    if not log_probs:
        return None
    return float(min(1.0, math.exp(sum(log_probs) / len(log_probs))))


def _chunk_geometry(recognizer: Any) -> tuple[int, int]:
    """Measure how many samples a stream needs for its first decoding step and for each
    further one (feeds silence to a throwaway stream; also warms the model up)."""
    step = _SAMPLE_RATE // 100
    zeros = np.zeros(step, dtype=np.float32)
    stream = recognizer.create_stream()
    fed = 0
    limit = 10 * _SAMPLE_RATE
    while not recognizer.is_ready(stream) and fed < limit:
        stream.accept_waveform(_SAMPLE_RATE, zeros)
        fed += step
    window = fed
    recognizer.decode_stream(stream)
    while not recognizer.is_ready(stream) and fed < limit:
        stream.accept_waveform(_SAMPLE_RATE, zeros)
        fed += step
    if fed >= limit:
        return 0, 0
    return window, fed - window


class _SherpaOnlineStream(STTStream):
    """Streaming recognition over a sherpa ``OnlineStream`` (one per utterance)."""

    def __init__(self, stt: SherpaOnnxSTT, *, language: str | None) -> None:
        self._sherpa = stt
        self._stream: Any = None
        self._offset = 0
        """Input samples before the current sherpa stream."""
        self._fed = 0
        """Samples fed into the current sherpa stream (no padding)."""
        self._segment_id = new_id("seg_")
        self._partial = ""
        self._speaking = False
        super().__init__(stt, language=language)

    # -------------------------------------------------------------- event loop
    async def _run(self) -> None:
        stt = self._sherpa
        await stt._run(stt._ensure_recognizer)
        while True:
            try:
                item = await self._input.recv()
            except ChanClosed:
                break
            if self.is_flush(item):
                self._finish(await stt._run(self._finalize_sync))
                continue
            assert isinstance(item, AudioFrame)
            chunks = [item.to_float32()]
            flush = False
            while not flush:  # batch whatever else is queued into one worker call
                try:
                    nxt = self._input.recv_nowait()
                except (asyncio.QueueEmpty, ChanClosed):
                    break
                if self.is_flush(nxt):
                    flush = True
                else:
                    assert isinstance(nxt, AudioFrame)
                    chunks.append(nxt.to_float32())
            samples = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            self._update(await stt._run(self._accept_sync, samples))
            if flush:
                self._finish(await stt._run(self._finalize_sync))

    def _update(self, snap: _Snapshot) -> None:
        if snap.text and not self._speaking:
            self._speaking = True
            self._emit(STTEvent(STTEventType.START_OF_SPEECH, segment_id=self._segment_id))
        if snap.endpoint:
            if snap.text:
                self._finish(snap)
            else:
                self._next_segment()
            return
        if snap.text and snap.text != self._partial and self._sherpa.capabilities.interim_results:
            self._partial = snap.text
            self._emit(
                STTEvent(STTEventType.INTERIM_TRANSCRIPT, self._transcript(snap), self._segment_id)
            )

    def _finish(self, snap: _Snapshot) -> None:
        """Final transcript (even an empty one: the cascade waits for it after a flush)."""
        segment = self._segment_id
        transcript = self._transcript(snap)
        self._emit(STTEvent(STTEventType.FINAL_TRANSCRIPT, transcript, segment))
        if self._speaking:
            end = Transcript(text="", start_time=snap.start, end_time=snap.end)
            self._emit(STTEvent(STTEventType.END_OF_SPEECH, end, segment))
        self._next_segment()

    def _next_segment(self) -> None:
        self._segment_id = new_id("seg_")
        self._partial = ""
        self._speaking = False

    def _transcript(self, snap: _Snapshot) -> Transcript:
        return Transcript(
            text=snap.text,
            language=snap.language or self._language or self._sherpa.language,
            confidence=snap.confidence,
            start_time=snap.start,
            end_time=snap.end,
            words=snap.words,
        )

    # ------------------------------------------------------------ worker thread
    def _accept_sync(self, samples: npt.NDArray[np.float32]) -> _Snapshot:
        stt = self._sherpa
        recognizer = stt._ensure_recognizer()
        try:
            if self._stream is None:
                self._stream = stt._new_stream(self._language)
            stream = self._stream
            stream.accept_waveform(_SAMPLE_RATE, samples)
            self._fed += len(samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            endpoint = stt.endpoint_detection and bool(recognizer.is_endpoint(stream))
            snap = stt._snapshot(
                recognizer.get_result_all(stream),
                self._offset / _SAMPLE_RATE,
                self._fed,
                endpoint=endpoint,
            )
            if endpoint:
                recognizer.reset(stream)  # same stream: result times stay stream-relative
            return snap
        except Exception as exc:
            raise _sherpa_error(exc, "streaming recognition") from exc

    def _finalize_sync(self) -> _Snapshot:
        """Decode the tail of the current utterance and start a new sherpa stream."""
        stt = self._sherpa
        recognizer = stt._ensure_recognizer()
        stream, self._stream = self._stream, None
        fed = self._fed
        offset = self._offset / _SAMPLE_RATE
        self._offset += fed
        self._fed = 0
        if stream is None or fed == 0:
            return _Snapshot("")
        try:
            pad = stt._padding(fed)
            if pad:
                stream.accept_waveform(_SAMPLE_RATE, np.zeros(pad, dtype=np.float32))
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            return stt._snapshot(recognizer.get_result_all(stream), offset, fed)
        except Exception as exc:
            raise _sherpa_error(exc, "finalizing a transcript") from exc


# ----------------------------------------------------------------------------- TTS
@register_provider(
    "tts",
    "sherpa-onnx",
    description="sherpa-onnx local TTS: Piper/VITS, Kokoro and Matcha voices (CPU, every OS)",
    default_model=DEFAULT_TTS_MODEL,
    models=_task_models("tts"),
    env=(),
    extra=_EXTRA,
    requires=("sherpa_onnx",),
    local=True,
    aliases=("sherpa",),
)
class SherpaOnnxTTS(TTS):
    """Local speech synthesis with sherpa-onnx ``OfflineTts`` (Piper/VITS, Kokoro, Matcha).

    Args:
        model: a catalog name (``"piper-en_US-libritts_r-medium"``, ``"kokoro-multi-lang-v1_0"``,
            ``"matcha-en_US-ljspeech"``...) or its release archive name, a local model
            directory, or an archive URL.
        voice: default speaker: an id (``"12"``) or, for Kokoro, a voice name
            (``"af_heart"``, ``"bm_george"``...).
        speed: speaking rate (1.0 = model default; > 1 is faster).
        kind: ``"tts-vits"``, ``"tts-kokoro"`` or ``"tts-matcha"`` for directories/URLs
            outside the catalog (guessed from the files when omitted).
        files: explicit files by role (``model``, ``tokens``, ``data_dir``, ``voices``,
            ``lexicon``, ``acoustic_model``, ``vocoder``, ``rule_fsts``).
        sha256: expected digest when ``model`` is a URL.
        sample_rate: output rate. Default: the model's (catalog, Piper ``.onnx.json``, or the
            kind's usual rate); audio at another rate is resampled.
        lang: Kokoro language hint for text without a lexicon entry (``"es"``, ``"fr"``...).
        num_threads: ONNX Runtime threads.
        execution_provider: ``"cpu"`` (default), ``"cuda"`` or ``"coreml"``.
        silence_scale: scales the pauses sherpa inserts between sentences of one request.
        max_num_sentences: sentences per synthesis step; each step's audio is emitted as
            soon as it is ready.
        chunk_duration: duration of the emitted audio chunks (s).
        clean_text: strip markdown/emoji before synthesis.
        trim_silence: trim per-sentence leading/trailing silence when streaming.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        speed: float = 1.0,
        kind: str | None = None,
        files: Mapping[str, str | os.PathLike[str]] | None = None,
        sha256: str | None = None,
        sample_rate: int | None = None,
        lang: str | None = None,
        num_threads: int = 2,
        execution_provider: str = "cpu",
        silence_scale: float = 0.2,
        max_num_sentences: int = 1,
        chunk_duration: float = 0.05,
        clean_text: bool = True,
        trim_silence: bool = True,
    ) -> None:
        _check_installed()
        if not 0.25 <= speed <= 4.0:
            raise ConfigurationError(f"sherpa-onnx: speed must be between 0.25 and 4, got {speed}")
        if num_threads < 1:
            raise ConfigurationError(f"sherpa-onnx: num_threads must be >= 1, got {num_threads}")
        if max_num_sentences < 1:
            raise ConfigurationError("sherpa-onnx: max_num_sentences must be >= 1")
        if chunk_duration <= 0:
            raise ConfigurationError(
                f"sherpa-onnx: chunk_duration must be > 0, got {chunk_duration}"
            )
        ref = _model_ref("tts", model or DEFAULT_TTS_MODEL, kind=kind, files=files, sha256=sha256)
        spec = ref.spec
        rate = sample_rate or (spec.sample_rate if spec is not None else None)
        if rate is None:
            rate = _piper_sample_rate(Path(ref.source)) or _TTS_DEFAULT_RATES[ref.kind]
        super().__init__(
            model=ref.label,
            sample_rate=rate,
            voice=voice or (spec.default_voice if spec is not None else None) or "0",
            clean_text=clean_text,
            trim_silence=trim_silence,
        )
        self.kind = ref.kind
        self.speed = float(speed)
        self.lang = lang
        self.num_threads = num_threads
        self.execution_provider = execution_provider
        self.silence_scale = silence_scale
        self.max_num_sentences = max_num_sentences
        self.chunk_duration = chunk_duration
        self.speakers = spec.speakers if spec is not None else ()
        """Voice names by speaker id (catalog Kokoro models)."""
        self.num_speakers: int | None = None
        """Speakers of the loaded model (``None`` until it is loaded)."""
        self.model_sample_rate: int | None = None
        """Native rate of the loaded model (``None`` until it is loaded)."""
        self._ref = ref
        self._worker = _Worker("sherpa-onnx-tts")
        self._tts: Any = None
        self.speaker_id(self.voice)  # validates the default voice early

    def speaker_id(self, voice: str | None) -> int:
        """Numeric speaker id for a voice name or id (``None``: the default voice)."""
        value = (voice or self.voice or "0").strip()
        if value.isdigit():
            sid = int(value)
        elif value in self.speakers:
            sid = self.speakers.index(value)
        else:
            known = f"; voices: {', '.join(self.speakers)}" if self.speakers else " (use an id)"
            raise ConfigurationError(f"unknown sherpa-onnx voice {value!r} for {self.model}{known}")
        if self.num_speakers is not None and sid >= self.num_speakers:
            raise ConfigurationError(
                f"sherpa-onnx voice {value!r}: {self.model} has {self.num_speakers} speaker(s)"
            )
        return sid

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        return _SherpaChunkedStream(self, text, voice=voice)

    async def warmup(self) -> None:
        """Download (first run only) and load the model, then synthesize a short text."""
        await self._worker.run(self._warmup_sync)

    async def aclose(self) -> None:
        """Stop the worker thread (a synthesis already running finishes in background)."""
        self._worker.close()
        self._tts = None

    # ------------------------------------------------------------ worker thread
    def _ensure_tts(self) -> Any:
        if self._tts is None:
            t0 = now()
            files = _materialize(self._ref)
            so = _import_sherpa()
            config = self._config(so, files)
            if not config.validate():
                raise ConfigurationError(
                    f"sherpa-onnx rejected the TTS configuration for {self.model} "
                    "(see the log above for the reason)"
                )
            try:
                tts = so.OfflineTts(config)
            except Exception as exc:
                raise _sherpa_error(exc, f"loading {self.model}") from exc
            self.num_speakers = int(tts.num_speakers)
            self.model_sample_rate = int(tts.sample_rate)
            self._tts = tts
            logger.info(
                "sherpa-onnx: loaded %s (%s, %d Hz, %d speakers) in %.2f s",
                self.model, self.kind, self.model_sample_rate, self.num_speakers, now() - t0,
            )  # fmt: skip
        return self._tts

    def _config(self, so: Any, files: Mapping[str, Path]) -> Any:
        f = {role: _joined(files, role) for role in (*_REQUIRED_FILES[self.kind], *_OPTIONAL_FILES)}
        model_config: dict[str, Any] = {
            "num_threads": self.num_threads,
            "provider": self.execution_provider,
        }
        if self.kind == "tts-vits":
            model_config["vits"] = so.OfflineTtsVitsModelConfig(
                model=f["model"], tokens=f["tokens"], data_dir=f["data_dir"], lexicon=f["lexicon"]
            )
        elif self.kind == "tts-kokoro":
            model_config["kokoro"] = so.OfflineTtsKokoroModelConfig(
                model=f["model"],
                voices=f["voices"],
                tokens=f["tokens"],
                data_dir=f["data_dir"],
                lexicon=f["lexicon"],
                lang=self.lang or "",
            )
        else:
            model_config["matcha"] = so.OfflineTtsMatchaModelConfig(
                acoustic_model=f["acoustic_model"],
                vocoder=f["vocoder"],
                tokens=f["tokens"],
                data_dir=f["data_dir"],
                lexicon=f["lexicon"],
            )
        return so.OfflineTtsConfig(
            model=so.OfflineTtsModelConfig(**model_config),
            rule_fsts=f["rule_fsts"],
            max_num_sentences=self.max_num_sentences,
            silence_scale=self.silence_scale,
        )

    def _warmup_sync(self) -> None:
        t0 = now()
        self._generate_sync(
            "Hello.", self.speaker_id(None), lambda samples: None, threading.Event()
        )
        logger.debug("sherpa-onnx: TTS warm-up took %.2f s", now() - t0)

    def _generate_sync(
        self,
        text: str,
        sid: int,
        on_audio: Callable[[bytes], None],
        stop: threading.Event,
    ) -> None:
        """Synthesize ``text``, handing s16le chunks at :attr:`sample_rate` to ``on_audio``."""
        tts = self._ensure_tts()
        sid = self.speaker_id(str(sid))
        so = _import_sherpa()
        config = so.GenerationConfig()
        config.sid = sid
        config.speed = self.speed
        config.silence_scale = self.silence_scale
        native = int(tts.sample_rate)
        resampler = StreamResampler(self.sample_rate, 1) if native != self.sample_rate else None

        def convert(samples: npt.NDArray[np.float32]) -> bytes:
            frame = AudioFrame.from_numpy(np.asarray(samples, dtype=np.float32), native)
            if resampler is not None:
                frame = resampler.push(frame)
            return frame.data

        def callback(samples: npt.NDArray[np.float32], progress: float) -> int:
            if stop.is_set():
                return 0  # 0 stops the generation, 1 continues
            data = convert(samples)  # the buffer is only valid during the callback
            if data:
                on_audio(data)
            return 1

        try:
            tts.generate(text, config, callback=callback)
        except Exception as exc:
            raise _sherpa_error(exc, "synthesis") from exc
        if resampler is not None and not stop.is_set():
            tail = resampler.flush()
            if tail:
                on_audio(tail.data)


def _piper_sample_rate(path: Path) -> int | None:
    """Output rate from a Piper voice's ``*.onnx.json`` next to the model (local dirs)."""
    if not path.is_dir():
        return None
    for config in sorted(path.glob("*.onnx.json")):
        try:
            rate = json.loads(config.read_text(encoding="utf-8"))["audio"]["sample_rate"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(rate, int) and rate > 0:
            return rate
    return None


class _SherpaChunkedStream(ChunkedStream):
    async def _run(self) -> None:
        tts = self._tts
        assert isinstance(tts, SherpaOnnxTTS)
        text = self.text.strip()
        if not any(ch.isalnum() for ch in text):
            return  # nothing to pronounce (sherpa would still return padding)
        sid = tts.speaker_id(self.voice)
        loop = asyncio.get_running_loop()
        chunks: Chan[bytes] = Chan()
        stop = threading.Event()

        def on_audio(data: bytes) -> None:
            _call_soon(loop, _send_if_open, chunks, data)

        def generate() -> None:
            try:
                tts._generate_sync(text, sid, on_audio, stop)
            finally:
                _call_soon(loop, chunks.close)

        step = max(1, round(tts.chunk_duration * tts.sample_rate)) * SAMPLE_WIDTH
        job = asyncio.ensure_future(tts._worker.run(generate))
        try:
            async for data in chunks:
                for start in range(0, len(data), step):
                    self._push_audio(data[start : start + step])
            await job
        finally:
            stop.set()  # a cancelled request stops at the next sentence
            if not job.done():
                job.cancel()


def _send_if_open(chan: Chan[bytes], data: bytes) -> None:
    if not chan.closed:
        chan.send_nowait(data)


def _call_soon(loop: asyncio.AbstractEventLoop, fn: Callable[..., Any], *args: Any) -> None:
    """``call_soon_threadsafe`` that tolerates a loop closed meanwhile (shutdown)."""
    try:
        loop.call_soon_threadsafe(fn, *args)
    except RuntimeError:
        pass


# ----------------------------------------------------------------------------- VAD
class _SherpaVADInference:
    """Per-stream sherpa ``VadModel`` (it keeps the recurrent state and hysteresis)."""

    def __init__(self, model: Any, window: int) -> None:
        self._model = model
        self._window = window

    def __call__(self, window: npt.NDArray[np.float32]) -> float:
        if window.shape != (self._window,):
            raise ValueError(f"expected {self._window} samples per window, got {window.shape}")
        return 1.0 if self._model.is_speech(window) else 0.0

    def reset(self) -> None:
        self._model.reset()


_VAD_WINDOWS = {"vad-silero": 512, "vad-ten": 256}


@register_provider(
    "vad",
    "sherpa-onnx",
    description="Silero / TEN VAD on sherpa-onnx's bundled ONNX Runtime (16 kHz, CPU)",
    default_model=DEFAULT_VAD_MODEL,
    models=_task_models("vad"),
    env=(),
    extra=_EXTRA,
    requires=("sherpa_onnx",),
    local=True,
    aliases=("sherpa",),
)
class SherpaOnnxVAD(VAD):
    """Silero or TEN VAD running inside sherpa-onnx (no ``onnxruntime`` package needed).

    sherpa's VAD reports speech / non-speech per window, so the "probability" seen by
    :class:`~voice_agent_next.vad.VADStream` is 1.0 or 0.0: ``activation_threshold`` is
    applied inside sherpa (with its fixed 0.15 hysteresis) and ``smoothing`` has no
    useful effect. Minimum speech/silence durations and padding work as usual.

    Args:
        model: ``"silero"`` (default, 32 ms windows) or ``"ten-vad"`` (16 ms windows), or a
            path to a local Silero/TEN ONNX file (then ``kind`` selects the family).
        kind: ``"vad-silero"`` or ``"vad-ten"`` for local files.
        num_threads: ONNX Runtime threads per stream.
        options: thresholds and durations (:class:`~voice_agent_next.vad.VADOptions`).
        **option_overrides: individual ``VADOptions`` fields, e.g. ``min_silence_duration=0.3``.
    """

    provider = _PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        kind: str | None = None,
        num_threads: int = 1,
        options: VADOptions | None = None,
        **option_overrides: float,
    ) -> None:
        _check_installed()
        ref = _model_ref("vad", model or DEFAULT_VAD_MODEL, kind=kind, files=None, sha256=None)
        opts = options or VADOptions()
        if option_overrides:
            known = {f.name for f in dataclasses.fields(VADOptions)}
            unknown = sorted(set(option_overrides) - known)
            if unknown:
                raise ConfigurationError(f"unknown VAD option(s) {unknown}; valid: {sorted(known)}")
            opts = dataclasses.replace(opts, **option_overrides)
        super().__init__(
            sample_rate=_SAMPLE_RATE,
            window_samples=_VAD_WINDOWS[ref.kind],
            options=opts,
            model=ref.label,
        )
        self.kind = ref.kind
        self.num_threads = num_threads
        self._ref = ref
        self._path: Path | None = None
        self._lock = threading.Lock()

    def _new_inference(self) -> _SherpaVADInference:
        return _SherpaVADInference(self._create_model(), self.window_samples)

    async def warmup(self) -> None:
        """Download (first run only) and load the model off the event loop, then run it once."""
        await asyncio.to_thread(self._warmup)

    def _warmup(self) -> None:
        self._new_inference()(np.zeros(self.window_samples, dtype=np.float32))

    def _model_path(self) -> Path:
        with self._lock:
            if self._path is None:
                self._path = _materialize(self._ref)["model"]
            return self._path

    def _create_model(self) -> Any:
        """A new sherpa ``VadModel`` (downloads the model on first use: blocking)."""
        path = self._model_path()
        so = _import_sherpa()
        family = {
            "model": str(path),
            "threshold": self.options.activation_threshold,
            # durations are handled by VADStream; sherpa needs positive values
            "min_silence_duration": 0.001,
            "min_speech_duration": 0.001,
            "window_size": self.window_samples,
            "max_speech_duration": 1e6,
        }
        if self.kind == "vad-silero":
            config = so.VadModelConfig(
                silero_vad=so.SileroVadModelConfig(**family),
                sample_rate=_SAMPLE_RATE,
                num_threads=self.num_threads,
            )
        else:
            config = so.VadModelConfig(
                ten_vad=so.TenVadModelConfig(**family),
                sample_rate=_SAMPLE_RATE,
                num_threads=self.num_threads,
            )
        if not config.validate():
            raise ConfigurationError(f"sherpa-onnx rejected the VAD configuration for {path}")
        try:
            return so.VadModel.create(config)
        except Exception as exc:
            raise _sherpa_error(exc, f"loading VAD model {path}") from exc
