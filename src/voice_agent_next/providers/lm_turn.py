"""Text end-of-turn detector from a small language model (local ONNX, no fine-tuning).

The detector writes the conversation in the model's chat format, the agent's last turn
followed by the user's transcript so far, **without** closing the user message, and reads
the probability that the model's next token is the end-of-message token
(``<|im_end|>``). A model that has seen many conversations assigns it a high probability
after "…for two people tonight." and almost none after "I would like to book a table," or
"my number is five five" — the idea behind TurnGPT and LiveKit's first (Qwen-based) text
detector, here with an off-the-shelf instruct model and a calibration fitted on eot-bench.

That raw probability is poorly calibrated (it is spread over many orders of magnitude),
so the detector maps it through ``sigmoid(a * ln(p) + b)``, fitted per model on LiveKit's
eot-bench (English, labels *eot* vs *hold*). Fuse it with an audio detector
(:class:`~voice_agent_next.turn.FusedTurnDetector`): on its own it is weaker than Smart
Turn (ROC-AUC 0.74 vs 0.83 on eot-bench English), fused it is better than either (0.88).

Models (Apache-2.0, int8 ONNX exported by Hugging Face for transformers.js, pinned):

* ``smollm2-135m`` (default): SmolLM2-135M-Instruct, 137 MB, ~10–30 ms per prediction on
  one CPU core;
* ``smollm2-360m``: SmolLM2-360M-Instruct, 365 MB, about 3x slower, no better fused.

Both are English-first: see ``docs/providers/lm-turn.md`` for the other languages.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.frame import AudioFrame
from ..chat import ChatContext
from ..errors import ConfigurationError, ProviderError
from ..models import ModelFile, register_model
from ..registry import register_provider
from ..turn import TurnDetector, turn_text
from ..utils.clock import now
from ..utils.deps import require
from ..utils.download import hf_file
from ..utils.log import logger

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "LMTurnDetector",
    "LMTurnModel",
    "format_prompt",
    "is_punctuated",
    "unpunctuated",
]


@dataclass(frozen=True)
class LMTurnModel:
    """A causal LM usable by :class:`LMTurnDetector` (ChatML chat format)."""

    repo: str
    revision: str
    onnx_file: str
    onnx_sha256: str
    onnx_size: int
    tokenizer_sha256: str
    tokenizer_size: int
    calibration: tuple[float, float]
    """``(a, b)`` of ``sigmoid(a * ln(p_end) + b)``, fitted on eot-bench English."""
    calibration_unpunctuated: tuple[float, float]
    """The same, fitted on eot-bench English lowercased and without punctuation: for STTs
    that don't punctuate (the model's end-of-message probability is much lower there)."""
    languages: str = "en"


MODELS: dict[str, LMTurnModel] = {
    "smollm2-135m": LMTurnModel(
        repo="HuggingFaceTB/SmolLM2-135M-Instruct",
        revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        onnx_file="onnx/model_int8.onnx",
        onnx_sha256="a7c33f9ef85d06734cc9d1f943f7e2ba57c77e769df727f4d3e217d9f672b0cc",
        onnx_size=137_147_867,
        tokenizer_sha256="9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        tokenizer_size=2_104_556,
        calibration=(0.2172, 1.3627),
        calibration_unpunctuated=(0.1733, 1.2938),
    ),
    "smollm2-360m": LMTurnModel(
        repo="HuggingFaceTB/SmolLM2-360M-Instruct",
        revision="a10cc1512eabd3dde888204e902eca88bddb4951",
        onnx_file="onnx/model_int8.onnx",
        onnx_sha256="cb7375a212a6583cbd96eaa739142f3a1a0d17cbd51043c60f081ea5ce6d21e3",
        onnx_size=364_564_558,
        tokenizer_sha256="9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        tokenizer_size=2_104_556,
        calibration=(0.1846, 0.7728),
        calibration_unpunctuated=(0.1471, 0.5295),
    ),
}
DEFAULT_MODEL = "smollm2-135m"
END_OF_MESSAGE = "<|im_end|>"
_PUNCTUATION = re.compile(r"[.?!,;:\u3002\uff1f\uff01\u3001\uff0c]")
_NOT_WORD = re.compile(r"[^\w\s']+")


def is_punctuated(text: str) -> bool:
    """The transcript has sentence punctuation (the STT punctuates)."""
    return _PUNCTUATION.search(text) is not None


def unpunctuated(text: str) -> str:
    """``text`` lowercased, without punctuation: how the unpunctuated calibration saw it."""
    return " ".join(_NOT_WORD.sub(" ", text.lower()).split())


def format_prompt(agent: str, user: str, *, max_agent_chars: int = 300) -> str:
    """The ChatML text whose next token is scored: the agent's last turn (its end, at most
    ``max_agent_chars``) and the user's open message."""
    out = ""
    agent = agent.strip()
    if agent:
        if len(agent) > max_agent_chars:
            agent = agent[len(agent) - max_agent_chars :].lstrip()
        out += f"<|im_start|>assistant\n{agent}{END_OF_MESSAGE}\n"
    return out + f"<|im_start|>user\n{user.strip()}"


@register_provider(
    "turn",
    "lm_turn",
    description="Text end-of-turn from a small LM's end-of-message probability (local ONNX)",
    default_model=DEFAULT_MODEL,
    models=tuple(MODELS),
    env=(),
    extra="text-turn",
    requires=("onnxruntime", "tokenizers"),
    local=True,
)
class LMTurnDetector(TurnDetector):
    """Text end-of-turn detector: a small causal LM's probability that the user's message
    ends here, calibrated (see the module docstring).

    Input: the last user message of ``chat_ctx`` (the transcript so far) and the agent
    message before it. Without a user transcript it returns ``1.0`` (nothing to judge).

    Args:
        model: ``"smollm2-135m"`` (default) or ``"smollm2-360m"``.
        model_path: a local ChatML causal-LM ``.onnx`` file (transformers.js export layout:
            ``input_ids``, ``attention_mask``, optional ``position_ids`` and
            ``past_key_values.*``; ``logits`` output) used instead of downloading
            ``model``; needs ``tokenizer_path`` too.
        tokenizer_path: its ``tokenizer.json``.
        calibration: ``(a, b)`` of ``sigmoid(a * ln(p) + b)`` for punctuated transcripts;
            ``None``: the model's fitted values (with ``model_path``: the raw probability).
        calibration_unpunctuated: the same for transcripts without any punctuation (STTs
            that don't punctuate, e.g. sherpa-onnx NeMo), which are lowercased first;
            ``None``: the model's fitted values (with ``model_path``: ``calibration``).
        threshold: calibrated probability at/above which the turn counts as complete.
        max_tokens: the prompt is cut to its last ``max_tokens`` tokens.
        num_threads: ONNX Runtime intra-op threads (1 keeps a voice agent's CPU usage
            predictable; the 135M model takes ~10-30 ms per prediction on one core).
    """

    provider = "lm_turn"
    modality = "text"

    def __init__(
        self,
        *,
        model: str | None = None,
        model_path: str | os.PathLike[str] | None = None,
        tokenizer_path: str | os.PathLike[str] | None = None,
        calibration: tuple[float, float] | None = None,
        calibration_unpunctuated: tuple[float, float] | None = None,
        threshold: float = 0.5,
        max_tokens: int = 128,
        num_threads: int = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError(f"threshold must be in [0, 1], got {threshold}")
        if max_tokens < 8:
            raise ConfigurationError(f"max_tokens must be >= 8, got {max_tokens}")
        if num_threads < 1:
            raise ConfigurationError(f"num_threads must be >= 1, got {num_threads}")
        self.spec: LMTurnModel | None = None
        if model_path is not None:
            if tokenizer_path is None:
                raise ConfigurationError("lm_turn: model_path needs tokenizer_path too")
            name = Path(model_path).stem
        else:
            name = (model or DEFAULT_MODEL).strip().lower()
            if name not in MODELS:
                raise ConfigurationError(
                    f"unknown lm_turn model {model!r}; available: {', '.join(MODELS)}"
                )
            self.spec = MODELS[name]
        super().__init__(model=name, threshold=threshold)
        self.model_path = Path(model_path) if model_path is not None else None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else None
        self.calibration = calibration or (self.spec.calibration if self.spec else None)
        self.calibration_unpunctuated = calibration_unpunctuated or (
            self.spec.calibration_unpunctuated if self.spec else self.calibration
        )
        self.max_tokens = max_tokens
        self.num_threads = num_threads
        self._session: Any = None
        self._tokenizer: Any = None
        self._end_id = 0
        self._inputs: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    def _files(self) -> tuple[Path, Path]:
        if self.spec is None:
            assert self.model_path is not None
            assert self.tokenizer_path is not None
            for path in (self.model_path, self.tokenizer_path):
                if not path.is_file():
                    raise ConfigurationError(f"lm_turn file not found: {path}")
            return self.model_path, self.tokenizer_path
        s = self.spec
        onnx = hf_file(s.repo, s.onnx_file, revision=s.revision, sha256=s.onnx_sha256)
        tok = hf_file(s.repo, "tokenizer.json", revision=s.revision, sha256=s.tokenizer_sha256)
        return onnx, tok

    def _load(self) -> None:
        ort = require("onnxruntime", extra="text-turn")
        tokenizers = require("tokenizers", extra="text-turn")
        onnx_path, tok_path = self._files()
        tokenizer = tokenizers.Tokenizer.from_file(os.fspath(tok_path))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        end_id = tokenizer.token_to_id(END_OF_MESSAGE)
        if end_id is None:
            raise ConfigurationError(f"{tok_path} has no {END_OF_MESSAGE} token (not ChatML)")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        t0 = now()
        try:
            session = ort.InferenceSession(
                os.fspath(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # onnxruntime errors derive from Exception only
            raise ProviderError(
                f"failed to load lm_turn model {onnx_path}: {exc}", provider=self.provider
            ) from exc
        self._inputs = {i.name: i for i in session.get_inputs()}
        if "input_ids" not in self._inputs:
            raise ConfigurationError(f"{onnx_path} has no input_ids input")
        self._tokenizer, self._end_id, self._session = tokenizer, end_id, session
        logger.debug("loaded lm_turn model %s in %.1f ms", onnx_path, (now() - t0) * 1e3)

    def _ensure_loaded(self) -> None:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._load()

    def _feed(self, ids: list[int]) -> dict[str, Any]:
        n = len(ids)
        feed: dict[str, Any] = {"input_ids": np.asarray([ids], dtype=np.int64)}
        if "attention_mask" in self._inputs:
            feed["attention_mask"] = np.ones((1, n), dtype=np.int64)
        if "position_ids" in self._inputs:
            feed["position_ids"] = np.arange(n, dtype=np.int64)[np.newaxis]
        for name, info in self._inputs.items():
            if name.startswith("past_key_values"):
                # an empty cache: (batch, kv heads, 0, head dim)
                heads, dim = info.shape[1], info.shape[3]
                if not isinstance(heads, int) or not isinstance(dim, int):
                    raise ConfigurationError(f"lm_turn: cannot size the cache input {name}")
                dtype = np.float16 if "float16" in info.type else np.float32
                feed[name] = np.zeros((1, heads, 0, dim), dtype=dtype)
        return feed

    def end_probability(self, agent: str, user: str) -> float:
        """Blocking: the raw probability that the user's message ends after ``user``."""
        self._ensure_loaded()
        ids = self._tokenizer.encode(format_prompt(agent, user), add_special_tokens=False).ids
        ids = ids[-self.max_tokens :]
        try:
            logits = self._session.run(["logits"], self._feed(ids))[0]
        except Exception as exc:
            raise ProviderError(f"lm_turn inference failed: {exc}", provider=self.provider) from exc
        last = np.asarray(logits[0, -1], dtype=np.float64)
        last -= last.max()
        return float(math.exp(last[self._end_id]) / np.exp(last).sum())

    def calibrate(self, p_end: float, *, punctuated: bool = True) -> float:
        """The calibrated end-of-turn probability of a raw end-of-message probability."""
        calibration = self.calibration if punctuated else self.calibration_unpunctuated
        if calibration is None:
            return p_end
        a, b = calibration
        x = a * math.log(max(p_end, 1e-12)) + b
        return 1.0 / (1.0 + math.exp(-x))

    def _infer(self, agent: str, user: str) -> float:
        punctuated = is_punctuated(user)
        if not punctuated:
            user = unpunctuated(user)
        return self.calibrate(self.end_probability(agent, user), punctuated=punctuated)

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        agent, user = turn_text(chat_ctx)
        if not user:
            return 1.0
        return await asyncio.to_thread(self._infer, agent, user)

    async def warmup(self) -> None:
        """Download (first run only) and load the model, then run one prediction."""
        await asyncio.to_thread(self._infer, "How can I help?", "Hello.")

    async def aclose(self) -> None:
        self._session = None


for _name, _m in MODELS.items():
    register_model(
        "lm_turn",
        _name,
        kind="turn",
        files=[
            ModelFile.from_hf(
                _m.repo,
                _m.onnx_file,
                revision=_m.revision,
                sha256=_m.onnx_sha256,
                size=_m.onnx_size,
            ),
            ModelFile.from_hf(
                _m.repo,
                "tokenizer.json",
                revision=_m.revision,
                sha256=_m.tokenizer_sha256,
                size=_m.tokenizer_size,
            ),
        ],
        license="Apache-2.0",
        languages=_m.languages,
        description=f"{_m.repo.split('/')[-1]} int8 ONNX, text end-of-turn (end-of-message "
        "probability)",
    )
