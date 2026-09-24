"""Alibaba Qwen3-TTS: local, audio-streaming text-to-speech with voice cloning (Apache-2.0).

Qwen3-TTS models generate 12.5 Hz speech codes (16 codebooks per 80 ms frame) that a
causal codec decodes to 24 kHz audio. The ``qwen-tts`` package renders a whole text before
returning; this provider streams instead: it collects the codes as the model generates
them and decodes them in small chunks (with left context, so the chunks join seamlessly),
so the first audio of a sentence is ready after a few frames.

Models (``model=``; a Hugging Face repo id or a local directory work too):

* ``0.6b-custom`` (default), ``1.7b-custom``: CustomVoice, nine built-in speakers
  (:data:`SPEAKERS`, default ``ryan``) and optional style instructions (``instruct=``).
* ``0.6b``, ``1.7b``: Base, zero-shot voice cloning from 3+ s of reference audio
  (``voice="me.wav"``; with ``ref_text=`` its transcript for better similarity).
* ``1.7b-design``: VoiceDesign, the voice is described in words (``voice="A warm, low
  female voice"``).

Ten languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese,
Spanish and Italian (``language="en"``...; default: detected from the text).

Usage::

    from voice_agent_next import create

    tts = create("tts", "qwen-tts")                               # 0.6B CustomVoice, ryan
    tts = create("tts", "qwen-tts/1.7b-custom", voice="vivian", instruct="Speak cheerfully")
    tts = create("tts", "qwen-tts/0.6b", voice="me.wav", ref_text="What I say in me.wav.")
    await tts.warmup()

There is no text alignment: word timings are estimates spread over each sentence's speech
span, as for the other local models.
"""

from __future__ import annotations

import os
import queue
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ConfigurationError, ProviderError
from ..registry import register_provider
from ..utils.clock import now
from ..utils.deps import is_installed, require
from ..utils.log import logger
from ._torch_tts import LocalTorchTTS, Stopped

__all__ = ["LANGUAGES", "MODELS", "SAMPLE_RATE", "SPEAKERS", "QwenTTS", "resolve_model"]

SAMPLE_RATE = 24_000
_EXTRA = "qwen-tts"
MODELS = {
    "0.6b-custom": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7b-custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "1.7b-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
"""Short model names and their Hugging Face repositories."""
_ALIASES = {
    "0.6b-base": "0.6b",
    "1.7b-base": "1.7b",
    "0.6b-customvoice": "0.6b-custom",
    "1.7b-customvoice": "1.7b-custom",
    "1.7b-voicedesign": "1.7b-design",
}
SPEAKERS = (
    "ryan", "aiden", "vivian", "serena", "uncle_fu", "dylan", "eric", "ono_anna", "sohee",
)  # fmt: skip
"""Built-in speakers of the CustomVoice models (ryan and aiden are native English)."""
DEFAULT_SPEAKER = "ryan"
LANGUAGES = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}
"""ISO 639-1 codes and the language names Qwen3-TTS uses."""
_FRAME_SAMPLES = 1920  # 24 kHz / 12.5 Hz
_VOICE_CACHE_SIZE = 8


def resolve_model(model: str | None) -> str:
    """The Hugging Face repo id (or local directory) of a model name."""
    name = (model or "0.6b-custom").strip()
    key = _ALIASES.get(name.lower(), name.lower())
    if key in MODELS:
        return MODELS[key]
    if "/" in name or Path(name).expanduser().is_dir():
        return name
    raise ConfigurationError(
        f"unknown Qwen3-TTS model {model!r}; known: {', '.join(MODELS)} "
        "(or a Hugging Face repo id / local directory)"
    )


def _check_installed() -> None:
    """Fail fast when the extra is missing, without importing torch."""
    for module, package in (("qwen_tts", "qwen-tts"), ("torch", "torch")):
        if not is_installed(module):
            require(module, extra=_EXTRA, package=package)


def _language(value: str | None) -> str:
    if value is None or value.strip().lower() == "auto":
        return "auto"
    key = value.strip().lower()
    key = LANGUAGES.get(key, key)
    if key not in LANGUAGES.values():
        known = ", ".join(sorted(LANGUAGES))
        raise ConfigurationError(f"unknown Qwen3-TTS language {value!r}; known: {known}, auto")
    return key


@register_provider(
    "tts",
    "qwen-tts",
    description="Alibaba Qwen3-TTS 0.6B/1.7B: local audio-streaming TTS with voice cloning",
    default_model="0.6b-custom",
    models=tuple(MODELS),
    env=(),
    extra=_EXTRA,
    requires=("qwen_tts", "torch"),
    local=True,
    aliases=("qwen3-tts",),
)
class QwenTTS(LocalTorchTTS):
    """Qwen3-TTS running locally with PyTorch (CUDA, Apple MPS or CPU; 24 kHz mono).

    Args:
        model: ``"0.6b-custom"`` (default), ``"1.7b-custom"``, ``"0.6b"``, ``"1.7b"``,
            ``"1.7b-design"``, a Hugging Face repo id or a local directory.
        voice: CustomVoice: a speaker (:data:`SPEAKERS`, default ``"ryan"``). Base: a
            reference audio file to clone (3+ s). VoiceDesign: a description of the voice.
        ref_text: transcript of the reference audio (Base). With it, the model continues
            the reference (better similarity); without it, only the speaker embedding is
            used.
        instruct: style instruction of CustomVoice models (``"Speak slowly and calmly"``).
        language: ``"en"``, ``"fr"``... or a name (``"english"``); default: auto-detect.
        device: ``"auto"`` (CUDA, then Apple MPS, then CPU), ``"cuda"``, ``"cuda:<n>"``,
            ``"mps"`` or ``"cpu"``.
        dtype: ``"auto"`` (bfloat16 on CUDA, float32 elsewhere), ``"bfloat16"``,
            ``"float16"`` or ``"float32"``.
        attn_implementation: ``"sdpa"`` (default) or ``"flash_attention_2"`` (needs the
            ``flash-attn`` package).
        first_chunk_frames: 80 ms frames decoded for the first audio chunk of a sentence
            (fewer: sooner first audio).
        chunk_frames: frames per following chunk (more: fewer, larger chunks).
        context_frames: frames of left context decoded again with every chunk, so chunk
            boundaries are seamless.
        temperature, top_k, top_p, repetition_penalty: sampling of the first codebook
            (default: the model's generation config).
        max_new_tokens: frame cap per sentence (default: from the text length).
        fast_code_predictor: generate the 15 residual codebooks of each frame with a
            plain sampling loop instead of Hugging Face ``generate`` (same sampling, several
            times faster: the difference between faster and slower than real time).
        split_sentences: synthesize long texts sentence by sentence.
        word_timings: attach estimated word timings to the audio.
        clean_text: strip markdown/emoji before synthesis.
    """

    provider = "qwen-tts"
    _thread_name = "qwen-tts"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | os.PathLike[str] | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        language: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
        first_chunk_frames: int = 2,
        chunk_frames: int = 6,
        context_frames: int = 25,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens: int | None = None,
        fast_code_predictor: bool = True,
        split_sentences: bool = True,
        word_timings: bool = True,
        clean_text: bool = True,
    ) -> None:
        repo = resolve_model(model)
        if dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ConfigurationError(f"unknown dtype {dtype!r}")
        for name, value in (
            ("first_chunk_frames", first_chunk_frames),
            ("chunk_frames", chunk_frames),
        ):
            if value < 1:
                raise ConfigurationError(f"{name} must be >= 1, got {value}")
        if context_frames < 0:
            raise ConfigurationError(f"context_frames must be >= 0, got {context_frames}")
        if temperature is not None and temperature <= 0:
            raise ConfigurationError(f"temperature must be > 0, got {temperature}")
        self.language = _language(language)
        _check_installed()
        super().__init__(
            model=repo,
            sample_rate=SAMPLE_RATE,
            voice=str(voice) if voice is not None else None,
            device=device,
            split_sentences=split_sentences,
            word_timings=word_timings,
            clean_text=clean_text,
        )
        self.ref_text = ref_text
        self.instruct = instruct
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.first_chunk_frames = first_chunk_frames
        self.chunk_frames = chunk_frames
        self.context_frames = context_frames
        self.max_new_tokens = max_new_tokens
        self.fast_code_predictor = fast_code_predictor
        self._sampling = {
            k: v
            for k, v in (
                ("temperature", temperature),
                ("top_k", top_k),
                ("top_p", top_p),
                ("repetition_penalty", repetition_penalty),
            )
            if v is not None
        }
        self.model_type: str | None = None
        """``"custom_voice"``, ``"base"`` or ``"voice_design"`` once loaded."""
        self._prompts: OrderedDict[str, Any] = OrderedDict()  # cloned voice -> prompt item

    async def load_voice(self, voice: str | os.PathLike[str]) -> None:
        """Encode a reference voice (Base models) ahead of its first use."""

        def run() -> None:
            self._clone_prompt(self._get_model(), str(voice))

        await self._submit(run)

    async def list_voices(self) -> list[str]:
        """Built-in speakers (CustomVoice models)."""
        return list(SPEAKERS)

    async def aclose(self) -> None:
        await super().aclose()
        self._prompts.clear()

    # ------------------------------------------------------------------ loading
    def _load(self, device: str) -> Any:
        qwen_tts = require("qwen_tts", extra=_EXTRA, package="qwen-tts")
        torch = require("torch", extra=_EXTRA)
        dtype = self.dtype
        if dtype == "auto":
            dtype = "bfloat16" if device.startswith("cuda") else "float32"
        try:
            tts = qwen_tts.Qwen3TTSModel.from_pretrained(
                self.model,
                device_map=device,
                dtype=getattr(torch, dtype),
                attn_implementation=self.attn_implementation,
            )
        except OSError as exc:  # unknown repo / missing files
            raise ConfigurationError(f"cannot load Qwen3-TTS model {self.model}: {exc}") from exc
        self.model_type = str(getattr(tts.model, "tts_model_type", "custom_voice"))
        predictor = getattr(getattr(tts.model, "talker", None), "code_predictor", None)
        if self.fast_code_predictor and predictor is not None:
            _FastCodePredictor(predictor, torch).install()
        if self.model_type == "base" and not self.voice:
            logger.warning(
                "qwen-tts: %s is a voice-cloning model: pass voice=<reference.wav>", self.model
            )
        self._prompts.clear()
        return tts

    def _warmup_sync(self) -> None:
        if self.model_type is None:
            self._get_model()
        if self.model_type == "base" and not self.voice:
            return  # nothing to say it with yet
        super()._warmup_sync()

    def _clone_prompt(self, model: Any, voice: str) -> Any:
        """The voice-clone prompt of a reference audio file (worker thread; cached)."""
        if self.model_type != "base":
            raise ConfigurationError(
                f"{self.model} does not clone voices; use a Base model (qwen-tts/0.6b)"
            )
        key = f"{voice}\0{self.ref_text or ''}"
        cached = self._prompts.get(key)
        if cached is not None:
            self._prompts.move_to_end(key)
            return cached
        path = Path(voice).expanduser()
        if not path.is_file():
            raise ConfigurationError(f"Qwen3-TTS reference audio {voice!r} does not exist")
        t0 = now()
        try:
            items = model.create_voice_clone_prompt(
                ref_audio=str(path),
                ref_text=self.ref_text,
                x_vector_only_mode=not self.ref_text,
            )
        except Exception as exc:
            raise ProviderError(
                f"cannot encode Qwen3-TTS voice {path.name}: {exc}", provider=self.provider
            ) from exc
        logger.info("qwen-tts: voice %s ready in %.2f s", path.name, now() - t0)
        self._prompts[key] = items[0]
        while len(self._prompts) > _VOICE_CACHE_SIZE:
            self._prompts.popitem(last=False)
        return items[0]

    # ------------------------------------------------------------------ synthesis
    def _call(self, model: Any, text: str, voice: str | None) -> tuple[Any, Any]:
        """``(generate function, keyword arguments)`` for this model type and voice, and
        the reference codes that precede the generated ones (voice cloning)."""
        kwargs: dict[str, Any] = dict(self._sampling)
        kwargs["max_new_tokens"] = self.max_new_tokens or 60 + 4 * len(text)
        kwargs["language"] = self.language
        if self.model_type == "base":
            if not voice:
                raise ConfigurationError(
                    f"{self.model} clones voices: pass voice=<reference audio file>"
                )
            prompt = self._clone_prompt(model, voice)
            kwargs.update(text=text, voice_clone_prompt=[prompt], non_streaming_mode=False)
            return (model.generate_voice_clone, kwargs), getattr(prompt, "ref_code", None)
        if self.model_type == "voice_design":
            if not voice:
                raise ConfigurationError(
                    f"{self.model} designs voices from a description: pass voice='<description>'"
                )
            kwargs.update(text=text, instruct=voice, non_streaming_mode=False)
            return (model.generate_voice_design, kwargs), None
        speaker = (voice or DEFAULT_SPEAKER).strip().lower()
        if Path(speaker).suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
            raise ConfigurationError(
                f"{self.model} has built-in speakers only; clone voices with a Base model "
                "(qwen-tts/0.6b)"
            )
        kwargs.update(text=text, speaker=speaker, non_streaming_mode=False)
        if self.instruct:
            kwargs["instruct"] = self.instruct
        return (model.generate_custom_voice, kwargs), None

    def _render(
        self, model: Any, text: str, voice: str | None, stop: threading.Event
    ) -> Iterable[np.ndarray]:
        (generate, kwargs), ref_code = self._call(model, text, voice)
        decoder = _StreamDecoder.of(model)
        if decoder is None:  # not a 12.5 Hz codec: no streaming
            wavs, _sr = generate(**kwargs)
            yield np.asarray(wavs[0], dtype=np.float32)
            return
        yield from decoder.stream(
            model,
            generate,
            kwargs,
            stop,
            ref_code=ref_code,
            first=self.first_chunk_frames,
            every=self.chunk_frames,
            context=self.context_frames,
        )


_END = object()


class _StreamDecoder:
    """Streams a Qwen3-TTS generation: a forward hook on the talker (the model that
    generates one 16-code frame per step) hands every frame to this thread, which decodes
    the frames in chunks with the causal 12.5 Hz codec while generation continues."""

    def __init__(self, tokenizer: Any, codec: Any, upsample: int) -> None:
        self.tokenizer = tokenizer  # qwen_tts.Qwen3TTSTokenizer
        self.codec = codec  # its decoder (a torch module: codes (B, 16, T) -> wav (B, 1, S))
        self.upsample = upsample

    @classmethod
    def of(cls, model: Any) -> _StreamDecoder | None:
        tokenizer = getattr(model.model, "speech_tokenizer", None)
        inner = getattr(tokenizer, "model", None)
        codec = getattr(inner, "decoder", None)
        get_type = getattr(inner, "get_model_type", None)
        if codec is None or get_type is None or "12hz" not in str(get_type()):
            return None
        upsample = int(getattr(inner, "get_decode_upsample_rate", lambda: _FRAME_SAMPLES)())
        return cls(tokenizer, codec, upsample)

    def stream(
        self,
        model: Any,
        generate: Any,
        kwargs: dict[str, Any],
        stop: threading.Event,
        *,
        ref_code: Any,
        first: int,
        every: int,
        context: int,
    ) -> Iterator[np.ndarray]:
        torch = require("torch", extra=_EXTRA)
        frames: queue.SimpleQueue[Any] = queue.SimpleQueue()
        abort = threading.Event()  # this consumer is gone: stop generating

        def on_step(_module: Any, _args: Any, output: Any) -> None:
            if stop.is_set() or abort.is_set():
                raise Stopped
            hidden = getattr(output, "hidden_states", None)
            codes = hidden[1] if isinstance(hidden, tuple) and len(hidden) > 1 else None
            if codes is not None:  # None on the prefill step
                frames.put(codes[0].detach().clone())

        def run() -> None:
            try:
                with _skip_final_decode(self.tokenizer):
                    generate(**kwargs)
            except BaseException as exc:  # Stopped included
                frames.put(exc)
            finally:
                frames.put(_END)

        handle = model.model.talker.register_forward_hook(on_step)
        worker = threading.Thread(target=run, name="qwen-tts-generate", daemon=True)
        worker.start()
        prefix = ref_code.to(self.codec_device) if ref_code is not None else None
        codes: list[Any] = []
        done = 0  # frames decoded
        finished = False
        try:
            while not finished:
                batch = [frames.get()]
                while not frames.empty():  # catch up with the generator
                    batch.append(frames.get())
                for item in batch:
                    if item is _END:
                        finished = True
                    elif isinstance(item, BaseException):
                        raise item
                    else:
                        codes.append(item)
                if stop.is_set():
                    return
                target = first if done == 0 else every
                if len(codes) - done >= target or (finished and len(codes) > done):
                    yield self._decode(torch, prefix, codes, done, len(codes), context)
                    done = len(codes)
        finally:
            abort.set()  # no-op once generation ended; else it stops at the next step
            worker.join()
            handle.remove()

    @property
    def codec_device(self) -> Any:
        return next(self.codec.parameters()).device

    def _decode(
        self, torch: Any, prefix: Any, codes: list[Any], start: int, end: int, context: int
    ) -> np.ndarray:
        """Audio of frames ``[start, end)``, decoded with up to ``context`` frames before."""
        new = torch.stack(codes[max(0, start - context) : end]).to(self.codec_device)
        left = min(start, context)
        if prefix is not None and left < context:  # continue from the reference audio
            extra = prefix[-(context - left) :]
            new = torch.cat([extra.to(new.dtype), new])
            left += int(extra.shape[0])
        with torch.inference_mode():
            wav = self.codec(new.clamp(min=0).T.unsqueeze(0))
        return wav[0, 0, left * self.upsample :].float().cpu().numpy()


class _FastCodePredictor:
    """Replaces ``code_predictor.generate`` (Hugging Face ``generate``) with a plain
    sampling loop over the same forward pass.

    For every 80 ms frame the talker calls its code predictor to generate the 15 residual
    codebooks, i.e. 15 decoding steps of a 5-layer model: small enough that the generic
    ``generate`` machinery (logits processors, stopping criteria, output bookkeeping)
    costs more than the model. Same sampling (temperature, top-k, top-p), same result
    type as far as the talker is concerned (``.sequences``).
    """

    def __init__(self, predictor: Any, torch: Any) -> None:
        self.predictor = predictor
        self.torch = torch
        self.original = predictor.generate

    def install(self) -> None:
        self.predictor.generate = self

    def __call__(
        self,
        inputs_embeds: Any = None,
        max_new_tokens: int = 15,
        do_sample: bool | None = True,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        temperature: float | None = 1.0,
        **_kwargs: Any,
    ) -> Any:
        torch = self.torch
        cache_module = require("transformers.cache_utils", extra=_EXTRA)
        cache = cache_module.DynamicCache()
        predictor = self.predictor
        tokens = []
        with torch.inference_mode():
            out = predictor(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
            for step in range(max_new_tokens):
                token = self._sample(out.logits[:, -1], do_sample, top_k, top_p, temperature)
                tokens.append(token)
                if step + 1 == max_new_tokens:
                    break
                out = predictor(
                    input_ids=token[:, None],
                    past_key_values=cache,
                    use_cache=True,
                    generation_steps=out.generation_steps,
                )
        return _Sequences(torch.stack(tokens, dim=-1))

    def _sample(
        self,
        logits: Any,
        do_sample: bool | None,
        top_k: int | None,
        top_p: float | None,
        temperature: float | None,
    ) -> Any:
        torch = self.torch
        logits = logits.float()
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature and temperature != 1.0:
            logits = logits / temperature
        if top_k and top_k < logits.shape[-1]:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and top_p < 1.0:
            ordered, index = torch.sort(logits, descending=True, dim=-1)
            cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
            drop = cumulative - ordered.softmax(dim=-1) > top_p  # keep the first token
            ordered = ordered.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, index, ordered)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class _Sequences:
    __slots__ = ("sequences",)

    def __init__(self, sequences: Any) -> None:
        self.sequences = sequences


class _skip_final_decode:
    """qwen-tts decodes the whole utterance once generation ends; the stream already did,
    so that decode is replaced by a no-op for the duration of the call."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __enter__(self) -> None:
        def decode(encoded: Any) -> tuple[list[np.ndarray], int]:
            n = len(encoded) if isinstance(encoded, list) else 1
            return [np.zeros(0, np.float32) for _ in range(n)], SAMPLE_RATE

        self.tokenizer.decode = decode

    def __exit__(self, *exc: object) -> None:
        try:
            del self.tokenizer.decode  # back to the class method
        except AttributeError:
            pass
