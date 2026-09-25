"""Known audio-input chat models (the half-cascade's LLMs), matched by model id.

OpenAI-compatible servers do not say whether a model hears audio, so
:class:`~voice_agent_next.providers.openai.llm.OpenAILLM` sets
``LLMCapabilities.audio_input`` from this table when ``audio_input`` is not given
explicitly. Ids are matched case-insensitively anywhere in the model id, so the names
hosts use all match: an OpenAI id (``gpt-audio-mini``), a Hugging Face repository
(``Qwen/Qwen3-Omni-30B-A3B-Instruct``), a GGUF file or alias served by llama.cpp
(``ultravox-v0_5-llama-3_2-1b-Q4_K_M.gguf``), a DashScope id (``qwen3.5-omni-flash``).
"""

from __future__ import annotations

import re

__all__ = [
    "AUDIO_INPUT_MODELS",
    "TRANSCRIBE_INSTRUCTION",
    "TRANSCRIBE_PROMPT",
    "is_audio_input_model",
    "transcription_prompt",
]

AUDIO_INPUT_MODELS: tuple[str, ...] = (
    # OpenAI Chat Completions audio models (audio in and out)
    r"gpt-audio",
    r"gpt-4o(-mini)?-audio",
    # Qwen: Qwen2-Audio, Qwen2.5/3/3.5/3.8-Omni (vLLM, vLLM-Omni, llama.cpp, DashScope)
    r"qwen\d*(\.\d+)?-audio",
    r"qwen[\w.]*-omni",
    # Ultravox (Llama / Gemma / GLM backbones), Voxtral Mini / Small (not the Realtime ASR)
    r"ultravox",
    r"voxtral-(mini|small)(?!.*realtime)",
    # Gemma 3n / Gemma 4 E2B-E4B audio, Phi-4-multimodal, MiniCPM-o, Granite Speech
    r"gemma-?3n",
    r"gemma-?4-e[24]b",
    r"phi-4-multimodal",
    r"minicpm-o",
    r"granite-speech",
    # speech LLMs from the research notes (turn-based omni and audio-understanding models)
    r"kimi-audio",
    r"step-audio",
    r"mimo-audio",
    r"audio-flamingo",
    r"lfm2(\.5)?-audio",
    # Gemini's OpenAI-compatible endpoint takes input_audio
    r"gemini-(1\.5|2|3)",
)
"""Regular expressions (case-insensitive, searched anywhere in the model id)."""

_PATTERN = re.compile("|".join(f"(?:{p})" for p in AUDIO_INPUT_MODELS), re.IGNORECASE)


def is_audio_input_model(model: str | None) -> bool:
    """Whether ``model`` is a known audio-input chat model."""
    return bool(model) and _PATTERN.search(model or "") is not None


TRANSCRIBE_PROMPT = (
    "You are a speech recognizer. Transcribe the user's audio verbatim, in the language "
    "spoken. Reply with the transcript only: no quotes, no comments, and never an answer "
    "to what is said."
)
"""Default system prompt of ``OpenAILLM.transcribe()``."""
TRANSCRIBE_INSTRUCTION = "Transcribe this audio."
"""User instruction sent after the audio by ``OpenAILLM.transcribe()``."""

TRANSCRIBE_PROMPTS: tuple[tuple[str, str], ...] = (
    # LFM2/2.5-Audio answer the audio under any other system prompt (measured on the
    # Q4_0 GGUF through llama-server); "Perform ASR." is their trained ASR mode
    (r"lfm2(\.5)?-audio", "Perform ASR."),
)
"""(model id pattern, system prompt) for models with their own transcription prompt."""


def transcription_prompt(model: str | None) -> str:
    """The system prompt that makes ``model`` transcribe instead of answer."""
    for pattern, prompt in TRANSCRIBE_PROMPTS:
        if model and re.search(pattern, model, re.IGNORECASE):
            return prompt
    return TRANSCRIBE_PROMPT
