"""mlx-lm: LLMs on the Apple silicon GPU through ``mlx_lm.server``'s OpenAI-compatible API.

Start the server (installed with the ``mlx`` extra) and point the agent at it::

    python -m mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit   # port 8080

    llm = create("llm", "mlx_lm")                                   # the server's --model
    llm = create("llm", "mlx_lm/mlx-community/Qwen3.5-4B-4bit")      # loaded on demand

Without a model, requests ask for ``default_model``: the model the server was started
with. A model id in the spec is loaded by the server on first use (downloading it from
the Hugging Face Hub if needed); :meth:`warmup` does that ahead of the first turn.
Server address: ``base_url=``, else ``MLX_LM_BASE_URL``, else
``http://127.0.0.1:8080/v1``. Streaming tool calls work with models whose chat template
mlx-lm can parse (Qwen3/3.5, Gemma 4, Mistral, GLM, Kimi...).

Why a server rather than in-process generation: token generation keeps the GPU busy for
seconds at a time, and in-process it would compete with the speech recognizer and the
synthesizer on the MLX thread (and the GIL) in the middle of a turn. The server runs it in
its own process, shares one loaded model between sessions and agents, and already
implements tool-call parsing and prompt caching; the client is the same OpenAI-compatible
code as Ollama, LM Studio and llama.cpp.
"""

from __future__ import annotations

from ..registry import register_provider
from .openai.llm import OpenAICompatibleLLM

__all__ = ["DEFAULT_BASE_URL", "MLXLMServerLLM"]

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


@register_provider(
    "llm",
    "mlx_lm",
    description="mlx-lm server on Apple silicon (OpenAI-compatible API)",
    models=(
        "mlx-community/Qwen3.5-4B-4bit",
        "mlx-community/Qwen3.5-9B-4bit",
        "mlx-community/gemma-4-e4b-it-4bit",
    ),
    extra="openai",
    requires=("openai",),
    local=True,
    aliases=("mlx_lm_server",),
)
class MLXLMServerLLM(OpenAICompatibleLLM):
    """``python -m mlx_lm.server`` (default port 8080). Thinking is off by default
    (``chat_template_kwargs={"enable_thinking": False}``): reasoning delays the first
    spoken word; pass ``extra={"chat_template_kwargs": None}`` to keep the template's own
    default."""

    provider = "mlx_lm"
    SYSTEM_MESSAGE_POLICY = "merge"  # Jinja chat templates: one leading system message
    DEFAULT_MODEL = "default_model"  # mlx_lm.server: the model given with --model
    DEFAULT_BASE_URL = DEFAULT_BASE_URL
    BASE_URL_ENV = ("MLX_LM_BASE_URL",)
    API_KEY_ENV = ()
    API_KEY_REQUIRED = False
    PRELOAD_ON_WARMUP = True
    DEFAULT_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}
    NOT_FOUND_HINT = "check the model id; the server loads Hugging Face MLX models on demand"
