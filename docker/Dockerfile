# syntax=docker/dockerfile:1.7
#
# voice-agent-next images: `van serve` on a slim Python base, non-root.
# Build from the repository root (docs/deploy/docker.md):
#
#   docker build -f docker/Dockerfile -t voice-agent-next .                  # CPU (local-cpu)
#   docker build -f docker/Dockerfile --target cuda -t voice-agent-next:cuda \
#     --build-arg EXTRAS="faster-whisper openai kokoro silero smart-turn webrtc resample cuda" \
#     --build-arg ONNXRUNTIME_GPU="onnxruntime-gpu[cuda,cudnn]~=1.26.0" .      # NVIDIA GPU
#   ... --build-arg BAKE_MODELS=preset:local-cpu                             # models inside
#
# Stages: `builder` installs the locked dependencies into /opt/venv with uv, `models`
# optionally downloads models, `runtime` copies only the venv and the models onto a fresh
# base: no compiler, no uv, no build caches in the final image.

ARG PYTHON_VERSION=3.12
ARG DEBIAN_RELEASE=trixie
ARG UV_VERSION=0.12.19

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ------------------------------------------------------------------------------ builder
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
# voice-agent-next extras (space separated). Default: the `local-cpu` preset (sherpa-onnx
# STT, Ollama through `openai`, Kokoro, Silero, Smart Turn) plus the WebRTC transport.
ARG EXTRAS="sherpa-onnx openai kokoro silero smart-turn webrtc resample"
# A pip requirement that replaces `onnxruntime` with a GPU build (CUDA image), or empty.
ARG ONNXRUNTIME_GPU=""
WORKDIR /src

# Dependencies first, from the lock file: this layer is rebuilt only when pyproject.toml,
# uv.lock or the build args change, not on every source change.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    set -eu; set --; \
    for extra in ${EXTRAS}; do set -- "$@" --extra "${extra}"; done; \
    uv sync --locked --no-dev --no-install-project "$@"; \
    if [ -n "${ONNXRUNTIME_GPU}" ]; then \
        uv pip uninstall --python /opt/venv/bin/python onnxruntime; \
        uv pip install --python /opt/venv/bin/python "${ONNXRUNTIME_GPU}"; \
    fi

# Then the project itself, as a regular (non-editable) install without touching the deps.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps . \
    && /opt/venv/bin/van providers --json > /dev/null

# ------------------------------------------------------------------------------- models
FROM builder AS models
# `van models download --for` targets to bake into the image (space separated), e.g.
# `preset:local-cpu` or `/src/agent.yaml`. Empty (default): models download on first use
# into /models, which you mount as a volume.
ARG BAKE_MODELS=""
ENV VAN_CACHE_DIR=/models \
    HF_HOME=/models/hf
RUN set -eu; mkdir -p /models/hf; \
    for target in ${BAKE_MODELS}; do /opt/venv/bin/van models download --for "${target}"; done

# ------------------------------------------------------------------------------ runtime
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS runtime
LABEL org.opencontainers.image.title="voice-agent-next" \
      org.opencontainers.image.description="Real-time speech-to-speech voice agents: van serve" \
      org.opencontainers.image.source="https://github.com/kadirnar/voice-agent-next" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN groupadd --system --gid 1000 van \
    && useradd --system --uid 1000 --gid van --home-dir /home/van --create-home \
       --shell /usr/sbin/nologin van

COPY --from=builder /opt/venv /opt/venv
COPY --from=models --chown=van:van /models /models
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/van-entrypoint
COPY --chmod=0755 docker/healthcheck.py /usr/local/bin/van-healthcheck

ENV PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VAN_CACHE_DIR=/models \
    HF_HOME=/models/hf

USER 1000:1000
WORKDIR /home/van
# openai-realtime, webrtc, websocket / telephony
EXPOSE 8000 8080 8765
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 CMD ["van-healthcheck"]
ENTRYPOINT ["van-entrypoint"]
# The offline mock engine over WebSocket; pass `--preset`, `--config`... to serve a real agent.
CMD ["--protocol", "websocket"]

# --------------------------------------------------------------------------------- cuda
# The NVIDIA driver comes from the host (NVIDIA Container Toolkit, `--gpus all`); the CUDA
# libraries come from pip wheels in the venv (the `cuda` extra, onnxruntime-gpu), which
# voice_agent_next.hardware loads without LD_LIBRARY_PATH: no multi-GB nvidia/cuda base.
FROM runtime AS cuda
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------------------- cpu
FROM runtime AS cpu
