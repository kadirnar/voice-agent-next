# Docker

`docker/` has two images and three compose stacks:

| File | What |
| --- | --- |
| `docker/Dockerfile` | The **CPU image** (target `cpu`, the default) and the **CUDA image** (target `cuda`). Both run [`van serve`](serving.md). |
| `docker/compose.yaml` | A **fully local stack**: Ollama, `van serve` over WebSocket and WebRTC with the `local-cpu` stack, and the browser demos |
| `docker/compose.gpu.yaml` | The **GPU variant** of that stack (an override file): the CUDA image, the `local-gpu` stack, Ollama on the GPU |
| `docker/compose.telephony.yaml` | A **phone agent** on Twilio Media Streams, with a TwiML webhook and an optional tunnel |
| `docker/config/*.yaml` | The agent configs the stacks mount at `/config/agent.yaml` |

CI (`.github/workflows/docker.yml`) builds both images on every pull request that touches
them and runs the CPU image and the local stack end to end. On `main` and on `v*` tags it
pushes them to `ghcr.io/kadirnar/voice-agent-next`.

## Quick start: a local agent in the browser

```bash
docker compose -f docker/compose.yaml up --build
```

Then open <http://localhost:8080>, allow the microphone and talk. `/websocket/` is the
[WebSocket demo](../transports/websocket.md) and `/webrtc/` is the
[WebRTC demo](../transports/webrtc.md). Both talk to the same agent config.

The first start downloads about 1 GB into two named volumes, which takes a few minutes:

* `ollama-pull` pulls the LLM, `LiquidAI/lfm2.5-1.2b-instruct` (~0.7 GB).
* `models` runs `van models download --for /config/agent.yaml`, which downloads the Kroko
  Zipformer STT, Kokoro, Silero VAD, Smart Turn and the `lm_turn` text model (~0.4 GB).

The agents start once both have finished. They run with `VAN_OFFLINE=1`, so no download
can ever happen during a call. Later starts use the volumes and work without network.

| Service | Port | What |
| --- | --- | --- |
| `web` | `8080` | nginx: the two demo pages on one origin, proxying `/ws/` and `/webrtc/{offer,config}` to the agents |
| `agent-ws` | `127.0.0.1:8765` | `van serve -p websocket` (van-ws/1): also for Python clients and for `/metrics` |
| `agent-webrtc` | (internal) `8080` | `van serve -p webrtc` |
| `ollama` | (internal) `11434` | The LLM server |

The ops routes of both agents are also reachable through `web`: `/ws/health`, `/ws/ready`,
`/ws/metrics`, and the same under `/webrtc/`.

The agents accept browser pages from `http://localhost:*` only
([Origin allow-list](serving.md#who-may-connect-the-origin-allow-list)). To open the demos
from another host name (an `https://` name on your LAN, for example), set
`VAN_ALLOWED_ORIGINS=https://agent.example.lan` (space-separated for several).

### Change the agent

Edit `docker/config/local-cpu.yaml`. It `extends` the [`local-cpu` preset](../presets.md), so
you only write what differs:

```yaml
extends: local-cpu
llm: {provider: "ollama/${OLLAMA_MODEL:-LiquidAI/lfm2.5-1.2b-instruct}", reasoning_effort: none}
agent:
  instructions: You are a pirate. Answer in one sentence.
```

To use another Ollama model, set `OLLAMA_MODEL`. Both the pull and the config read it.
`reasoning_effort: none` keeps models that think by default (Qwen3.5) from reasoning
before every answer:

```bash
OLLAMA_MODEL=qwen3.5:4b docker compose -f docker/compose.yaml up
```

Other variables: `WEB_PORT` (default 8080), `VAN_IMAGE` (the image to run),
`OLLAMA_VERSION`, `NGINX_VERSION`. Compose also reads them from `docker/.env`.

### WebRTC and the Docker network

Signalling (`/webrtc/offer`) goes through nginx, but the media is UDP between the browser
and the `agent-webrtc` container. On a **Linux** host, a browser on the same machine reaches
the container's address on the Docker bridge, so the demo works as it is.
**Docker Desktop** (macOS, Windows) does not route to container addresses. There, use the
WebSocket demo, or give the WebRTC server a TURN server (`ice_servers`,
[WebRTC](../transports/webrtc.md)). For production WebRTC, run the agent with
`network_mode: host` or behind a TURN server.

## GPU variant

```bash
docker compose -f docker/compose.yaml -f docker/compose.gpu.yaml up --build
```

It needs the NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. The override changes four things:

* The agents run the CUDA image.
* The config is `docker/config/local-gpu.yaml`, which extends the
  [`local-gpu` preset](../presets.md): faster-whisper `large-v3-turbo` on CUDA and Kokoro.
* Ollama runs `qwen3.5:4b` (thinking off) on the GPU.
* Ollama and the agents reserve the GPUs (`deploy.resources.reservations.devices`).

The first start downloads about 3.4 GB for the LLM and 1.8 GB of models.

## The images

| | CPU (`cpu`) | CUDA (`cuda`) |
| --- | --- | --- |
| Tags | `latest`, `main`, `X.Y.Z`, `X.Y`, `sha-…` | the same with `-cuda` (`latest-cuda`, ...) |
| Platforms | `linux/amd64`, `linux/arm64` | `linux/amd64` |
| Size (uncompressed, amd64) | 515 MB; 789 MB with `BAKE_MODELS=preset:local-cpu` (measured before `text-turn` was added: +32 MB of packages, +137 MB baked model) | 3.9 GB (cuBLAS, cuDNN, CUDA runtime wheels) |
| Extras | `sherpa-onnx openai kokoro silero smart-turn text-turn webrtc resample`: the `local-cpu` preset and WebRTC | `faster-whisper openai kokoro silero smart-turn webrtc resample cuda`: the `local-gpu` preset, cuBLAS 12 |
| ONNX Runtime | `onnxruntime` (CPU) | `onnxruntime-gpu[cuda,cudnn]` 1.26 (CUDA 12, cuDNN 9) |

Both images are built the same way:

* **Base.** `python:3.12-slim-trixie`, with the locked dependencies (`uv.lock`) installed
  by uv.
* **Multi-stage build.** The final image gets only the virtual environment in
  `/opt/venv`: no compiler, no uv, no caches. The dependency layer is rebuilt only when
  `pyproject.toml`, `uv.lock` or the extras change.
* **Non-root.** The server runs as user `van` (uid 1000).
* **Entrypoint.** `van serve --host 0.0.0.0`, with `--protocol websocket` as the default
  command. Without a source, that serves the offline mock engine.
* **Health check.** `HEALTHCHECK` calls `GET /health` on the served port. The port comes
  from `--port`, else from the default port of the protocol. Set `VAN_HEALTH_PORT` to
  override it.
* **Model cache.** Models go to `VAN_CACHE_DIR=/models`, and the Hugging Face cache to
  `HF_HOME=/models/hf`.

The CUDA image is **not** built on the multi-GB `nvidia/cuda` base. The CUDA libraries
come from NVIDIA's pip wheels in the virtual environment, which `voice_agent_next.hardware`
loads without `LD_LIBRARY_PATH` ([hardware](../hardware.md)). The driver comes from the host
through the Container Toolkit. The image sets `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`. The host driver must support CUDA 12, and
Blackwell GPUs need driver 570 or newer.

### Running

The entrypoint passes flags to `van serve`, runs any other `van` command, and runs any
program on `PATH`:

```bash
IMAGE=ghcr.io/kadirnar/voice-agent-next

# the offline mock engine over WebSocket (the default command)
docker run --rm -p 8765:8765 $IMAGE

# OpenAI Realtime protocol in front of a cloud engine: clients must send VAN_SERVER_API_KEY
docker run --rm -p 8000:8000 -e OPENAI_API_KEY -e VAN_SERVER_API_KEY $IMAGE \
  -p openai-realtime --engine openai/gpt-realtime

# the local-cpu stack, with Ollama running on the host
docker run --rm -p 8765:8765 -v van-models:/models \
  -e OLLAMA_HOST=http://host.docker.internal:11434 --add-host host.docker.internal:host-gateway \
  $IMAGE --preset local-cpu

docker run --rm $IMAGE providers                             # any van command
docker run --rm $IMAGE models download --for preset:local-cpu --dry-run
docker run --rm --gpus all $IMAGE:latest-cuda doctor         # is the GPU used?
```

The entrypoint binds `0.0.0.0` inside the container, so the
[secure defaults](serving.md#listening-beyond-this-machine) apply: `-p openai-realtime`
needs `--api-key` or `VAN_SERVER_API_KEY` (or `--insecure` behind an authenticating proxy),
and the other protocols log a warning. Browser pages from other origins than `localhost`
need `--allowed-origin` (or `VAN_ALLOWED_ORIGINS`, space-separated).

`van serve` drains on `SIGTERM` ([graceful drain](serving.md#graceful-drain)). Give
`docker stop -t` (or the Kubernetes `terminationGracePeriodSeconds`) more time than
`--drain-timeout`, which defaults to 30 s.

Run **one worker per container** and scale with replicas behind a load balancer. That way
every `/metrics` scrape reaches one known process ([workers](serving.md#worker-processes)).

## Models: bake them in, or keep them on a volume

By default an image holds no models. They download on first use into `/models`, so mount
a volume there to keep them across containers:

```bash
docker run -v van-models:/models IMAGE models download --for preset:local-cpu
docker run -v van-models:/models -e VAN_OFFLINE=1 ... IMAGE --preset local-cpu
```

The `BAKE_MODELS` build argument downloads models **into the image** at build time. It
takes `van models download --for` targets, separated by spaces. `preset:NAME` selects every
model of a preset:

```bash
docker build -f docker/Dockerfile --build-arg BAKE_MODELS=preset:local-cpu -t van:local-cpu .
docker run -e VAN_OFFLINE=1 -e OLLAMA_HOST=... van:local-cpu --preset local-cpu
```

| | Baked (`BAKE_MODELS`) | Volume (default) |
| --- | --- | --- |
| Image size | + the models (`local-cpu`: +~250 MB; `local-gpu`: +~1.8 GB) | code only |
| Cold start | no download: starts offline, the same everywhere | first container downloads |
| Model updates | rebuild the image | `van models download` into the volume |
| Best for | autoscaling, air-gapped clusters, reproducible releases | development, several configs sharing one cache |

A named volume mounted on `/models` of a baked image starts with the baked models: Docker
copies them into an empty volume. With a bind mount, the host directory must be writable
by uid 1000.

## Building the images yourself

```bash
# CPU (the default target)
docker build -f docker/Dockerfile -t voice-agent-next .

# CUDA
docker build -f docker/Dockerfile --target cuda -t voice-agent-next:cuda \
  --build-arg EXTRAS="faster-whisper openai kokoro silero smart-turn webrtc resample cuda" \
  --build-arg ONNXRUNTIME_GPU="onnxruntime-gpu[cuda,cudnn]~=1.26.0" .
```

| Build argument | Default | What |
| --- | --- | --- |
| `EXTRAS` | `sherpa-onnx openai kokoro silero smart-turn text-turn webrtc resample` | voice-agent-next extras to install (see `pyproject.toml`), for example add `anthropic google` for cloud LLMs |
| `ONNXRUNTIME_GPU` | empty | A pip requirement that replaces `onnxruntime` with a GPU build |
| `BAKE_MODELS` | empty | `van models download --for` targets to bake in |
| `PYTHON_VERSION` | `3.12` | Kokoro needs Python < 3.14 |
| `DEBIAN_RELEASE` | `trixie` | The Debian release of the `python:*-slim` base |
| `UV_VERSION` | pinned | The uv used at build time only |

The build context is the repository root. `.dockerignore` sends only `pyproject.toml`,
`uv.lock`, `README.md`, `LICENSE`, `src/` and the two container scripts.

## Telephony (Twilio)

```bash
export TWILIO_ACCOUNT_SID=AC... TWILIO_AUTH_TOKEN=...      # or in docker/.env
docker compose -f docker/compose.telephony.yaml --profile tunnel up --build
docker compose -f docker/compose.telephony.yaml logs tunnel   # prints https://<name>.trycloudflare.com
```

The stack runs `van serve -p twilio` behind nginx. nginx serves two routes:

* `/twiml` answers the call with `<Connect><Stream url="wss://HOST/stream">`.
* `/stream` proxies the media-stream WebSocket to the agent.

The `tunnel` profile starts a free Cloudflare quick tunnel that gives the stack a public
HTTPS URL. In the Twilio console, set your number's **A call comes in** webhook to
`https://<public host>/twiml`, then call the number.

Without the tunnel, put your own TLS proxy in front of port 8080 and set
`PUBLIC_HOST=voice.example.com`. The TwiML then points at `wss://voice.example.com/stream`.
If `PUBLIC_HOST` is empty, the TwiML uses the `Host` header that the tunnel or proxy
forwards.

The credentials let the agent end calls through Twilio's REST API. The agent's `/ready`
and `/metrics` stay on `127.0.0.1:8765`, off the public URL. The agent config is
`docker/config/telephony.yaml`. For lower latency on real phone lines, extend a cloud
preset there (`extends: cloud-fast`) and pass its API keys in the compose file. Telnyx,
Vonage and Plivo work the same way with `--protocol telnyx` and so on, plus their own
call markup ([telephony](../transports/telephony.md)).

## Limitations

* The CUDA image is built in CI but not run on a GPU, because GitHub has no GPU runners.
  CI checks that the CUDA libraries and the CUDA execution provider are in place.
* The CUDA image is `linux/amd64` only. NVIDIA's pip wheels for Jetson and Grace are not
  covered.
* The local stacks run two agent processes, one for WebSocket and one for WebRTC. Each
  loads its own models. Remove a service you don't need.
* The WebRTC media path depends on the Docker network (see above).
