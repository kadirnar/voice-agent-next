# Model manager (`van models`)

Local providers download their weights on first use: Silero VAD, Smart Turn, Kokoro,
faster-whisper and the sherpa-onnx catalog. `van models` shows what they can download,
what is already cached and how much disk it uses. It can also download models ahead of
time, check them against their pinned SHA-256 digests and clean up.

```console
$ van models list --kind vad
$ van models download kokoro/v1.0-int8 silero
$ van models download --for agent.yaml          # everything a config needs
$ van models verify
$ van models du
$ van models prune                              # dry run: shows what would be deleted
$ van models prune --yes
$ van models path
```

## Where models live

| Location | What | Override |
|---|---|---|
| model cache: `~/.cache/voice-agent-next/models` (Linux), `~/Library/Caches/voice-agent-next/models` (macOS), `%LOCALAPPDATA%\voice-agent-next\Cache\models` (Windows) | Silero, Kokoro, sherpa-onnx (extracted archives), Smart Turn without `huggingface_hub` (`hf/…`) | `VAN_CACHE_DIR` |
| Hugging Face cache: `~/.cache/huggingface/hub` | faster-whisper repositories, Smart Turn (when `huggingface_hub` is installed) | `HF_HUB_CACHE`, `HF_HOME` |

`van models path` prints the model cache, `van models path --hf` prints the Hugging Face
cache and `van models path <model>` prints where a model's files are (or will be).

The Hugging Face cache is shared with other programs. `van models` reports what the
catalog's models use there, but it deletes from it only when you pass `--include-hf`.
If you pass a custom `download_root` to faster-whisper, the model manager cannot see
that model.

Set `VAN_OFFLINE=1` to turn off all downloads. Providers then use only cached files, and
`van models download` fails for models that are not cached.

## Commands

### `list`

```console
$ van models list [--provider sherpa-onnx] [--kind stt|tts|vad|turn] [--cached] [--json]
```

Each catalog model is listed by name (`<provider>/<model>`, the same string you pass to
`create()` or put in a config), with its kind, its state, its download size, its license
and its languages. The state is `cached`, `partial` (some files are missing) or
`missing`. `(HF)` marks a model that is stored in the Hugging Face cache.

### `download`

```console
$ van models download <name>... [--for <config|components|spec>]... [--force] [--dry-run]
```

A name can be a catalog name (`sherpa-onnx/moonshine-tiny-en`), a model id that only one
provider uses (`moonshine-tiny-en`), or a provider or spec without a model (`silero`,
`kokoro`), which selects that provider's default model(s). Cached models are skipped
unless you pass `--force`. Downloads show a progress bar, are written atomically and are
checked against their SHA-256 digests. Interrupted downloads start again from the
beginning.

With `--for`, you download everything a deployment needs:

| `--for` value | Meaning |
|---|---|
| `agent.yaml` / `.toml` / `.json` | every component of the config (failover lists included) |
| `stt=whisper/small,vad=silero,turn=smart-turn` | these components (default models when no model is given) |
| `kokoro/v1.0-int8`, `sherpa-onnx` | a spec: that model, or the provider's default model of every kind |

Cloud providers need nothing. If a local component has no catalog entry (a local path, a
custom URL, a model served by Ollama or vLLM), you get a note and nothing is downloaded
for it.

#### Docker images and offline machines

```dockerfile
ENV VAN_CACHE_DIR=/models HF_HOME=/models/hf
RUN van models download --for /app/agent.yaml
ENV VAN_OFFLINE=1
```

Copy the model cache (and the Hugging Face cache, if you use faster-whisper) to the
offline machine, point `VAN_CACHE_DIR` / `HF_HOME` at the copies and set `VAN_OFFLINE=1`.

### `verify`

```console
$ van models verify [<name>...] [--json]
```

Checks every cached model (or only the models you name):

* single files and Smart Turn files are hashed and compared with the SHA-256 pinned in the
  catalog;
* extracted archives (sherpa-onnx) are checked against the archive digest recorded when
  they were extracted, and every file the model needs must be present;
* Hugging Face repositories (faster-whisper): large files are stored under their SHA-256
  in the HF cache, so they are hashed and compared with that name.

A file is `ok`, `unverified` (it is present but there is no digest to compare against),
`missing` or `corrupt`. The command exits with code 1 when a file is missing or corrupt.
To fix a broken model, run `van models download --force <name>`.

### `prune`

```console
$ van models prune [--yes] [--model <name>]... [--all] [--no-partial] [--no-unused]
                   [--older-than DAYS] [--include-hf] [--json]
```

By default `prune` is a dry run: it lists what it would delete and how much space that
frees. Pass `--yes` to delete. It considers:

* **partial** files: interrupted downloads (`.*.part`) and interrupted extractions
  (`.*.tmp`) older than one hour. Newer ones may belong to a download that is still
  running.
* **unused** files: anything in a provider's folder (`silero/`, `kokoro/`,
  `sherpa-onnx/`, `hf/`) that no catalog model uses, such as older model versions, pins
  replaced in a newer release, or custom URLs. Other folders at the top of the cache
  (such as Kokoro's copy of the espeak-ng data) are never touched.
* **models**, only when you ask: `--model <name>` (repeatable) or `--all` for every cached
  catalog model. Models stored in the Hugging Face cache are kept and reported unless you
  also pass `--include-hf`.

`--older-than DAYS` keeps anything that was read or written in the last `DAYS` days.
Most file systems update access times only coarsely, so treat this as "not used for
days", not as an exact timestamp.

### `du`

```console
$ van models du [--json]
```

Shows disk usage per provider in the model cache and in the Hugging Face cache, plus
unused, partial and other files in the model cache.

## Python API

Other tools can use the manager directly (`voice_agent_next.models`):

```python
from voice_agent_next import models

req = models.models_for_config("agent.yaml")  # or models_for("stt=...,tts=...")
for info in req.models:
    if not models.model_status(info).cached:
        models.download_model(info)
print(req.notes)
```

`catalog()`, `get_model()`, `resolve_models()`, `model_status()`, `verify_model()`,
`plan_prune()` / `apply_prune()` and `disk_usage()` back the CLI commands.

## Declaring models in a provider

A provider registers the files it downloads at import time, next to its pinned URLs:

```python
from ..models import ModelFile, register_model

register_model(
    "silero",  # provider name, as in @register_provider
    "v6.2",  # model id, as in create("vad", "silero/v6.2")
    kind="vad",
    files=[
        ModelFile.from_url(
            URL, subdir="silero", filename="silero_vad_v6.2.onnx", sha256=SHA256, size=2_327_524
        )
    ],
    license="MIT",
    languages="any",
)
```

| Constructor | The provider downloads with |
|---|---|
| `ModelFile.from_url(url, subdir=, filename=, sha256=, size=)` | `utils.download.download` |
| `ModelFile.from_archive(url, subdir=, sha256=, required=)` | `utils.download.download_archive` |
| `ModelFile.from_hf(repo, filename, revision=, sha256=)` | `utils.download.hf_file` |
| `ModelFile.from_hf_repo(repo, revision=, patterns=, required=)` | `huggingface_hub.snapshot_download` |

The arguments must match what the provider passes to the download function, so that both
look in the same place. `tests/test_models.py` checks this for the built-in providers.
