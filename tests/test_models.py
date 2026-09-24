"""Model manager (`voice_agent_next.models`, `van models`): offline, on fake caches."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from voice_agent_next import models
from voice_agent_next.cli.main import app
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.models import ModelFile, register_model
from voice_agent_next.utils.download import DownloadError, cache_dir

COMMIT = "a" * 40
MODEL_BYTES = b"onnx-weights" * 500
MODEL_SHA = hashlib.sha256(MODEL_BYTES).hexdigest()
VOICES_BYTES = b"voices" * 300
VOICES_SHA = hashlib.sha256(VOICES_BYTES).hexdigest()


def _tar_bz2(files: dict[str, bytes], top: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


ARCHIVE = _tar_bz2({"encoder.onnx": b"enc", "tokens.txt": b"a 0\n"}, "fake-asr")
ARCHIVE_SHA = hashlib.sha256(ARCHIVE).hexdigest()
PAYLOADS = {
    "https://example.com/m/model-v2.onnx": MODEL_BYTES,
    "https://example.com/m/voices.bin": VOICES_BYTES,
    "https://example.com/r/fake-asr.tar.bz2": ARCHIVE,
}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.delenv("VAN_OFFLINE", raising=False)
    return tmp_path


@pytest.fixture
def fake_catalog(env: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, models.ModelInfo]:
    """An isolated catalog of fake models (the real providers are not imported)."""
    monkeypatch.setattr(models, "_MODELS", {})
    monkeypatch.setattr(models, "_LOADED", True)
    tts = register_model(
        "fakeprov",
        "v2",
        kind="tts",
        files=[
            ModelFile.from_url(
                "https://example.com/m/model-v2.onnx",
                subdir="fakeprov/r1",
                sha256=MODEL_SHA,
                size=len(MODEL_BYTES),
            ),
            ModelFile.from_url(
                "https://example.com/m/voices.bin",
                subdir="fakeprov/r1",
                sha256=VOICES_SHA,
                size=len(VOICES_BYTES),
            ),
        ],
        license="MIT",
        languages="en",
        aliases=("latest",),
    )
    asr = register_model(
        "fake-sherpa",
        "asr",
        kind="stt",
        files=[
            ModelFile.from_archive(
                "https://example.com/r/fake-asr.tar.bz2",
                subdir="fake-sherpa",
                sha256=ARCHIVE_SHA,
                required=("encoder.onnx", "tokens.txt"),
            )
        ],
        size=len(ARCHIVE),
    )
    turn = register_model(
        "fakehf",
        "turn-v1",
        kind="turn",
        files=[
            ModelFile.from_hf(
                "org/turn", "turn-v1.onnx", revision=COMMIT, sha256=MODEL_SHA, size=12
            )
        ],
    )
    repo = register_model(
        "fakehf",
        "whisper-tiny",
        kind="stt",
        files=[
            ModelFile.from_hf_repo(
                "org/whisper-tiny", patterns=("model.bin", "config.json"), required=("model.bin",)
            )
        ],
    )
    return {"tts": tts, "asr": asr, "turn": turn, "repo": repo}


def mock_client(calls: list[str] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        payload = PAYLOADS.get(str(request.url))
        if payload is None:
            return httpx.Response(404)
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    return httpx.Client(transport=httpx.MockTransport(handler))


def hf_snapshot(root: Path, repo: str, files: dict[str, bytes], *, ref: str = "main") -> Path:
    """Lay out ``files`` like huggingface_hub does (blobs + snapshot, symlinks if possible)."""
    base = root / "hf" / f"models--{repo.replace('/', '--')}"
    snap = base / "snapshots" / COMMIT
    snap.mkdir(parents=True)
    (base / "refs").mkdir()
    (base / "refs" / ref).write_text(COMMIT)
    (base / "blobs").mkdir()
    for name, data in files.items():
        blob = base / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        try:
            (snap / name).symlink_to(os.path.relpath(blob, snap))
        except OSError:  # no symlink privilege (Windows): huggingface_hub copies too
            (snap / name).write_bytes(data)
    return base


def age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    targets = [path, *path.rglob("*")] if path.is_dir() else [path]
    for p in targets:
        os.utime(p, (t, t))


# ------------------------------------------------------------------------ catalog
def test_register_model_validates_and_describes(fake_catalog: dict[str, Any]) -> None:
    tts = fake_catalog["tts"]
    assert tts.name == "fakeprov/v2"
    assert tts.size == len(MODEL_BYTES) + len(VOICES_BYTES)
    assert fake_catalog["asr"].name == "fake-sherpa/asr"
    assert fake_catalog["asr"].provider == "fake_sherpa"
    assert fake_catalog["asr"].files[0].filename == "fake-asr"  # extracted directory
    assert fake_catalog["repo"].size is None
    with pytest.raises(ValueError, match="kind"):
        register_model("x", "y", kind="speech", files=[ModelFile.from_url("https://e/x")])
    with pytest.raises(ValueError, match="no files"):
        register_model("x", "y", kind="stt", files=[])


def test_lookup_by_name_alias_provider_and_ambiguity(fake_catalog: dict[str, Any]) -> None:
    assert models.get_model("fakeprov/v2") is fake_catalog["tts"]
    assert models.get_model("FakeProv/latest") is fake_catalog["tts"]
    assert models.get_model("fake_sherpa/asr") is fake_catalog["asr"]
    assert models.get_model("turn-v1") is fake_catalog["turn"]
    assert models.get_model("fakeprov") is fake_catalog["tts"]  # the provider's only model
    with pytest.raises(ConfigurationError, match="ambiguous"):
        models.get_model("fakehf")
    with pytest.raises(ConfigurationError, match="unknown model"):
        models.get_model("nope/nothing")
    assert [m.model for m in models.catalog(kind="stt")] == ["asr", "whisper-tiny"]
    assert [m.model for m in models.catalog(provider="fakehf", kind="turn")] == ["turn-v1"]


def test_real_catalog_declares_every_local_download() -> None:
    from voice_agent_next.providers.kokoro import KOKORO_MODELS
    from voice_agent_next.providers.sherpa_onnx import SHERPA_MODELS
    from voice_agent_next.providers.smart_turn import MODEL_SHA256

    everything = models.catalog()
    names = [m.name for m in everything]
    assert len(names) == len(set(names))
    by_provider: dict[str, int] = {}
    for m in everything:
        by_provider[m.provider] = by_provider.get(m.provider, 0) + 1
        assert m.license, m.name
        assert m.size, m.name
        for f in m.files:
            if f.source != "hf-repo":
                assert f.sha256 and len(f.sha256) == 64, (m.name, f)
    assert by_provider["sherpa_onnx"] == len(SHERPA_MODELS)
    assert by_provider["kokoro"] == len(KOKORO_MODELS)
    assert by_provider["smart_turn"] == len(MODEL_SHA256)
    assert by_provider["silero"] == 1
    assert by_provider["faster_whisper"] == 10
    assert models.get_model("whisper/turbo").name == "faster-whisper/large-v3-turbo"
    stem = "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8"
    assert models.get_model(f"sherpa/{stem}").name == "sherpa-onnx/nemo-fastconformer-en-80ms"


def test_catalog_paths_match_what_providers_download(env: Path) -> None:
    """The model manager must look where the providers write."""
    silero = models.get_model("silero/v6.2")
    assert models.model_status(silero).files[0].path == cache_dir() / "silero/silero_vad_v6.2.onnx"
    kokoro = models.get_model("kokoro/v1.0-int8")
    paths = [fs.path for fs in models.model_status(kokoro).files]
    assert paths == [
        cache_dir() / "kokoro/model-files-v1.1/kokoro-v1.0.int8.onnx",
        cache_dir() / "kokoro/model-files-v1.1/voices-v1.0.bin",
    ]
    asr = models.get_model("sherpa-onnx/nemo-fastconformer-en-80ms")
    (f,) = asr.files
    assert f.source == "archive"
    assert models.file_status(f).path == (
        cache_dir()
        / "sherpa-onnx/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8"
    )
    assert "tokens.txt" in f.required


# ------------------------------------------------------------------------ download
def test_download_status_and_verify(fake_catalog: dict[str, Any]) -> None:
    tts, asr = fake_catalog["tts"], fake_catalog["asr"]
    assert models.model_status(tts).state == "missing"
    seen: list[tuple[str, int, int | None]] = []
    calls: list[str] = []
    client = mock_client(calls)
    paths = models.download_model(
        tts, client=client, progress=lambda f, done, total: seen.append((f.label, done, total))
    )
    assert [p.read_bytes() for p in paths] == [MODEL_BYTES, VOICES_BYTES]
    assert seen[-1] == ("voices.bin", len(VOICES_BYTES), len(VOICES_BYTES))
    assert {label for label, _, _ in seen} == {"model-v2.onnx", "voices.bin"}
    status = models.model_status(tts)
    assert status.state == "cached"
    assert status.size_on_disk == tts.size
    models.download_model(tts, client=client)
    assert len(calls) == 2  # cache hits

    (root,) = models.download_model(asr, client=client)
    assert (root / "encoder.onnx").read_bytes() == b"enc"
    assert models.model_status(asr).cached
    assert [c.status for c in models.verify_model(asr)] == ["ok"]
    assert [c.status for c in models.verify_model(tts)] == ["ok", "ok"]

    paths[0].write_bytes(b"tampered")
    (root / "tokens.txt").unlink()
    checks = models.verify_model(tts)
    assert [c.status for c in checks] == ["corrupt", "ok"]
    assert "sha256" in checks[0].detail
    (check,) = models.verify_model(asr)
    assert check.status == "corrupt" and "tokens.txt" in check.detail
    paths[1].unlink()
    assert models.model_status(tts).state == "partial"
    assert models.verify_model(tts)[1].status == "missing"


def test_offline_download_of_missing_model_fails(
    fake_catalog: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAN_OFFLINE", "1")
    assert models.is_offline()
    calls: list[str] = []
    with pytest.raises(DownloadError, match="VAN_OFFLINE"):
        models.download_model(fake_catalog["tts"], client=mock_client(calls))
    assert calls == []


def test_hf_file_and_repo_in_the_hf_cache(fake_catalog: dict[str, Any], env: Path) -> None:
    turn, repo = fake_catalog["turn"], fake_catalog["repo"]
    assert models.model_status(turn).state == "missing"
    base = env / "hf" / "models--org--turn"
    (base / "snapshots" / COMMIT).mkdir(parents=True)
    (base / "snapshots" / COMMIT / "turn-v1.onnx").write_bytes(MODEL_BYTES)
    (fs,) = models.model_status(turn).files
    assert fs.present and fs.in_hf_cache and fs.size_on_disk == len(MODEL_BYTES)
    assert [c.status for c in models.verify_model(turn)] == ["ok"]
    # the same file downloaded without huggingface_hub lives in the model cache
    (base / "snapshots" / COMMIT / "turn-v1.onnx").unlink()
    fallback = cache_dir() / "hf/org/turn" / COMMIT / "turn-v1.onnx"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(MODEL_BYTES)
    (fs,) = models.model_status(turn).files
    assert fs.present and not fs.in_hf_cache and fs.path == fallback

    weights = b"ctranslate2" * 100
    hf_snapshot(env, "org/whisper-tiny", {"model.bin": weights, "config.json": b"{}"})
    status = models.model_status(repo)
    assert status.cached and status.hf_size_on_disk == len(weights) + 2
    (check,) = models.verify_model(repo)
    blob = env / "hf/models--org--whisper-tiny/blobs" / hashlib.sha256(weights).hexdigest()
    snapshot_file = env / "hf/models--org--whisper-tiny/snapshots" / COMMIT / "model.bin"
    if snapshot_file.is_symlink():
        assert check.status == "ok"
        blob.write_bytes(b"bit rot")
        (check,) = models.verify_model(repo)
        assert check.status == "corrupt" and "model.bin" in check.detail
    else:  # copies carry no digest in their name
        assert check.status == "unverified"
    snapshot_file.unlink()
    assert models.model_status(repo).state == "missing"
    assert models.verify_model(repo)[0].status == "corrupt"  # incomplete snapshot


# ------------------------------------------------------------------------ prune / du
def test_prune_plan_and_apply(fake_catalog: dict[str, Any], env: Path) -> None:
    client = mock_client()
    models.download_model(fake_catalog["tts"], client=client)
    models.download_model(fake_catalog["asr"], client=client)
    root = cache_dir()
    old_part = root / "fakeprov/r1/.model-v2.onnx.x1y2.part"
    old_part.write_bytes(b"12345")
    age(old_part, 7200)
    young_part = root / "fakeprov/r1/.voices.bin.abcd.part"  # a download in progress
    young_part.write_bytes(b"1")
    stale_tmp = root / "fake-sherpa/.fake-asr.q1.tmp"
    stale_tmp.mkdir()
    (stale_tmp / "encoder.onnx").write_bytes(b"half")
    age(stale_tmp, 7200)
    old_version = root / "fakeprov/r0/model-v1.onnx"
    old_version.parent.mkdir()
    old_version.write_bytes(b"v1" * 10)
    other = root / "espeak-ng-data-1.0"
    other.mkdir()
    (other / "phontab").write_bytes(b"data")

    items = models.plan_prune()
    by_path = {i.path: i for i in items}
    assert set(by_path) == {old_part, stale_tmp, old_version.parent}
    assert by_path[old_part].reason == "partial" and by_path[old_part].size == 5
    assert by_path[stale_tmp].reason == "partial"
    assert by_path[old_version.parent].reason == "unused"
    assert by_path[old_version.parent].size == 20
    assert models.plan_prune(partial=False, unused=False) == []
    assert models.plan_prune(partial=False, older_than=86_400) == []  # used recently
    age(old_version.parent, 3 * 86_400)
    assert [i.path for i in models.plan_prune(partial=False, older_than=86_400)] == [
        old_version.parent
    ]

    usage = models.disk_usage()
    assert usage.by_provider == {
        "fakeprov": len(MODEL_BYTES) + len(VOICES_BYTES),
        "fake_sherpa": models.model_status(fake_catalog["asr"]).size_on_disk,
    }
    assert usage.unused == 20
    assert usage.partial == 5 + 4 + 1  # disk_usage counts young partial files too
    assert usage.other >= 4
    assert usage.total == sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

    freed = models.apply_prune(items)
    assert freed == 5 + 4 + 20
    assert not old_part.exists() and not stale_tmp.exists() and not old_version.parent.exists()
    assert young_part.exists() and other.exists()
    assert models.model_status(fake_catalog["tts"]).cached  # models are never pruned implicitly

    items = models.plan_prune(models=["fakeprov/v2"])
    assert {i.reason for i in items} == {"model"}
    assert len(items) == 2 and all(i.model == "fakeprov/v2" for i in items)
    models.apply_prune(items)
    assert models.model_status(fake_catalog["tts"]).state == "missing"
    items = models.plan_prune(all_models=True, partial=False, unused=False)
    assert [i.model for i in items] == ["fake-sherpa/asr"]


def test_prune_keeps_the_hf_cache_unless_asked(fake_catalog: dict[str, Any], env: Path) -> None:
    base = hf_snapshot(env, "org/whisper-tiny", {"model.bin": b"w" * 50, "config.json": b"{}"})
    assert models.plan_prune(all_models=True) == []
    usage = models.disk_usage()
    assert usage.hf_by_provider == {"fakehf": 52}
    assert usage.by_provider == {}
    (item,) = models.plan_prune(models=["fakehf/whisper-tiny"], include_hf=True)
    assert item.path == base and item.in_hf_cache
    models.apply_prune([item])
    assert not base.exists()


# ------------------------------------------------------------------------ requirements
def test_models_for_specs_and_configs() -> None:
    req = models.models_for("stt=whisper/small,vad=silero,turn_detector=smart-turn")
    assert [m.name for m in req.models] == [
        "faster-whisper/small",
        "silero/v6.2",
        "smart-turn/smart-turn-v3.2-cpu",
    ]
    cfg = {
        "stt": ["sherpa-onnx/moonshine-tiny-en", {"provider": "deepgram", "model": "nova-3"}],
        "llm": "openai/gpt-4.1-mini",
        "tts": {"provider": "kokoro", "model": "v1.0-int8", "voice": "af_bella"},
        "vad": "sherpa/silero",
    }
    names = [m.name for m in models.models_for_config(cfg).models]
    assert names == ["sherpa-onnx/moonshine-tiny-en", "kokoro/v1.0-int8", "sherpa-onnx/silero"]
    # a provider without a model: its default model of every kind
    defaults = [m.name for m in models.models_for("sherpa-onnx").models]
    assert defaults == [
        "sherpa-onnx/nemo-fastconformer-en-80ms",
        "sherpa-onnx/piper-en_US-libritts_r-medium",
        "sherpa-onnx/silero",
    ]
    assert [m.name for m in models.models_for("kokoro/v1.1-zh").models] == ["kokoro/v1.1-zh"]
    custom = models.models_for_spec("stt", "sherpa-onnx//opt/models/my-asr")
    assert custom.models == [] and "not in the model catalog" in custom.notes[0]
    assert models.models_for_spec("stt", "deepgram/nova-3").models == []
    assert models.resolve_models("silero") == [models.get_model("silero/v6.2")]
    with pytest.raises(ConfigurationError):
        models.models_for("stt:whisper,x=y")


def test_models_for_config_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("stt: faster-whisper/tiny.en\nllm: ollama/llama3.2\ntts: kokoro\n")
    req = models.models_for(str(path))
    assert [m.name for m in req.models] == ["faster-whisper/tiny.en", "kokoro/v1.0"]
    assert any("ollama" in n for n in req.notes)


# ------------------------------------------------------------------------ CLI
def invoke(*args: str) -> Any:
    # CI sets FORCE_COLOR and the CLI's console is created at import: swap in a plain one
    # so assertions see the same text on every runner
    from rich.console import Console

    import voice_agent_next.cli.models as cli_models

    saved = cli_models.console
    cli_models.console = Console(no_color=True, force_terminal=False, width=250)
    try:
        return CliRunner().invoke(app, ["models", *args], env={"COLUMNS": "250"})
    finally:
        cli_models.console = saved


@pytest.fixture
def fake_downloads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route the CLI's downloads to the mock HTTP transport."""
    from voice_agent_next.utils import download as dl

    calls: list[str] = []
    client = mock_client(calls)
    real = dl.download

    def download(url: str, **kwargs: Any) -> Path:
        kwargs["client"] = client
        return real(url, **kwargs)

    monkeypatch.setattr(models, "download", download)
    monkeypatch.setattr(dl, "download", download)  # used by download_archive
    return calls


def test_cli_list_download_verify(fake_catalog: dict[str, Any], fake_downloads: list[str]) -> None:
    result = invoke("list", "--json")
    assert result.exit_code == 0, result.output
    rows = {r["name"]: r for r in json.loads(result.stdout)}
    assert rows["fakeprov/v2"]["state"] == "missing"
    assert rows["fakeprov/v2"]["license"] == "MIT"

    result = invoke("download", "fakeprov/v2", "fake-sherpa/asr", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would download fakeprov/v2" in result.stdout and fake_downloads == []

    result = invoke("download", "fakeprov/v2", "fake-sherpa/asr")
    assert result.exit_code == 0, result.output
    assert "done    fakeprov/v2" in result.stdout
    assert len(fake_downloads) == 3
    result = invoke("download", "fakeprov/latest")
    assert "cached  fakeprov/v2" in result.stdout and len(fake_downloads) == 3

    result = invoke("list", "--cached")
    assert "fakeprov/v2" in result.stdout and "whisper-tiny" not in result.stdout
    assert "cached" in result.stdout

    result = invoke("verify")
    assert result.exit_code == 0, result.output
    assert "sha256 matches" in result.stdout
    models.model_status(fake_catalog["tts"]).files[0].path.write_bytes(b"x")
    result = invoke("verify", "--json")
    assert result.exit_code == 1
    statuses = {(r["model"], r["file"]): r["status"] for r in json.loads(result.stdout)}
    assert statuses[("fakeprov/v2", "model-v2.onnx")] == "corrupt"
    assert statuses[("fake-sherpa/asr", "fake-asr")] == "ok"

    result = invoke("download", "nope/nothing")
    assert result.exit_code == 1 and "unknown model" in result.output
    result = invoke("download")
    assert result.exit_code != 0


def test_cli_download_respects_offline(
    fake_catalog: dict[str, Any], fake_downloads: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAN_OFFLINE", "1")
    result = invoke("download", "fakeprov/v2")
    assert result.exit_code == 1
    assert "VAN_OFFLINE" in result.output and fake_downloads == []
    assert "downloads are disabled" in invoke("list").stdout


def test_cli_prune_path_du(fake_catalog: dict[str, Any], env: Path) -> None:
    models.download_model(fake_catalog["tts"], client=mock_client())
    stale = cache_dir() / "fakeprov/old-release"
    stale.mkdir()
    (stale / "model.onnx").write_bytes(b"123")

    result = invoke("prune")
    assert result.exit_code == 0, result.output
    assert "would delete" in result.stdout and "--yes" in result.stdout
    assert stale.exists()
    result = invoke("prune", "--yes", "--json")
    data = json.loads(result.stdout)
    assert data["deleted"] and data["bytes"] == 3
    assert [i["reason"] for i in data["items"]] == ["unused"]
    assert not stale.exists()
    assert "nothing to prune" in invoke("prune").stdout

    hf_snapshot(env, "org/whisper-tiny", {"model.bin": b"w" * 10, "config.json": b"{}"})
    result = invoke("prune", "--all")
    assert "fakeprov/v2" in result.stdout
    assert "kept in the Hugging Face cache" in result.stdout and "whisper-tiny" in result.stdout

    assert invoke("path").stdout.strip() == str(cache_dir())
    assert invoke("path", "--hf").stdout.strip() == str(env / "hf")
    lines = invoke("path", "fakeprov/v2").stdout.splitlines()
    assert lines == [
        str(cache_dir() / "fakeprov/r1/model-v2.onnx"),
        str(cache_dir() / "fakeprov/r1/voices.bin"),
    ]

    result = invoke("du", "--json")
    assert result.exit_code == 0, result.output
    usage = json.loads(result.stdout)
    assert usage["by_provider"] == {"fakeprov": len(MODEL_BYTES) + len(VOICES_BYTES)}
    assert usage["hf_by_provider"] == {"fakehf": 12}
    result = invoke("du")
    assert "fakeprov" in result.stdout and "total" in result.stdout


def test_download_progress_callback(env: Path) -> None:
    from voice_agent_next.utils.download import download

    seen: list[tuple[int, int | None]] = []
    download(
        "https://example.com/m/model-v2.onnx",
        client=mock_client(),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen and seen[-1] == (len(MODEL_BYTES), len(MODEL_BYTES))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
