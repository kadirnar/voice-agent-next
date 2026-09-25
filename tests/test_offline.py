"""One offline switch: ``VAN_OFFLINE`` or ``HF_HUB_OFFLINE`` forbids every download (#141)."""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent_next import models
from voice_agent_next.bench.asr_datasets import load_asr_dataset
from voice_agent_next.bench.eot_datasets import load_eot_dataset
from voice_agent_next.bench.quality_datasets import load_quality_dataset
from voice_agent_next.utils import is_offline
from voice_agent_next.utils.download import DownloadError, download


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VAN_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path))


@pytest.mark.parametrize("var", ["VAN_OFFLINE", "HF_HUB_OFFLINE"])
@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_either_variable_turns_offline_mode_on(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    assert not is_offline()
    monkeypatch.setenv(var, value)
    assert is_offline() and models.is_offline()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_falsy_values_keep_downloads_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("VAN_OFFLINE", value)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert not is_offline()


def test_hf_hub_offline_blocks_downloads_and_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(DownloadError, match="not cached"):
        download("https://example.invalid/model.bin")
    with pytest.raises(DownloadError, match="not cached"):
        load_quality_dataset("voicebench-advbench-smoke")
    with pytest.raises(DownloadError, match="not cached"):
        load_eot_dataset("eot-bench-en")
    with pytest.raises(DownloadError, match="not cached"):
        load_asr_dataset("librispeech-test-clean-smoke")


def test_doctor_reports_offline_mode_from_hf_hub_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from voice_agent_next import doctor

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    names = [c.name for c in doctor.model_checks()]
    assert "offline mode" not in names
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert "offline mode" in [c.name for c in doctor.model_checks()]
