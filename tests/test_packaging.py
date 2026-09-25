"""Release packaging: the version, and what the wheel and sdist contain (issue #51).

The distributions are built in-process with hatchling's PEP 517 hooks (dev dependencies):
no network, no subprocess, about a second. The release workflow repeats the checks on the
real `uv build` output and installs the wheel on Linux, macOS and Windows.
"""

from __future__ import annotations

import re
import tarfile
import tomllib
import zipfile
from email.parser import HeaderParser
from pathlib import Path

import pytest

import voice_agent_next

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not PYPROJECT.is_file(), reason="needs the source tree (pyproject.toml)"
)

# PEP 440 public version (canonical form, as `uv version` writes it)
_PEP440 = re.compile(
    r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$",
)
MAX_WHEEL_BYTES = 4_000_000
MAX_SDIST_BYTES = 4_000_000


def _project() -> dict[str, object]:
    with PYPROJECT.open("rb") as f:
        project: dict[str, object] = tomllib.load(f)["project"]
    return project


def test_version_is_single_sourced_from_pyproject() -> None:
    version = voice_agent_next.__version__
    assert isinstance(version, str)
    assert _PEP440.match(version), version
    # installed metadata (editable or not) mirrors pyproject.toml, the only place it is set
    assert version == _project()["version"]


def test_changelog_exists_with_unreleased_or_current_section() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.startswith("# Changelog")
    version = str(_project()["version"])
    if ".dev" in version:
        assert "## [Unreleased]" in changelog
    else:  # a release commit: `git-cliff --tag v<version>` wrote its section
        assert f"## [{version}]" in changelog


@pytest.fixture(scope="module")
def dists(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    hatch_build = pytest.importorskip("hatchling.build")
    pytest.importorskip("hatch_fancy_pypi_readme")
    out = tmp_path_factory.mktemp("dist")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)  # PEP 517 hooks run from the project root
        wheel = out / hatch_build.build_wheel(str(out))
        sdist = out / hatch_build.build_sdist(str(out))
    return {"wheel": wheel, "sdist": sdist}


def _metadata(text: str) -> tuple[dict[str, list[str]], str]:
    msg = HeaderParser().parsestr(text)
    fields: dict[str, list[str]] = {}
    for key, value in msg.items():
        fields.setdefault(key, []).append(value)
    return fields, str(msg.get_payload())


def test_wheel_metadata(dists: dict[str, Path]) -> None:
    wheel = dists["wheel"]
    version = voice_agent_next.__version__
    assert wheel.name == f"voice_agent_next-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as zf:
        meta_text = zf.read(f"voice_agent_next-{version}.dist-info/METADATA").decode()
        entry_points = zf.read(f"voice_agent_next-{version}.dist-info/entry_points.txt").decode()
    fields, description = _metadata(meta_text)

    assert fields["Name"] == ["voice-agent-next"]
    assert fields["Version"] == [version]
    assert fields["Requires-Python"] == [">=3.11"]
    assert fields["License-Expression"] == ["Apache-2.0"]
    assert fields["License-File"] == ["LICENSE"]
    assert fields["Description-Content-Type"] == ["text/markdown"]
    assert "Typing :: Typed" in fields["Classifier"]
    urls = dict(u.split(", ", 1) for u in fields["Project-URL"])
    assert {"Homepage", "Documentation", "Repository", "Issues", "Changelog"} <= urls.keys()
    assert all(u.startswith("https://") for u in urls.values())

    # the README is the long description, with every link absolute (PyPI has no repo around it)
    assert description.lstrip().startswith("# voice-agent-next")
    links = re.findall(r"\]\(([^)]+)\)", description)
    assert links
    relative = [link for link in links if not link.startswith(("https://", "http://", "#"))]
    assert not relative, relative

    assert "van = voice_agent_next.cli.main:main" in entry_points


def test_wheel_contents(dists: dict[str, Path]) -> None:
    wheel = dists["wheel"]
    assert wheel.stat().st_size < MAX_WHEEL_BYTES
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    top_level = {n.split("/", 1)[0] for n in names}
    version = voice_agent_next.__version__
    assert top_level == {"voice_agent_next", f"voice_agent_next-{version}.dist-info"}
    assert "voice_agent_next/py.typed" in names
    assert "voice_agent_next/cli/main.py" in names
    assert f"voice_agent_next-{version}.dist-info/licenses/LICENSE" in names
    assert not [n for n in names if "__pycache__" in n or n.endswith((".pyc", ".wav", ".onnx"))]


def test_sdist_contents(dists: dict[str, Path]) -> None:
    sdist = dists["sdist"]
    assert sdist.stat().st_size < MAX_SDIST_BYTES
    version = voice_agent_next.__version__
    prefix = f"voice_agent_next-{version}/"
    with tarfile.open(sdist) as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    assert all(n.startswith(prefix) for n in names)
    rel = {n[len(prefix) :] for n in names}
    for required in ("pyproject.toml", "PKG-INFO", "README.md", "CHANGELOG.md", "LICENSE"):
        assert required in rel, required
    assert "src/voice_agent_next/py.typed" in rel
    top_level = {n.split("/", 1)[0] for n in rel}
    assert top_level <= {
        "src",
        "pyproject.toml",
        "PKG-INFO",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        ".gitignore",
    }, top_level
    assert not [n for n in rel if "__pycache__" in n or n.endswith((".pyc", ".wav", ".onnx"))]
