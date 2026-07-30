"""The version lives in ./VERSION, and must stay out of the Docker cache key.

The Dockerfile bind-mounts pyproject.toml and uv.lock into the `uv sync` layer, and
BuildKit keys that layer -- ~8GB of dependency installation -- on their contents. A
version string in either file therefore puts a full torch reinstall behind every
release. These tests fail if one creeps back.
"""

import os
import re
import tomllib

from pathlib import Path

import pytest

from src.params import DOCKER_IMAGE, VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def test_version_file_holds_a_semver() -> None:
    raw = (REPO_ROOT / "VERSION").read_text().strip()
    assert SEMVER.match(raw), f"VERSION must be MAJOR.MINOR.PATCH, got {raw!r}"
    assert raw == VERSION


@pytest.mark.skipif(
    os.getenv("TRAINER_IMAGE") is not None,
    reason="TRAINER_IMAGE overrides the tag on purpose",
)
def test_image_tag_follows_the_version_file() -> None:
    assert f"yolo-trainer:{VERSION}" == DOCKER_IMAGE


def test_pyproject_declares_the_version_dynamic() -> None:
    """A static version here would put an 8GB reinstall behind every bump."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = data["project"]

    assert "version" not in project, (
        "pyproject.toml is bind-mounted into the uv sync layer; a version here"
        " invalidates ~8GB of cached dependencies on every release."
        " Keep it in ./VERSION and declare it dynamic."
    )
    assert "version" in project.get("dynamic", [])


def test_lockfile_carries_no_project_version() -> None:
    """uv.lock is the other half of the cache key."""
    data = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    entries = [p for p in data["package"] if p["name"] == "template-yolov8"]

    assert len(entries) == 1
    assert "version" not in entries[0], (
        "uv.lock records a version for the project again -- re-run `uv lock`"
        " after confirming pyproject.toml declares the version dynamic."
    )
