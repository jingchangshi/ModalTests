"""Local gates for the P0 checkout resolution and the P2 task-to-execution match.

The claim under test is narrow and checkable without a GPU, a network, or a Modal account:
a run either executes exactly the stages its task declares, or it says so. Each test here
fails if the code goes back to reading the host machine, to resolving a path from the
current working directory, or to reporting a stage it did not run as a success.

Nothing here imports real Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))


class _PassthroughImage:
    """An Image stand-in that chains every builder method and records the build."""

    last_build: dict = {}

    @classmethod
    def from_dockerfile(cls, dockerfile, **kwargs):
        _PassthroughImage.last_build = {"dockerfile": dockerfile, **kwargs}
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _PassthroughVolume:
    """A Volume stand-in, so importing the entrypoint does not need a Modal account."""

    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _RecordingFunction:
    """A Modal function stand-in that records every remote call it receives."""

    def __init__(self, name: str, calls: dict) -> None:
        self.name = name
        self._calls = calls

    def remote(self, *args, **kwargs):
        self._calls["remote"] += 1
        self._calls["names"].append(self.name)
        raise AssertionError(f"{self.name}.remote was called")


def _install_stub_modal(calls: dict) -> None:
    """Register a Modal stub that counts remote calls instead of making them."""
    module = types.ModuleType("modal")
    module.Image = _PassthroughImage
    module.Volume = _PassthroughVolume
    module.method = lambda **kwargs: (lambda fn: fn)
    module.is_local = lambda: True

    class _App:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **kwargs):
            return lambda fn: _RecordingFunction(fn.__name__, calls)

        def local_entrypoint(self, **kwargs):
            return lambda fn: fn

    module.App = _App
    sys.modules["modal"] = module


@pytest.fixture(scope="module")
def run_demo():
    """Import ``run_demo`` once against a recording Modal stub."""
    calls = {"remote": 0, "names": []}
    _install_stub_modal(calls)
    sys.modules.pop("run_demo", None)
    module = importlib.import_module("run_demo")
    module._calls = calls
    yield module
    staging = getattr(module, "STAGING", None)
    if staging is not None and Path(staging).exists():
        module.cleanup(module.SNAPSHOT)
    sys.modules.pop("run_demo", None)


def _checkout(root: Path) -> Path:
    """Create a directory that looks enough like a checkout to be accepted."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = \"tilefoundry\"\n")
    return root


def test_the_explicit_argument_wins_over_everything(run_demo, tmp_path: Path, monkeypatch) -> None:
    """A path the caller names is used even when the environment names another."""
    explicit = _checkout(tmp_path / "explicit")
    other = _checkout(tmp_path / "environment")
    monkeypatch.setenv("TILEFOUNDRY_REPO", str(other))
    assert run_demo.resolve_tilefoundry_repo(str(explicit)) == explicit.resolve()


def test_the_environment_variable_is_used_when_no_argument_is_given(
    run_demo, tmp_path: Path, monkeypatch
) -> None:
    """The documented quick start exports the variable and passes no flag."""
    checkout = _checkout(tmp_path / "from-env")
    monkeypatch.setenv("TILEFOUNDRY_REPO", str(checkout))
    assert run_demo.resolve_tilefoundry_repo() == checkout.resolve()


def test_the_sibling_checkout_beside_the_repository_is_found(
    run_demo, tmp_path: Path, monkeypatch
) -> None:
    """With no argument and no variable, a sibling clone is discovered automatically.

    The documented quick start clones both repositories under one workspace directory, so
    the lookup must reach the directory that holds the ModalTests root, not the demo
    directory. A demo root nested one level deeper proves which of the two is used.
    """
    monkeypatch.delenv("TILEFOUNDRY_REPO", raising=False)
    repo = tmp_path / "workspace" / "ModalTests"
    demo = repo / "tilefoundry_demo"
    demo.mkdir(parents=True)
    sibling = _checkout(tmp_path / "workspace" / "TileFoundry")
    assert run_demo.resolve_tilefoundry_repo(demo_root=demo) == sibling.resolve()


def test_a_missing_checkout_raises_before_any_image_is_built(
    run_demo, tmp_path: Path, monkeypatch
) -> None:
    """No candidate is fatal, and the message says what to do rather than what failed."""
    monkeypatch.delenv("TILEFOUNDRY_REPO", raising=False)
    demo = tmp_path / "lonely" / "tilefoundry_demo"
    demo.mkdir(parents=True)
    with pytest.raises(run_demo.CheckoutNotFound) as caught:
        run_demo.resolve_tilefoundry_repo(demo_root=demo)
    message = str(caught.value)
    assert message.startswith(
        "no TileFoundry checkout was found, so no Modal image was built."
    )
    # The fix has to be in the message: a bare failure would leave the reader guessing.
    assert "TILEFOUNDRY_REPO" in message
    assert "git clone" in message
    assert "pyproject.toml" in message


def test_a_candidate_without_the_marker_is_not_accepted(
    run_demo, tmp_path: Path, monkeypatch
) -> None:
    """A directory that merely has the name is not a checkout."""
    monkeypatch.delenv("TILEFOUNDRY_REPO", raising=False)
    demo = tmp_path / "workspace" / "ModalTests" / "tilefoundry_demo"
    demo.mkdir(parents=True)
    (tmp_path / "workspace" / "TileFoundry").mkdir(parents=True)
    with pytest.raises(run_demo.CheckoutNotFound):
        run_demo.resolve_tilefoundry_repo(demo_root=demo)


def test_resolution_ignores_the_current_working_directory(
    run_demo, tmp_path: Path, monkeypatch
) -> None:
    """Two different working directories must resolve the same checkout.

    This is the cross-server property: a new machine has a different home directory and a
    different place to run from, so nothing may depend on where the command was typed.
    """
    monkeypatch.delenv("TILEFOUNDRY_REPO", raising=False)
    repo = tmp_path / "ws" / "ModalTests"
    demo = repo / "tilefoundry_demo"
    demo.mkdir(parents=True)
    sibling = _checkout(tmp_path / "ws" / "TileFoundry")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    first = run_demo.resolve_tilefoundry_repo(demo_root=demo)
    monkeypatch.chdir(demo)
    second = run_demo.resolve_tilefoundry_repo(demo_root=demo)
    assert first == second == sibling.resolve()


def test_the_empty_environment_variable_falls_through(
    run_demo, tmp_path: Path, monkeypatch
) -> None:
    """An exported-but-empty variable is not a path, so it must not shadow the sibling."""
    repo = tmp_path / "ws" / "ModalTests"
    demo = repo / "tilefoundry_demo"
    demo.mkdir(parents=True)
    sibling = _checkout(tmp_path / "ws" / "TileFoundry")
    monkeypatch.setenv("TILEFOUNDRY_REPO", "")
    assert run_demo.resolve_tilefoundry_repo(demo_root=demo) == sibling.resolve()
