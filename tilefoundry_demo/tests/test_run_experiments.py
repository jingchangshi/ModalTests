"""Local gates for the P1 reference-data contract.

The claim: the RMSNorm reference tensors are generated per run, in a Modal CPU container,
and a generation that fails stops the run before any GPU is allocated. Every test here
fails if the code goes back to uploading committed tensors, to reading a shared directory,
or to reaching a GPU stage after generation reported a problem.

Nothing here imports real Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))


class _PassthroughImage:
    """An Image stand-in that chains every builder method.

    from_dockerfile is defined because importing run_experiments imports run_demo, and
    that module builds its image at import time from the frozen snapshot. Stubbing it
    keeps the test offline while still exercising the real import path.
    """

    @classmethod
    def from_dockerfile(cls, dockerfile, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _RecordingVolume:
    """A Volume stand-in that records uploads and commits instead of performing them."""

    uploads: list = []
    commits = 0
    reloads = 0

    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    @contextlib.contextmanager
    def batch_upload(self, **kwargs):
        class _Batch:
            @staticmethod
            def put_file(local: str, remote: str) -> None:
                _RecordingVolume.uploads.append((Path(local).name, remote))

        yield _Batch()

    def commit(self) -> None:
        _RecordingVolume.commits += 1

    def reload(self) -> None:
        _RecordingVolume.reloads += 1


class _RecordingFunction:
    """A Modal function stand-in that records its calls and returns a scripted value."""

    def __init__(self, name: str, calls: dict, scripted: dict) -> None:
        self.name = name
        self._calls = calls
        self._scripted = scripted

    def remote(self, *args, **kwargs):
        self._calls["remote"] += 1
        self._calls["names"].append(self.name)
        if self.name in self._scripted:
            value = self._scripted[self.name]
            return value(*args, **kwargs) if callable(value) else value
        raise AssertionError(f"{self.name}.remote was called, but it was not expected")


def _install_stub_modal(calls: dict, scripted: dict) -> None:
    """Register a Modal stub whose Functions replay *scripted* results."""
    module = types.ModuleType("modal")
    module.Image = _PassthroughImage
    module.Volume = _RecordingVolume
    module.method = lambda **kwargs: (lambda fn: fn)
    module.is_local = lambda: True

    class _App:
        def __init__(self, name: str) -> None:
            self.name = name

        def function(self, **kwargs):
            return lambda fn: _RecordingFunction(fn.__name__, calls, scripted)

        def local_entrypoint(self, **kwargs):
            return lambda fn: fn

    module.App = _App
    sys.modules["modal"] = module


@pytest.fixture(scope="module")
def experiments():
    """Import run_experiments once against a recording Modal stub."""
    calls = {"remote": 0, "names": []}
    scripted: dict = {}
    _install_stub_modal(calls, scripted)
    sys.modules.pop("run_experiments", None)
    module = importlib.import_module("run_experiments")
    module._calls = calls
    module._scripted = scripted
    module._volume = _RecordingVolume
    yield module
    sys.modules.pop("run_experiments", None)


@pytest.fixture(autouse=True)
def _reset_volume(monkeypatch, tmp_path: Path, experiments):
    """Clear the recorded Volume traffic and keep run evidence out of the checkout.

    main() reserves a local directory per run id, exclusively by design. Pointing the
    module HERE at a temporary directory keeps these tests from writing under the
    repository and from colliding with a run id an earlier attempt already claimed.
    """
    _RecordingVolume.uploads = []
    _RecordingVolume.commits = 0
    _RecordingVolume.reloads = 0
    monkeypatch.setattr(experiments, "HERE", tmp_path)
    yield


def _generated(run_id: str) -> dict:
    """A generation record shaped like a successful one."""
    return {
        "run_id": run_id,
        "returncode": 0,
        "files": {
            "x.pt": {"size": 10, "sha256": "a" * 64},
            "expected.pt": {"size": 20, "sha256": "b" * 64},
            "model.safetensors": {"size": 30, "sha256": "c" * 64},
        },
        "generator": {"script": "make_reference.py"},
        "versions": {"torch": "2.9.1"},
        "provenance": {"seed": 0},
    }


def test_the_reference_tensors_are_not_uploaded(experiments) -> None:
    """The generator produces them; the checkout must not carry any of them."""
    uploaded = set(experiments.SOURCE_FILES)
    for name in experiments.REFERENCE_FILES:
        assert name not in uploaded, f"{name} must be generated, not uploaded"


def test_the_generation_inputs_are_uploaded(experiments) -> None:
    """The generator needs its script and the published config in the container."""
    experiments._volume.uploads = []
    experiments.upload_source("run-abc")
    names = {name for name, _ in experiments._volume.uploads}
    assert "make_reference.py" in names
    assert "config.json" in names
    for reference in experiments.REFERENCE_FILES:
        assert reference not in names


def test_every_upload_lands_under_the_run_prefix(experiments) -> None:
    """Two runs must not share a prefix, or one could read the other data."""
    experiments._volume.uploads = []
    experiments.upload_source("run-one")
    experiments.upload_source("run-two")
    prefixes = {remote.rsplit("/", 1)[0] for _, remote in experiments._volume.uploads}
    assert len(prefixes) == 2, prefixes
    for prefix in prefixes:
        assert prefix.startswith("experiments/")
        assert "/cases/rmsnorm_fp8" in prefix


def test_the_prefix_carries_the_run_id(experiments) -> None:
    """The run id is what makes the prefix run-scoped, so it must be the only variable."""
    assert experiments.run_prefix("abc") == "experiments/abc/cases/rmsnorm_fp8"
    assert experiments.run_prefix("xyz") == "experiments/xyz/cases/rmsnorm_fp8"


def test_an_unsafe_run_id_is_refused(experiments) -> None:
    """A run id becomes a path component on both sides, so it is validated."""
    for bad in ("../escape", "a/b", "", "with space"):
        with pytest.raises(ValueError):
            experiments.run_prefix(bad)


def test_a_generation_error_stops_the_run_before_any_gpu(experiments) -> None:
    """The central P1 guarantee: a failed generation costs no GPU time."""
    experiments._calls["remote"] = 0
    experiments._calls["names"].clear()
    experiments._scripted["reference_generator"] = {
        "run_id": "run-fail",
        "returncode": 1,
        "error": "make_reference.py exited 1",
    }
    try:
        with pytest.raises(SystemExit) as caught:
            experiments.main(experiment="all", run_id="run-fail-p1")
    finally:
        experiments._scripted.clear()
    message = str(caught.value)
    assert "reference generation failed" in message
    assert "No GPU was allocated" in message, message
    assert experiments._calls["names"] == ["reference_generator"], experiments._calls["names"]


def test_a_generation_without_digests_stops_the_run(experiments) -> None:
    """A record that claims success but names no digest proves nothing, so it stops too."""
    experiments._calls["remote"] = 0
    experiments._calls["names"].clear()
    experiments._scripted["reference_generator"] = {
        "run_id": "run-empty",
        "returncode": 0,
        "files": {},
    }
    try:
        with pytest.raises(SystemExit) as caught:
            experiments.main(experiment="all", run_id="run-empty-p1")
    finally:
        experiments._scripted.clear()
    assert "no digest" in str(caught.value)
    assert experiments._calls["names"] == ["reference_generator"]


def test_a_clean_generation_then_reaches_the_gpu_stages(experiments) -> None:
    """The gate must not block a run whose generation is good."""
    experiments._calls["remote"] = 0
    experiments._calls["names"].clear()
    experiments._scripted["reference_generator"] = _generated("run-good")
    experiments._scripted["correctness_experiment"] = {"returncode": 0}
    experiments._scripted["twin_experiment"] = {"returncode": 0}
    experiments._scripted["benchmark_experiment"] = {"returncode": 0}
    try:
        experiments.main(experiment="all", run_id="run-good-p1")
    finally:
        experiments._scripted.clear()
    assert experiments._calls["names"] == [
        "reference_generator",
        "correctness_experiment",
        "twin_experiment",
        "benchmark_experiment",
    ]


def test_the_generator_runs_without_a_device(experiments) -> None:
    """Generation must ask for no GPU, so a broken generator costs only container time."""
    source = (DEMO_ROOT / "run_experiments.py").read_text()
    decorator = source.split("def reference_generator", 1)[0].rsplit("@app.function(", 1)[1]
    decorator = decorator.split(")", 1)[0]
    assert "gpu=" not in decorator, f"the generator must not request a GPU: {decorator}"


def test_a_failed_experiment_is_reported_as_a_failure(experiments) -> None:
    """A stage that returned an error must not leave the command with a success exit."""
    experiments._calls["remote"] = 0
    experiments._calls["names"].clear()
    experiments._scripted["reference_generator"] = _generated("run-bad")
    experiments._scripted["correctness_experiment"] = {"raised": "Traceback: boom"}
    try:
        with pytest.raises(SystemExit) as caught:
            experiments.main(experiment="correctness", run_id="run-bad-p1")
    finally:
        experiments._scripted.clear()
    assert "correctness" in str(caught.value)


def test_an_unknown_experiment_is_refused_before_anything_runs(experiments) -> None:
    """A typo costs no upload, no container, and no GPU."""
    experiments._calls["remote"] = 0
    experiments._calls["names"].clear()
    experiments._volume.uploads = []
    with pytest.raises(SystemExit) as caught:
        experiments.main(experiment="benchmrk", run_id="run-typo")
    assert "unknown experiment" in str(caught.value)
    assert experiments._calls["remote"] == 0
    assert experiments._volume.uploads == []
