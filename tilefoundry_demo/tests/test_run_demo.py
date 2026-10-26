"""Local gates for the entrypoint's pre-GPU validation.

``run_demo`` imports Modal, so these tests install a stub module that records every remote
call and refuses to perform one. That makes the central P1 claim checkable without a client,
a network, or a GPU: a task that does not validate never reaches ``.remote()``.

Nothing here imports real Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))


class _RecordingFunction:
    """A Modal function stand-in that fails the test if anything calls it remotely."""

    def __init__(self, name: str, calls: dict) -> None:
        self.name = name
        self._calls = calls

    def remote(self, *args, **kwargs):
        self._calls["remote"] += 1
        self._calls["names"].append(self.name)
        raise AssertionError(f"{self.name}.remote was called, but no remote work was expected")


class _PassthroughImage:
    """An Image stand-in that records how it was built, and chains every builder method."""

    last_build: dict = {}

    @classmethod
    def from_dockerfile(cls, dockerfile, **kwargs):
        _PassthroughImage.last_build = {"dockerfile": dockerfile, **kwargs}
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class _PassthroughVolume:
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


def _install_stub_modal(calls: dict) -> types.ModuleType:
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
    return module


@pytest.fixture(scope="module")
def run_demo():
    """Import ``run_demo`` once against a recording Modal stub.

    The module freezes the real source tree at import time, which is the design's own
    decision for CLI compatibility. One import per module keeps that cost bounded, and the
    staging is removed at teardown so the test run leaves no snapshot behind.
    """
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


@pytest.mark.parametrize(
    "requested", ["../demo_report.py", "demo_report.py", "/etc/passwd", "tasks/../demo_report.py"]
)
def test_a_task_outside_tasks_is_refused(run_demo, requested: str) -> None:
    """A spec is a reviewed file in tasks/, not any path the caller names."""
    with pytest.raises(SystemExit) as caught:
        run_demo.load_and_validate_task(requested)
    assert "outside" in str(caught.value)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s["stages"][2].update(type="shell"), "not one of"),
        (lambda s: s["stages"][2]["params"].update(case_id="/bin/sh"), "not a registered case"),
        (lambda s: s.update(gpu={"type": "H200", "count": 5}), "gpu.count"),
        (lambda s: s.update(schema_version=99), "schema_version"),
        (lambda s: s["stages"][2].update(shell=True), "never accepted"),
        (lambda s: s.update(env={"AWS_SECRET_ACCESS_KEY": "x"}), "never accepted"),
    ],
)
def test_an_invalid_task_is_refused_without_a_remote_call(
    run_demo, tmp_path: Path, mutate, message: str
) -> None:
    """Validation happens locally, so a bad spec costs no image build and no GPU."""
    spec = json.loads((DEMO_ROOT / "tasks/square.json").read_text())
    mutate(spec)
    path = DEMO_ROOT / "tasks" / "_rejected_for_test.json"
    path.write_text(json.dumps(spec))
    try:
        with pytest.raises(SystemExit) as caught:
            run_demo.load_and_validate_task(str(path.relative_to(DEMO_ROOT)))
        assert message in str(caught.value)
    finally:
        path.unlink()
    assert run_demo._calls["remote"] == 0


def test_a_late_invalid_stage_allocates_no_gpu(run_demo) -> None:
    """A valid GPU stage followed by an invalid one must reject the whole spec."""
    spec = json.loads((DEMO_ROOT / "tasks/square.json").read_text())
    spec["stages"].append({"stage_id": "extra", "type": "shell", "params": {"command": "id"}})
    path = DEMO_ROOT / "tasks" / "_late_invalid_for_test.json"
    path.write_text(json.dumps(spec))
    try:
        with pytest.raises(SystemExit):
            run_demo.load_and_validate_task(str(path.relative_to(DEMO_ROOT)))
    finally:
        path.unlink()
    assert run_demo._calls["remote"] == 0


def test_the_shipped_tasks_load(run_demo) -> None:
    """Both reviewed specs resolve, and the CPU-only one declares no device."""
    gpu_task = run_demo.load_and_validate_task("tasks/square.json")
    cpu_task = run_demo.load_and_validate_task("tasks/square_cpu.json")
    assert gpu_task["gpu"]["count"] == 1
    assert cpu_task["gpu"]["count"] == 0


def test_the_default_task_matches_the_cli_default(run_demo) -> None:
    """Bare ``modal run run_demo.py`` must resolve the same spec the design names."""
    assert run_demo.DEFAULT_TASK == "tasks/square.json"
    assert run_demo.load_and_validate_task(run_demo.DEFAULT_TASK)["task_id"] == "square"


def test_the_main_signature_keeps_task_and_existing_flags(run_demo) -> None:
    """Adding ``--task`` must not drop the flags the README already documents."""
    import inspect

    parameters = inspect.signature(run_demo.main).parameters
    assert {"cpu_only", "run_id", "task"} <= set(parameters)
    assert parameters["task"].default == run_demo.DEFAULT_TASK
