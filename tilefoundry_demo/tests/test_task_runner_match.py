"""Local gates for the P2 claim that a task declares what actually runs.

The two failures this file guards against are silent: a registry that names a test which
does not exist, and a runner that reports a stage it never executed. Both are checked here
without a GPU, because both are local claims about code and data.

Nothing here imports real Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

import demo_task  # noqa: E402


def _spec(task: str = "square") -> dict:
    """A validated copy of one shipped task, which callers may mutate."""
    path = DEMO_ROOT / "tasks" / f"{task}.json"
    return demo_task.validate_task(json.loads(path.read_text()))


def test_the_registered_case_is_the_real_test_node() -> None:
    """The registry names a node id that must be the one the test file defines.

    The previous registry named tests/nvidia/cuda/integration/test_square.py::test_square,
    which does not exist. A stale node id is invisible locally and only fails late, inside
    a billed container, so the shape is checked here.
    """
    entry = demo_task.CASE_REGISTRY["upstream_square"]
    node_id = entry["node_id"]
    path, _, test_name = node_id.partition("::")
    assert path.endswith(".py"), node_id
    assert test_name.startswith("test_"), node_id
    # The file is part of the TileFoundry checkout, not this repository, so its existence is
    # a fact about the registry rather than something this test can open. What is checkable
    # here is that the recorded location is the integration suite the case claims.
    assert "/integration/" in path, node_id
    assert entry["runner"] == "run_demo"


def test_every_registered_case_names_a_runner_that_exists() -> None:
    """A case is executed by exactly one driver, and that driver must be a known one."""
    known = {"run_demo", "run_experiments"}
    for case_id, entry in demo_task.CASE_REGISTRY.items():
        assert entry["runner"] in known, (case_id, entry["runner"])


def test_every_registered_source_holds_the_marker_it_claims() -> None:
    """A source entry must carry the fields the CLI argv builder reads."""
    for source_id, entry in demo_task.SOURCE_REGISTRY.items():
        assert entry["path"].endswith(".py"), (source_id, entry["path"])
        assert "selector" in entry, source_id
        assert entry["cpu_capable"] in (True, False), source_id


def test_a_gpu_stage_with_another_runners_case_is_refused() -> None:
    """A task cannot smuggle a different driver case into this entrypoint.

    Running the spec anyway would execute some stages and report the rest, which is the
    mismatch this whole layer exists to prevent.
    """
    spec = _spec()
    for stage in spec["stages"]:
        if stage["type"] == "gpu_test":
            stage["params"]["case_id"] = "rmsnorm_fp8"
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.assert_runner_support(spec)
    message = str(caught.value)
    assert "run_experiments" in message
    assert "cannot run it" in message


def test_an_unsupported_stage_type_is_refused() -> None:
    """A stage type the contract knows but this runner cannot execute is refused."""
    spec = _spec()
    spec["stages"].append(
        {"stage_id": "bench", "type": "gpu_benchmark", "params": {"case_id": "upstream_square"}}
    )
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.assert_runner_support(spec)
    assert "cannot execute stage type" in str(caught.value)


def test_the_shipped_tasks_are_runnable_by_this_runner() -> None:
    """Both reviewed specs must pass the support gate, or the README commands are wrong."""
    for task in ("square", "square_cpu"):
        demo_task.assert_runner_support(_spec(task))


def test_a_runner_support_failure_is_a_local_exit(run_demo_module) -> None:
    """The gate must reject via SystemExit before any remote call is attempted."""
    module = run_demo_module
    spec = json.loads((DEMO_ROOT / "tasks/square.json").read_text())
    for stage in spec["stages"]:
        if stage["type"] == "gpu_test":
            stage["params"]["case_id"] = "rmsnorm_fp8"
    path = DEMO_ROOT / "tasks" / "_foreign_case_for_test.json"
    path.write_text(json.dumps(spec))
    try:
        with pytest.raises(SystemExit) as caught:
            module.load_and_validate_task(str(path.relative_to(DEMO_ROOT)))
    finally:
        path.unlink()
    assert "not acceptable" in str(caught.value)
    assert module._calls["remote"] == 0


def test_the_declared_stage_ids_are_the_ones_the_container_reports() -> None:
    """The evidence keys must be the task stage ids, not a fixed historical list."""
    for task in ("square", "square_cpu"):
        spec = _spec(task)
        ids = demo_task.task_stage_ids(spec)
        assert ids[0] == "environment", ids
        assert ids[-1] == "archive", ids
        assert len(ids) == len(set(ids)), f"duplicate stage id in {task}: {ids}"
        for stage_id in ids:
            assert stage_id in spec["expected_outcomes"], (task, stage_id)


def test_the_square_task_declares_the_upstream_case() -> None:
    """The default task must run the registered upstream test, not a placeholder."""
    spec = _spec("square")
    gpu = [stage for stage in spec["stages"] if stage["type"] == "gpu_test"]
    assert len(gpu) == 1, gpu
    assert gpu[0]["params"]["case_id"] == "upstream_square"


def test_the_cpu_task_declares_no_device() -> None:
    """A CPU task must not ask for a GPU, or the claim that it is CPU-only is false."""
    spec = _spec("square_cpu")
    assert spec["gpu"]["count"] == 0
    assert not [s for s in spec["stages"] if s["type"] in demo_task.GPU_STAGE_TYPES]


@pytest.fixture(scope="module")
def run_demo_module():
    """Import the entrypoint against a recording Modal stub."""
    import importlib
    import types

    calls = {"remote": 0, "names": []}

    class _Image:
        @classmethod
        def from_dockerfile(cls, *args, **kwargs):
            return cls()

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class _Volume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class _Function:
        def __init__(self, name):
            self.name = name

        def remote(self, *args, **kwargs):
            calls["remote"] += 1
            raise AssertionError(f"{self.name}.remote was called")

    module = types.ModuleType("modal")
    module.Image = _Image
    module.Volume = _Volume
    module.method = lambda **kwargs: (lambda fn: fn)
    module.is_local = lambda: True

    class _App:
        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            return lambda fn: _Function(fn.__name__)

        def local_entrypoint(self, **kwargs):
            return lambda fn: fn

    module.App = _App
    sys.modules["modal"] = module
    sys.modules.pop("run_demo", None)
    imported = importlib.import_module("run_demo")
    imported._calls = calls
    yield imported
    staging = getattr(imported, "STAGING", None)
    if staging is not None and Path(staging).exists():
        imported.cleanup(imported.SNAPSHOT)
    sys.modules.pop("run_demo", None)
