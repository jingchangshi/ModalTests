"""Contract tests for the container orchestration, runnable with no GPU.

These tests import ``demo_container`` and drive it through a temporary
TileFoundry-shaped tree with a stub test process, so they catch wiring errors —
a misspelled helper, a wrong keyword, a stage that silently reports ``NOT_RUN``
— before an H200 is ever allocated.

Nothing here imports torch, tilefoundry, or Modal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_container  # noqa: E402
import demo_report  # noqa: E402
import demo_task  # noqa: E402

DEMO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def fake_source(tmp_path: Path) -> Path:
    """A tree with the fixtures the demo points at, but no compiler behind them."""
    root = tmp_path / "tilefoundry"
    (root / "tests/nvidia/cuda/fixtures/cli/products").mkdir(parents=True)
    (root / "tests/nvidia/cuda/fixtures/cli/products/h200_sxm.py").write_text("")
    (root / "tests/nvidia/cuda/integration").mkdir(parents=True)
    (root / "tests/nvidia/cuda/fixtures/tir").mkdir(parents=True)
    (root / "tests/nvidia/cuda/fixtures/tir/square.py").write_text("")
    return root


def _spec(**overrides) -> dict:
    """A validated CPU task, so the orchestration tests exercise the real contract."""
    spec = {
        "schema_version": 1,
        "task_id": "square_cpu",
        "description": "analyze and check on CPU",
        "source_snapshot": {"mode": "local", "expected_sha256": None},
        "target": "nvidia.h200_sxm",
        "gpu": {"type": "H200", "count": 0},
        "stages": [
            {"stage_id": "environment", "type": "environment", "params": {}},
            {
                "stage_id": "analyze",
                "type": "hir_analyze",
                "params": {"source_id": "h200_model", "analyses": ["compute-cost", "memory"], "dims": {}},
            },
            {
                "stage_id": "check",
                "type": "hir_check",
                "params": {
                    "source_id": "h200_model",
                    "input_ids": [],
                    "comparisons": [{"out": "output", "predicates": [{"fn": "nan_inf", "bounds": {}}]}],
                    "dims": {},
                },
            },
            {"stage_id": "archive", "type": "archive", "params": {}},
        ],
        "inputs": [],
        "expected_outcomes": {},
        "artifacts": {"include": ["logs", "results", "generated"], "required": []},
        "limits": dict(demo_task.DEFAULT_LIMITS),
    }
    spec.update(overrides)
    return demo_task.validate_task(spec)


def _cpu_task() -> dict:
    """The validated CPU task an orchestration payload carries."""
    return _spec()


def _gpu_task() -> dict:
    """The validated GPU task the shipped spec defines, as a payload carries it."""
    return demo_task.validate_task(
        demo_task.load_task(str(DEMO_ROOT / "tasks" / "square.json"))
    )


def test_analysis_command_targets_the_declared_source(fake_source: Path, tmp_path: Path) -> None:
    """The analyze stage runs the argv the task definition produced, from the source root."""
    seen = {}

    def fake_run(command, *, cwd, log, env=None):
        seen["command"] = command
        seen["cwd"] = cwd
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text("stub\n")
        return 0

    spec = _spec()
    analyze = spec["stages"][1]
    argv = demo_task.stage_command(analyze, {"report_dir": tmp_path / "results", "inputs": {}})
    original = demo_container._run_command
    demo_container._run_command = fake_run
    try:
        demo_container.run_analysis(
            fake_source,
            tmp_path / "results",
            tmp_path / "logs",
            argv=argv,
            analyses=analyze["params"]["analyses"],
        )
    finally:
        demo_container._run_command = original

    command = seen["command"]
    assert command[:3] == ["tilefoundry", "analyze", demo_task._source_selector("h200_model")]
    assert "--compute-cost" in command and "--memory" in command and "--json" in command
    assert "--analysis" not in command
    assert seen["cwd"] == fake_source


def test_analysis_missing_report_is_recorded_not_raised(fake_source: Path, tmp_path: Path) -> None:
    """A zero exit code with no report is evidence of a gap, not an exception."""
    spec = _spec()
    analyze = spec["stages"][1]
    argv = demo_task.stage_command(analyze, {"report_dir": tmp_path / "results", "inputs": {}})
    original = demo_container._run_command
    demo_container._run_command = lambda *args, **kwargs: 0
    try:
        analysis = demo_container.run_analysis(
            fake_source,
            tmp_path / "results",
            tmp_path / "logs",
            argv=argv,
            analyses=analyze["params"]["analyses"],
        )
    finally:
        demo_container._run_command = original
    assert analysis["json_ok"] is False
    assert demo_report.evaluate_analysis(analysis)[0] != demo_report.STATUS_PASS


def test_analysis_reads_back_the_executed_analyses(fake_source: Path, tmp_path: Path) -> None:
    """The stage reports what the JSON says was executed, not what was requested."""
    results = tmp_path / "results"
    results.mkdir()

    def fake_run(command, *, cwd, log, env=None):
        Path(command[3]).write_text(
            json.dumps(
                {
                    "target": "nvidia.h200_sxm",
                    "module": "Model",
                    "requested": ["compute-cost", "memory"],
                    "executed": ["compute-cost", "memory"],
                    "functions": [{"calls": [{"compute-cost": {}, "memory": {}}]}],
                    "source": "annotated",
                }
            )
        )
        return 0

    spec = _spec()
    analyze = spec["stages"][1]
    argv = demo_task.stage_command(analyze, {"report_dir": results, "inputs": {}})
    original = demo_container._run_command
    demo_container._run_command = fake_run
    try:
        analysis = demo_container.run_analysis(
            fake_source, results, tmp_path / "logs",
            argv=argv,
            analyses=analyze["params"]["analyses"],
        )
    finally:
        demo_container._run_command = original
    assert analysis["function_count"] == 1
    assert analysis["call_count"] == 1
    assert demo_report.evaluate_analysis(analysis)[0] == demo_report.STATUS_PASS


def test_run_check_reads_the_report_it_asked_for(fake_source: Path, tmp_path: Path) -> None:
    """A check stage is judged by the JSON the CLI wrote, not by its exit code alone."""
    results = tmp_path / "results"
    results.mkdir()

    def fake_run(command, *, cwd, log, env=None):
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "passed": True,
                    "runs": [
                        {
                            "outputs": [
                                {
                                    "path": "output",
                                    "fns": [{"fn": "nan_inf", "nan": 0.0, "inf": 0.0, "passed": True}],
                                }
                            ]
                        }
                    ],
                }
            )
        )
        return 0

    spec = _spec()
    check_stage = spec["stages"][2]
    argv = demo_task.stage_command(check_stage, {"report_dir": results, "inputs": {}})
    original = demo_container._run_command
    demo_container._run_command = fake_run
    try:
        check = demo_container.run_check(fake_source, results, tmp_path / "logs", argv=argv)
    finally:
        demo_container._run_command = original
    assert check["json_ok"] is True
    assert check["passed"] is True
    assert check["predicates"] == [{"out": "output", "fn": "nan_inf", "passed": True}]


def test_cmake_step_needs_a_command_and_a_clean_stderr(tmp_path: Path) -> None:
    """The dumped command proves the attempt; the dumped stderr proves the outcome."""
    build = tmp_path / "build"
    build.mkdir()
    assert demo_container._cmake_step_succeeded(build, "cmake-build") is False
    (build / "cmake-build.cmd.txt").write_text("cmake --build")
    assert demo_container._cmake_step_succeeded(build, "cmake-build") is False
    (build / "cmake-build.stderr").write_text("")
    assert demo_container._cmake_step_succeeded(build, "cmake-build") is True
    (build / "cmake-build.stderr").write_text("error: nvcc failed")
    assert demo_container._cmake_step_succeeded(build, "cmake-build") is False


def test_dump_build_dir_follows_the_suite_dump_layout(tmp_path: Path) -> None:
    """The build directory is derived from the dump root the suite really writes."""
    nested = tmp_path / "generated" / "test_tir_control_flow" / "for_if_square_matches_torch" / "build"
    nested.mkdir(parents=True)
    assert demo_container._dump_build_dir(tmp_path / "generated") == nested


def test_dump_build_dir_reports_a_missing_dump_instead_of_guessing(tmp_path: Path) -> None:
    """An empty dump root yields a path that cannot contain evidence."""
    derived = demo_container._dump_build_dir(tmp_path / "generated")
    assert not derived.exists()
    assert demo_container._cmake_step_succeeded(derived, "cmake-build") is False


def test_run_reports_missing_stages_without_claiming_success(tmp_path: Path, monkeypatch) -> None:
    """The orchestration path itself, with every external call stubbed out."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    volume = tmp_path / "volume"
    volume.mkdir()
    payload = {
        "run_id": "test-run",
        "task": _gpu_task(),
        "snapshot": {
            "commit": "0" * 40,
            "branch": "main",
            "dirty": False,
            "dirty_entries": [],
            "snapshot_digest": "d" * 64,
            "snapshot_file_count": 1,
            "cutlass_sha": "e406c186f510a15091cce01f782020ceb7ba8eb5",
        },
        "inventory": [{"path": "pyproject.toml", "size": 23, "sha256": demo_report.sha256_file(source / "pyproject.toml")}],
    }
    monkeypatch.setenv("TILEFOUNDRY_DEMO_VOLUME", str(volume))

    def fake_environment(source_root, log):
        return {
            "imports_ok": True,
            "target_registered": True,
            "tools_ok": True,
            "cuda_available": True,
            "gpu_name": "NVIDIA H200",
            "compute_capability": [9, 0],
            "python_version": "3.12.0",
            "tilefoundry_version": "0.0.0+demo.dddd",
            "tilefoundry_nvidia_version": "0.0.0+demo.dddd",
            # Origins follow the real source root, so `_run`'s provenance re-check sees a
            # coherent install and the assertions below test orchestration, not this stub.
            "tilefoundry_origin": f"{source_root}/src/tilefoundry/__init__.py",
            "tilefoundry_nvidia_origin": (
                f"{source_root}/targets/nvidia/tilefoundry_nvidia/__init__.py"
            ),
            "origin_inside_snapshot": True,
            "installations": {
                "tilefoundry": {"url": f"file://{source_root}", "dir_info": {"editable": True}},
                "tilefoundry-nvidia": {
                    "url": f"file://{source_root}/targets/nvidia",
                    "dir_info": {"editable": True},
                },
            },
        }

    def fake_analysis(source_root, results_dir, logs_dir, *, argv, analyses):
        return {
            "returncode": 0,
            "json_ok": True,
            "executed": ["compute-cost", "memory"],
            "target": "nvidia.h200_sxm",
            "command": argv[2],
            "argv": argv,
        }

    def fake_test(source_root, results_dir, logs_dir, dump_root, *, node_id):
        return {
            "pytest": {
                "returncode": 0,
                "elapsed_seconds": 1.0,
                "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
                "command": node_id,
            },
            "compile": {
                "calls": [{"runtime_name": "M.host"}],
                "runtime_name": "M.host",
                "cu_files": ["module.device_0.cu"],
                "cmake_configured": True,
                "cmake_built": True,
                "target_arch": "sm_90",
                "libraries": [{"path": "generated/libM.so", "size": 1, "sha256": "ab"}],
            },
            "execution": {"runtime_calls": 1, "synchronize_calls": 1, "errors": []},
            "correctness": {
                "assert_close_calls": 1,
                "comparisons": [{"rtol": 0, "atol": 0, "shapes_match": True}],
                "errors": [],
            },
        }

    monkeypatch.setattr(demo_container, "collect_environment", fake_environment)
    monkeypatch.setattr(demo_container, "run_analysis", fake_analysis)
    monkeypatch.setattr(demo_container, "run_test", fake_test)

    manifest = demo_container._run(payload, source)
    # The GPU path must attach the inventory comparison itself; the environment
    # observer does not produce it, so a passing source-integrity gate here
    # proves the attachment, not the stub.
    assert manifest["evidence"]["environment"]["source_check"]["post_image_build"] == {
        "missing": [],
        "unexpected": [],
        "modified": [],
    }
    assert manifest["environment_status"] == "PASS"
    assert manifest["analysis_status"] == "PASS"
    assert manifest["compile_status"] == "PASS"
    assert manifest["execution_status"] == "PASS"
    assert manifest["correctness_status"] == "PASS"
    assert manifest["overall_status"] == "PASS", manifest["overall_reason"]
    assert "artifacts_bundle" in manifest
    archived = volume / manifest["artifacts_bundle"]["archive_path"]
    assert archived.is_file()
    assert manifest["artifacts_bundle"]["archive_sha256"] == demo_report.sha256_file(archived)
    assert manifest["run_id"] == "test-run"


def test_run_fails_when_the_installed_tree_lost_a_selected_file(tmp_path: Path, monkeypatch) -> None:
    """A tree that does not match the digested inventory is not a passing run."""
    source = tmp_path / "src"
    source.mkdir()
    volume = tmp_path / "volume"
    volume.mkdir()
    payload = {
        "run_id": "drift-run",
        "task": _cpu_task(),
        "snapshot": {
            "commit": "0" * 40,
            "branch": "main",
            "dirty": False,
            "dirty_entries": [],
            "snapshot_digest": "d" * 64,
            "snapshot_file_count": 1,
            "cutlass_sha": "e406c186f510a15091cce01f782020ceb7ba8eb5",
        },
        "inventory": [{"path": "src/missing.py", "size": 1, "sha256": "aa"}],
    }
    monkeypatch.setenv("TILEFOUNDRY_DEMO_VOLUME", str(volume))
    monkeypatch.setattr(
        demo_container,
        "collect_environment",
        lambda source_root, log: {
            "imports_ok": True,
            "target_registered": True,
            "tools_ok": True,
            "origin_inside_snapshot": True,
            # A provenance-clean stub: these tests assert about the inventory drift, so the
            # installation record must not fail on its own and mask the intended reason.
            "installations": {
                "tilefoundry": {"url": f"file://{source}", "dir_info": {"editable": True}},
                "tilefoundry-nvidia": {
                    "url": f"file://{source}/targets/nvidia",
                    "dir_info": {"editable": True},
                },
            },
            "tilefoundry_origin": f"{source}/src/tilefoundry/__init__.py",
            "tilefoundry_nvidia_origin": f"{source}/targets/nvidia/tilefoundry_nvidia/__init__.py",
            "tilefoundry_version": "0.0.0+demo.dddd",
            "tilefoundry_nvidia_version": "0.0.0+demo.dddd",
        },
    )
    monkeypatch.setattr(demo_container, "run_analysis", lambda *args: {})
    monkeypatch.setattr(demo_container, "run_test", lambda *args: {})

    manifest = demo_container._run(payload, source)
    assert manifest["environment_status"] == "FAIL"
    assert "missing" in manifest["overall_reason"]
    assert manifest["overall_status"] != "PASS"


def test_compiled_arch_reads_the_nvcc_flag_as_cmake_quotes_it(tmp_path: Path) -> None:
    """The real dump quotes the flag, so the parser matches a substring, not a prefix."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "cmake-build.stdout").write_text(
        'cd /w && /usr/local/cuda/bin/nvcc -x cu -c device_0.cu '
        '"--generate-code=arch=compute_90,code=[compute_90,sm_90]" -o device_0.cu.o\n'
    )
    assert demo_container._compiled_arch(build) == "sm_90"


def test_compiled_arch_returns_none_without_the_flag(tmp_path: Path) -> None:
    """A build log with no architecture yields no claim, which the gate rejects."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "cmake-build.stdout").write_text("cmake --build . --verbose\n")
    assert demo_container._compiled_arch(build) is None
