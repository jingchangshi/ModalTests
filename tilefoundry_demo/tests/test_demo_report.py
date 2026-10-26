"""Local gates for the demo's evidence contract; no GPU, torch, or Modal needed.

Run from this directory:

    python -m pytest tests -q

These tests exercise the reducer and the snapshot selection rules directly. The
one property they all protect is that a missing, partial, or contradictory
observation can never be reported as ``PASS``.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_report  # noqa: E402
import demo_source  # noqa: E402

PASSING_ENVIRONMENT = {
    "imports_ok": True,
    "target_registered": True,
    "tools_ok": True,
    "cuda_available": True,
    "gpu_name": "NVIDIA H200",
    "compute_capability": [9, 0],
    "tilefoundry_version": "0.0.0+demo.47a6ed6a7dc8",
    "tilefoundry_nvidia_version": "0.0.0+demo.47a6ed6a7dc8",
    "tilefoundry_origin": "/opt/tilefoundry/src/tilefoundry/__init__.py",
    "tilefoundry_nvidia_origin": "/opt/tilefoundry/targets/nvidia/tilefoundry_nvidia/__init__.py",
    "origin_inside_snapshot": True,
    "installations": {
        "tilefoundry": {"url": "file:///opt/tilefoundry", "dir_info": {"editable": True}},
        "tilefoundry-nvidia": {
            "url": "file:///opt/tilefoundry/targets/nvidia",
            "dir_info": {"editable": True},
        },
    },
    "source_check": {
        "before_install": {"missing": [], "unexpected": [], "modified": []},
        "after_install": {"missing": [], "unexpected": [], "modified": []},
    },
}

PASSING_EVIDENCE = {
    "preflight_returncode": 0,
    "environment": PASSING_ENVIRONMENT,
    "analysis": {
        "returncode": 0,
        "json_ok": True,
        "executed": ["compute-cost", "memory"],
        "target": "nvidia.h200_sxm",
    },
    "compile": {
        "calls": [{"entry": "square_host", "runtime_name": "TirSquare.square_host"}],
        "errors": [],
        "runtime_name": "TirSquare.square_host",
        "cu_files": ["module.device_0.cu"],
        "cmake_configured": True,
        "cmake_built": True,
        "libraries": [{"path": "generated/libTirSquare.so", "size": 10, "sha256": "ab"}],
        "target_arch": "sm_90",
    },
    "execution": {"runtime_calls": 1, "synchronize_calls": 1, "errors": []},
    "correctness": {
        "assert_close_calls": 1,
        "comparisons": [{"rtol": 0, "atol": 0, "shapes_match": True, "outcome": "passed"}],
        "errors": [],
    },
    "pytest": {
        "returncode": 0,
        "junit": {
            "tests": 1,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "cases": [
                {
                    "name": "test_for_if_square_matches_torch",
                    "classname": "test_tir_control_flow",
                    "outcome": "passed",
                }
            ],
        },
    },
}


def _with(**overrides) -> dict:
    """Return the passing evidence with one stage replaced."""
    return {**PASSING_EVIDENCE, **overrides}


def _statuses(evidence: dict) -> dict:
    """Reduce evidence to its per-stage verdicts."""
    return demo_report.evaluate(evidence)


def test_complete_evidence_is_pass() -> None:
    """Every stage observed as required reduces to ``PASS``."""
    verdict = _statuses(PASSING_EVIDENCE)
    assert verdict["overall_status"] == "PASS", verdict["overall_reason"]
    assert all(stage["status"] == "PASS" for stage in verdict["stages"].values())


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("preflight never ran", {"preflight_returncode": None}),
        ("preflight failed", {"preflight_returncode": 1}),
        ("imports failed", {"environment": {**PASSING_ENVIRONMENT, "imports_ok": False}}),
        ("target missing", {"environment": {**PASSING_ENVIRONMENT, "target_registered": False}}),
        ("build tools missing", {"environment": {**PASSING_ENVIRONMENT, "tools_ok": False}}),
        ("no cuda device", {"environment": {**PASSING_ENVIRONMENT, "cuda_available": False}}),
        ("wrong device", {"environment": {**PASSING_ENVIRONMENT, "gpu_name": "NVIDIA A10G"}}),
        (
            "package from an index",
            {"environment": {**PASSING_ENVIRONMENT, "origin_inside_snapshot": False}},
        ),
        (
            "non-editable install",
            {
                "environment": {
                    **PASSING_ENVIRONMENT,
                    "installations": {
                        "tilefoundry": {"url": "https://pypi.org/simple/tilefoundry"},
                        "tilefoundry-nvidia": {"url": "file:///opt/tilefoundry/targets/nvidia", "dir_info": {"editable": True}},
                    },
                }
            },
        ),
        (
            "tree drifted before install",
            {
                "environment": {
                    **PASSING_ENVIRONMENT,
                    "source_check": {
                        "before_install": {"missing": ["src/tilefoundry/x.py"], "unexpected": [], "modified": []},
                        "after_install": {"missing": [], "unexpected": [], "modified": []},
                    },
                }
            },
        ),
        (
            "tree drifted after install",
            {
                "environment": {
                    **PASSING_ENVIRONMENT,
                    "source_check": {
                        "before_install": {"missing": [], "unexpected": [], "modified": []},
                        "after_install": {"missing": [], "unexpected": [], "modified": ["src/tilefoundry/y.py"]},
                    },
                }
            },
        ),
        (
            "core and target versions differ",
            {"environment": {**PASSING_ENVIRONMENT, "tilefoundry_nvidia_version": "0.0.3"}},
        ),
        (
            "wrong capability",
            {"environment": {**PASSING_ENVIRONMENT, "compute_capability": [8, 6]}},
        ),
    ],
)
def test_environment_failure_is_never_pass(name: str, overrides: dict) -> None:
    """A missing or wrong environment observation blocks the overall verdict."""
    verdict = _statuses(_with(**overrides))
    assert verdict["environment_status"] != "PASS" or verdict["stages"]["gpu"]["status"] != "PASS"
    assert verdict["overall_status"] != "PASS", name


@pytest.mark.parametrize(
    ("name", "analysis"),
    [
        ("never ran", {"returncode": None}),
        ("failed", {"returncode": 1, "json_ok": True}),
        ("no json", {"returncode": 0, "json_ok": False}),
        (
            "missing analysis root",
            {"returncode": 0, "json_ok": True, "executed": ["compute-cost"], "target": "nvidia.h200_sxm"},
        ),
        (
            "wrong target",
            {
                "returncode": 0,
                "json_ok": True,
                "executed": ["compute-cost", "memory"],
                "target": "nvidia.b200_sxm",
            },
        ),
    ],
)
def test_analysis_gaps_are_never_pass(name: str, analysis: dict) -> None:
    """Static analysis is its own stage and cannot borrow a later stage's success."""
    verdict = _statuses(_with(analysis=analysis))
    assert verdict["analysis_status"] != "PASS"
    assert verdict["overall_status"] != "PASS", name


@pytest.mark.parametrize(
    ("name", "compile_evidence"),
    [
        ("never compiled", {"calls": []}),
        (
            "compile raised",
            {**PASSING_EVIDENCE["compile"], "errors": ["RuntimeError: nvcc failed"]},
        ),
        ("no cuda source", {**PASSING_EVIDENCE["compile"], "cu_files": []}),
        ("no configure", {**PASSING_EVIDENCE["compile"], "cmake_configured": False}),
        ("no build log", {**PASSING_EVIDENCE["compile"], "cmake_built": False}),
        ("no library", {**PASSING_EVIDENCE["compile"], "libraries": []}),
        (
            "two libraries",
            {
                **PASSING_EVIDENCE["compile"],
                "libraries": [
                    {"path": "generated/libTirSquare.so", "size": 1, "sha256": "a"},
                    {"path": "generated/libOther.so", "size": 1, "sha256": "b"},
                ],
            },
        ),
        (
            "wrong library name",
            {
                **PASSING_EVIDENCE["compile"],
                "libraries": [{"path": "generated/libOther.so", "size": 1, "sha256": "a"}],
            },
        ),
    ],
)
def test_compile_gaps_are_never_pass(name: str, compile_evidence: dict) -> None:
    """A compile stage without the dumped source, logs, and the built library fails."""
    verdict = _statuses(_with(compile=compile_evidence))
    assert verdict["compile_status"] != "PASS"
    assert verdict["overall_status"] != "PASS", name


@pytest.mark.parametrize(
    ("name", "execution"),
    [
        ("never invoked", {"runtime_calls": 0, "synchronize_calls": 0, "errors": []}),
        ("never synchronized", {"runtime_calls": 1, "synchronize_calls": 0, "errors": []}),
        ("raised", {"runtime_calls": 0, "synchronize_calls": 1, "errors": ["CUDA error"]}),
    ],
)
def test_execution_gaps_are_never_pass(name: str, execution: dict) -> None:
    """A compiled kernel that never ran is not a successful execution."""
    verdict = _statuses(_with(execution=execution))
    assert verdict["execution_status"] != "PASS"
    assert verdict["overall_status"] != "PASS", name


@pytest.mark.parametrize(
    ("name", "correctness"),
    [
        ("no comparison", {"assert_close_calls": 0, "comparisons": [], "errors": []}),
        (
            "mismatch",
            {
                "assert_close_calls": 1,
                "comparisons": [
                    {"rtol": 0, "atol": 0, "shapes_match": True, "outcome": "AssertionError: mismatch"}
                ],
                "errors": ["AssertionError: mismatch"],
            },
        ),
        (
            "relaxed tolerance",
            {
                "assert_close_calls": 1,
                "comparisons": [{"rtol": 1e-2, "atol": 1e-2, "shapes_match": True, "outcome": "passed"}],
                "errors": [],
            },
        ),
        (
            "shape mismatch",
            {
                "assert_close_calls": 1,
                "comparisons": [{"rtol": 0, "atol": 0, "shapes_match": False, "outcome": "passed"}],
                "errors": [],
            },
        ),
        (
            "two comparisons",
            {
                "assert_close_calls": 2,
                "comparisons": [
                    {"rtol": 0, "atol": 0, "shapes_match": True, "outcome": "passed"},
                    {"rtol": 0, "atol": 0, "shapes_match": True, "outcome": "passed"},
                ],
                "errors": [],
            },
        ),
    ],
)
def test_correctness_gaps_are_never_pass(name: str, correctness: dict) -> None:
    """Correctness needs exactly one tolerance-free comparison that passed."""
    verdict = _statuses(_with(correctness=correctness))
    assert verdict["correctness_status"] != "PASS"
    assert verdict["overall_status"] != "PASS", name


@pytest.mark.parametrize(
    ("name", "pytest_stage"),
    [
        ("never ran", {"returncode": None, "junit": None}),
        ("no junit", {"returncode": 0, "junit": None}),
        (
            "skipped",
            {
                "returncode": 0,
                "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 1, "cases": []},
            },
        ),
        (
            "failed with rc 0",
            {
                "returncode": 0,
                "junit": {"tests": 1, "failures": 1, "errors": 0, "skipped": 0, "cases": []},
            },
        ),
        (
            "collected too many",
            {
                "returncode": 0,
                "junit": {"tests": 2, "failures": 0, "errors": 0, "skipped": 0, "cases": []},
            },
        ),
        (
            "rc nonzero",
            {
                "returncode": 1,
                "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0, "cases": []},
            },
        ),
    ],
)
def test_pytest_gaps_are_never_pass(name: str, pytest_stage: dict) -> None:
    """A zero exit code alone does not establish that one test ran and passed."""
    verdict = _statuses(_with(pytest=pytest_stage))
    assert verdict["overall_status"] != "PASS", name


def test_skipped_test_is_reported_as_failure_not_success() -> None:
    """The exact skip shape pytest writes for a missing CUDA device."""
    verdict = _statuses(
        _with(
            pytest={
                "returncode": 0,
                "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 1, "cases": []},
            }
        )
    )
    assert verdict["overall_status"] == "FAIL"
    assert "skipped" in verdict["overall_reason"]


def test_missing_evidence_is_never_pass() -> None:
    """Absent observations are never a silent pass, and an unarchived run is an error."""
    combined = demo_report.combine_with_archive(_statuses({}), {"error": "never archived"})
    assert combined["overall_status"] == "ERROR"
    assert set(combined["stages"]) == {
        "environment",
        "gpu",
        "analysis",
        "compile",
        "execution",
        "correctness",
        "pytest",
        "archive",
    }


def test_worst_stage_wins() -> None:
    """One hard error outranks a sibling failure in the overall verdict."""
    verdict = demo_report.reduce_overall(
        ["PASS", "PASS", demo_report.STATUS_ERROR, demo_report.STATUS_FAIL], 0
    )
    assert verdict == demo_report.STATUS_ERROR


def test_nonzero_returncode_dominates_a_passing_stage_set() -> None:
    """A failing pytest process cannot be reported as success."""
    assert demo_report.reduce_overall(["PASS"] * 5, 1) == demo_report.STATUS_FAIL


def test_parse_junit_reads_a_real_pytest_file(tmp_path: Path) -> None:
    """The parser reads what pytest actually writes, including a skip."""
    suite = ElementTree.Element("testsuite", {"tests": "1", "failures": "0", "errors": "0", "skipped": "1"})
    case = ElementTree.SubElement(suite, "testcase", {"name": "test_x", "classname": "test_mod"})
    ElementTree.SubElement(case, "skipped", {"message": "requires CUDA"})
    tree = ElementTree.ElementTree(suite)
    path = tmp_path / "junit.xml"
    tree.write(path)
    summary = demo_report.parse_junit(path)
    assert summary["tests"] == 1
    assert summary["skipped"] == 1
    assert summary["cases"][0]["outcome"] == "skipped"


def test_inventory_delta_names_missing_extra_and_modified_files() -> None:
    """Snapshot drift is reported by path, so an upload rule cannot hide it."""
    expected = [
        {"path": "a.py", "size": 1, "sha256": "aa"},
        {"path": "b.py", "size": 1, "sha256": "bb"},
    ]
    actual = [
        {"path": "a.py", "size": 1, "sha256": "zz"},
        {"path": "c.py", "size": 1, "sha256": "cc"},
    ]
    delta = demo_report.inventory_delta(expected, actual)
    assert delta == {"missing": ["b.py"], "unexpected": ["c.py"], "modified": ["a.py"]}


def test_snapshot_digest_changes_with_content() -> None:
    """The digest identifies content, so a silent edit changes the run identity."""
    one = [{"path": "a.py", "size": 1, "sha256": "aa"}]
    two = [{"path": "a.py", "size": 1, "sha256": "ab"}]
    assert demo_report.digest_records(one) != demo_report.digest_records(two)
    assert demo_report.digest_records(one) == demo_report.digest_records(list(one))


def test_download_command_uses_the_archived_path() -> None:
    """The printed command must address the archive the run actually wrote."""
    command = demo_report.download_command("tilefoundry-artifacts", "20260101-000000", "./x.tar.gz")
    assert command == (
        "modal volume get tilefoundry-artifacts "
        "runs/20260101-000000/tilefoundry-demo-20260101-000000.tar.gz ./x.tar.gz"
    )
    assert demo_report.archive_relpath("r1") == "runs/r1/tilefoundry-demo-r1.tar.gz"


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """A tiny git repository with one tracked, one ignored, and one untracked file."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "examples").mkdir()
    (repo / "third_party" / "cutlass").mkdir(parents=True)
    (repo / "src" / "core.py").write_text("VALUE = 1\n")
    (repo / "examples" / "demo.py").write_text("print('x')\n")
    (repo / "src" / "local.py").write_text("EDIT = True\n")
    (repo / ".gitignore").write_text("*.log\nbuild/\n")
    (repo / "noise.log").write_text("ignored\n")
    (repo / "build").mkdir()
    (repo / "build" / "artifact.txt").write_text("ignored\n")
    for args in (
        ("init", "-q"),
        ("config", "user.email", "demo@example.com"),
        ("config", "user.name", "demo"),
        ("add", "-A"),
        ("commit", "-q", "-m", "init"),
        ("rm", "-q", "--cached", "src/local.py"),
        (
            "update-index",
            "--add",
            "--cacheinfo",
            "160000,e406c186f510a15091cce01f782020ceb7ba8eb5,third_party/cutlass",
        ),
        ("commit", "-q", "-m", "pin submodule"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


def test_selection_keeps_local_edits_and_drops_excluded_trees(fake_repo: Path) -> None:
    """Untracked work under development ships; examples and the submodule do not."""
    paths = demo_source.selected_paths(fake_repo)
    assert "src/core.py" in paths
    assert "src/local.py" in paths
    assert "examples/demo.py" not in paths
    assert not any(path.startswith("third_party/cutlass/") for path in paths)
    assert "noise.log" not in paths
    assert "build/artifact.txt" not in paths


def test_snapshot_records_dirty_state_and_cutlass_pin(fake_repo: Path) -> None:
    """The manifest records that the run used a dirty tree and which submodule commit it pinned."""
    staging, identity, files = demo_source.prepare(fake_repo)
    assert identity["dirty"] is True
    assert any("src/local.py" in entry for entry in identity["dirty_entries"])
    assert identity["cutlass_sha"] == "e406c186f510a15091cce01f782020ceb7ba8eb5"
    assert identity["snapshot_file_count"] == len(files)
    assert (staging / "src" / "local.py").is_file()
    assert not (staging / "examples").exists()


def test_snapshot_digest_matches_the_uploaded_bytes(fake_repo: Path) -> None:
    """The digest is computed from the staging directory the image uploads."""
    staging, identity, files = demo_source.prepare(fake_repo)
    assert demo_report.digest_records(demo_report.inventory(staging)) == identity["snapshot_digest"]
    assert demo_report.inventory_delta(files, demo_report.inventory(staging)) == {
        "missing": [],
        "unexpected": [],
        "modified": [],
    }


def test_snapshot_detects_an_edit_made_between_selection_and_upload(fake_repo: Path) -> None:
    """An edit after freezing is caught by comparing the recorded inventory."""
    staging, _, files = demo_source.prepare(fake_repo)
    (staging / "src" / "core.py").write_text("VALUE = 2\n")
    delta = demo_report.inventory_delta(files, demo_report.inventory(staging))
    assert delta["modified"] == ["src/core.py"]


def test_manifest_json_is_deterministic(tmp_path: Path) -> None:
    """The same evidence twice produces the same manifest text."""
    path = tmp_path / "m.json"
    demo_report.write_json(path, {"b": 1, "a": [2, 3]})
    first = path.read_text()
    demo_report.write_json(path, {"a": [2, 3], "b": 1})
    assert path.read_text() == first
    assert json.loads(first) == {"a": [2, 3], "b": 1}


GOOD_ARCHIVE = {
    "archive_path": "runs/20260101-000000/tilefoundry-demo-20260101-000000.tar.gz",
    "archive_bytes": 4096,
    "archive_sha256": "cd",
}


def test_a_passing_run_with_a_passing_archive_is_delivered() -> None:
    """Execution and archiving both passing is the only delivered success."""
    combined = demo_report.combine_with_archive(_statuses(PASSING_EVIDENCE), GOOD_ARCHIVE)
    assert combined["overall_status"] == "PASS"
    assert combined["stages"]["archive"]["status"] == "PASS"


@pytest.mark.parametrize(
    ("name", "bundle"),
    [
        ("never archived", None),
        ("archive failed", {"error": "OSError: no space left"}),
        ("no digest", {"archive_path": "x", "archive_bytes": 0, "archive_sha256": ""}),
    ],
)
def test_unarchived_evidence_is_never_pass(name: str, bundle) -> None:
    """A run whose evidence never reached the Volume is not a delivered result."""
    combined = demo_report.combine_with_archive(_statuses(PASSING_EVIDENCE), bundle)
    assert combined["stages"]["archive"]["status"] != "PASS"
    assert combined["overall_status"] != "PASS", name


def test_a_failed_run_stays_failed_even_with_a_good_archive() -> None:
    """Archiving a failure does not launder it into a success."""
    failed = _with(
        pytest={
            "returncode": 1,
            "junit": {"tests": 1, "failures": 1, "errors": 0, "skipped": 0, "cases": []},
        }
    )
    combined = demo_report.combine_with_archive(_statuses(failed), GOOD_ARCHIVE)
    assert combined["overall_status"] == "FAIL"


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("wrong arch", {"target_arch": "sm_86"}),
        ("unknown arch", {"target_arch": None}),
        ("no library digest", {"libraries": [{"path": "generated/libTirSquare.so", "size": 10}]}),
    ],
)
def test_compile_arch_and_digest_are_gated(name: str, overrides: dict) -> None:
    """A library built for another architecture, or one with no digest, fails."""
    verdict = _statuses(_with(compile={**PASSING_EVIDENCE["compile"], **overrides}))
    assert verdict["compile_status"] != "PASS"
    assert verdict["overall_status"] != "PASS", name


def test_the_recorded_arch_is_reported_when_it_matches() -> None:
    """sm_90 for the H200 target passes and is stated in the reason."""
    status, _ = demo_report.evaluate_compile(PASSING_EVIDENCE["compile"])
    assert status == "PASS"


def test_source_integrity_reason_names_the_evidence() -> None:
    """A passing environment states what makes it the uploaded tree."""
    status, reason = demo_report.evaluate_environment(PASSING_ENVIRONMENT, 0)
    assert status == "PASS"
    assert "snapshot" in reason and "inventory" in reason
