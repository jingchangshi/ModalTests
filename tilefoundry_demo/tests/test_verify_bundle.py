"""Local gates for the bundle verifier, using a synthetic but real tarball.

These tests build an evidence bundle on disk, archive it, and check that the
verifier accepts a matching printed result and rejects each way the archive can
disagree with it. No GPU, torch, or Modal runtime is involved.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_report  # noqa: E402
import verify_bundle  # noqa: E402

RUN_ID = "20260101-000000"
EVIDENCE = {
    "preflight_returncode": 0,
    "environment": {
        "imports_ok": True,
        "target_registered": True,
        "tools_ok": True,
        "cuda_available": True,
        "gpu_name": "NVIDIA H200",
        "compute_capability": [9, 0],
        "python_version": "3.12.15",
        "torch_version": "2.9.1+cu128",
        "torch_cuda_version": "12.8",
    },
    "analysis": {"returncode": 0, "json_ok": True, "executed": ["compute-cost", "memory"], "target": "nvidia.h200_sxm"},
    "compile": {
        "calls": [{"runtime_name": "TirSquare.square_host"}],
        "errors": [],
        "runtime_name": "TirSquare.square_host",
        "cu_files": ["module.device_0.cu"],
        "cmake_configured": True,
        "cmake_built": True,
        "libraries": [{"path": "generated/build/build/libTirSquare.so", "size": 3, "sha256": "aa"}],
    },
    "execution": {"runtime_calls": 1, "synchronize_calls": 1, "errors": []},
    "correctness": {
        "assert_close_calls": 1,
        "comparisons": [{"rtol": 0, "atol": 0, "shapes_match": True, "outcome": "passed"}],
        "errors": [],
    },
    "pytest": {
        "returncode": 0,
        "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0, "cases": []},
    },
}

JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" '
    'failures="0" skipped="0" tests="1"><testcase classname="m" name="test_x" time="1.0"/>'
    "</testsuite></testsuites>"
)


def build_bundle(root: Path, evidence: dict | None = None) -> Path:
    """Materialise one evidence directory with every member the verifier requires."""
    evidence = evidence or EVIDENCE
    bundle = root / f"tilefoundry-demo-{RUN_ID}"
    (bundle / "results").mkdir(parents=True)
    (bundle / "logs").mkdir()
    (bundle / "generated/build/build").mkdir(parents=True)
    (bundle / "generated/test/build").mkdir(parents=True)
    (bundle / "results/analysis.json").write_text(json.dumps({"target": "nvidia.h200_sxm", "executed": ["compute-cost", "memory"]}))
    (bundle / "results/junit.xml").write_text(JUNIT)
    (bundle / "results/test-results.json").write_text(json.dumps({"compile": evidence["compile"], "execution": evidence["execution"], "correctness": evidence["correctness"]}))
    (bundle / "logs/compile-and-test.log").write_text("1 passed\n")
    (bundle / "generated/build/build/libTirSquare.so").write_bytes(b"\x7fELF")
    (bundle / "generated/test/build/module.device_0.cu").write_text("__global__ void k() {}\n")
    verdict = demo_report.evaluate(evidence)
    manifest = {
        "run_id": RUN_ID,
        "source_commit": "0" * 40,
        "source_dirty": False,
        "source_snapshot_sha256": "d" * 64,
        "target": "nvidia.h200_sxm",
        "gpu": "NVIDIA H200",
        "compute_capability": [9, 0],
        "python_version": "3.12.15",
        "torch_version": "2.9.1+cu128",
        "cuda_version": "12.8",
        "pytest_returncode": 0,
        "overall_status": verdict["overall_status"],
        "stage_details": verdict["stages"],
        "evidence": evidence,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    archive = root / f"{bundle.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=bundle.name)
    return archive


def printed_result(archive: Path) -> dict:
    """Build the envelope the container prints for *archive*."""
    verdict = demo_report.evaluate(EVIDENCE)
    bundle_record = {
        "archive_path": f"runs/{RUN_ID}/{archive.name}",
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": verify_bundle.sha256_file(archive),
    }
    delivered = demo_report.combine_with_archive(verdict, bundle_record)
    return {
        "run_id": RUN_ID,
        "source_commit": "0" * 40,
        "source_dirty": False,
        "source_snapshot_sha256": "d" * 64,
        "target": "nvidia.h200_sxm",
        "gpu": "NVIDIA H200",
        "compute_capability": [9, 0],
        "python_version": "3.12.15",
        "torch_version": "2.9.1+cu128",
        "cuda_version": "12.8",
        "pytest_returncode": 0,
        "artifacts_bundle": bundle_record,
        "overall_status": delivered["overall_status"],
        "stage_details": delivered["stages"],
    }


def test_a_matching_bundle_and_result_verify(tmp_path: Path) -> None:
    """A coherent run verifies, including the delivered verdict."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    assert result["overall_status"] == "PASS"
    assert verify_bundle.verify(archive, result, tmp_path / "work") == []


def test_a_tampered_archive_digest_is_rejected(tmp_path: Path) -> None:
    """The printed digest must be the digest of the downloaded file."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    result["artifacts_bundle"]["archive_sha256"] = "0" * 64
    failures = verify_bundle.verify(archive, result, tmp_path / "work")
    assert any("digest" in failure for failure in failures)


def test_a_mismatched_run_id_is_rejected(tmp_path: Path) -> None:
    """The printed run id must be the directory inside the archive."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    result["run_id"] = "20260101-000001"
    failures = verify_bundle.verify(archive, result, tmp_path / "work")
    assert any("run id" in failure or "run_id" in failure for failure in failures)


def test_an_archive_claiming_success_without_the_library_is_rejected(tmp_path: Path) -> None:
    """A bundle that lost the built library cannot verify as a pass."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    (tmp_path / f"tilefoundry-demo-{RUN_ID}" / "generated/build/build/libTirSquare.so").unlink()
    (tmp_path / f"tilefoundry-demo-{RUN_ID}" / "generated/test/build/module.device_0.cu").unlink()
    rebuilt = tmp_path / "rebuilt.tar.gz"
    with tarfile.open(rebuilt, "w:gz") as tar:
        tar.add(tmp_path / f"tilefoundry-demo-{RUN_ID}", arcname=f"tilefoundry-demo-{RUN_ID}")
    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(rebuilt)
    result["artifacts_bundle"]["archive_bytes"] = rebuilt.stat().st_size
    failures = verify_bundle.verify(rebuilt, result, tmp_path / "work2")
    assert any("shared library" in failure for failure in failures)
    assert any("CUDA source" in failure for failure in failures)


def test_a_skipped_test_in_the_archive_is_rejected(tmp_path: Path) -> None:
    """An archive whose JUnit shows a skip cannot verify as a pass."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    (tmp_path / f"tilefoundry-demo-{RUN_ID}" / "results/junit.xml").write_text(
        '<?xml version="1.0"?><testsuites><testsuite tests="1" errors="0" failures="0" skipped="1">'
        '<testcase classname="m" name="test_x"><skipped message="requires CUDA"/></testcase>'
        "</testsuite></testsuites>"
    )
    rebuilt = tmp_path / "rebuilt.tar.gz"
    with tarfile.open(rebuilt, "w:gz") as tar:
        tar.add(tmp_path / f"tilefoundry-demo-{RUN_ID}", arcname=f"tilefoundry-demo-{RUN_ID}")
    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(rebuilt)
    result["artifacts_bundle"]["archive_bytes"] = rebuilt.stat().st_size
    failures = verify_bundle.verify(rebuilt, result, tmp_path / "work3")
    assert any("JUnit" in failure for failure in failures)


def test_an_archive_that_overstates_its_own_verdict_is_rejected(tmp_path: Path) -> None:
    """A manifest claiming PASS while its evidence says otherwise is a discrepancy."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    manifest_path = tmp_path / f"tilefoundry-demo-{RUN_ID}" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["overall_status"] = "PASS"
    manifest["evidence"] = {**EVIDENCE, "pytest": {"returncode": 1, "junit": None}}
    manifest_path.write_text(json.dumps(manifest))
    rebuilt = tmp_path / "rebuilt.tar.gz"
    with tarfile.open(rebuilt, "w:gz") as tar:
        tar.add(tmp_path / f"tilefoundry-demo-{RUN_ID}", arcname=f"tilefoundry-demo-{RUN_ID}")
    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(rebuilt)
    result["artifacts_bundle"]["archive_bytes"] = rebuilt.stat().st_size
    failures = verify_bundle.verify(rebuilt, result, tmp_path / "work4")
    assert any("overall_status" in failure for failure in failures)
