"""Local gates for the bundle verifier, using a synthetic but real tarball.

These tests build an evidence bundle on disk, archive it, and check that the
verifier accepts a matching printed result and rejects each way the archive can
disagree with it. No GPU, torch, or Modal runtime is involved.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_report  # noqa: E402
import verify_bundle  # noqa: E402

RUN_ID = "20260101-000000"
SHARED_LIBRARY = b"\x7fELF\x02\x01\x01\x00fake shared object body\n"
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
        "tilefoundry_version": "0.0.0+demo.dddd",
        "tilefoundry_nvidia_version": "0.0.0+demo.dddd",
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
            "post_image_build": {"missing": [], "unexpected": [], "modified": []},
            "post_execution": {"missing": [], "unexpected": [], "modified": []},
        },
    },
    "analysis": {"returncode": 0, "json_ok": True, "executed": ["compute-cost", "memory"], "target": "nvidia.h200_sxm"},
    "compile": {
        "calls": [{"runtime_name": "TirSquare.square_host"}],
        "errors": [],
        "runtime_name": "TirSquare.square_host",
        "cu_files": ["module.device_0.cu"],
        "cmake_configured": True,
        "cmake_built": True,
        # The real digest, because check_libraries recomputes it from the archived .so.
        "libraries": [
            {
                "path": "generated/build/build/libTirSquare.so",
                "size": len(SHARED_LIBRARY),
                "sha256": hashlib.sha256(SHARED_LIBRARY).hexdigest(),
            }
        ],
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
        "junit": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0, "cases": []},
    },
}

JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" '
    'failures="0" skipped="0" tests="1"><testcase classname="m" name="test_x" time="1.0"/>'
    "</testsuite></testsuites>"
)


def record_members(bundle: Path) -> list[dict]:
    """List the bundle's members the way the container's ``_file_inventory`` does."""
    return [
        {
            "path": f"{bundle.name}/{path.relative_to(bundle).as_posix()}",
            "size": path.stat().st_size,
            "sha256": verify_bundle.sha256_file(path),
        }
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and not path.is_symlink()
        # A file cannot record its own digest; the verifier hashes the manifest itself.
        and path.name != "manifest.json"
    ]


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
    (bundle / "generated/build/build/libTirSquare.so").write_bytes(SHARED_LIBRARY)
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
    manifest["files"] = record_members(bundle)
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


def _extract_all(archive: Path, workdir: Path) -> Path:
    """Extract *archive* under *workdir* and return its bundle directory."""
    return verify_bundle.extract(archive, workdir)


def _rearchive(bundle: Path) -> Path:
    """Repack *bundle* in place, replacing the archive the fixture wrote."""
    archive = bundle.parent / f"{bundle.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=bundle.name)
    return archive


def test_a_member_missing_from_the_archive_is_rejected(tmp_path: Path) -> None:
    """A dropped log makes the archive unverifiable even when every other field matches."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    (bundle / "logs/compile-and-test.log").unlink()
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("missing archive member" in item for item in failures)
    assert any("listed member is missing" in item for item in failures)


def test_a_member_added_to_the_archive_is_rejected(tmp_path: Path) -> None:
    """An unlisted extra file was not present when the run finished."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    (bundle / "results/extra.json").write_text("{}")
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("does not list" in item and "extra.json" in item for item in failures)


def test_a_member_rewritten_after_the_manifest_is_rejected(tmp_path: Path) -> None:
    """Same names and sizes, different bytes: only the per-member digest catches this."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    log = bundle / "logs/compile-and-test.log"
    original = log.read_text()
    log.write_text(original.replace("1 passed", "1 failed"))
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("compile-and-test.log" in item and "recorded sha256" in item for item in failures)


def test_a_shared_library_without_a_compile_record_is_rejected(tmp_path: Path) -> None:
    """A .so nobody's compile call produced is unattributed code."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    (bundle / "generated/build/build/libStray.so").write_bytes(b"\x7fELFstray")
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("which no compile record names" in item for item in failures)


def test_a_shared_library_whose_digest_drifted_is_rejected(tmp_path: Path) -> None:
    """The archived .so must be the bytes the compiler reported."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    library = bundle / "generated/build/build/libTirSquare.so"
    library.write_bytes(library.read_bytes() + b"\x00")
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("differs from the compile record" in item for item in failures)


def test_an_empty_file_list_is_rejected(tmp_path: Path) -> None:
    """A bundle that lists nothing cannot be audited, so it does not verify."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"] = []
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("cannot be audited" in item for item in failures)


def test_a_duplicate_file_entry_is_rejected(tmp_path: Path) -> None:
    """Two entries for one path leave the real digest ambiguous."""
    archive = build_bundle(tmp_path)
    result = printed_result(archive)
    bundle = _extract_all(archive, tmp_path / "work")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["files"].append(dict(manifest["files"][0]))
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    archive = _rearchive(bundle)

    result["artifacts_bundle"]["archive_sha256"] = verify_bundle.sha256_file(archive)
    result["artifacts_bundle"]["archive_bytes"] = archive.stat().st_size
    failures = verify_bundle.verify(archive, result, tmp_path / "work2")
    assert any("twice" in item for item in failures)
