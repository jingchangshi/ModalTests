"""Run the TileFoundry H200 demo inside one Modal container and archive evidence.

The container does three things in order and never merges their outcomes
together: it proves the software stack is the uploaded snapshot, it runs HIR
static analysis, and it runs the upstream TIR integration test on a real GPU.
``demo_report.evaluate`` then maps the recorded observations onto one status per
stage, so a stage that did not pass cannot be reported as an overall success.

Every stage records what it observed before it records a verdict, and the
evidence bundle is archived on both the success and the failure path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from demo_report import (
    combine_with_archive,
    evaluate,
    inventory,
    inventory_delta,
    parse_junit,
    sha256_file,
    write_json,
)

EVIDENCE_FILE = "test-results.json"
MANIFEST_FILE = "manifest.json"
ANALYSIS_REPORT = "analysis.json"
ANALYSIS_SOURCE = "tests/nvidia/cuda/fixtures/cli/products/h200_sxm.py:Model"
TEST_TARGET = "tests/nvidia/cuda/integration/test_tir_control_flow.py::test_for_if_square_matches_torch"
TARGET_NAME = "nvidia.h200_sxm"
BUILD_PREFIX = "tilefoundry_build_"


def _run_command(command: list[str], *, cwd: Path, log: Path, env: dict | None = None) -> int:
    """Run one command, appending its combined output to *log*, and return its exit code."""
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env={**os.environ, **(env or {})},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write(f"\n[exit {process.returncode}]\n")
    return process.returncode


def _version_line(command: list[str], *, needle: str | None = None) -> str | None:
    """Return one line of a tool's version output, or ``None`` when it is absent.

    ``nvcc --version`` puts the release on the last line, below a banner, so a
    needle selects the informative line instead of the first one.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [
        line.strip()
        for line in (result.stdout or result.stderr).splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    if needle is not None:
        for line in lines:
            if needle in line:
                return line
        return None
    return lines[0]


def _origin(module) -> str | None:
    """Return the file a module was imported from."""
    path = getattr(module, "__file__", None)
    return str(Path(path).resolve()) if path else None


def _install_origin(name: str) -> dict | None:
    """Return the PEP 610 record of where a distribution came from, if any."""
    import importlib.metadata as metadata

    try:
        text = metadata.distribution(name).read_text("direct_url.json")
    except Exception:  # noqa: BLE001 - absence of the record is itself the observation
        return None
    return json.loads(text) if text else None


def collect_environment(source_root: Path, log: Path) -> dict:
    """Observe the software stack: versions, imports, build tools, and package origin."""
    import importlib.metadata as metadata
    import platform

    environment: dict = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    for tool, command, needle in (
        ("nvcc_version", ["nvcc", "--version"], "release"),
        ("cmake_version", ["cmake", "--version"], None),
        ("ninja_version", ["ninja", "--version"], None),
        ("gxx_version", ["g++", "--version"], None),
    ):
        environment[tool] = _version_line(command, needle=needle)

    cutlass_home = os.environ.get("TILEFOUNDRY_CUTLASS_INCLUDE")
    environment["tilefoundry_cutlass_include"] = cutlass_home
    cutlass = Path(cutlass_home or "/opt/cutlass/include")
    environment["cutlass_include_present"] = cutlass.is_dir()
    environment["cutlass_header_count"] = len(list(cutlass.rglob("*.hpp"))) if cutlass.is_dir() else 0

    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"python: {sys.executable} {environment['python_version']}\n")
        for key in ("nvcc_version", "cmake_version", "ninja_version", "gxx_version"):
            handle.write(f"{key}: {environment[key]}\n")
        for name in ("tilefoundry", "tilefoundry-nvidia"):
            try:
                handle.write(f"{name} version: {metadata.version(name)}\n")
            except metadata.PackageNotFoundError:
                handle.write(f"{name} version: NOT INSTALLED\n")

    try:
        import tilefoundry
        import tilefoundry_nvidia

        environment["tilefoundry_version"] = metadata.version("tilefoundry")
        environment["tilefoundry_nvidia_version"] = metadata.version("tilefoundry-nvidia")
        environment["tilefoundry_origin"] = _origin(tilefoundry)
        environment["tilefoundry_nvidia_origin"] = _origin(tilefoundry_nvidia)
        environment["imports_ok"] = True
        environment["imports_error"] = None
    except Exception as error:  # noqa: BLE001 - the container reports, the reducer decides
        environment.update({"imports_ok": False, "imports_error": f"{type(error).__name__}: {error}"})
        return environment

    environment["installations"] = {
        name: _install_origin(name) for name in ("tilefoundry", "tilefoundry-nvidia")
    }
    origins = (environment["tilefoundry_origin"], environment["tilefoundry_nvidia_origin"])
    environment["origin_inside_snapshot"] = all(
        origin is not None and origin.startswith(str(source_root) + os.sep) for origin in origins
    )

    try:
        from tilefoundry_nvidia import CudaTarget

        target = CudaTarget(TARGET_NAME)
        available = {value.identity for value in CudaTarget.available()}
        environment["target_registered"] = TARGET_NAME in available
        environment["target_available"] = sorted(available)
        environment["target_arch"] = str(getattr(target, "arch", None))
    except Exception as error:  # noqa: BLE001
        environment.update(
            {"target_registered": False, "target_error": f"{type(error).__name__}: {error}"}
        )

    environment["tools_ok"] = all(
        environment.get(tool)
        for tool in ("nvcc_version", "cmake_version", "ninja_version", "gxx_version")
    )

    try:
        import torch

        environment["torch_version"] = torch.__version__
        environment["torch_cuda_version"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        if environment["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            environment["gpu_name"] = properties.name
            environment["compute_capability"] = [properties.major, properties.minor]
            environment["gpu_memory_bytes"] = properties.total_memory
            environment["gpu_count"] = torch.cuda.device_count()
    except Exception as error:  # noqa: BLE001
        environment.update({"torch_error": f"{type(error).__name__}: {error}", "cuda_available": False})

    environment["nvidia_smi"] = _version_line(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"]
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"cuda_available: {environment.get('cuda_available')}\n")
        handle.write(f"gpu: {environment.get('gpu_name')} {environment.get('compute_capability')}\n")
        handle.write(f"torch: {environment.get('torch_version')} cuda {environment.get('torch_cuda_version')}\n")
        handle.write(f"nvidia-smi: {environment.get('nvidia_smi')}\n")
    return environment


def run_analysis(source_root: Path, results_dir: Path, logs_dir: Path) -> dict:
    """Run the shipped HIR analysis and read back the JSON it wrote."""
    report = results_dir / ANALYSIS_REPORT
    returncode = _run_command(
        [
            sys.executable,
            "-m",
            "tilefoundry.cli",
            "analyze",
            ANALYSIS_SOURCE,
            str(report),
            "--compute-cost",
            "--memory",
            "--json",
        ],
        cwd=source_root,
        log=logs_dir / "analysis.log",
    )
    analysis: dict = {"returncode": returncode, "command": ANALYSIS_SOURCE, "log": "logs/analysis.log"}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - a missing report is the observation
        analysis.update({"json_ok": False, "json_error": f"{type(error).__name__}: {error}"})
        return analysis
    functions = payload.get("functions") or ()
    analysis.update(
        {
            "json_ok": True,
            "target": payload.get("target"),
            "module": payload.get("module"),
            "requested": payload.get("requested"),
            "executed": payload.get("executed"),
            "function_count": len(functions),
            "call_count": sum(len(function.get("calls") or ()) for function in functions),
            "annotated_source_bytes": len(payload.get("source") or ""),
        }
    )
    return analysis


def _dump_build_dir(dump_root: Path) -> Path:
    """Locate the CMake build the suite's autouse dump scope wrote.

    ``tests/conftest.py`` roots each test at ``<dump_root>/<file_stem>/<test>``,
    and ``link_modules`` nests its own ``build`` scope under that. The path is
    derived from the dump root itself rather than assumed, so a layout change in
    the suite surfaces as missing evidence instead of a wrong verdict.
    """
    candidates = sorted(dump_root.glob(f"*/*/build"))
    return candidates[0] if candidates else dump_root / "missing" / "build"


def _cmake_step_succeeded(build_dir: Path, step: str) -> bool:
    """Report whether one CMake step ran and left no failure in its dumps.

    ``link_modules`` dumps ``<step>.cmd.txt`` *before* running the step, so the
    presence of that file only proves the step was attempted. Success is the
    attempt plus a stderr dump with no ``error:``/``Error`` line in it; a step
    that fails also aborts compilation, so the library check below would fail
    too. Both are required rather than either one.
    """
    if not (build_dir / f"{step}.cmd.txt").exists():
        return False
    stderr_path = build_dir / f"{step}.stderr"
    if not stderr_path.exists():
        return False
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    return not any(marker in stderr for marker in ("error:", "Error", "FAILED"))


def _compiled_arch(build_dir: Path) -> str | None:
    """Read the architecture nvcc was told to generate code for.

    The CMake project sets ``CUDA_ARCHITECTURES`` from the target's arch, and the
    dumped configure/build output carries the resulting ``--generate-code``
    flags. This reads those flags rather than trusting the target declaration, so
    a mismatch between the Target and what nvcc compiled is visible in evidence.
    """
    for name in ("cmake-build.stdout", "cmake-configure.stdout"):
        path = build_dir / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in ("arch=compute_", "--generate-code=compute_"):
            start = text.find(marker)
            if start == -1:
                continue
            tail = text[start + len(marker) :]
            digits = ""
            for character in tail:
                if not character.isdigit():
                    break
                digits += character
            if digits:
                return f"sm_{digits}"
    return None


def run_test(source_root: Path, results_dir: Path, logs_dir: Path, dump_root: Path) -> dict:
    """Run the upstream TIR integration test with only the observing plugin added."""
    evidence_path = results_dir / EVIDENCE_FILE
    junit = results_dir / "junit.xml"
    build_root = Path(os.environ["TILEFOUNDRY_DEMO_BUILD_ROOT"])
    build_root.mkdir(parents=True, exist_ok=True)
    for stale in build_root.glob(f"{BUILD_PREFIX}*"):
        shutil.rmtree(stale, ignore_errors=True)

    env = {
        "TMPDIR": str(build_root),
        "TILEFOUNDRY_DEMO_EVIDENCE": str(evidence_path),
        "TILEFOUNDRY_DEMO_DUMP_ROOT": str(dump_root),
        "TILEFOUNDRY_DEMO_EVIDENCE_ROOT": str(dump_root.parent),
        "TILEFOUNDRY_DEMO_BUILD_ROOT": str(build_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parent),
        "TILEFOUNDRY_CACHE_DIR": str(build_root / "cache"),
    }
    started = time.monotonic()
    returncode = _run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            TEST_TARGET,
            "-v",
            "-s",
            "-rs",
            "-p",
            "demo_pytest_plugin",
            "--rootdir",
            str(source_root),
            f"--junitxml={junit}",
        ],
        cwd=source_root,
        log=logs_dir / "compile-and-test.log",
        env=env,
    )
    elapsed = time.monotonic() - started

    try:
        observed = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        observed = {"read_error": f"{type(error).__name__}: {error}"}
    try:
        junit_summary = parse_junit(junit)
    except Exception as error:  # noqa: BLE001
        junit_summary = {"error": f"{type(error).__name__}: {error}"}

    runtime_name = (observed.get("compile") or {}).get("runtime_name")
    build_capture = (observed.get("compile") or {}).get("build_capture") or {}
    build_dir = _dump_build_dir(dump_root)
    compile_evidence = {
        "calls": (observed.get("compile") or {}).get("calls") or [],
        "errors": (observed.get("compile") or {}).get("errors") or [],
        "runtime_name": runtime_name,
        "cmake_configured": _cmake_step_succeeded(build_dir, "cmake-configure"),
        "cmake_built": _cmake_step_succeeded(build_dir, "cmake-build"),
        "build_directory": build_dir.relative_to(dump_root.parent).as_posix(),
        "capture_directory": build_capture.get("directory"),
        "cu_files": sorted(path.name for path in build_dir.glob("*.cu")),
        "libraries": build_capture.get("libraries") or [],
        "build_source": build_capture.get("source"),
        "target_arch": _compiled_arch(build_dir),
    }
    if build_capture.get("error"):
        compile_evidence["errors"] = [*compile_evidence["errors"], build_capture["error"]]
    return {
        "pytest": {
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 3),
            "junit": junit_summary,
            "command": TEST_TARGET,
        },
        "compile": compile_evidence,
        "execution": observed.get("execution") or {},
        "correctness": observed.get("correctness") or {},
    }


def _bundle(root: Path, run_id: str, volume_root: Path) -> dict:
    """Archive the evidence directory as one tarball and publish it to the Volume."""
    archive = Path("/tmp") / f"tilefoundry-demo-{run_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname=f"tilefoundry-demo-{run_id}")
    destination = volume_root / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    remote = destination / archive.name
    if remote.exists():
        raise FileExistsError(f"{remote} already exists; a run id must name one run")
    shutil.copy2(archive, remote)
    return {
        "archive_path": remote.relative_to(volume_root).as_posix(),
        "archive_bytes": remote.stat().st_size,
        "archive_sha256": sha256_file(remote),
    }


def _tail(path: Path, limit: int = 3000) -> str:
    """Return the end of a log file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def main() -> int:
    """Run the demo from the process environment, for use outside the entrypoint."""
    manifest = _run(
        {
            "run_id": os.environ["TILEFOUNDRY_DEMO_RUN_ID"],
            "snapshot": json.loads(Path(os.environ["TILEFOUNDRY_DEMO_SNAPSHOT"]).read_text()),
            "inventory": json.loads(Path(os.environ["TILEFOUNDRY_DEMO_INVENTORY"]).read_text()),
        },
        Path(os.environ["TILEFOUNDRY_DEMO_SOURCE"]),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["overall_status"] == "PASS" else 1


def preflight(source_root: Path, payload: dict) -> dict:
    """Prove the uploaded snapshot is what got installed, without touching a GPU."""
    root = Path("/tmp/tilefoundry-demo") / payload["run_id"] / "preflight"
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    expected = payload.get("inventory") or []
    environment = collect_environment(source_root, logs_dir / "environment.log")
    environment["source_check"] = {
        "before_install": inventory_delta(expected, inventory(source_root)) if expected else None,
        "after_install": inventory_delta(expected, inventory(source_root)) if expected else None,
    }
    environment["preflight_returncode"] = 0 if environment.get("imports_ok") else 1
    statuses = evaluate(
        {
            "environment": environment,
            "preflight_returncode": environment["preflight_returncode"],
            "analysis": {"returncode": None},
            "compile": {"calls": []},
            "execution": {},
            "correctness": {},
            "pytest": {"returncode": None},
        }
    )
    environment["environment_status"] = statuses["environment_status"]
    environment["environment_reason"] = statuses["stages"]["environment"]["reason"]
    return environment


def run(source_root: Path, payload: dict) -> dict:
    """Run analysis and the GPU test, archive evidence, and return the manifest."""
    return _run(payload, source_root)


def _run(payload: dict, source_root: Path) -> dict:
    """Run every container stage, archive what it observed, and return the manifest."""
    volume_root = Path(os.environ.get("TILEFOUNDRY_DEMO_VOLUME", "/artifacts"))
    run_id = payload["run_id"]
    expected_inventory = payload.get("inventory") or []
    snapshot = payload["snapshot"]

    root = Path("/tmp/tilefoundry-demo") / run_id
    results_dir = root / "results"
    logs_dir = root / "logs"
    dump_root = root / "generated"
    for path in (results_dir, logs_dir, dump_root):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TILEFOUNDRY_DEMO_BUILD_ROOT"] = str(root / "build")

    started = time.monotonic()
    evidence: dict = {
        "preflight_returncode": None,
        "environment": None,
        "analysis": None,
        "compile": None,
        "execution": None,
        "correctness": None,
        "pytest": None,
    }
    source_check: dict = {}
    exception_text: str | None = None

    try:
        source_check["before_install"] = inventory_delta(expected_inventory, inventory(source_root))
        environment = collect_environment(source_root, logs_dir / "environment.log")
        evidence["environment"] = environment
        evidence["preflight_returncode"] = 0 if environment.get("imports_ok") else 1
        if evidence["preflight_returncode"] == 0:
            evidence["analysis"] = run_analysis(source_root, results_dir, logs_dir)
            observed = run_test(source_root, results_dir, logs_dir, dump_root)
            evidence.update(
                {
                    "pytest": observed["pytest"],
                    "compile": observed["compile"],
                    "execution": observed["execution"],
                    "correctness": observed["correctness"],
                }
            )
        source_check["after_install"] = inventory_delta(expected_inventory, inventory(source_root))
    except BaseException as error:  # noqa: BLE001 - a crash is evidence, not a silent abort
        exception_text = f"{type(error).__name__}: {error}\n\n{_tail(logs_dir / 'compile-and-test.log')}"
        (logs_dir / "exception.txt").write_text(exception_text + "\n", encoding="utf-8")

    if evidence["environment"] is not None:
        evidence["environment"]["source_check"] = source_check
    statuses = evaluate(evidence)
    elapsed = round(time.monotonic() - started, 3)
    environment = evidence["environment"] or {}

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_commit": snapshot["commit"],
        "source_branch": snapshot["branch"],
        "source_dirty": snapshot["dirty"],
        "source_dirty_entries": snapshot["dirty_entries"],
        "source_snapshot_sha256": snapshot["snapshot_digest"],
        "source_snapshot_files": snapshot["snapshot_file_count"],
        "cutlass_sha": snapshot["cutlass_sha"],
        "source_check": source_check,
        "target": TARGET_NAME,
        "gpu_requested": "H200",
        "gpu": environment.get("gpu_name"),
        "compute_capability": environment.get("compute_capability"),
        "python_version": environment.get("python_version"),
        "torch_version": environment.get("torch_version"),
        "cuda_version": environment.get("torch_cuda_version"),
        "nvcc_version": environment.get("nvcc_version"),
        "tilefoundry_version": environment.get("tilefoundry_version"),
        "tilefoundry_nvidia_version": environment.get("tilefoundry_nvidia_version"),
        "pytest_returncode": (evidence["pytest"] or {}).get("returncode"),
        "analysis_command": ANALYSIS_SOURCE,
        "test_command": TEST_TARGET,
        "elapsed_seconds": elapsed,
        "test_elapsed_seconds": (evidence["pytest"] or {}).get("elapsed_seconds"),
        "exception": exception_text,
        "environment_status": statuses["environment_status"],
        "analysis_status": statuses["analysis_status"],
        "compile_status": statuses["compile_status"],
        "execution_status": statuses["execution_status"],
        "correctness_status": statuses["correctness_status"],
        "overall_status": statuses["overall_status"],
        "overall_reason": statuses["overall_reason"],
        "stage_details": statuses["stages"],
        "notes": {
            "analysis_vs_execution": (
                "analysis_status covers HIR static analysis only; it is never evidence that a "
                "kernel ran"
            ),
            "compile_measurement": (
                "test_elapsed_seconds covers the whole pytest process, including the first CUDA "
                "module load and the torch import; it is not a compile-only time"
            ),
            "snapshot": (
                "source_snapshot_sha256 digests the files uploaded into the image; source_check "
                "compares that inventory with the tree the container installed and ran"
            ),
        },
    }
    manifest["artifacts"] = inventory(root)
    write_json(results_dir / "stage-status.json", statuses)
    write_json(root / MANIFEST_FILE, {**manifest, "evidence": evidence})

    bundle: dict
    try:
        bundle = _bundle(root, run_id, volume_root)
    except BaseException as error:  # noqa: BLE001
        bundle = {"error": f"{type(error).__name__}: {error}"}

    final = combine_with_archive(statuses, bundle)
    manifest["artifacts_bundle"] = bundle
    manifest["overall_status"] = final["overall_status"]
    manifest["overall_reason"] = final["overall_reason"]
    manifest["stage_details"] = statuses["stages"]
    manifest["notes"]["archive_verdict"] = (
        "the manifest inside the archive states the verdict of the executed stages, because it "
        "was written before the archive existed; the returned envelope adds the archive verdict "
        "and verify_bundle.py re-derives both from the downloaded copy"
    )
    write_json(results_dir / "stage-status.json", final)
    write_json(root / MANIFEST_FILE, {**manifest, "evidence": evidence})

    return {**manifest, "evidence": evidence, **final, "stages": final["stages"]}


if __name__ == "__main__":
    raise SystemExit(main())
