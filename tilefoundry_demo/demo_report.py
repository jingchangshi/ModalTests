"""Pure status reduction and archive helpers for the TileFoundry H200 demo.

Every value here comes from an observation recorded elsewhere: a subprocess exit
code, a parsed JSON report, a pytest JUnit file, or a file digest. Nothing in
this module runs a compiler, imports torch, or talks to Modal, so the whole
module is exercised by local tests without a GPU.

The one rule that matters: a run is ``PASS`` only when every required stage is
``PASS`` and the observed evidence supports it. A stage that produced partial
evidence stays ``FAIL``/``ERROR``/``BLOCKED`` and drags ``overall_status`` with it.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

SCHEMA_VERSION = 1

STATUS_NOT_RUN = "NOT_RUN"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_ERROR = "ERROR"
STATUS_BLOCKED = "BLOCKED"

STAGE_FIELDS = (
    "environment_status",
    "analysis_status",
    "compile_status",
    "execution_status",
    "correctness_status",
)

EXPECTED_TARGET = "nvidia.h200_sxm"
EXPECTED_GPU = "H200"
EXPECTED_COMPUTE_CAPABILITY = (9, 0)

SKIP_PARTS = frozenset(
    {"__pycache__", ".git", ".pytest_cache", "test_results", ".torch_extensions", "ccache"}
)

_SEVERITY = (
    (STATUS_ERROR, 4),
    (STATUS_FAIL, 3),
    (STATUS_BLOCKED, 2),
    (STATUS_NOT_RUN, 1),
    (STATUS_PASS, 0),
)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of one file's content."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_records(records: list[dict]) -> str:
    """Digest selected-source records into one snapshot identity.

    Each record states a POSIX relative path and the SHA-256 of its content, so
    the digest changes when a file is added, removed, renamed, or edited.
    """
    canonical = "\n".join(
        f"{record['path']}\0{record['size']}\0{record['sha256']}"
        for record in sorted(records, key=lambda item: item["path"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def inventory(root: Path, *, skip: frozenset[str] = SKIP_PARTS) -> list[dict]:
    """List every regular file under *root*, skipping runtime scratch directories."""
    entries = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if any(part in skip for part in relative.parts):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def inventory_delta(expected: list[dict], actual: list[dict]) -> dict:
    """Compare two inventories and name the files that differ.

    ``missing`` files were selected but never arrived; ``unexpected`` files
    arrived but were never selected, which is what an upload rule that differs
    from the selection rule looks like; ``modified`` files arrived changed.
    """
    left = {record["path"]: record for record in expected}
    right = {record["path"]: record for record in actual}
    return {
        "missing": sorted(set(left) - set(right)),
        "unexpected": sorted(set(right) - set(left)),
        "modified": sorted(
            path for path in set(left) & set(right) if left[path]["sha256"] != right[path]["sha256"]
        ),
    }


def _worst(statuses: list[str]) -> str:
    """Return the most severe status in *statuses*, preferring a recorded cause."""
    severities = dict(_SEVERITY)
    unknown = [status for status in statuses if status not in severities]
    if unknown:
        raise ValueError(f"evaluate: unknown status {unknown[0]!r}")
    return max(statuses, key=lambda status: severities[status], default=STATUS_NOT_RUN)


def evaluate_environment(environment: dict | None, preflight_returncode: int | None) -> tuple[str, str]:
    """Judge the recorded environment claim, not the process that produced it."""
    if preflight_returncode is None:
        return STATUS_NOT_RUN, "container preflight never ran"
    if preflight_returncode != 0:
        return STATUS_FAIL, f"container preflight exited {preflight_returncode}"
    if not environment:
        return STATUS_ERROR, "preflight exited 0 but recorded no environment"
    if not environment.get("imports_ok"):
        return STATUS_FAIL, "TileFoundry or TileFoundry-NVIDIA failed to import"
    if not environment.get("target_registered"):
        return STATUS_FAIL, f"{EXPECTED_TARGET} is not a registered target"
    if not environment.get("tools_ok"):
        return STATUS_FAIL, "nvcc, cmake, ninja, or g++ is missing from the image"
    return STATUS_PASS, "imports, target registration, and build tools are present"


def evaluate_gpu(environment: dict | None) -> tuple[str, str]:
    """Judge the device the kernel actually ran on, as reported by CUDA."""
    if not environment:
        return STATUS_NOT_RUN, "no device observation"
    if not environment.get("cuda_available"):
        return STATUS_BLOCKED, "torch reports no CUDA device"
    name = str(environment.get("gpu_name") or "")
    if EXPECTED_GPU not in name.upper():
        return STATUS_FAIL, f"device is {name!r}, expected {EXPECTED_GPU}"
    capability = environment.get("compute_capability")
    if capability is None or tuple(capability) != EXPECTED_COMPUTE_CAPABILITY:
        return STATUS_FAIL, f"compute capability is {capability}, expected sm_90"
    return STATUS_PASS, f"{name} reports compute capability {tuple(capability)}"


def evaluate_analysis(analysis: dict) -> tuple[str, str]:
    """Judge the HIR analysis by its exit code and the JSON it claims to have written."""
    if analysis.get("returncode") is None:
        return STATUS_NOT_RUN, "tilefoundry analyze was never run"
    if analysis["returncode"] != 0:
        return STATUS_FAIL, f"tilefoundry analyze exited {analysis['returncode']}"
    if not analysis.get("json_ok"):
        return STATUS_ERROR, "analyze exited 0 but wrote no parseable JSON report"
    executed = set(analysis.get("executed") or ())
    missing = {"compute-cost", "memory"} - executed
    if missing:
        return STATUS_FAIL, f"analysis report does not state {sorted(missing)}"
    if analysis.get("target") != EXPECTED_TARGET:
        return STATUS_FAIL, f"analysis ran against target {analysis.get('target')!r}"
    return STATUS_PASS, "analyze wrote JSON for compute-cost and memory on the H200 target"


def evaluate_compile(compile_evidence: dict) -> tuple[str, str]:
    """Judge compilation from the recorded call, the dumps, and the built library."""
    if not compile_evidence.get("calls"):
        return STATUS_NOT_RUN, "tilefoundry.compile was never called"
    if compile_evidence.get("errors"):
        return STATUS_ERROR, f"tilefoundry.compile raised: {compile_evidence['errors'][0]}"
    runtime_name = compile_evidence.get("runtime_name")
    if not runtime_name:
        return STATUS_ERROR, "tilefoundry.compile returned no named runtime"
    if not compile_evidence.get("cu_files"):
        return STATUS_ERROR, "no generated CUDA source was dumped"
    if not compile_evidence.get("cmake_configured"):
        return STATUS_ERROR, "no successful CMake configure was dumped"
    if not compile_evidence.get("cmake_built"):
        return STATUS_ERROR, "no successful CMake build was dumped"
    libraries = compile_evidence.get("libraries") or []
    if not libraries:
        return STATUS_ERROR, "no built shared library was captured"
    if len(libraries) > 1:
        return STATUS_ERROR, f"captured {len(libraries)} libraries, expected exactly one"
    expected = f"lib{runtime_name.split('.')[0]}.so"
    if libraries[0]["path"].rsplit("/", 1)[-1] != expected:
        return STATUS_FAIL, f"built library is {libraries[0]['path']!r}, expected {expected}"
    return STATUS_PASS, f"CMake and nvcc built {expected} from the dumped CUDA source"


def evaluate_archive(bundle: dict | None) -> tuple[str, str]:
    """Judge the archive by the digest and size the Volume copy actually reported."""
    if not bundle:
        return STATUS_ERROR, "the archive stage recorded nothing"
    if bundle.get("error"):
        return STATUS_ERROR, f"the archive could not be written: {bundle['error']}"
    if not bundle.get("archive_sha256") or not bundle.get("archive_bytes"):
        return STATUS_ERROR, "the archive reported no digest or size"
    return STATUS_PASS, f"archived {bundle['archive_bytes']} bytes as {bundle['archive_path']}"


def evaluate_execution(execution: dict) -> tuple[str, str]:
    """Judge execution by a runtime call that was followed by a device synchronize."""
    if execution.get("errors"):
        return STATUS_ERROR, f"runtime call raised: {execution['errors'][0]}"
    if not execution.get("runtime_calls"):
        return STATUS_NOT_RUN, "the compiled runtime was never invoked"
    if not execution.get("synchronize_calls"):
        return STATUS_ERROR, "the runtime was invoked but the device was never synchronized"
    return STATUS_PASS, f"{execution['runtime_calls']} runtime call(s) completed and synchronized"


def evaluate_correctness(correctness: dict) -> tuple[str, str]:
    """Judge correctness from the one comparison the upstream test performs.

    The gate is the comparison the test itself makes, observed with both
    tolerances pinned to zero: no exact-zero comparison can pass unless the
    compiled kernel produced the reference values.
    """
    if correctness.get("errors"):
        return STATUS_FAIL, f"comparison raised: {correctness['errors'][0]}"
    calls = correctness.get("assert_close_calls")
    if not calls:
        return STATUS_NOT_RUN, "no comparison against the torch reference was performed"
    if calls != 1:
        return STATUS_FAIL, f"{calls} comparisons, expected one"
    (comparison,) = correctness.get("comparisons") or [{}]
    if (comparison.get("rtol"), comparison.get("atol")) != (0, 0):
        return STATUS_FAIL, (
            f"comparison used rtol={comparison.get('rtol')!r} atol={comparison.get('atol')!r}, "
            "expected a tolerance-free comparison"
        )
    if not comparison.get("shapes_match"):
        return STATUS_FAIL, "the comparison saw operands of different shapes"
    source = correctness.get("reference_source") or "unobserved"
    return STATUS_PASS, f"one rtol=0 atol=0 comparison passed (reference {source})"


def reduce_overall(stage_statuses: list[str], pytest_returncode: int | None) -> str:
    """Reduce stage statuses into one verdict; partial success is never ``PASS``."""
    if pytest_returncode is None:
        return _worst([*stage_statuses, STATUS_NOT_RUN])
    if pytest_returncode != 0:
        return _worst([*stage_statuses, STATUS_FAIL])
    return _worst([*stage_statuses, STATUS_PASS])


def parse_junit(path: Path) -> dict:
    """Read the counts of a JUnit XML report written by pytest."""
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    cases = []
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key, 0) or 0)
        for case in suite.iter("testcase"):
            outcome = "passed"
            for label in ("failure", "error", "skipped"):
                if case.find(label) is not None:
                    outcome = label
            cases.append({"name": case.get("name"), "classname": case.get("classname"), "outcome": outcome})
    return {**totals, "cases": cases}


def evaluate_pytest(pytest: dict) -> tuple[str, str]:
    """Judge the pytest stage: exactly one case, and that case passed."""
    returncode = pytest.get("returncode")
    if returncode is None:
        return STATUS_NOT_RUN, "pytest was never run"
    junit = pytest.get("junit")
    if not junit:
        return STATUS_ERROR, "pytest wrote no JUnit report"
    if junit.get("error"):
        return STATUS_ERROR, f"the JUnit report could not be read: {junit['error']}"
    if junit.get("tests") != 1:
        return STATUS_FAIL, f"pytest collected {junit.get('tests')} cases, expected one"
    if junit.get("skipped"):
        return STATUS_FAIL, "the test was skipped, so no kernel ran"
    if junit.get("failures") or junit.get("errors"):
        return STATUS_FAIL, "the test failed"
    if returncode != 0:
        return STATUS_FAIL, f"pytest exited {returncode} with a passing JUnit case"
    return STATUS_PASS, "exactly one case passed and pytest exited 0"


def evaluate(evidence: dict) -> dict:
    """Map recorded evidence onto per-stage statuses and one execution verdict.

    The verdict covers only what the container executed. Archiving is judged
    separately by :func:`evaluate_archive`, because an archive cannot state the
    outcome of writing itself.

    Args:
        evidence: Environment, analysis, compile, execution, correctness, and
            pytest observations, each a plain mapping as recorded in the container.

    Returns:
        One ``(status, reason)`` pair per stage plus ``overall_status`` and
        ``overall_reason`` for the executed stages.
    """
    stages = {
        "environment": evaluate_environment(
            evidence.get("environment"), evidence.get("preflight_returncode")
        ),
        "gpu": evaluate_gpu(evidence.get("environment")),
        "analysis": evaluate_analysis(evidence.get("analysis") or {}),
        "compile": evaluate_compile(evidence.get("compile") or {}),
        "execution": evaluate_execution(evidence.get("execution") or {}),
        "correctness": evaluate_correctness(evidence.get("correctness") or {}),
        "pytest": evaluate_pytest(evidence.get("pytest") or {}),
    }
    pipeline = [stages[name][0] for name in stages]
    overall = reduce_overall(pipeline, (evidence.get("pytest") or {}).get("returncode"))
    reasons = [f"{name}: {reason}" for name, (status, reason) in stages.items() if status != STATUS_PASS]
    return {
        "stages": {name: {"status": status, "reason": reason} for name, (status, reason) in stages.items()},
        "environment_status": stages["environment"][0],
        "analysis_status": stages["analysis"][0],
        "compile_status": stages["compile"][0],
        "execution_status": stages["execution"][0],
        "correctness_status": stages["correctness"][0],
        "overall_status": overall,
        "overall_reason": "; ".join(reasons) if reasons else "every executed stage passed on recorded evidence",
    }


def combine_with_archive(verdict: dict, archive: dict | None) -> dict:
    """Fold the archive verdict into a delivered-run verdict.

    A run whose evidence never reached the Volume cannot be delivered as a
    success, so the archive failure outranks a passing execution.
    """
    status, reason = evaluate_archive(archive)
    overall = _worst([verdict["overall_status"], status])
    reasons = [] if overall == STATUS_PASS else [f"archive: {reason}" if status != STATUS_PASS else verdict["overall_reason"]]
    return {
        **verdict,
        "stages": {**verdict["stages"], "archive": {"status": status, "reason": reason}},
        "overall_status": overall,
        "overall_reason": "; ".join(reasons) if reasons else verdict["overall_reason"],
    }


def archive_relpath(run_id: str) -> str:
    """The Volume-relative path of one run's immutable archive."""
    return f"runs/{run_id}/tilefoundry-demo-{run_id}.tar.gz"


def download_command(volume_name: str, run_id: str, local_path: str) -> str:
    """The exact local command that retrieves one run's archive."""
    return f"modal volume get {volume_name} {archive_relpath(run_id)} {local_path}"


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
