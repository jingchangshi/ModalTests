"""Check a downloaded evidence bundle against the result the container printed.

The bundle's own manifest was written before the archive existed, so it can
never contain its own digest. This checker closes that gap from outside: it
re-derives the stage verdict from the archived evidence, compares the archived
manifest fields with the printed result, and verifies that the digest the
container printed for the archive is the digest of the file on disk.

Run it after ``modal volume get``:

    python verify_bundle.py results/<run-id>/tilefoundry-demo-<run-id>.tar.gz \\
        --result results/<run-id>/remote-result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_report  # noqa: E402

MATCHED_FIELDS = (
    "run_id",
    "source_commit",
    "source_dirty",
    "source_snapshot_sha256",
    "target",
    "gpu",
    "compute_capability",
    "python_version",
    "torch_version",
    "cuda_version",
    "pytest_returncode",
)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: Path, destination: Path) -> Path:
    """Extract one bundle tarball and return its single top-level directory."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        roots = {Path(name).parts[0] for name in names}
        if len(roots) != 1:
            raise SystemExit(f"archive holds {len(roots)} top-level entries, expected one: {sorted(roots)}")
        tar.extractall(destination, filter="data")
    return destination / roots.pop()


def verify(archive: Path, result: dict, workdir: Path) -> list[str]:
    """Return every discrepancy between the archive and the printed result."""
    failures: list[str] = []
    bundle = extract(archive, workdir)
    manifest = json.loads((bundle / "manifest.json").read_text())
    evidence = manifest.get("evidence") or {}

    for name in ("manifest.json", "results/analysis.json", "results/junit.xml", "results/test-results.json", "logs/compile-and-test.log"):
        if not (bundle / name).exists():
            failures.append(f"missing archive member: {name}")

    for field in MATCHED_FIELDS:
        archived = manifest.get(field)
        printed = result.get(field)
        if archived != printed:
            failures.append(f"{field}: archive says {archived!r}, result says {printed!r}")

    bundle_info = result.get("artifacts_bundle") or {}
    if bundle_info.get("archive_sha256") != sha256_file(archive):
        failures.append("the archive digest does not match the digest the container printed")
    if bundle_info.get("archive_bytes") != archive.stat().st_size:
        failures.append("the archive size does not match the size the container printed")

    if result.get("run_id") != bundle.name.removeprefix("tilefoundry-demo-"):
        failures.append("the top-level archive directory does not match the run id")

    underived = demo_report.evaluate(evidence)
    for stage, verdict in underived["stages"].items():
        printed = (result.get("stage_details") or {}).get(stage, {}).get("status")
        if verdict["status"] != printed:
            failures.append(f"stage {stage}: archived evidence says {verdict['status']}, result says {printed}")

    if manifest.get("overall_status") != underived["overall_status"]:
        failures.append(
            "the archived manifest's overall_status does not match the verdict its own evidence "
            "produces"
        )
    delivered = demo_report.combine_with_archive(underived, result.get("artifacts_bundle"))
    if delivered["overall_status"] != result.get("overall_status"):
        failures.append(
            "the delivered verdict does not follow from the archived evidence plus the printed "
            f"archive record: {delivered['overall_status']} vs {result.get('overall_status')}"
        )

    archived_analysis = json.loads((bundle / "results/analysis.json").read_text())
    if archived_analysis.get("target") != "nvidia.h200_sxm":
        failures.append(f"the archived analysis names target {archived_analysis.get('target')!r}")

    junit = demo_report.parse_junit(bundle / "results/junit.xml")
    if junit["tests"] != 1 or junit["skipped"] or junit["failures"] or junit["errors"]:
        failures.append(f"the archived JUnit report does not show one passing test: {junit}")

    observed = json.loads((bundle / "results/test-results.json").read_text())
    compile_calls = (observed.get("compile") or {}).get("calls") or []
    if not compile_calls:
        failures.append("the archived evidence records no tilefoundry.compile call")
    if not (observed.get("execution") or {}).get("runtime_calls"):
        failures.append("the archived evidence records no runtime invocation")
    comparisons = (observed.get("correctness") or {}).get("comparisons") or []
    if not comparisons or comparisons[0].get("rtol") != 0 or comparisons[0].get("atol") != 0:
        failures.append("the archived comparison was not tolerance-free")

    library = bundle / "generated/build/build"
    if not any(path.suffix == ".so" for path in library.glob("*.so")):
        failures.append("the archive holds no built shared library")
    if not any(bundle.glob("generated/**/*.cu")):
        failures.append("the archive holds no generated CUDA source")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Check one bundle and print a verdict with every discrepancy it found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--result", type=Path, required=True, help="the remote-result.json the run printed")
    parser.add_argument("--workdir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    workdir = arguments.workdir or arguments.archive.parent / "extracted"
    result = json.loads(arguments.result.read_text())
    failures = verify(arguments.archive, result, workdir)
    if failures:
        print("VERIFY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"VERIFY OK: {arguments.archive.name} matches {arguments.result.name}")
    print(f"  run_id: {result.get('run_id')}")
    print(f"  overall_status: {result.get('overall_status')}")
    print(f"  archive sha256: {result['artifacts_bundle']['archive_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
