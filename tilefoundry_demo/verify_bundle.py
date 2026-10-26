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


#: The members a bundle must carry for the evidence to be checkable at all.
REQUIRED_MEMBERS = (
    "manifest.json",
    "results/analysis.json",
    "results/junit.xml",
    "results/test-results.json",
    "logs/compile-and-test.log",
)


def check_file_list(bundle: Path, manifest: dict) -> list[str]:
    """Re-hash every archived member against the list the bundle recorded.

    The bundle cannot contain its own digest, so it lists its members instead. This is the
    check that makes that list worth having: it catches a member dropped in transit, a
    member added afterwards, and a member rewritten with its tar header left alone, which a
    whole-archive digest alone would also catch but could not localise.
    """
    failures: list[str] = []
    recorded = manifest.get("files")
    if not isinstance(recorded, list) or not recorded:
        return ["the archived manifest lists no files, so its members cannot be audited"]

    expected: dict[str, dict] = {}
    for entry in recorded:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path:
            failures.append(f"the archived manifest has a file entry with no path: {entry!r}")
            continue
        if path in expected:
            failures.append(f"the archived manifest lists {path!r} twice")
        expected[path] = entry

    top = bundle.name
    actual = {
        f"{top}/{path.relative_to(bundle).as_posix()}": path
        for path in bundle.rglob("*")
        if path.is_file() and not path.is_symlink()
        # A file cannot record its own digest, so the manifest is exempt from its own list.
        # It is still checked below against the printed overall_status and evidence.
        and not (path.parent == bundle and path.name == "manifest.json")
    }

    for path in sorted(set(expected) - set(actual)):
        failures.append(f"a listed member is missing from the archive: {path}")
    for path in sorted(set(actual) - set(expected)):
        failures.append(f"the archive holds a member the manifest does not list: {path}")

    for path in sorted(set(expected) & set(actual)):
        entry = expected[path]
        file = actual[path]
        if entry.get("size") != file.stat().st_size:
            failures.append(
                f"{path}: recorded size {entry.get('size')}, archived size {file.stat().st_size}"
            )
        digest = sha256_file(file)
        if entry.get("sha256") != digest:
            failures.append(f"{path}: recorded sha256 {entry.get('sha256')}, archived {digest}")
    return failures


def check_libraries(bundle: Path, observed: dict) -> list[str]:
    """Bind each archived shared library to the digest the compiler reported."""
    failures: list[str] = []
    libraries = (observed.get("compile") or {}).get("libraries") or []
    if not libraries:
        return ["the archived evidence records no compiled library, so the .so is unattributed"]
    by_name = {}
    for record in libraries:
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            by_name[Path(record["path"]).name] = record
    built = sorted(bundle.rglob("*.so"))
    if not built:
        return ["the archive holds no built shared library"]
    for library in built:
        record = by_name.get(library.name)
        if record is None:
            failures.append(f"the archive holds {library.name}, which no compile record names")
            continue
        if record.get("sha256") != sha256_file(library):
            failures.append(f"{library.name}: the archive digest differs from the compile record")
    return failures


def verify(archive: Path, result: dict, workdir: Path) -> list[str]:
    """Return every discrepancy between the archive and the printed result."""
    failures: list[str] = []
    bundle = extract(archive, workdir)
    manifest = json.loads((bundle / "manifest.json").read_text())
    evidence = manifest.get("evidence") or {}

    for name in REQUIRED_MEMBERS:
        if not (bundle / name).exists():
            failures.append(f"missing archive member: {name}")

    # P0-5 closure: the archive's own digest cannot live inside it, so every member is
    # re-hashed against the list the bundle carries.
    failures += check_file_list(bundle, manifest)

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

    failures += check_libraries(bundle, observed)
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
