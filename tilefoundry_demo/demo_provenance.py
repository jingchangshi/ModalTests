"""Prove the installed packages are the uploaded snapshot, and fail closed.

The image installs the snapshot as two editable distributions, ``tilefoundry`` and
``tilefoundry-nvidia``. A PEP 610 ``direct_url.json`` that merely says *editable* is not
enough: an editable install of some other checkout, a site-packages copy of the released
wheel, and a stale local build all report ``editable`` or report nothing, and each would
let the manifest name source the compiler never saw.

This module states the four things that have to be true at once:

1. each distribution's PEP 610 URL is exactly the expected directory inside the snapshot;
2. each imported module's file resolves inside that same directory;
3. the two distributions carry one version, not two versions that happen to be equal;
4. the installed tree still matches the inventory that was digested before upload.

Nothing here imports tilefoundry, torch, or Modal: it judges observations that the container
collects. That keeps the whole rule testable on a machine with no GPU.
"""

from __future__ import annotations

import posixpath
from pathlib import PurePosixPath

#: Distribution name -> the directory it must be installed from, relative to the snapshot.
EXPECTED_ORIGINS = {
    "tilefoundry": ".",
    "tilefoundry-nvidia": "targets/nvidia",
}


def _normalized(path: object) -> str | None:
    """Return a normalised POSIX path for *path*, or None if it is not a usable path."""
    if not isinstance(path, str) or not path:
        return None
    # A Windows-style separator would silently normalise into something unrelated.
    if "\\" in path:
        return None
    return posixpath.normpath(path)


def check_installation_url(name: str, record: object, source_root: str) -> list[str]:
    """Check one distribution's PEP 610 record against the directory it must come from."""
    expected = _normalized(f"{source_root}/{EXPECTED_ORIGINS[name]}")
    if not isinstance(record, dict):
        return [f"{name}: no PEP 610 direct_url.json record; inheritance cannot be proven"]
    url = record.get("url")
    actual = _normalized(url[len("file://") :]) if isinstance(url, str) and url.startswith("file://") else None
    if actual is None:
        return [f"{name}: direct_url url {url!r} is not a file:// URL inside the snapshot"]
    if actual != expected:
        return [f"{name}: installed from {actual}, expected {expected}"]
    if not record.get("dir_info", {}).get("editable"):
        return [f"{name}: {actual} is a non-editable install, so its files may be stale copies"]
    return []


def check_import_origin(name: str, origin: object, source_root: str) -> list[str]:
    """Check that an imported module's file resolves inside the snapshot directory."""
    module_root = _normalized(f"{source_root}/{EXPECTED_ORIGINS[name]}")
    actual = _normalized(origin)
    if actual is None:
        return [f"{name}: no import origin was recorded"]
    if not module_root or not (actual == module_root or actual.startswith(module_root + "/")):
        return [f"{name}: imported from {actual}, which is outside {module_root}"]
    return []


def check_provenance(environment: dict, source_root: str) -> list[str]:
    """Return every reason the installed packages are not the uploaded snapshot.

    An empty list means the four conditions hold. Any entry is a failure: this check has no
    warning state, because a package of unknown origin is a package the evidence cannot
    describe.
    """
    failures: list[str] = []

    installations = environment.get("installations")
    if not isinstance(installations, dict):
        return ["no installations record at all"]
    for name in EXPECTED_ORIGINS:
        failures += check_installation_url(name, installations.get(name), source_root)

    for name, key in (
        ("tilefoundry", "tilefoundry_origin"),
        ("tilefoundry-nvidia", "tilefoundry_nvidia_origin"),
    ):
        failures += check_import_origin(name, environment.get(key), source_root)

    version = environment.get("tilefoundry_version")
    nvidia_version = environment.get("tilefoundry_nvidia_version")
    if not version:
        failures.append("tilefoundry: no installed version was recorded")
    if not nvidia_version:
        failures.append("tilefoundry-nvidia: no installed version was recorded")
    if version and nvidia_version and version != nvidia_version:
        failures.append(
            f"core and target versions differ: {version!r} vs {nvidia_version!r}, "
            "so one of them is not this snapshot"
        )

    source_check = environment.get("source_check")
    if not isinstance(source_check, dict):
        return failures + ["no inventory comparison of the installed tree"]

    from demo_report import PHASE_POST_IMAGE_BUILD, PHASE_POST_EXECUTION

    # The post-image-build comparison must always exist: it is the first thing that can
    # prove the install matches the digest. The post-execution one cannot exist during a
    # CPU preflight, which stops before any execution, so its absence there is expected
    # and is reported separately rather than counted as a failure.
    observed = source_check.get(PHASE_POST_IMAGE_BUILD)
    if not isinstance(observed, dict):
        failures.append(f"no {PHASE_POST_IMAGE_BUILD} inventory comparison")

    for phase in (PHASE_POST_IMAGE_BUILD, PHASE_POST_EXECUTION):
        delta = source_check.get(phase)
        if not isinstance(delta, dict):
            continue
        for kind in ("missing", "unexpected", "modified"):
            drifted = delta.get(kind) or []
            if drifted:
                failures.append(
                    f"the installed tree changed {phase}: {len(drifted)} {kind} file(s), "
                    f"first {drifted[0]!r}"
                )
    return failures


def expected_module_root(name: str, source_root: str) -> PurePosixPath:
    """The directory one distribution's code must live in, for error messages."""
    return PurePosixPath(_normalized(f"{source_root}/{EXPECTED_ORIGINS[name]}") or source_root)
