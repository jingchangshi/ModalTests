"""The build inputs the image is allowed to read, and the proof they are frozen.

The image is built from a Dockerfile, a build context, a pinned CUTLASS commit, and the
Python source it installs. If any of those comes from a different place than the others,
the manifest can name one revision while the compiler runs another. P0-2 closes that gap by
making every input a file inside one frozen snapshot, and by digesting those files so the
manifest can state what the build actually consumed.

Nothing here imports Modal, torch, or TileFoundry, so the whole rule is exercised by local
tests on a machine with no GPU and no Modal client.
"""

from __future__ import annotations

import re
from pathlib import Path

from demo_report import sha256_file

# Every build input the image needs must be inside the frozen snapshot. A missing one would
# force the build to read the live checkout, which is the defect this module exists to stop.
REQUIRED_BUILD_INPUTS = (
    "Dockerfile",
    "pyproject.toml",
    "targets/nvidia/pyproject.toml",
)

# Third-party bootstrap requirements. These are image configuration, not source, and are
# resolved from the index at build time; they are listed here so the manifest says so.
BOOTSTRAP_PIP = (
    "setuptools>=68",
    "setuptools-scm>=8",
    "wheel",
    "ortools>=9.15",
    "transformers>=5.13,<5.15",
)

CUTLASS_PIN_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def validate_cutlass_pin(pin: object) -> str:
    """Return the CUTLASS gitlink SHA, or raise if it cannot pin a fetch.

    ``third_party/cutlass`` is an unpopulated gitlink whose headers the image fetches. An
    empty or abbreviated pin would make the image fetch whatever ``HEAD`` happens to be,
    which is exactly the drift the snapshot exists to exclude.
    """
    if not isinstance(pin, str) or not CUTLASS_PIN_PATTERN.match(pin):
        raise ValueError(
            f"cutlass pin: {pin!r} is not a 40-character lowercase hex gitlink SHA"
        )
    return pin


def frozen_image_inputs(staging: Path, snapshot: dict) -> dict:
    """Digest the build inputs that live in the frozen snapshot.

    Args:
        staging: The frozen snapshot directory the image is built from.
        snapshot: The snapshot identity, whose ``cutlass_sha`` must be a real pin.

    Returns:
        The ``build_inputs`` record the manifest carries, naming the Dockerfile digest,
        the snapshot that is also the build context, and the bootstrap requirements.

    Raises:
        ValueError: If the CUTLASS pin is not a full SHA.
        RuntimeError: If a required build input is absent from the snapshot.
    """
    staging = Path(staging)
    cutlass_sha = validate_cutlass_pin(snapshot.get("cutlass_sha"))

    files: dict[str, dict] = {}
    for relative in REQUIRED_BUILD_INPUTS:
        path = staging / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"frozen_image_inputs: the snapshot has no regular file {relative!r}; "
                "the image would have to read it from the live checkout"
            )
        files[relative] = {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return {
        "dockerfile": files["Dockerfile"],
        "build_inputs": files,
        "context_dir": str(staging),
        "context_digest": snapshot.get("snapshot_digest"),
        "cutlass_sha": cutlass_sha,
        "bootstrap_pip_requirements": list(BOOTSTRAP_PIP),
        "note": (
            "Dockerfile, build context, CUTLASS pin and the installed Python source all "
            "come from this one frozen snapshot, not from the live checkout"
        ),
    }


def assert_dockerfile_unchanged(staging: Path, build_inputs: dict) -> None:
    """Re-hash the staged Dockerfile and require the digest the manifest recorded.

    Called immediately before the path is handed to the image builder, so a Dockerfile
    rewritten between freezing and building cannot produce an image the manifest
    misdescribes.
    """
    recorded = (build_inputs.get("dockerfile") or {}).get("sha256")
    if not recorded:
        raise RuntimeError("assert_dockerfile_unchanged: build inputs record no Dockerfile digest")
    current = sha256_file(Path(staging) / "Dockerfile")
    if current != recorded:
        raise RuntimeError(
            "the staged Dockerfile changed after it was digested: "
            f"recorded {recorded}, now {current}"
        )
