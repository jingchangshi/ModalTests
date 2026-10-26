"""Select, freeze, and identify the local TileFoundry source snapshot.

The demo must run the source on this machine, not whatever a remote clone
happens to hold. One function builds a staging directory from git's own list of
files, and the same directory is what gets hashed and what gets uploaded, so the
digest in the manifest cannot drift from the bytes the container installs.

Nothing here imports Modal, torch, or TileFoundry, so it runs on a machine with
no GPU and no TileFoundry install.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from demo_report import digest_records, inventory

EXCLUDED_PREFIXES = ("examples/",)
EXCLUDED_EXACT = frozenset({"third_party/cutlass"})


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run one git command in *repo* and return its decoded stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo} (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def selected_paths(repo: Path) -> list[str]:
    """Return the repository-relative paths this demo ships to the container.

    Tracked files below the excluded prefixes are selected, plus untracked files
    that git does not ignore: a local edit under development is part of the
    snapshot, and dropping it silently would make the manifest describe source
    the container never saw.
    """
    tracked = set(_git(repo, "ls-files", "-z").split("\0")) - {""}
    untracked = set(_git(repo, "ls-files", "-z", "--others", "--exclude-standard").split("\0")) - {""}
    selected = []
    for path in sorted(tracked | untracked):
        if path in EXCLUDED_EXACT or path.startswith(EXCLUDED_PREFIXES):
            continue
        if not (repo / path).is_file():
            continue
        selected.append(path)
    return selected


def git_identity(repo: Path) -> dict:
    """Record the commit, branch, dirty state, and pinned submodules of *repo*."""
    porcelain = [line for line in _git(repo, "status", "--porcelain").splitlines() if line.strip()]
    gitlinks = {}
    for line in _git(repo, "ls-files", "-s").splitlines():
        mode, sha, _, path = line.split(maxsplit=3)
        if mode == "160000":
            gitlinks[path] = sha
    return {
        "repo": str(repo.resolve()),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "commit": _git(repo, "rev-parse", "HEAD").strip(),
        "dirty": bool(porcelain),
        "dirty_entries": porcelain,
        "submodules": gitlinks,
        "cutlass_sha": gitlinks.get("third_party/cutlass"),
    }


def build_snapshot(repo: Path, staging: Path) -> list[dict]:
    """Copy the selected files into *staging* and return their recorded inventory."""
    repo = Path(repo)
    staging = Path(staging)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for relative in selected_paths(repo):
        source = repo / relative
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return inventory(staging)


def snapshot_identity(repo: Path, staging: Path, files: list[dict]) -> dict:
    """Describe one frozen snapshot: git identity, selection policy, and its digest."""
    repo = Path(repo)
    return {
        **git_identity(repo),
        "snapshot_digest": digest_records(files),
        "snapshot_file_count": len(files),
        "snapshot_bytes": sum(record["size"] for record in files),
        "staging_dir": str(Path(staging).resolve()),
        "selection_policy": {
            "command": "git ls-files -z + git ls-files -z --others --exclude-standard",
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "excluded_exact": sorted(EXCLUDED_EXACT),
            "excluded_note": (
                "examples/ is left out of the Python source distribution by MANIFEST.in; "
                "third_party/cutlass is an unpopulated gitlink whose headers the image "
                "fetches at the pinned SHA"
            ),
        },
    }


def prepare(repo: Path, root: Path | None = None) -> tuple[Path, dict, list[dict]]:
    """Freeze *repo* into a staging directory and return it with its identity.

    The caller owns the returned staging directory and should remove it when the
    run ends; :func:`cleanup` does that for the path recorded in the identity.
    """
    root = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="tilefoundry-demo-src-"))
    staging = root / "tilefoundry"
    before = git_identity(repo)
    files = build_snapshot(repo, staging)
    after = git_identity(repo)
    if before["commit"] != after["commit"] or before["dirty_entries"] != after["dirty_entries"]:
        raise RuntimeError(
            "prepare: the working tree changed while it was being frozen; run the demo again"
        )
    return staging, snapshot_identity(repo, staging, files), files


def cleanup(identity: dict) -> None:
    """Remove the staging directory one snapshot was frozen into."""
    staging = identity.get("staging_dir")
    if not staging:
        return
    root = Path(staging).parent
    if root.name.startswith("tilefoundry-demo-src-") and root.is_dir():
        shutil.rmtree(root, ignore_errors=True)
