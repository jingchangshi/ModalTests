"""Select, freeze, and identify the local TileFoundry source snapshot.

The demo must run the source on this machine, not whatever a remote clone
happens to hold. One function freezes the tree into a staging directory, and the
same directory is what gets hashed and what gets uploaded, so the digest in the
manifest cannot drift from the bytes the container installs.

Nothing here imports Modal, torch, or TileFoundry, so it runs on a machine with
no GPU and no TileFoundry install.

Two properties are enforced rather than assumed:

* **Containment.** A git name is text from the working tree, so every name is
  validated as a plain relative POSIX path before it is used, and every read
  happens through a directory descriptor opened with ``O_NOFOLLOW`` relative to
  the repository root. A symlink whose target resolves outside the repository (or
  a dangling one) fails the snapshot instead of being copied.
* **Frozen bytes.** Selection, a source fingerprint pass, the copy, and a second
  fingerprint pass must agree on the same selected set, the same git identity, and
  the same per-file content. This detects an edit observed between two passes; it
  is not a claim that no malicious writer can ever defeat the check.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from demo_report import SKIP_PARTS, digest_records, inventory

EXCLUDED_PREFIXES = ("examples/",)
EXCLUDED_EXACT = frozenset({"third_party/cutlass"})

# An editable install writes ``*.egg-info``/``.pth`` into the checkout. They are
# produced by the image, never authored, and uploading them would make the
# snapshot digest depend on what a previous container run happened to write.
RUNTIME_SCRATCH_GLOBS = ("*.egg-info", "*.egg-link", "*.pth")

UNTRACKED_SUFFIXES = (
    ".py",
    ".pyi",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".rst",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".c",
    ".cc",
    ".cpp",
    ".cmake",
)
UNTRACKED_BASENAMES = frozenset({"Dockerfile", "CMakeLists.txt", "MANIFEST.in", "LICENSE", "NOTICE"})

INCLUDE_RULE = (
    "tracked files from `git ls-files -z` plus untracked files from "
    "`git ls-files -z --others --exclude-standard`, minus excluded prefixes "
    f"{list(EXCLUDED_PREFIXES)}, minus excluded exact {sorted(EXCLUDED_EXACT)}, minus "
    "runtime scratch parts/paths, minus untracked files whose suffix is not one of "
    f"{list(UNTRACKED_SUFFIXES)} and whose basename is not one of {sorted(UNTRACKED_BASENAMES)}"
)

REASON_EXCLUDED_PREFIX = "excluded_prefix"
REASON_GITLINK = "gitlink"
REASON_RUNTIME_SCRATCH = "runtime_scratch"
REASON_UNTRACKED_NOT_ALLOWLISTED = "untracked_not_allowlisted"
REASON_TRACKED_MISSING = "tracked_missing"

_IDENTITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

_HEX40 = re.compile(r"[0-9a-f]{40}")


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


def _git_names(text: str) -> list[str]:
    """Split NUL-separated ``git ls-files -z`` output into names."""
    return [name for name in text.split("\0") if name]


def _reject(relative: str, field: str, detail: str) -> ValueError:
    """Build the one error shape a rejected source name produces."""
    return ValueError(f"source {relative!r}: {field}: {detail}")


def validate_source_path(repo: Path, relative: str) -> Path:
    """Return the file a validated git name resolves to, or raise ``ValueError``.

    The name is judged as text first, because resolving it is exactly the step an
    escape needs. Only a name of known-safe shape is resolved, and the resolved
    path is compared with the resolved root using ``is_relative_to``: a string
    prefix test would accept ``/repo-evil`` for root ``/repo``, and would say
    nothing about a symlinked parent directory.
    """
    if not isinstance(relative, str) or relative == "":
        raise _reject(relative, "path", "is empty")
    if "\0" in relative:
        raise _reject(relative, "path", "contains a NUL byte")
    if "\n" in relative or "\r" in relative:
        raise _reject(relative, "path", "contains a newline")
    if "\\" in relative:
        raise _reject(relative, "path", "contains a backslash")
    if relative.startswith("/"):
        raise _reject(relative, "path", "is absolute")
    parts = relative.split("/")
    if any(part == ".." for part in parts):
        raise _reject(relative, "path", "contains a '..' component")
    if any(part == "." for part in parts):
        raise _reject(relative, "path", "contains a '.' component")
    if any(part == "" for part in parts):
        raise _reject(relative, "path", "contains an empty component")

    root = Path(repo).resolve()
    try:
        resolved = root.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise _reject(relative, "path", f"does not resolve ({type(error).__name__})") from None
    if resolved == root or not resolved.is_relative_to(root):
        raise _reject(relative, "path", "resolves outside the repository root")
    return resolved


def check_symlink(root: Path, relative: str) -> None:
    """Reject a selected name whose link (or whose parent directory's link) escapes.

    ``validate_source_path`` resolved the final target, so this recovers the other
    half of the observation: which root-relative component carried a link, and
    whether that link stayed inside the repository.
    """
    root = Path(root).resolve()
    current = root
    for part in relative.split("/"):
        current = current / part
        if not current.is_symlink():
            continue
        link_target = Path(os.readlink(current))
        absolute = link_target if link_target.is_absolute() else (current.parent / link_target)
        absolute = absolute.resolve()
        name = current.relative_to(root).as_posix()
        if not absolute.is_relative_to(root):
            raise _reject(relative, "symlink", f"{name} points outside the repository root")
        if not absolute.exists():
            raise _reject(relative, "symlink", f"{name} is dangling")


def _open_beneath(root: Path, relative: str, reported: str) -> tuple[int, os.stat_result]:
    """Open *relative* beneath *root* without following any symlink, and stat it.

    Directory descriptors plus ``dir_fd`` make the composed path unforgeable: no
    component can be swapped for a link between two ``stat`` calls because no
    intermediate path lookup happens again. ``O_NOFOLLOW`` on the final component
    rejects a substitution, and the regular-file check rejects devices, FIFOs,
    sockets and directories.
    """
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = relative.split("/")
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = following
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode):
            raise _reject(reported, "type", "not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd, observed


def _hash_fd(fd: int, relative: str) -> tuple[int, str]:
    """Hash an open descriptor from offset zero and return ``(size, sha256)``."""
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        digest.update(block)
        size += len(block)
    if size != os.fstat(fd).st_size:
        raise RuntimeError(f"source {relative!r}: read: short read while hashing")
    return size, digest.hexdigest()


def _identity(observed: os.stat_result) -> dict:
    """Return the stat fields a rewrite of the same path would change."""
    return {field: getattr(observed, field) for field in _IDENTITY_FIELDS}


def _locate(repo: Path, relative: str) -> tuple[Path, Path, str]:
    """Return ``(root, resolved, resolved_relative)`` for one selected name."""
    root = Path(repo).resolve()
    check_symlink(root, relative)
    resolved = validate_source_path(repo, relative)
    return root, resolved, resolved.relative_to(root).as_posix()


def fingerprint_sources(repo: Path, paths: list[str]) -> dict[str, dict]:
    """Hash every selected file in the repository, recording where it resolved.

    The fingerprint is the first and last observation of the freeze: it records
    the content, the resolution and the stat identity of each selected name, so an
    edit that leaves ``git status --porcelain`` unchanged still changes it.
    """
    fingerprints: dict[str, dict] = {}
    for relative in paths:
        root, resolved, resolved_relative = _locate(repo, relative)
        try:
            fd, observed = _open_beneath(root, resolved_relative, relative)
        except OSError as error:
            raise RuntimeError(
                f"source {relative!r}: read: cannot open ({type(error).__name__}: {error})"
            ) from None
        try:
            size, sha256 = _hash_fd(fd, relative)
        finally:
            os.close(fd)
        fingerprints[relative] = {
            "path": relative,
            "resolved_relative": resolved_relative,
            "size": size,
            "sha256": sha256,
            **_identity(observed),
        }
    return fingerprints


def copy_checked(repo: Path, relative: str, staging: Path) -> dict:
    """Copy one source file into *staging*, hashing the bytes while they are read.

    The source is opened once and copied from that same descriptor, so the staged
    bytes are the hashed bytes. The descriptor is statted before and after the
    copy: a change to any identity field means the read spanned a rewrite, and the
    staged file no longer represents one version of the source.
    """
    root, resolved, resolved_relative = _locate(repo, relative)
    try:
        fd, before = _open_beneath(root, resolved_relative, relative)
    except OSError as error:
        raise RuntimeError(
            f"source {relative!r}: read: cannot open ({type(error).__name__}: {error})"
        ) from None
    target = Path(staging) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        with open(target, "xb") as handle:
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                digest.update(block)
                size += len(block)
                handle.write(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    changed = [field for field in _IDENTITY_FIELDS if getattr(before, field) != getattr(after, field)]
    if changed:
        raise RuntimeError(
            f"source {relative!r}: race: {', '.join(changed)} changed while it was being copied"
        )
    if size != before.st_size:
        # The copy read more or fewer bytes than the file held when it was opened:
        # the read spanned a rewrite that happened to leave the metadata alone.
        raise RuntimeError(
            f"source {relative!r}: race: read {size} bytes from a file of {before.st_size}"
        )
    staged_size = target.stat().st_size
    if staged_size != size:
        raise RuntimeError(f"source {relative!r}: write: staged {staged_size} bytes, read {size}")
    if _locate(repo, relative)[1] != resolved:
        raise RuntimeError(f"source {relative!r}: race: the path resolves elsewhere after copying")
    return {
        "path": relative,
        "resolved_relative": resolved_relative,
        "size": size,
        "sha256": digest.hexdigest(),
    }


def is_runtime_scratch(relative: str) -> bool:
    """Report whether a selected name is runtime scratch rather than authored source."""
    parts = relative.split("/")
    if any(part in SKIP_PARTS for part in parts):
        return True
    return any(Path(part).match(pattern) for part in parts for pattern in RUNTIME_SCRATCH_GLOBS)


def untracked_allowed(relative: str) -> bool:
    """Report whether an untracked file is source the demo is allowed to ship.

    An untracked file is a local edit under development, so dropping it silently
    would make the manifest describe source the container never saw. It is also the
    only file class git cannot vouch for, so the allowlist keeps a stray data blob
    or a local notes file out of the upload.
    """
    name = relative.rsplit("/", 1)[-1]
    return name in UNTRACKED_BASENAMES or name.endswith(UNTRACKED_SUFFIXES)


def exclusion_reason(relative: str, tracked: bool) -> str | None:
    """Return why a validated git name is not uploaded, or ``None`` to include it."""
    if relative in EXCLUDED_EXACT:
        return REASON_GITLINK
    if relative.startswith(EXCLUDED_PREFIXES):
        return REASON_EXCLUDED_PREFIX
    if is_runtime_scratch(relative):
        return REASON_RUNTIME_SCRATCH
    if not tracked and not untracked_allowed(relative):
        return REASON_UNTRACKED_NOT_ALLOWLISTED
    return None


def select_sources(repo: Path) -> tuple[list[str], list[dict]]:
    """Return the uploaded paths and one record per omitted name.

    Every raw git name is validated *before* any exclusion applies: rule order is
    observable, and an exclusion that ran first would be the place a hostile name
    hides, because an excluded path is never resolved and so never checked.

    Returns:
        The sorted selected paths and the sorted ``{path, tracked, reason}``
        records for everything left out.
    """
    repo = Path(repo)
    tracked = set(_git_names(_git(repo, "ls-files", "-z")))
    untracked = set(_git_names(_git(repo, "ls-files", "-z", "--others", "--exclude-standard")))
    for relative in sorted(tracked | untracked):
        validate_source_path(repo, relative)

    selected: list[str] = []
    skips: list[dict] = []
    for relative in sorted(tracked | untracked):
        reason = exclusion_reason(relative, relative in tracked)
        if reason is None:
            selected.append(relative)
        else:
            skips.append({"path": relative, "tracked": relative in tracked, "reason": reason})
    return selected, skips


def missing_tracked(repo: Path, paths: list[str]) -> list[dict]:
    """Record tracked files that are selected by git but absent from the tree."""
    records = []
    for relative in paths:
        if not (Path(repo) / relative).exists():
            records.append({"path": relative, "tracked": True, "reason": REASON_TRACKED_MISSING})
    return records


def selected_paths(repo: Path) -> list[str]:
    """Return the repository-relative paths this demo ships to the container."""
    return select_sources(repo)[0]


def git_identity(repo: Path) -> dict:
    """Record the commit, branch, dirty state, and pinned submodules of *repo*."""
    porcelain = [line for line in _git(repo, "status", "--porcelain").splitlines() if line.strip()]
    gitlinks = {}
    for line in _git(repo, "ls-files", "-s").splitlines():
        mode, sha, _, path = line.split(maxsplit=3)
        if mode == "160000":
            gitlinks[path] = sha
    return {
        "repo": str(Path(repo).resolve()),
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
        copy_checked(repo, relative, staging)
    return inventory(staging, skip=frozenset())


def materialized_links(fingerprints: dict[str, dict]) -> list[dict]:
    """List the selected names whose stored bytes came from another path."""
    return [
        {"path": relative, "resolved_relative": record["resolved_relative"]}
        for relative, record in sorted(fingerprints.items())
        if record["resolved_relative"] != relative
    ]


def snapshot_identity(
    repo: Path,
    staging: Path,
    files: list[dict],
    *,
    skips: list[dict] | None = None,
    links: list[dict] | None = None,
    git: dict | None = None,
) -> dict:
    """Describe one frozen snapshot: git identity, selection policy, and its digest."""
    repo = Path(repo)
    identity = git if git is not None else git_identity(repo)
    return {
        **identity,
        "snapshot_digest": digest_records(files),
        "snapshot_file_count": len(files),
        "snapshot_bytes": sum(record["size"] for record in files),
        "staging_dir": str(Path(staging).resolve()),
        "selection_policy": {
            "command": "git ls-files -z + git ls-files -z --others --exclude-standard",
            "include_rule": INCLUDE_RULE,
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "excluded_exact": sorted(EXCLUDED_EXACT),
            "skip_parts": sorted(SKIP_PARTS),
            "skipped": sorted(skips or [], key=lambda record: record["path"]),
            "materialized_links": sorted(links or [], key=lambda record: record["path"]),
            "excluded_note": (
                "examples/ is left out of the Python source distribution by MANIFEST.in; "
                "third_party/cutlass is an unpopulated gitlink whose headers the image "
                "fetches at the pinned SHA"
            ),
        },
    }


def require_stable(first: dict[str, dict], second: dict[str, dict], first_git: dict, second_git: dict) -> None:
    """Raise when two fingerprint passes disagree about the same repository."""
    for field in ("commit", "dirty_entries", "cutlass_sha"):
        if first_git.get(field) != second_git.get(field):
            raise RuntimeError(
                f"prepare: git {field} changed while the tree was being frozen; run the demo again"
            )
    if set(first) != set(second):
        added = sorted(set(second) - set(first))
        removed = sorted(set(first) - set(second))
        raise RuntimeError(
            "prepare: the selected file set changed while the tree was being frozen "
            f"(added {added[:3]}, removed {removed[:3]})"
        )
    for relative in sorted(first):
        left, right = first[relative], second[relative]
        if left["sha256"] != right["sha256"] or left["size"] != right["size"]:
            raise RuntimeError(f"prepare: source {relative!r}: content changed between selection and upload")
        if left["resolved_relative"] != right["resolved_relative"]:
            raise RuntimeError(f"prepare: source {relative!r}: resolution changed between selection and upload")
        changed = [field for field in _IDENTITY_FIELDS if left[field] != right[field]]
        if changed:
            raise RuntimeError(
                f"prepare: source {relative!r}: {', '.join(changed)} changed between selection and upload"
            )


def require_copied(first: dict[str, dict], staged: list[dict]) -> None:
    """Raise when the staged bytes are not the bytes the first pass hashed."""
    by_path = {record["path"]: record for record in staged}
    if set(by_path) != set(first):
        missing = sorted(set(first) - set(by_path))
        extra = sorted(set(by_path) - set(first))
        raise RuntimeError(
            f"prepare: staged file set differs (missing {missing[:3]}, extra {extra[:3]})"
        )
    for relative in sorted(first):
        copied = by_path[relative]
        if copied["sha256"] != first[relative]["sha256"] or copied["size"] != first[relative]["size"]:
            raise RuntimeError(f"prepare: source {relative!r}: the staged bytes differ from the hashed bytes")


def prepare(repo: Path, root: Path | None = None) -> tuple[Path, dict, list[dict]]:
    """Freeze *repo* into a staging directory and return it with its identity.

    Three observations must agree before a snapshot is accepted: a fingerprint
    pass before copying, the copy itself, and a fingerprint pass after it, with the
    selected set and the git identity stable across the whole freeze. A partially
    written staging directory is removed on any failure, so a failed freeze can
    never be mistaken for a usable snapshot.

    The caller owns the returned staging directory and should remove it when the
    run ends; :func:`cleanup` does that for the path recorded in the identity.
    """
    repo = Path(repo)
    owned_root = root is None
    rootpath = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="tilefoundry-demo-src-"))
    staging = rootpath / "tilefoundry"
    try:
        before_git = git_identity(repo)
        paths, skips = select_sources(repo)
        before = fingerprint_sources(repo, paths)
        staged = build_snapshot(repo, staging)
        require_copied(before, staged)
        after_paths, after_skips = select_sources(repo)
        after = fingerprint_sources(repo, after_paths)
        after_git = git_identity(repo)
        require_stable(before, after, before_git, after_git)
        if after_skips != skips:
            raise RuntimeError("prepare: the recorded selection changed while the tree was frozen")
        final = inventory(staging, skip=frozenset())
        if final != staged:
            raise RuntimeError("prepare: the staged inventory changed after it was written")
        skips = sorted([*skips, *missing_tracked(repo, after_paths)], key=lambda record: record["path"])
    except BaseException:
        shutil.rmtree(staging if not owned_root else rootpath, ignore_errors=True)
        raise
    identity = snapshot_identity(
        repo,
        staging,
        final,
        skips=skips,
        links=materialized_links(before),
        git=before_git,
    )
    return staging, identity, final


def cleanup(identity: dict) -> None:
    """Remove the staging directory one snapshot was frozen into."""
    staging = identity.get("staging_dir")
    if not staging:
        return
    root = Path(staging).parent
    if root.name.startswith("tilefoundry-demo-src-") and root.is_dir():
        shutil.rmtree(root, ignore_errors=True)


def pinned_sha(value: str | None, field: str = "cutlass_sha") -> str:
    """Return a required 40-character lowercase hex gitlink SHA or raise."""
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        raise ValueError(
            f"source {field!r}: pin: expected 40 lowercase hex characters, observed {value!r}"
        )
    return value
