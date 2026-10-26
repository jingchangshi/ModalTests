"""Local gates for the source-freeze contract: containment and frozen bytes.

Each test performs one of the attacks or races the snapshot claims to reject and
asserts the rejection, so the claims in ``demo_source`` are executable rather than
asserted. Every case runs against a throwaway git repository on this machine: no
GPU, no Modal, no torch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_report  # noqa: E402
import demo_source  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    """Run one git command in *repo*, failing the test when it fails."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A tiny git repository with tracked source, an excluded tree, and one gitlink."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "third_party" / "cutlass").mkdir(parents=True)
    (root / "src" / "a.py").write_text("A = 1\n")
    (root / "src" / "core.py").write_text("VALUE = 1\n")
    (root / "examples" / "demo.py").write_text("print('x')\n")
    (root / ".gitignore").write_text("*.log\nbuild/\n")
    (root / "noise.log").write_text("ignored\n")
    (root / "build").mkdir()
    (root / "build" / "artifact.txt").write_text("ignored\n")
    for args in (
        ("init", "-q"),
        ("config", "user.email", "demo@example.com"),
        ("config", "user.name", "demo"),
        ("add", "-A"),
        ("commit", "-q", "-m", "init"),
        (
            "update-index",
            "--add",
            "--cacheinfo",
            "160000,e406c186f510a15091cce01f782020ceb7ba8eb5,third_party/cutlass",
        ),
        ("commit", "-q", "-m", "pin submodule"),
    ):
        _git(root, *args)
    return root


@pytest.fixture()
def scratch(tmp_path: Path) -> Path:
    """A private parent directory for the staging tree."""
    path = tmp_path / "freeze"
    path.mkdir()
    return path


def _stage_git_name(repo: Path, name: str) -> None:
    """Make git report one hostile name without ever calling a shell.

    ``git ls-files -z`` output is NUL separated, so a NUL inside a name is
    indistinguishable from a separator: staging the name plus a trailing NUL
    reproduces exactly what a hostile ``.git/index`` would hand the selector.
    """
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "seed.py").write_text("SEED = 1\n")
    _git(repo, "add", "src/seed.py")


def test_rejects_absolute_git_path(monkeypatch, repo: Path) -> None:
    """An absolute git name is refused before it can be resolved."""
    monkeypatch.setattr(
        demo_source,
        "_git_names",
        lambda text: ["/etc/passwd"] + [name for name in text.split("\0") if name],
    )
    with pytest.raises(ValueError) as caught:
        demo_source.select_sources(repo)
    message = str(caught.value)
    assert "source '/etc/passwd'" in message
    assert "absolute" in message


def test_rejects_parent_git_path(monkeypatch, repo: Path) -> None:
    """A traversal name is refused instead of being normalised away."""
    monkeypatch.setattr(
        demo_source,
        "_git_names",
        lambda text: ["src/../../secret.py"] + [name for name in text.split("\0") if name],
    )
    with pytest.raises(ValueError) as caught:
        demo_source.select_sources(repo)
    assert "'..' component" in str(caught.value)


def test_rejects_backslash_and_nul_and_dot_names(repo: Path) -> None:
    """Every other unsafe spelling is refused by the same validator."""
    for name, field in (
        ("src\\a.py", "backslash"),
        ("src/a\x00.py", "NUL"),
        ("src/./a.py", "'.' component"),
        ("src//a.py", "empty component"),
        ("", "empty"),
        ("src/a\n.py", "newline"),
    ):
        with pytest.raises(ValueError) as caught:
            demo_source.validate_source_path(repo, name)
        assert field in str(caught.value), name


def test_rejects_external_symlink(repo: Path) -> None:
    """A file link and a directory link that leave the repository are both refused."""
    outside = repo.parent / "outside.txt"
    outside.write_text("secret\n")
    (repo / "src" / "leak.py").symlink_to(outside)
    with pytest.raises(ValueError) as caught:
        demo_source.validate_source_path(repo, "src/leak.py")
    assert "outside the repository root" in str(caught.value)

    (repo / "src" / "link").symlink_to(repo.parent, target_is_directory=True)
    with pytest.raises(ValueError) as caught:
        demo_source.validate_source_path(repo, "src/link/outside.txt")
    assert "outside the repository root" in str(caught.value)


def test_rejects_dangling_symlink(repo: Path) -> None:
    """A link with no target cannot be frozen, so it is an error, not a skip."""
    (repo / "src" / "dangling.py").symlink_to(repo / "src" / "gone.py")
    _git(repo, "add", "src/dangling.py")
    with pytest.raises(ValueError) as caught:
        demo_source.select_sources(repo)
    assert "does not resolve" in str(caught.value)


def test_rejects_link_swap_during_copy(repo: Path, scratch: Path) -> None:
    """Replacing a selected file with an escaping link after selection is caught."""
    source = repo / "src" / "a.py"
    outside = repo.parent / "passwd"
    outside.write_text("root:x:0:0\n")
    paths, _ = demo_source.select_sources(repo)
    assert "src/a.py" in paths
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError) as caught:
        demo_source.build_snapshot(repo, scratch / "staging")
    assert "outside the repository root" in str(caught.value)
    assert not (scratch / "staging" / "src" / "a.py").exists()


def test_detects_same_porcelain_edit(monkeypatch, repo: Path, scratch: Path) -> None:
    """An untracked edit between passes keeps porcelain identical and is still caught."""
    (repo / "new.py").write_text("old\n")
    real_fingerprint = demo_source.fingerprint_sources
    calls = {"count": 0}

    def fingerprint(repo_path, paths):
        # Both passes hash the same "old" bytes; only the copy in between reads the
        # edited file. ``git status --porcelain`` is unchanged the whole time, so a
        # commit/porcelain comparison alone could never see this edit.
        result = real_fingerprint(repo_path, paths)
        calls["count"] += 1
        if calls["count"] == 1:
            (Path(repo_path) / "new.py").write_text("new\n")
        return result

    monkeypatch.setattr(demo_source, "fingerprint_sources", fingerprint)
    before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    with pytest.raises(RuntimeError) as caught:
        demo_source.prepare(repo, scratch)
    assert "the staged bytes differ from the hashed bytes" in str(caught.value)
    after = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    assert before == after == "?? new.py\n"
    assert not (scratch / "tilefoundry").exists()


def test_detects_copy_race(monkeypatch, repo: Path, scratch: Path) -> None:
    """Rewriting a tracked file to the same length during the read is caught."""
    real_read = os.read
    real_fingerprint = demo_source.fingerprint_sources
    state = {"phase": "fingerprint", "rewritten": False}

    def fingerprint(repo_path, paths):
        state["phase"] = "fingerprint"
        try:
            return real_fingerprint(repo_path, paths)
        finally:
            state["phase"] = "copy"

    def read(fd, size):
        block = real_read(fd, size)
        if state["phase"] == "copy" and not state["rewritten"] and block:
            state["rewritten"] = True
            (repo / "src" / "a.py").write_text("A = 2\n")
        return block

    monkeypatch.setattr(demo_source, "fingerprint_sources", fingerprint)

    monkeypatch.setattr(demo_source.os, "read", read)
    with pytest.raises(RuntimeError):
        # The first fingerprint pass hashes, the copy spans the rewrite, and the
        # second pass is unreachable -- so the copy stage itself must raise.
        demo_source.prepare(repo, scratch)
    assert state["rewritten"] is True
    assert not (scratch / "tilefoundry").exists()


def test_records_untracked_skips(repo: Path) -> None:
    """An allowlisted untracked file ships; a data blob is skipped and recorded."""
    (repo / "notes.py").write_text("NOTE = 1\n")
    (repo / "weights.bin").write_bytes(b"\x00\x01")
    paths, skips = demo_source.select_sources(repo)
    assert "notes.py" in paths
    assert "weights.bin" not in paths
    record = next(skip for skip in skips if skip["path"] == "weights.bin")
    assert record == {
        "path": "weights.bin",
        "tracked": False,
        "reason": "untracked_not_allowlisted",
    }


def test_no_hidden_uploaded_files(repo: Path, scratch: Path) -> None:
    """A scratch file under a runtime directory is skipped, recorded, and never uploaded."""
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "leak.py").write_text("LEAK = 1\n")
    staging, identity, files = demo_source.prepare(repo, scratch)
    uploaded = {record["path"] for record in demo_report.inventory(staging, skip=frozenset())}
    assert ".pytest_cache/leak.py" not in uploaded
    assert {record["path"] for record in files} == uploaded
    skipped = {record["path"]: record for record in identity["selection_policy"]["skipped"]}
    assert skipped[".pytest_cache/leak.py"]["reason"] == "runtime_scratch"
    assert not (staging / ".pytest_cache").exists()


def test_selection_keeps_local_edits_and_drops_excluded_trees(repo: Path) -> None:
    """Untracked work under development ships; examples and the submodule do not."""
    (repo / "src" / "local.py").write_text("EDIT = True\n")
    paths = demo_source.selected_paths(repo)
    assert "src/core.py" in paths
    assert "src/local.py" in paths
    assert "examples/demo.py" not in paths
    assert not any(path.startswith("third_party/cutlass/") for path in paths)
    assert "noise.log" not in paths
    assert "build/artifact.txt" not in paths


def test_snapshot_records_dirty_state_and_cutlass_pin(repo: Path, scratch: Path) -> None:
    """The manifest records that the run used a dirty tree and which submodule commit it pinned."""
    (repo / "src" / "local.py").write_text("EDIT = True\n")
    staging, identity, files = demo_source.prepare(repo, scratch)
    assert identity["dirty"] is True
    assert any("src/local.py" in entry for entry in identity["dirty_entries"])
    assert identity["cutlass_sha"] == "e406c186f510a15091cce01f782020ceb7ba8eb5"
    assert identity["snapshot_file_count"] == len(files)
    assert (staging / "src" / "local.py").is_file()
    assert not (staging / "examples").exists()


def test_snapshot_digest_matches_the_uploaded_bytes(repo: Path, scratch: Path) -> None:
    """The digest is computed from the staging directory the image uploads."""
    staging, identity, files = demo_source.prepare(repo, scratch)
    assert demo_report.digest_records(demo_report.inventory(staging, skip=frozenset())) == identity["snapshot_digest"]
    assert demo_report.inventory_delta(files, demo_report.inventory(staging, skip=frozenset())) == {
        "missing": [],
        "unexpected": [],
        "modified": [],
    }


def test_snapshot_detects_an_edit_made_between_selection_and_upload(repo: Path, scratch: Path) -> None:
    """An edit after freezing is caught by comparing the recorded inventory."""
    staging, _, files = demo_source.prepare(repo, scratch)
    (staging / "src" / "core.py").write_text("VALUE = 2\n")
    delta = demo_report.inventory_delta(files, demo_report.inventory(staging, skip=frozenset()))
    assert delta["modified"] == ["src/core.py"]


def test_internal_symlink_is_materialized_as_a_regular_file(repo: Path, scratch: Path) -> None:
    """A link that stays inside the repository ships as bytes, and the link is recorded."""
    (repo / "src" / "alias.py").symlink_to(repo / "src" / "core.py")
    _git(repo, "add", "src/alias.py")
    staging, identity, files = demo_source.prepare(repo, scratch)
    staged = staging / "src" / "alias.py"
    assert staged.is_file() and not staged.is_symlink()
    assert staged.read_bytes() == (repo / "src" / "core.py").read_bytes()
    assert identity["selection_policy"]["materialized_links"] == [
        {"path": "src/alias.py", "resolved_relative": "src/core.py"}
    ]
    assert {record["path"] for record in files} >= {"src/alias.py", "src/core.py"}


def test_failure_removes_partial_staging(repo: Path, scratch: Path, monkeypatch) -> None:
    """A freeze that fails part way leaves no staging directory behind."""
    real_copy = demo_source.copy_checked
    seen = {"count": 0}

    def copy_checked(repo_path, relative, staging):
        seen["count"] += 1
        if seen["count"] == 2:
            raise RuntimeError("source 'src/core.py': race: simulated")
        return real_copy(repo_path, relative, staging)

    monkeypatch.setattr(demo_source, "copy_checked", copy_checked)
    with pytest.raises(RuntimeError):
        demo_source.prepare(repo, scratch)
    assert not (scratch / "tilefoundry").exists()


def test_selection_policy_records_every_omitted_path(repo: Path, scratch: Path) -> None:
    """Each omission states the path, whether git tracks it, and the reason."""
    _, identity, _ = demo_source.prepare(repo, scratch)
    policy = identity["selection_policy"]
    assert policy["include_rule"].startswith("tracked files from `git ls-files -z`")
    reasons = {record["path"]: record["reason"] for record in policy["skipped"]}
    assert reasons["examples/demo.py"] == "excluded_prefix"
    assert reasons["third_party/cutlass"] == "gitlink"
    # ``noise.log`` is git-ignored, so git never offers it for selection; the
    # allowlist skip is asserted on a name git does report (see
    # test_records_untracked_skips).
    assert [record["path"] for record in policy["skipped"]] == sorted(reasons)
    assert policy["skip_parts"] == sorted(demo_report.SKIP_PARTS)


def test_pinned_sha_requires_forty_lowercase_hex() -> None:
    """A missing or malformed CUTLASS pin is rejected instead of passed through."""
    good = "e406c186f510a15091cce01f782020ceb7ba8eb5"
    assert demo_source.pinned_sha(good) == good
    for bad in (None, "", "0" * 39, "E406C186F510A15091CCE01F782020CEB7BA8EB5", "z" * 40):
        with pytest.raises(ValueError) as caught:
            demo_source.pinned_sha(bad)
        assert "pin" in str(caught.value)
