"""Local gates for the observing plugin, runnable with no GPU and no torch.

The plugin is the only thing standing between "the test passed" and "the demo
can prove the test passed", so its capture logic is tested directly here. The
torch and tilefoundry instrumentation is deliberately not exercised: those
imports do not exist on this machine, and pretending otherwise would test a
mock rather than the real compiler.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_pytest_plugin  # noqa: E402


@pytest.fixture()
def built(tmp_path: Path) -> tuple[Path, Path]:
    """One ``tilefoundry_build_*`` directory holding one built library."""
    build_root = tmp_path / "build"
    library = build_root / "tilefoundry_build_M_split_abc" / "build" / "libM.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELF-library")
    return build_root, tmp_path


def test_capture_copies_the_library_out_of_the_build_root(built) -> None:
    """One build directory yields one library, recorded with its digest."""
    build_root, evidence_root = built
    captured = demo_pytest_plugin._capture_build(build_root, evidence_root)
    assert captured["libraries"] == [
        {
            "path": "generated/build/build/libM.so",
            "size": len(b"\x7fELF-library"),
            "sha256": captured["libraries"][0]["sha256"],
        }
    ]
    assert (evidence_root / "generated/build/build/libM.so").read_bytes() == b"\x7fELF-library"


def test_capture_counts_each_library_once(built) -> None:
    """The copy lands outside the scanned root, so no library is reported twice."""
    build_root, evidence_root = built
    first = demo_pytest_plugin._capture_build(build_root, evidence_root)
    second = demo_pytest_plugin._capture_build(build_root, evidence_root)
    assert len(first["libraries"]) == 1
    assert len(second["libraries"]) == 1


def test_capture_refuses_an_ambiguous_build_root(built) -> None:
    """Two build directories are an error, not a choice."""
    build_root, evidence_root = built
    (build_root / "tilefoundry_build_M_split_xyz" / "build").mkdir(parents=True)
    captured = demo_pytest_plugin._capture_build(build_root, evidence_root)
    assert "error" in captured
    assert "libraries" not in captured


def test_capture_refuses_an_empty_build_root(tmp_path: Path) -> None:
    """No build directory is an error, not an empty success."""
    build_root = tmp_path / "build"
    build_root.mkdir()
    captured = demo_pytest_plugin._capture_build(build_root, tmp_path)
    assert "found 0" in captured["error"]


def test_capture_refuses_a_destination_inside_the_build_root(built) -> None:
    """A destination under TMPDIR would make the scan see its own copy."""
    build_root, _ = built
    inside = build_root / "evidence"
    inside.mkdir()
    captured = demo_pytest_plugin._capture_build(build_root, inside)
    assert "error" in captured


def test_evidence_flush_writes_the_observed_shape(tmp_path: Path, monkeypatch) -> None:
    """The evidence document is JSON with the sections the reducer reads."""
    target = tmp_path / "test-results.json"
    monkeypatch.setenv(demo_pytest_plugin.EVIDENCE_ENV, str(target))
    demo_pytest_plugin._STATE.clear()
    state = demo_pytest_plugin._state()
    state["compile"]["errors"].append("RuntimeError: nvcc failed")
    demo_pytest_plugin._flush()
    written = json.loads(target.read_text())
    assert set(written) == {"compile", "execution", "correctness"}
    assert written["compile"]["errors"] == ["RuntimeError: nvcc failed"]
    assert written["correctness"]["assert_close_calls"] == 0
    demo_pytest_plugin._STATE.clear()
