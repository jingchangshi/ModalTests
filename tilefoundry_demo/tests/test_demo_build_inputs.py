"""Local gates for the P0-2/P0-3 build-input and run-identifier contracts.

Both defects these tests cover used to be undetectable without paying for a GPU run:

* P0-2: the image took its Dockerfile from the live checkout while the Python source came
  from a frozen snapshot, so the manifest could name one revision and the compiler run
  another.
* P0-3: a run id flowed into a local directory and a Volume path unvalidated, and two runs
  in the same second silently shared a namespace.

Nothing here imports Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_build_inputs  # noqa: E402
import demo_report  # noqa: E402

PIN = "e406c186f510a15091cce01f782020ceb7ba8eb5"


@pytest.fixture()
def frozen(tmp_path: Path) -> Path:
    """A snapshot tree holding the three build inputs the image requires."""
    staging = tmp_path / "snapshot"
    (staging / "targets/nvidia").mkdir(parents=True)
    (staging / "Dockerfile").write_text("FROM nvidia/cuda:12.9.0-devel-ubuntu24.04\n")
    (staging / "pyproject.toml").write_text("[project]\nname = 'tilefoundry'\n")
    (staging / "targets/nvidia/pyproject.toml").write_text("[project]\nname = 'tilefoundry-nvidia'\n")
    return staging


def test_build_inputs_come_from_the_frozen_snapshot(frozen: Path) -> None:
    """The recorded Dockerfile digest is the digest of the staged file."""
    record = demo_build_inputs.frozen_image_inputs(frozen, {"cutlass_sha": PIN, "snapshot_digest": "abc"})

    dockerfile = record["dockerfile"]
    assert dockerfile["path"] == "Dockerfile"
    assert dockerfile["sha256"] == demo_report.sha256_file(frozen / "Dockerfile")
    assert record["context_dir"] == str(frozen)
    assert record["cutlass_sha"] == PIN
    assert set(record["build_inputs"]) == set(demo_build_inputs.REQUIRED_BUILD_INPUTS)
    assert record["bootstrap_pip_requirements"]


def test_a_live_dockerfile_cannot_substitute_for_the_frozen_one(frozen: Path, tmp_path: Path) -> None:
    """A Dockerfile edited in the working tree after freezing is not what gets built.

    This is the P0-2 defect stated as a test: the record must name the staged bytes, so
    rewriting some other copy cannot change the digest the manifest reports.
    """
    record = demo_build_inputs.frozen_image_inputs(frozen, {"cutlass_sha": PIN, "snapshot_digest": "abc"})
    live = tmp_path / "live"
    live.mkdir()
    (live / "Dockerfile").write_text("FROM scratch\n")

    assert record["dockerfile"]["sha256"] != demo_report.sha256_file(live / "Dockerfile")
    assert record["context_dir"] == str(frozen)


def test_a_dockerfile_rewritten_after_freezing_is_rejected(frozen: Path) -> None:
    """Re-hashing before the build catches a change between freeze and build."""
    record = demo_build_inputs.frozen_image_inputs(frozen, {"cutlass_sha": PIN, "snapshot_digest": "abc"})
    demo_build_inputs.assert_dockerfile_unchanged(frozen, record)

    (frozen / "Dockerfile").write_text("FROM nvidia/cuda:12.9.0-devel-ubuntu24.04\nRUN false\n")
    with pytest.raises(RuntimeError) as caught:
        demo_build_inputs.assert_dockerfile_unchanged(frozen, record)
    assert "changed after it was digested" in str(caught.value)


@pytest.mark.parametrize("pin", ["", None, "e406c18", "E406C186F510A15091CCE01F782020CEB7BA8EB5", "z" * 40])
def test_an_unusable_cutlass_pin_is_rejected(frozen: Path, pin: object) -> None:
    """An absent or abbreviated pin would fetch whatever HEAD happens to be."""
    with pytest.raises(ValueError) as caught:
        demo_build_inputs.frozen_image_inputs(frozen, {"cutlass_sha": pin, "snapshot_digest": "abc"})
    assert "40-character lowercase hex" in str(caught.value)


@pytest.mark.parametrize("missing", demo_build_inputs.REQUIRED_BUILD_INPUTS)
def test_a_missing_build_input_is_rejected(frozen: Path, missing: str) -> None:
    """A build input absent from the snapshot would have to be read from the live tree."""
    (frozen / missing).unlink()
    with pytest.raises(RuntimeError) as caught:
        demo_build_inputs.frozen_image_inputs(frozen, {"cutlass_sha": PIN, "snapshot_digest": "abc"})
    assert missing in str(caught.value)


def test_a_symlinked_build_input_is_rejected(frozen: Path, tmp_path: Path) -> None:
    """A link could point at a Dockerfile outside the snapshot."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("FROM scratch\n")
    (frozen / "Dockerfile").unlink()
    (frozen / "Dockerfile").symlink_to(elsewhere)

    with pytest.raises(RuntimeError) as caught:
        demo_build_inputs.frozen_image_inputs(frozen, {"cutlass_sha": PIN, "snapshot_digest": "abc"})
    assert "regular file" in str(caught.value)


@pytest.mark.parametrize(
    "run_id",
    ["../escape", "/tmp/x", "a/b", "a\\b", "a b", "a;id", "a$(id)", "", "x" * 65, "-leading", ".hidden"],
)
def test_an_unsafe_run_id_is_rejected(run_id: str, tmp_path: Path) -> None:
    """Run ids become path components in two places, so the alphabet is closed."""
    with pytest.raises(ValueError):
        demo_report.validate_run_id(run_id)
    with pytest.raises(ValueError):
        demo_report.reserve_directory(tmp_path, run_id)


@pytest.mark.parametrize("run_id", ["20261026-034027", "square", "rmsnorm_fp8", "a", "A_1-2", "x" * 64])
def test_a_safe_run_id_is_accepted(run_id: str, tmp_path: Path) -> None:
    """The generated id shape and ordinary names all pass."""
    assert demo_report.validate_run_id(run_id) == run_id
    assert demo_report.reserve_directory(tmp_path, run_id).is_dir()


def test_a_taken_run_id_is_never_overwritten(tmp_path: Path) -> None:
    """Reusing an id would mix two runs' evidence under one name."""
    demo_report.reserve_directory(tmp_path, "run")
    with pytest.raises(FileExistsError):
        demo_report.reserve_directory(tmp_path, "run")


def test_a_taken_run_id_is_a_collision_even_when_empty(tmp_path: Path) -> None:
    """An empty existing directory still owns the name."""
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        demo_report.reserve_directory(tmp_path, "run")


def test_a_symlinked_run_id_is_a_collision(tmp_path: Path) -> None:
    """A link planted at the run path must not be followed."""
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "run").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(FileExistsError):
        demo_report.reserve_directory(tmp_path, "run")


def test_a_generated_run_id_is_suffixed_until_free(tmp_path: Path) -> None:
    """Two runs in the same second get two namespaces."""
    base = "20261026-034027-123456"
    assert demo_report.resolve_run_id(tmp_path, base) == base

    demo_report.reserve_directory(tmp_path, base)
    assert demo_report.resolve_run_id(tmp_path, base) == f"{base[:60]}-001"

    demo_report.reserve_directory(tmp_path, f"{base[:60]}-001")
    assert demo_report.resolve_run_id(tmp_path, base) == f"{base[:60]}-002"
