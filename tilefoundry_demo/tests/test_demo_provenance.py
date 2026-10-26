"""Local gates for the P0-4 provenance rule: the image ran the uploaded snapshot.

Each test breaks exactly one of the four conditions and requires the check to fail. The
point of the rule is that *none* of these states is a warning: a package of unknown origin
is a package the evidence cannot describe, so it must not reach ``PASS``.

Nothing here imports Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_provenance  # noqa: E402
import demo_report  # noqa: E402

ROOT = "/opt/tilefoundry"


def clean() -> dict:
    """An environment that satisfies every provenance condition."""
    return {
        "tilefoundry_version": "0.0.0+demo.47a6ed6a7dc8",
        "tilefoundry_nvidia_version": "0.0.0+demo.47a6ed6a7dc8",
        "tilefoundry_origin": f"{ROOT}/src/tilefoundry/__init__.py",
        "tilefoundry_nvidia_origin": f"{ROOT}/targets/nvidia/tilefoundry_nvidia/__init__.py",
        "installations": {
            "tilefoundry": {"url": f"file://{ROOT}", "dir_info": {"editable": True}},
            "tilefoundry-nvidia": {
                "url": f"file://{ROOT}/targets/nvidia",
                "dir_info": {"editable": True},
            },
        },
        "source_check": {
            "post_image_build": {"missing": [], "unexpected": [], "modified": []},
            "post_execution": {"missing": [], "unexpected": [], "modified": []},
        },
    }


def test_a_clean_install_has_nothing_to_report() -> None:
    """The baseline: no failure entries at all."""
    assert demo_provenance.check_provenance(clean(), ROOT) == []


def test_core_installed_from_the_wrong_directory_is_rejected() -> None:
    """The core distribution must come from the snapshot root, not a sibling checkout."""
    environment = clean()
    environment["installations"]["tilefoundry"] = {
        "url": "file:///opt/other-tilefoundry",
        "dir_info": {"editable": True},
    }
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("installed from /opt/other-tilefoundry" in item for item in failures)


def test_target_installed_from_the_core_root_is_rejected() -> None:
    """The target must come from targets/nvidia specifically."""
    environment = clean()
    environment["installations"]["tilefoundry-nvidia"] = {
        "url": f"file://{ROOT}",
        "dir_info": {"editable": True},
    }
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("tilefoundry-nvidia" in item and "expected" in item for item in failures)


def test_a_site_packages_copy_is_rejected() -> None:
    """A real (non-editable) install is a stale copy, whatever its version string says."""
    environment = clean()
    environment["installations"]["tilefoundry"] = {
        "url": f"file://{ROOT}",
        "dir_info": {"editable": False},
    }
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("non-editable" in item for item in failures)


def test_a_missing_pep610_record_is_rejected() -> None:
    """Absence of the record is not permission: it is unproven inheritance."""
    environment = clean()
    environment["installations"]["tilefoundry"] = None
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("no PEP 610" in item for item in failures)


def test_a_non_file_url_is_rejected() -> None:
    """A wheel URL from an index is not the snapshot."""
    environment = clean()
    environment["installations"]["tilefoundry"] = {
        "url": "https://files.pythonhosted.org/tilefoundry-0.1.whl",
        "dir_info": {"editable": True},
    }
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("not a file:// URL" in item for item in failures)


def test_an_import_outside_the_snapshot_is_rejected() -> None:
    """The module that actually imported must sit inside the distribution root."""
    environment = clean()
    environment["tilefoundry_origin"] = "/usr/lib/python3.12/site-packages/tilefoundry/__init__.py"
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("which is outside" in item for item in failures)


def test_a_prefix_sibling_directory_is_not_inside_the_snapshot() -> None:
    """``/opt/tilefoundry-evil`` shares a string prefix and is still outside."""
    environment = clean()
    environment["tilefoundry_origin"] = "/opt/tilefoundry-evil/src/tilefoundry/__init__.py"
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("which is outside" in item for item in failures)


def test_mismatched_versions_are_rejected() -> None:
    """Core and target must be one build, not two that happen to both be installed."""
    environment = clean()
    environment["tilefoundry_nvidia_version"] = "0.0.0+demo.ffffffffffff"
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("versions differ" in item for item in failures)


def test_installed_tree_drift_is_rejected() -> None:
    """A file changed in the installed tree means the tree is not the digested snapshot."""
    environment = clean()
    environment["source_check"]["post_execution"] = {
        "missing": [],
        "unexpected": ["src/tilefoundry/extra.py"],
        "modified": [],
    }
    failures = demo_provenance.check_provenance(environment, ROOT)
    assert any("post_execution" in item and "unexpected" in item for item in failures)


def test_the_reducer_fails_a_run_with_provenance_failures() -> None:
    """The environment stage must FAIL, not pass on the weaker prefix flag alone."""
    environment = json.loads(json.dumps(clean()))
    environment.update(
        {
            "imports_ok": True,
            "target_registered": True,
            "tools_ok": True,
            "origin_inside_snapshot": True,
            "provenance_failures": ["tilefoundry: installed from /opt/other, expected /opt/tilefoundry"],
        }
    )
    status, reason = demo_report.evaluate_environment(environment, 0)
    assert status == demo_report.STATUS_FAIL
    assert "not the uploaded snapshot" in reason


def test_the_reducer_passes_an_environment_with_no_provenance_failures() -> None:
    """An empty failure list must not be mistaken for missing evidence."""
    environment = json.loads(json.dumps(clean()))
    environment.update(
        {
            "imports_ok": True,
            "target_registered": True,
            "tools_ok": True,
            "origin_inside_snapshot": True,
            "provenance_failures": [],
        }
    )
    status, reason = demo_report.evaluate_environment(environment, 0)
    assert status == demo_report.STATUS_PASS, reason
