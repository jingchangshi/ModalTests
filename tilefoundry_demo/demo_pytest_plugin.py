"""Observe the upstream test without changing it, and record what it really did.

The demo runs TileFoundry's own integration test unchanged. This plugin is the
only instrumentation: it wraps ``tilefoundry.compile``, the runtime object that
compile returns, ``torch.cuda.synchronize``, and ``torch.testing.assert_close``.
Every wrapper records what it saw and then delegates with the same arguments and
the same return value. Nothing is suppressed, no tolerance is relaxed, and an
exception is recorded and then re-raised.

Evidence is written after each observation so an aborted session still leaves
what was recorded before the abort.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

EVIDENCE_ENV = "TILEFOUNDRY_DEMO_EVIDENCE"
DUMP_ROOT_ENV = "TILEFOUNDRY_DEMO_DUMP_ROOT"
EVIDENCE_ROOT_ENV = "TILEFOUNDRY_DEMO_EVIDENCE_ROOT"
BUILD_ENV = "TILEFOUNDRY_DEMO_BUILD_ROOT"
BUILD_PREFIX = "tilefoundry_build_"

_STATE: dict = {}


def _state() -> dict:
    """The evidence document, created on first use."""
    if not _STATE:
        _STATE.update(
            {
                "compile": {"calls": [], "errors": [], "runtime_name": None},
                "execution": {"runtime_calls": 0, "synchronize_calls": 0, "errors": []},
                "correctness": {
                    "assert_close_calls": 0,
                    "comparisons": [],
                    "errors": [],
                    "reference_source": "x.square() computed by the upstream test in torch",
                },
            }
        )
    return _STATE


def _flush() -> None:
    """Write the evidence document beside the other run artifacts."""
    path = Path(os.environ[EVIDENCE_ENV])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_state(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _RuntimeProxy:
    """A transparent stand-in for the compiled runtime that counts calls.

    Attributes and the call itself are delegated unchanged; the wrapper only
    records that the call returned, which is what distinguishes "the kernel ran"
    from "the kernel was compiled".
    """

    def __init__(self, inner) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def __call__(self, *args, **kwargs):
        state = _state()
        try:
            result = object.__getattribute__(self, "_inner")(*args, **kwargs)
        except BaseException as error:
            state["execution"]["errors"].append(f"{type(error).__name__}: {error}")
            _flush()
            raise
        state["execution"]["runtime_calls"] += 1
        _flush()
        return result

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_inner"))


def _capture_build(build_root: Path, evidence_root: Path) -> dict:
    """Copy the build directory the compile just produced, before anything else runs.

    ``tilefoundry.compile`` builds in a fresh ``tilefoundry_build_*`` directory
    under ``TMPDIR`` and keeps no reference to it. Copying it here, in the same
    process that compiled and before the kernel executes, is the only moment
    when the directory is known to still be the one that produced the library.

    The copy lands under ``<evidence_root>/generated/build``, which is outside
    ``TMPDIR`` so the scan below cannot pick the same library up twice.
    """
    if evidence_root.resolve().is_relative_to(build_root.resolve()) or evidence_root.resolve() == build_root.resolve():
        return {"error": f"evidence root {evidence_root} is inside the build root {build_root}"}
    candidates = sorted(path for path in build_root.glob(f"{BUILD_PREFIX}*") if path.is_dir())
    if len(candidates) != 1:
        return {"error": f"expected one build directory under {build_root}, found {len(candidates)}"}
    source = candidates[0]
    destination = evidence_root / "generated" / "build"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    libraries = [
        {
            "path": path.relative_to(evidence_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted((destination / "build").glob("*.so"))
    ]
    return {
        "source": str(source),
        "directory": destination.relative_to(evidence_root).as_posix(),
        "libraries": libraries,
    }


def _instrument_compile() -> None:
    """Wrap ``tilefoundry.compile`` so its call, result, and failure are recorded."""
    import tilefoundry

    original = tilefoundry.compile
    if getattr(original, "_tilefoundry_demo_wrapped", False):
        return

    def compile_(mod, /, *args, **kwargs):
        state = _state()
        try:
            runtime = original(mod, *args, **kwargs)
        except BaseException as error:
            state["compile"]["errors"].append(f"{type(error).__name__}: {error}")
            _flush()
            raise
        name = getattr(runtime, "name", None)
        state["compile"]["calls"].append(
            {
                "entry": getattr(mod, "entry", None),
                "module": getattr(mod, "name", None),
                "runtime_name": name,
            }
        )
        state["compile"]["runtime_name"] = name
        evidence_root = os.environ.get(EVIDENCE_ROOT_ENV)
        build_root = os.environ.get(BUILD_ENV)
        if evidence_root and build_root:
            state["compile"]["build_capture"] = _capture_build(Path(build_root), Path(evidence_root))
        _flush()
        return _RuntimeProxy(runtime)

    compile_._tilefoundry_demo_wrapped = True
    tilefoundry.compile = compile_


def _instrument_torch() -> None:
    """Wrap the device synchronize and the reference comparison the test uses."""
    import torch

    state = _state()
    original_sync = torch.cuda.synchronize
    if not getattr(original_sync, "_tilefoundry_demo_wrapped", False):

        def synchronize(*args, **kwargs):
            result = original_sync(*args, **kwargs)
            state["execution"]["synchronize_calls"] += 1
            _flush()
            return result

        synchronize._tilefoundry_demo_wrapped = True
        torch.cuda.synchronize = synchronize

    original_close = torch.testing.assert_close
    if getattr(original_close, "_tilefoundry_demo_wrapped", False):
        return

    def assert_close(actual, expected, *args, **kwargs):
        state["correctness"]["assert_close_calls"] += 1
        record = {
            "rtol": kwargs.get("rtol", args[0] if args else None),
            "atol": kwargs.get("atol", args[1] if len(args) > 1 else None),
            "shapes_match": tuple(getattr(actual, "shape", ())) == tuple(getattr(expected, "shape", ())),
            "dtypes": [str(getattr(actual, "dtype", None)), str(getattr(expected, "dtype", None))],
        }
        try:
            result = original_close(actual, expected, *args, **kwargs)
        except BaseException as error:
            record["outcome"] = f"{type(error).__name__}: {error}"
            state["correctness"]["comparisons"].append(record)
            state["correctness"]["errors"].append(record["outcome"])
            _flush()
            raise
        record["outcome"] = "passed"
        state["correctness"]["comparisons"].append(record)
        _flush()
        return result

    assert_close._tilefoundry_demo_wrapped = True
    torch.testing.assert_close = assert_close


def pytest_configure(config) -> None:
    """Redirect the suite's dump root into the evidence bundle, before collection."""
    root = os.environ.get(DUMP_ROOT_ENV)
    _state()["dumps_root"] = root
    if not root:
        return
    import tests.conftest as suite_conftest

    suite_conftest._RESULTS_ROOT = Path(root)


def pytest_collection_finish(session) -> None:
    """Record which node IDs were collected, so a silent skip is visible."""
    _state()["collected"] = [item.nodeid for item in session.items]
    _flush()


def pytest_runtest_logreport(report) -> None:
    """Record the per-phase outcome pytest itself recorded."""
    if report.when != "call" and report.outcome == "passed":
        return
    _state().setdefault("reports", []).append(
        {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
    )
    _flush()


def pytest_runtest_setup(item) -> None:
    """Instrument before the test body so a compile in setup is still observed."""
    _instrument_compile()
    _instrument_torch()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Record the exit status and write the final evidence document."""
    state = _state()
    state["exitstatus"] = int(exitstatus)
    _flush()
