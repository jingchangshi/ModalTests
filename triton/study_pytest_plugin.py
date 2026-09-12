"""Pytest-side instrumentation for Triton compiler experiments."""

import json
import os
import platform
import time
from pathlib import Path

import pytest


RESULTS_DIR = Path(os.environ.get("TRITON_STUDY_RESULTS_DIR", "/tmp/triton-study/results"))


def _write_json(name, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(data, indent=2, sort_keys=True))


def pytest_sessionstart(session):
    import torch
    import triton

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    _write_json(
        "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": [props.major, props.minor],
            "gpu_memory_bytes": props.total_memory,
        },
    )

    stages_seen = []

    def inspect_stages(*args):
        # Triton's hook is also called with no arguments when its value is
        # incorporated into the compiler cache key. Return a stable key then.
        if not args:
            return "modal-triton-study-v1"

        if len(args) != 5:
            record = {"warning": "unexpected add_stages_inspection_hook signature", "argc": len(args)}
        else:
            _backend, stages, options, language, capability = args
            record = {
                "stages": list(stages.keys()),
                "language": str(language),
                "capability": str(capability),
                "options": repr(options),
            }

        if record not in stages_seen:
            stages_seen.append(record)
            _write_json("pipeline.json", stages_seen)
        return None

    runtime_knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    if runtime_knobs is not None and hasattr(runtime_knobs, "add_stages_inspection_hook"):
        runtime_knobs.add_stages_inspection_hook = inspect_stages
    else:
        _write_json(
            "pipeline.json",
            [{"warning": "Triton build does not expose add_stages_inspection_hook"}],
        )


def pytest_runtest_setup(item):
    item._triton_study_start_ns = time.perf_counter_ns()


def pytest_runtest_teardown(item, nextitem):
    start_ns = getattr(item, "_triton_study_start_ns", None)
    if start_ns is None:
        return
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "pytest_wall_times.jsonl").open("a") as f:
        f.write(json.dumps({"test": item.nodeid, "wall_ms": elapsed_ms}) + "\n")


@pytest.fixture
def triton_study_metrics():
    def record(**metrics):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with (RESULTS_DIR / "kernel_metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")

    return record
