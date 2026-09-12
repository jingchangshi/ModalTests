"""Pytest-side instrumentation for Triton compiler experiments."""

import hashlib
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
    hook_key = "modal-triton-study-pipeline-inspection-v2"
    hook_hash = hashlib.sha256(hook_key.encode("utf-8")).hexdigest()

    def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
        """Triton 3.8-compatible pipeline inspection hook.

        Triton calls this hook in two distinct modes:
        1. With no stage arguments, to obtain a stable (key, hash) pair used in
           JIT/compiler cache keys.
        2. With backend/stages/options/language/capability populated, to inspect
           the concrete compilation pipeline.
        """
        if all(arg is None for arg in (stages, options, language, capability)):
            return hook_key, hook_hash

        record = {
            "stages": list(stages.keys()) if hasattr(stages, "keys") else repr(stages),
            "language": str(language),
            "capability": str(capability),
            "options": repr(options),
            "backend": type(self).__name__ if self is not None else None,
        }
        if record not in stages_seen:
            stages_seen.append(record)
            _write_json("pipeline.json", stages_seen)
        return None

    runtime_knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
    if runtime_knobs is not None and hasattr(runtime_knobs, "add_stages_inspection_hook"):
        runtime_knobs.add_stages_inspection_hook = inspect_stages_hook
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
