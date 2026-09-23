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


def _get_cpu_model():
    """Return a stable CPU model string without spawning an external process."""
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    return platform.processor() or platform.machine()


def _get_loadavg():
    """Sample the current 1/5/15-minute load averages and normalized 1m load."""
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        return {
            "loadavg_1m": None,
            "loadavg_5m": None,
            "loadavg_15m": None,
            "normalized_loadavg_1m": None,
        }

    cpu_count = os.cpu_count()
    return {
        "loadavg_1m": load1,
        "loadavg_5m": load5,
        "loadavg_15m": load15,
        "normalized_loadavg_1m": load1 / cpu_count if cpu_count else None,
    }


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
            "cpu_model": _get_cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "hostname": platform.node(),
            "machine": platform.machine(),
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
    item._triton_study_load_before = _get_loadavg()
    item._triton_study_start_ns = time.perf_counter_ns()


def pytest_runtest_teardown(item, nextitem):
    start_ns = getattr(item, "_triton_study_start_ns", None)
    if start_ns is None:
        return

    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    record = {
        "test": item.nodeid,
        "wall_ms": elapsed_ms,
        "load_before": getattr(item, "_triton_study_load_before", {}),
        "load_after": _get_loadavg(),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "test_runtime.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


@pytest.fixture
def triton_study_metrics():
    def record(**metrics):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with (RESULTS_DIR / "kernel_metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")

    return record
