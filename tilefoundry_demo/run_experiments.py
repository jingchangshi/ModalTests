#!/usr/bin/env python3
"""Run the RMSNorm + FP8 tutorial experiments on a real H200.

This is the P2 driver. It reuses ``run_demo``'s frozen image, so the experiments see
exactly the environment the P0/P1 gates already verified: the same source snapshot, the
same digest, the same installed packages. It adds no new build path.

Each experiment writes one JSON record. A record carries the command that was run, the
exit status, and the tool's own output, so a reader can re-run it and compare. Nothing
here predicts performance: ``benchmark`` records what ``torch.cuda.Event`` measured on the
device that ran, and it records the baseline it compared against.

Usage:
    modal run run_experiments.py --experiment correctness
    modal run run_experiments.py --experiment twin
    modal run run_experiments.py --experiment benchmark
    modal run run_experiments.py --experiment all
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

import run_demo

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
GPU_SPEC = "H200"
# The benchmark runs many timed repeats; the H200 is fast but the Triton compile plus the
# PyTorch baseline need room.
TIMEOUT_SECONDS = 1800
VOLUME_NAME = run_demo.VOLUME_NAME

app = modal.App("tilefoundry-rmsnorm-fp8")

CASE = "rmsnorm_fp8"

if modal.is_local():
    IMAGE = run_demo.IMAGE
else:
    IMAGE = modal.Image.debian_slim()

# Modal rejects any local mount that follows another one, and `run_demo.IMAGE` already
# ends in an `add_local_python_source`. The case files therefore travel through the same
# Volume the demo archives to, uploaded by `main` before the GPU is requested. This keeps
# `run_demo`'s frozen image chain untouched, so the experiments see exactly the
# environment the P0/P1 gates verified.
CASE_REMOTE = f"cases/{CASE}"
CASE_LOCAL = CASES / CASE


UPLOADED_FILES = (
    "config.json",
    "x.pt",
    "expected.pt",
    "model.safetensors",
    "rms_norm_quant.py",
    "rms_norm_quant_opt.py",
    "twin.py",
    "benchmark.py",
    "make_reference.py",
    "agent_candidate.py",
)


def upload_case() -> None:
    """Publish the case files to the Volume so every stage reads identical bytes.

    A file absent from the case directory is skipped rather than fatal, because the
    optional ones (the agent candidate) exist only for the experiments that use them.
    """
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for name in UPLOADED_FILES:
            path = CASE_LOCAL / name
            if path.exists():
                batch.put_file(str(path), f"{CASE_REMOTE}/{name}")


CASE_DIR = Path("/cases") / CASE


def ingest_case() -> Path:
    """Copy the case files from the mounted Volume into a private working directory.

    The upload happens once per experiment run, and every stage in that run reads the
    same bytes, so two stages cannot silently disagree about the input tensors. The copy
    is per call so a stage that writes next to its inputs cannot affect the next stage.
    """
    import shutil
    import tempfile

    source = Path("/artifacts") / CASE_REMOTE
    names = ("config.json", "x.pt", "expected.pt", "model.safetensors")
    missing = [name for name in names if not (source / name).exists()]
    if missing:
        raise RuntimeError(f"the Volume has no case files: missing {', '.join(missing)}")
    work = Path(tempfile.mkdtemp(prefix=f"{CASE}-"))
    for name in names:
        shutil.copy2(source / name, work / name)
    # The authored modules are the case's own code, so they travel with it. They are
    # read from the same Volume copy, not from the image, so an experiment always runs
    # the case revision that was uploaded with it.
    for peer in source.iterdir():
        if peer.is_file() and peer.suffix in {".py", ".json", ".txt"} and not (work / peer.name).exists():
            shutil.copy2(peer, work / peer.name)
    return work


def coarse(value):
    """Reduce a record to JSON primitives before it crosses the process boundary.

    The container has torch and this host does not, so returning a tensor or a numpy
    scalar makes Modal fail to deserialise the result. Coercing here keeps the record
    readable on a machine that has neither library.
    """
    if isinstance(value, dict):
        return {str(key): coarse(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [coarse(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _tail(text: str, limit: int = 4000) -> str:
    """Keep the end of a log, which is where compilers and stack traces say why."""
    if text is None:
        return ""
    return text if len(text) <= limit else "..." + text[-limit:]


def guard(body, label: str):
    """Run one stage and turn an exception into a record instead of a bare raise.

    The container has torch and the host does not, so Modal cannot deserialise a torch
    exception object here; the failure would surface as "torch is not available" and hide
    the real cause. Returning the traceback as text keeps the cause readable.
    """
    import traceback

    try:
        return body()
    except BaseException:
        # BaseException, not Exception: a subprocess wrapper or a C extension can raise
        # something outside the usual hierarchy, and letting it escape loses the cause to
        # a deserialisation error on a host that has no torch.
        return {"stage": label, "raised": traceback.format_exc()}


def _correctness() -> dict:
    """P2-1 and P2-2 on the GPU: does the corrected HIR check, and does the wrong one fail?

    The two runs differ only in the authored module. If both passed, the check would not
    be discriminating; if both failed, the checker or the reference would be broken.
    """
    import subprocess
    import sys

    case = ingest_case()
    records: dict[str, object] = {
        "case": CASE,
        "device": "cuda",
        "python": str(sys.version),
        "torch": None,
        "runs": [],
    }
    import torch

    # str() and int(): these cross the process boundary, and the host that reads the
    # record has no torch, so a version object or a numpy scalar would fail to load.
    records["torch"] = str(torch.__version__)
    records["gpu_name"] = str(torch.cuda.get_device_name(0))
    records["compute_capability"] = [int(part) for part in torch.cuda.get_device_capability(0)]

    common = [
        "--inputs",
        "files:x.pt",
        "--weights",
        "ckpt:.",
        "--expected",
        "expected.pt",
        "--device",
        "cuda",
        "--out",
        "output[0]",
        "--fn",
        "equal",
        "--out",
        "output[1]",
        "--fn",
        "allclose",
        "--atol",
        "1e-6",
        "--rtol",
        "1e-6",
    ]
    for label, module in (
        ("wrong", "rms_norm_quant.py:Wrong"),
        ("corrected", "rms_norm_quant.py:RmsNormQuant"),
    ):
        command = ["tilefoundry", "check", module, *common]
        completed = subprocess.run(
            command, cwd=case, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
        records["runs"].append(
            {
                "label": label,
                "module": module,
                "command": command,
                "returncode": completed.returncode,
                "stdout": _tail(completed.stdout),
                "stderr": _tail(completed.stderr),
            }
        )

    # `analyze` reads the authored program alone: it takes a SOURCE selector and report
    # flags, and no input tensors or device, because it costs the program rather than
    # running it. Both placement variants are costed so the comparison is between two
    # reports of the same step, not between a report and a measurement.
    for label, selector in (
        ("analysis_naive", "rms_norm_quant_opt.py:Naive"),
        ("analysis_fused", "rms_norm_quant_opt.py:Fused"),
        # The agent's own placement is priced by the same report, so its claim is checked
        # by the compiler rather than asserted here. It exists only for P2-5.
        *(
            [("analysis_agent", "agent_candidate.py:SplitMeta")]
            if (case / "agent_candidate.py").exists()
            else []
        ),
    ):
        report = case / f"{label}.txt"
        analysis = [
            "tilefoundry",
            "analyze",
            selector,
            str(report),
            "--compute-cost",
            "--memory",
            "--roofline",
        ]
        completed = subprocess.run(
            analysis, cwd=case, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
        # The documented invocation writes the report to a file and keeps the summary
        # lines on grep. Read those lines back so the record carries the totals a reader
        # can compare against the tutorial, not a truncated tail of the whole report.
        summary = ""
        if report.exists():
            summary = "\n".join(
                line
                for line in report.read_text().splitlines()
                if line.startswith("# memory ") or line.startswith("# roofline ")
            )
        records[label] = {
            "command": analysis,
            "returncode": completed.returncode,
            "summary": summary,
            "stdout": _tail(completed.stdout),
            "stderr": _tail(completed.stderr),
        }

    # P2-5: the agent-authored candidate must clear the same gate. It is a separate
    # module so the tutorial's own Fused definition stays untouched and both remain
    # available for comparison.
    if (case / "agent_candidate.py").exists():
        # The checked variant, not the 264-row analysis variant: `check` binds x.pt and
        # expected.pt, which carry 2 rows.
        command = [
            "tilefoundry",
            "check",
            "agent_candidate.py:SplitMetaChecked",
            "--inputs",
            "files:x.pt",
            "--weights",
            "ckpt:.",
            "--expected",
            "expected.pt",
            "--device",
            "cuda",
            "--out",
            "output[0]",
            "--fn",
            "equal",
            "--out",
            "output[1]",
            "--fn",
            "allclose",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-6",
        ]
        completed = subprocess.run(
            command, cwd=case, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
        records["agent_candidate"] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": _tail(completed.stdout),
            "stderr": _tail(completed.stderr),
        }

    return coarse(records)


def _twin() -> dict:
    """P2-3: run the Triton twin through the checker on the device.

    The twin must produce a passed comparison, and the record must show the Triton kernel
    actually launched, so a skipped or mocked kernel cannot pass as a run.
    """
    import subprocess
    import sys

    case = ingest_case()
    command = [
        "tilefoundry",
        "check",
        "twin.py:Fast",
        "--inputs",
        "files:x.pt",
        "--weights",
        "ckpt:.",
        "--expected",
        "expected.pt",
        "--device",
        "cuda",
        "--out",
        "output[0]",
        "--fn",
        "equal",
        "--out",
        "output[1]",
        "--fn",
        "allclose",
        "--atol",
        "1e-6",
        "--rtol",
        "1e-6",
    ]
    completed = subprocess.run(
        command, cwd=case, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
    )
    import torch

    record = {
        "case": CASE,
        "command": command,
        "returncode": completed.returncode,
        "stdout": _tail(completed.stdout),
        "stderr": _tail(completed.stderr),
        "torch": str(torch.__version__),
        "gpu_name": str(torch.cuda.get_device_name(0)),
    }
    try:
        import triton

        record["triton"] = triton.__version__
    except Exception as error:  # pragma: no cover - the image either has Triton or not
        record["triton"] = f"unavailable: {error}"
    return coarse(record)


def _benchmark() -> dict:
    """P2-4: measured latency of the fused step on the device.

    The measurement is ``torch.cuda.Event`` around a launched kernel, after a warmup, over
    many repeats, reported as median and P10/P90. The baseline is named in the record: it
    is the unfused PyTorch composition of the same step, so the comparison states what it
    compares rather than implying a claim about the compiler.
    """
    import subprocess
    import sys

    case = ingest_case()
    # benchmark.py already prints one JSON object on stdout; there is no --json flag.
    command = [sys.executable, "benchmark.py"]
    completed = subprocess.run(
        command, cwd=case, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
    )
    record = {
        "case": CASE,
        "command": command,
        "returncode": completed.returncode,
        "stdout": _tail(completed.stdout, 20000),
        "stderr": _tail(completed.stderr),
    }
    # benchmark.py pretty-prints one JSON object across many lines, so parse the whole
    # stdout rather than its last line.
    try:
        measurement = json.loads(completed.stdout)
    except Exception:
        measurement = None
    record["measurement"] = measurement
    if isinstance(measurement, dict):
        arms = measurement.get("arms") or {}
        record["summary_ms"] = {
            arm: {
                key: values.get(key)
                for key in ("median_ms", "p10_ms", "p90_ms", "repeats", "warmup")
            }
            for arm, values in arms.items()
        }
        record["agreement"] = measurement.get("agreement")
    return coarse(record)


@app.function(image=IMAGE, gpu=GPU_SPEC, timeout=TIMEOUT_SECONDS, volumes={"/artifacts": run_demo.artifacts})
def correctness_experiment() -> dict:
    """P2-1, P2-2 and P2-5 on the device."""
    return guard(_correctness, "correctness")


@app.function(image=IMAGE, gpu=GPU_SPEC, timeout=TIMEOUT_SECONDS, volumes={"/artifacts": run_demo.artifacts})
def twin_experiment() -> dict:
    """P2-3 on the device."""
    return guard(_twin, "twin")


@app.function(image=IMAGE, gpu=GPU_SPEC, timeout=TIMEOUT_SECONDS, volumes={"/artifacts": run_demo.artifacts})
def benchmark_experiment() -> dict:
    """P2-4 on the device."""
    return guard(_benchmark, "benchmark")


@app.local_entrypoint()
def main(experiment: str = "all") -> None:
    """Run one experiment and print its record, plus the path where it was archived."""
    wanted = (
        ["correctness", "twin", "benchmark"]
        if experiment == "all"
        else [experiment]
    )
    unknown = [name for name in wanted if name not in {"correctness", "twin", "benchmark"}]
    if unknown:
        raise SystemExit(f"unknown experiment(s): {', '.join(unknown)}")

    upload_case()
    stamp = run_demo._timestamp()
    results: dict[str, object] = {"run_stamp": stamp, "experiments": {}}
    if "correctness" in wanted:
        results["experiments"]["correctness"] = correctness_experiment.remote()
    if "twin" in wanted:
        results["experiments"]["twin"] = twin_experiment.remote()
    if "benchmark" in wanted:
        results["experiments"]["benchmark"] = benchmark_experiment.remote()

    out_dir = HERE / "results" / f"experiments-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiments.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nexperiment record: {out_path}")
