#!/usr/bin/env python3
"""Run the RMSNorm + FP8 tutorial experiments on a real H200.

This is the P2 driver. It reuses run_demo frozen image, so the experiments see exactly
the environment the P0/P1 gates already verified: the same source snapshot, the same
digest, the same installed packages. It adds no new build path.

Each experiment writes one JSON record. A record carries the command that was run, the
exit status, and the tool own output, so a reader can re-run it and compare. Nothing
here predicts performance: benchmark records what torch.cuda.Event measured on the
device that ran, and it records the baseline it compared against.

The reference tensors are GENERATED, NOT SHIPPED. A fresh clone holds only the case
source and config.json; a Modal CPU Function runs make_reference.py once per run and
publishes the three files under a run-scoped Volume prefix. The GPU stages read only that
prefix, so two runs can never share mutable inputs and a stale directory from an earlier
run cannot be mistaken for this run data.

Usage:
    modal run run_experiments.py --experiment all
    modal run run_experiments.py --experiment correctness
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

import run_demo
from demo_report import sha256_file, validate_run_id, write_json

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
GPU_SPEC = "H200"
TIMEOUT_SECONDS = 1800
GENERATE_TIMEOUT_SECONDS = 900
VOLUME_NAME = run_demo.VOLUME_NAME
VOLUME_ROOT = "/artifacts"

app = modal.App("tilefoundry-rmsnorm-fp8")

CASE = "rmsnorm_fp8"

if modal.is_local():
    IMAGE = run_demo.IMAGE
else:
    IMAGE = modal.Image.debian_slim()

CASE_LOCAL = CASES / CASE

# Only the case source and configuration travel. The reference tensors are produced in
# the container from make_reference.py, so no checkout can carry a previous machine data
# into the run.
SOURCE_FILES = (
    "config.json",
    "make_reference.py",
    "rms_norm_quant.py",
    "rms_norm_quant_opt.py",
    "twin.py",
    "benchmark.py",
    "agent_candidate.py",
)

REFERENCE_FILES = ("x.pt", "expected.pt", "model.safetensors")

# The shape/dtype contract the shipped case depends on. A generator that produced
# something else would make tilefoundry check fail later, on a GPU, with a message about
# tensors rather than about generation.
EXPECTED_REFERENCE = {
    "x": {"shape": [2, 7168], "dtype": "torch.bfloat16"},
    "expected_count": 2,
    "quant": {"shape": [2, 7168], "dtype": "torch.float8_e4m3fn"},
    "scale": {"shape": [2, 56], "dtype": "torch.float32"},
    "gamma": {"shape": [1, 7168], "dtype": "torch.bfloat16"},
}

REFERENCE_MANIFEST = "reference-manifest.json"


def run_prefix(run_id: str) -> str:
    """The Volume-relative prefix one run owns, and nobody else writes to."""
    validate_run_id(run_id)
    return f"experiments/{run_id}/cases/{CASE}"


def upload_source(run_id: str) -> dict:
    """Publish the case source files to this run own Volume prefix.

    Returns the SHA-256 of each uploaded file, so the generation record can name the
    exact bytes it ran against.
    """
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    prefix = run_prefix(run_id)
    published: dict[str, str] = {}
    with volume.batch_upload(force=True) as batch:
        for name in SOURCE_FILES:
            path = CASE_LOCAL / name
            # The agent candidate exists only for the experiments that use it.
            if not path.is_file():
                continue
            batch.put_file(str(path), f"{prefix}/{name}")
            published[name] = sha256_file(path)
    for required in ("make_reference.py", "config.json"):
        if required not in published:
            raise SystemExit(
                f"{CASE_LOCAL / required} is missing; the reference data cannot be generated"
            )
    return published


def verify_reference(work: Path) -> dict:
    """Check generated files against the case contract, or explain what is wrong.

    Runs in the generating container, where torch is available. Every failure raises, so
    a generation that produced the wrong thing can never reach a GPU stage.

    Args:
        work: The directory make_reference.py wrote into.

    Returns:
        The manifest to record beside the generated data.

    Raises:
        RuntimeError: If a file is absent, empty, unreadable, or off-contract.
    """
    import safetensors
    import torch
    import transformers
    from safetensors.torch import load_file

    missing = [name for name in REFERENCE_FILES if not (work / name).is_file()]
    if missing:
        raise RuntimeError(f"make_reference.py did not write {', '.join(missing)}")
    empty = [name for name in REFERENCE_FILES if (work / name).stat().st_size == 0]
    if empty:
        raise RuntimeError(f"make_reference.py wrote empty files: {', '.join(empty)}")

    def check(label: str, tensor, expected: dict) -> None:
        """Refuse a tensor whose shape or dtype is not the case contract."""
        shape = [int(size) for size in tensor.shape]
        if shape != expected["shape"]:
            raise RuntimeError(f"{label}: shape {shape} is not {expected['shape']}")
        if str(tensor.dtype) != expected["dtype"]:
            raise RuntimeError(f"{label}: dtype {tensor.dtype} is not {expected['dtype']}")

    check(
        "x.pt",
        torch.load(work / "x.pt", weights_only=True, map_location="cpu"),
        EXPECTED_REFERENCE["x"],
    )

    values = torch.load(work / "expected.pt", weights_only=True, map_location="cpu")
    if not isinstance(values, (list, tuple)) or len(values) != EXPECTED_REFERENCE[
        "expected_count"
    ]:
        raise RuntimeError(
            "expected.pt: expected a two-element sequence [quant, scale], observed "
            f"{type(values).__name__}"
        )
    quant, scale = values
    check("expected.pt[0] (quant)", quant, EXPECTED_REFERENCE["quant"])
    check("expected.pt[1] (scale)", scale, EXPECTED_REFERENCE["scale"])

    weights = load_file(str(work / "model.safetensors"))
    if "gamma" not in weights:
        raise RuntimeError(f"model.safetensors: no gamma key, only {sorted(weights)}")
    check("model.safetensors[gamma]", weights["gamma"], EXPECTED_REFERENCE["gamma"])

    return {
        "files": {
            name: {
                "size": (work / name).stat().st_size,
                "sha256": sha256_file(work / name),
            }
            for name in REFERENCE_FILES
        },
        "contract": EXPECTED_REFERENCE,
        "generator": {
            "script": "make_reference.py",
            "script_sha256": sha256_file(work / "make_reference.py"),
            "config_sha256": sha256_file(work / "config.json"),
        },
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "safetensors": str(safetensors.__version__),
        },
        # The generator writes this beside its output. Carrying it here is what lets a
        # reader tell this run provenance from the historical committed record.
        "provenance": json.loads(
            (work / "reference-provenance.json").read_text(encoding="utf-8")
        ),
    }

def generate_reference(run_id: str, source_digests: dict) -> dict:
    """Generate and validate the reference data for one run. Runs in the container.

    A CPU Function on purpose: H200 time buys nothing for a torch reduction, and
    generating on CPU keeps the GPU path from depending on a mutable shared directory.
    """
    import subprocess
    import sys

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    prefix = run_prefix(run_id)
    source = Path(VOLUME_ROOT) / prefix
    work = Path("/tmp") / f"{CASE}-{run_id}"
    work.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        if (source / name).is_file():
            (work / name).write_bytes((source / name).read_bytes())

    completed = subprocess.run(
        [sys.executable, "make_reference.py"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=GENERATE_TIMEOUT_SECONDS,
    )
    record: dict = {
        "run_id": run_id,
        "command": [sys.executable, "make_reference.py"],
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "source_sha256": dict(sorted(source_digests.items())),
    }
    if completed.returncode != 0:
        record["error"] = f"make_reference.py exited {completed.returncode}"
        return record
    try:
        record.update(verify_reference(work))
    except BaseException as error:  # noqa: BLE001 - a bad generation is a record, not a crash
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    # Publish beside the source, in this run own prefix. Committing here is what makes
    # the files visible to the GPU Functions that run afterwards.
    for name in REFERENCE_FILES:
        (source / name).write_bytes((work / name).read_bytes())
    write_json(source / REFERENCE_MANIFEST, record)
    volume.commit()
    return record


def load_reference(run_id: str) -> dict:
    """Read this run reference manifest in a GPU Function, refusing stale data.

    Args:
        run_id: The run whose prefix to read.

    Returns:
        The manifest the generator wrote.

    Raises:
        RuntimeError: If this run has no generated data of its own.
    """
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    # Another container wrote these files, so the mount has to be refreshed before the
    # read; otherwise a long-lived container could serve an unchanged view.
    volume.reload()
    manifest_path = Path(VOLUME_ROOT) / run_prefix(run_id) / REFERENCE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(
            f"run {run_id!r} has no generated reference data at {manifest_path}; "
            "the generator did not run or did not publish"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ingest_case(run_id: str) -> Path:
    """Copy this run case files into a private working directory.

    Every reference byte is verified against the digest the generator recorded, so a
    container that somehow sees an older or partial directory fails here rather than
    producing a comparison against the wrong reference.

    Args:
        run_id: The run whose prefix to read.

    Returns:
        A writable copy of the case, carrying only this run generated data.

    Raises:
        RuntimeError: If a required file is missing or its digest disagrees.
    """
    import shutil
    import tempfile

    manifest = load_reference(run_id)
    source = Path(VOLUME_ROOT) / run_prefix(run_id)
    work = Path(tempfile.mkdtemp(prefix=f"{CASE}-"))
    for name in (*SOURCE_FILES, *REFERENCE_FILES):
        peer = source / name
        if not peer.is_file():
            if name in REFERENCE_FILES:
                raise RuntimeError(f"run {run_id!r} is missing generated file {name}")
            continue
        shutil.copy2(peer, work / name)

    recorded = manifest.get("files") or {}
    for name in REFERENCE_FILES:
        expected = (recorded.get(name) or {}).get("sha256")
        observed = sha256_file(work / name)
        if expected != observed:
            raise RuntimeError(
                f"{name}: digest {observed} does not match this run manifest {expected}; "
                "refusing to compare against data that is not the data this run generated"
            )
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
    exception object here; the failure would surface as a torch is not available message
    and hide the real cause. Returning the traceback as text keeps the cause readable.
    """
    import traceback

    try:
        return body()
    except BaseException:
        # BaseException, not Exception: a subprocess wrapper or a C extension can raise
        # something outside the usual hierarchy, and letting it escape loses the cause to
        # a deserialisation error on a host that has no torch.
        return {"stage": label, "raised": traceback.format_exc()}

def _correctness(run_id: str) -> dict:
    """P2-1 and P2-2 on the GPU: does the corrected HIR check, and does the wrong one fail?

    The two runs differ only in the authored module. If both passed, the check would not
    be discriminating; if both failed, the checker or the reference would be broken.
    """
    import subprocess
    import sys

    case = ingest_case(run_id)
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

    # analyze reads the authored program alone: it takes a SOURCE selector and report
    # flags, and no input tensors or device, because it costs the program rather than
    # running it. Both placement variants are costed so the comparison is between two
    # reports of the same step, not between a report and a measurement.
    for label, selector in (
        ("analysis_naive", "rms_norm_quant_opt.py:Naive"),
        ("analysis_fused", "rms_norm_quant_opt.py:Fused"),
        # The agent own placement is priced by the same report, so its claim is checked
        # by the compiler rather than asserted here. It exists only for P2-5.
        *((
            [("analysis_agent", "agent_candidate.py:SplitMeta")]
            if (case / "agent_candidate.py").exists()
            else []
        )),
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
    # module so the tutorial own Fused definition stays untouched and both remain
    # available for comparison.
    if (case / "agent_candidate.py").exists():
        # The checked variant, not the 264-row analysis variant: check binds x.pt and
        # expected.pt, which carry 2 rows.
        command = [
            "tilefoundry",
            "check",
            "agent_candidate.py:SplitMetaChecked",
            *common,
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


def _twin(run_id: str) -> dict:
    """P2-3: run the Triton twin through the checker on the device.

    The twin must produce a passed comparison, and the record must show the Triton kernel
    actually launched, so a skipped or mocked kernel cannot pass as a run.
    """
    import subprocess
    import sys

    case = ingest_case(run_id)
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


def _benchmark(run_id: str) -> dict:
    """P2-4: measured latency of the fused step on the device.

    The measurement is torch.cuda.Event around a launched kernel, after a warmup, over
    many repeats, reported as median and P10/P90. The baseline is named in the record: it
    is the unfused PyTorch composition of the same step, so the comparison states what it
    compares rather than implying a claim about the compiler.
    """
    import subprocess
    import sys

    case = ingest_case(run_id)
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


@app.function(
    image=IMAGE,
    timeout=GENERATE_TIMEOUT_SECONDS,
    volumes={VOLUME_ROOT: run_demo.artifacts},
)
def reference_generator(run_id: str, source_digests: dict) -> dict:
    """Generate and validate this run reference data, with no GPU allocated."""
    return coarse(guard(lambda: generate_reference(run_id, source_digests), "reference"))


@app.function(
    image=IMAGE,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_ROOT: run_demo.artifacts},
)
def correctness_experiment(run_id: str) -> dict:
    """P2-1, P2-2 and P2-5 on the device."""
    return guard(lambda: _correctness(run_id), "correctness")


@app.function(
    image=IMAGE,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_ROOT: run_demo.artifacts},
)
def twin_experiment(run_id: str) -> dict:
    """P2-3 on the device."""
    return guard(lambda: _twin(run_id), "twin")


@app.function(
    image=IMAGE,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_ROOT: run_demo.artifacts},
)
def benchmark_experiment(run_id: str) -> dict:
    """P2-4 on the device."""
    return guard(lambda: _benchmark(run_id), "benchmark")

EXPERIMENTS = ("correctness", "twin", "benchmark")


def assert_reference_ready(record: dict, run_id: str) -> None:
    """Stop the run unless this run reference data was generated and validated.

    This is the gate that makes a generation failure cost nothing: it runs on the host,
    after the CPU Function returned and before any GPU Function is called. A GPU stage
    can therefore never read a stale directory from an earlier run, because it is never
    reached at all when generation failed.

    Args:
        record: The generation record the CPU Function returned.
        run_id: The run whose data it describes.

    Raises:
        SystemExit: Naming why the data is unusable.
    """
    if record.get("error"):
        raise SystemExit(
            f"reference generation failed for run {run_id!r}: {record['error']}. "
            "No GPU was allocated and no experiment was run."
        )
    files = record.get("files") or {}
    missing = [name for name in REFERENCE_FILES if not (files.get(name) or {}).get("sha256")]
    if missing:
        raise SystemExit(
            f"reference generation for run {run_id!r} recorded no digest for "
            f"{', '.join(missing)}; no GPU was allocated."
        )


@app.local_entrypoint()
def main(experiment: str = "all", run_id: str = "") -> None:
    """Generate this run reference data, then run the requested experiments on a GPU.

    Order matters and is the contract: claim a run namespace, publish the case source,
    generate and validate the reference tensors on CPU, and only then ask for a GPU. A
    generation failure exits before any device stage, so a failed run can never be
    reported as a completed one.
    """
    wanted = list(EXPERIMENTS) if experiment == "all" else [experiment]
    unknown = [name for name in wanted if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(
            f"unknown experiment(s): {', '.join(unknown)}; "
            f"choose from all, {', '.join(EXPERIMENTS)}"
        )

    # The run id namespaces both the Volume prefix and the local result directory. A
    # supplied id must be safe; a generated one is re-suffixed until it is free, so two
    # runs can never share a prefix.
    results_root = HERE / "results"
    generated = not run_id
    if generated:
        from demo_report import resolve_run_id

        run_id = resolve_run_id(results_root, run_demo._timestamp())
    else:
        validate_run_id(run_id)
    from demo_report import reserve_directory

    try:
        reserve_directory(results_root, f"experiments-{run_id}")
    except FileExistsError:
        raise SystemExit(f"run id {run_id!r} already has experiment evidence under {results_root}") from None

    print(f"run_id: {run_id}")
    print(f"volume prefix: {run_prefix(run_id)}")

    source_digests = upload_source(run_id)
    print(f"uploaded case source: {sorted(source_digests)}")

    # CPU stage: no gpu= argument. A failure here leaves no GPU allocated.
    generation = reference_generator.remote(run_id, source_digests)
    assert_reference_ready(generation, run_id)
    print(
        json.dumps(
            {
                "reference": {
                    "run_id": run_id,
                    "files": generation["files"],
                    "versions": generation["versions"],
                    "generator": generation["generator"],
                }
            },
            indent=2,
            sort_keys=True,
        )
    )

    results: dict[str, object] = {
        "run_id": run_id,
        "volume_prefix": run_prefix(run_id),
        "reference": generation,
        "experiments": {},
    }
    if "correctness" in wanted:
        results["experiments"]["correctness"] = correctness_experiment.remote(run_id)
    if "twin" in wanted:
        results["experiments"]["twin"] = twin_experiment.remote(run_id)
    if "benchmark" in wanted:
        results["experiments"]["benchmark"] = benchmark_experiment.remote(run_id)

    out_dir = results_root / f"experiments-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiments.json"
    write_json(out_path, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nexperiment record: {out_path}")

    failed = [
        name
        for name, record in results["experiments"].items()
        if isinstance(record, dict)
        and (record.get("raised") or record.get("returncode") not in (None, 0))
    ]
    if failed:
        raise SystemExit(
            f"experiment(s) failed: {', '.join(failed)}; see {out_path}"
        )
