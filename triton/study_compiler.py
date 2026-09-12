"""Run a reproducible Triton GPU compiler experiment on Modal.

The experiment keeps Triton's volatile compiler scratch data on /tmp, captures
stage IR and compiler diagnostics, records structured metadata/timing, and only
copies an immutable tarball to the Modal Volume after compilation has stopped.
"""

import json

from modal_env import app, artifacts, triton_image


@app.function(
    image=triton_image,
    gpu="A10G",
    timeout=1800,
    volumes={"/artifacts": artifacts},
)
def study(test_path: str, detail: str = "stage"):
    import datetime
    import hashlib
    import json
    import os
    import pathlib
    import shutil
    import subprocess
    import tarfile

    if detail not in {"stage", "pass"}:
        raise ValueError("detail must be 'stage' or 'pass'")

    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = pathlib.Path(f"/tmp/triton-study/{run_id}")
    dump_dir = root / "dumps"
    results_dir = root / "results"
    logs_dir = root / "logs"
    for path in (dump_dir, results_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TRITON_ALWAYS_COMPILE": "1",
            "TRITON_KERNEL_DUMP": "1",
            "TRITON_DUMP_DIR": str(dump_dir),
            "TRITON_STUDY_RESULTS_DIR": str(results_dir),
            "MLIR_ENABLE_TIMING": "1",
            "LLVM_ENABLE_TIMING": "1",
        }
    )

    if detail == "pass":
        env["MLIR_ENABLE_DUMP"] = "1"
        env["MLIR_DUMP_PATH"] = str(logs_dir / "mlir-pass-ir.log")
        env["LLVM_IR_ENABLE_DUMP"] = "1"

    command = [
        "python",
        "-m",
        "pytest",
        "-v",
        "-s",
        "-p",
        "study_pytest_plugin",
        test_path,
    ]

    compiler_log = logs_dir / "compiler.log"
    with compiler_log.open("w") as log:
        proc = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    artifact_inventory = []
    for path in sorted(dump_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("tmp."):
            continue
        artifact_inventory.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    environment = {}
    environment_file = results_dir / "environment.json"
    if environment_file.exists():
        environment = json.loads(environment_file.read_text())

    pipeline = []
    pipeline_file = results_dir / "pipeline.json"
    if pipeline_file.exists():
        pipeline = json.loads(pipeline_file.read_text())

    kernel_metrics = []
    metrics_file = results_dir / "kernel_metrics.jsonl"
    if metrics_file.exists():
        kernel_metrics = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "detail": detail,
        "test_path": test_path,
        "pytest_returncode": proc.returncode,
        "status": "passed" if proc.returncode == 0 else "failed",
        "environment": environment,
        "pipeline": pipeline,
        "kernel_metrics": kernel_metrics,
        "artifacts": artifact_inventory,
        "notes": {
            "cold_metric": "Python wall time from first Triton launch through CUDA synchronize; includes JIT, module load, launch, and first execution.",
            "warm_metric": "triton.testing.do_bench after the first launch in the same process; the in-process JIT kernel cache is warm even though TRITON_ALWAYS_COMPILE forces compiler disk-cache misses for new compilations.",
            "pass_timing": "MLIR/LLVM timing output is preserved in logs/compiler.log.",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    archive_name = f"triton-study-{run_id}.tar.gz"
    archive = pathlib.Path("/tmp") / archive_name
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname=f"triton-study-{run_id}")

    remote_dir = pathlib.Path(f"/artifacts/studies/{run_id}")
    remote_dir.mkdir(parents=True, exist_ok=True)
    remote_archive = remote_dir / archive_name
    shutil.copy2(archive, remote_archive)
    artifacts.commit()

    return {
        "run_id": run_id,
        "status": manifest["status"],
        "archive": str(remote_archive.relative_to("/artifacts")),
        "artifact_count": len(artifact_inventory),
        "compiler_log": "logs/compiler.log",
    }


@app.local_entrypoint()
def main(test_path: str = "tests/test_add.py", detail: str = "stage"):
    result = study.remote(test_path, detail)
    print(json.dumps(result, indent=2))
    print("\nDownload:")
    print(
        "modal volume get triton-artifacts "
        f"{result['archive']} ./triton-study-{result['run_id']}.tar.gz"
    )
