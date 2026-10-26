"""One-command TileFoundry H200 demo: local source in, GPU evidence out.

Run it from this directory:

    modal run run_demo.py

On this machine the module freezes the local TileFoundry checkout into a
staging directory, digests it, and builds the image from that exact directory,
so what the container installs cannot drift from what the manifest records. In
the container the same module re-imports with ``modal.is_local()`` false, which
skips the staging work.

The container then installs the snapshot, proves the installed packages are that
snapshot, runs the shipped HIR analysis, and runs the upstream TIR integration
test on one H200. Evidence is archived to a Modal Volume and the printed result
carries the exact ``modal volume get`` command for it.

Cost control lives here: ``--cpu-only`` stops after the no-GPU preflight, so a
broken image is never paid for with a GPU, and the GPU path is a single bounded
invocation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from demo_report import download_command  # noqa: E402
from demo_report import evaluate, resolve_run_id, reserve_directory, validate_run_id, write_json  # noqa: E402
from demo_build_inputs import assert_dockerfile_unchanged, frozen_image_inputs  # noqa: E402
from demo_task import (  # noqa: E402
    TaskValidationError,
    load_task,
    runner_inventory,
    runner_sha,
    task_spec_sha,
    validate_task,
)

DEFAULT_TILEFOUNDRY_REPO = Path(
    os.environ.get("TILEFOUNDRY_REPO", "/home/shijingchang/workspace/TileFoundry")
)
VOLUME_NAME = "tilefoundry-artifacts"
SOURCE_ROOT = "/opt/tilefoundry"
GPU_TIMEOUT_SECONDS = 600
CPU_TIMEOUT_SECONDS = 1800
GPU_SPEC = "H200"


def package_version(snapshot: dict) -> str:
    """Derive a deterministic PEP 440 version for a source tree with no ``.git``.

    ``setuptools-scm`` cannot read a version out of an uploaded snapshot, so the
    demo injects one. It is derived from the snapshot digest, which makes two
    different trees produce two different versions and one tree always the same
    one, and it is recorded in the manifest beside the digest.
    """
    return f"0.0.0+demo.{snapshot['snapshot_digest'][:12]}"


app = modal.App("tilefoundry-h200-demo")
artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

if modal.is_local():
    from demo_source import cleanup, prepare

    STAGING, SNAPSHOT, SNAPSHOT_FILES = prepare(DEFAULT_TILEFOUNDRY_REPO)
    SNAPSHOT["package_version"] = package_version(SNAPSHOT)
    # The P0-2 proof: the Dockerfile and the build context come from the same frozen
    # directory as the Python source, so the image cannot be built from one revision while
    # the manifest names another. `frozen_image_inputs` digests those bytes and refuses a
    # snapshot that is missing a build input.
    SNAPSHOT["build_inputs"] = frozen_image_inputs(STAGING, SNAPSHOT)
    # Re-hash immediately before the path is handed to the builder.
    assert_dockerfile_unchanged(STAGING, SNAPSHOT["build_inputs"])
    write_json(HERE / "results" / ".snapshot" / "source-snapshot.json", SNAPSHOT)
    IMAGE = (
        modal.Image.from_dockerfile(
            str(STAGING / "Dockerfile"),
            context_dir=str(STAGING),
            build_args={"CUTLASS_SHA": SNAPSHOT["build_inputs"]["cutlass_sha"]},
            ignore=[],
        )
        .pip_install(
            "setuptools>=68",
            "setuptools-scm>=8",
            "wheel",
            "ortools>=9.15",
            "transformers>=5.13,<5.15",
        )
        .env(
            {
                "TILEFOUNDRY_CUTLASS_INCLUDE": "/opt/cutlass/include",
                "PYTHONDONTWRITEBYTECODE": "1",
                "SETUPTOOLS_SCM_PRETEND_VERSION": SNAPSHOT["package_version"],
            }
        )
        .add_local_dir(str(STAGING), SOURCE_ROOT, copy=True, ignore=[])
        .run_commands(
            "cd /opt/tilefoundry && python -m pip install --no-deps --no-build-isolation "
            "-e . -e targets/nvidia",
            "cd /opt/tilefoundry && find . -name '*.egg-info' -prune -exec rm -rf {} +",
            "python -c 'import tilefoundry, tilefoundry_nvidia; print(tilefoundry.__file__)'",
        )
        # Every module run_demo imports must be present, because the container re-imports
        # this file. A missing one fails at container import, before any stage runs.
        .add_local_python_source(
            "demo_build_inputs",
            "demo_container",
            "demo_provenance",
            "demo_pytest_plugin",
            "demo_report",
            "demo_source",
            "demo_task",
        )
    )
else:
    STAGING = None
    SNAPSHOT = {}
    SNAPSHOT_FILES = []
    IMAGE = modal.Image.debian_slim()


@app.function(image=IMAGE, volumes={"/artifacts": artifacts}, timeout=CPU_TIMEOUT_SECONDS)
def preflight(payload: dict) -> dict:
    """Prove the uploaded snapshot installs, imports, and registers the H200 target.

    No GPU is reserved for this function, so a broken dependency or a wrong
    install origin costs nothing beyond container time.
    """
    if modal.is_local():
        raise RuntimeError("preflight runs in the container")
    import sys

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import demo_container

    result = demo_container.preflight(Path(SOURCE_ROOT), payload)
    artifacts.commit()
    return _plain(result)


@app.function(image=IMAGE, gpu=GPU_SPEC, volumes={"/artifacts": artifacts}, timeout=GPU_TIMEOUT_SECONDS)
def run_on_h200(payload: dict) -> dict:
    """Run the HIR analysis and the upstream TIR test on one H200, then archive evidence."""
    if modal.is_local():
        raise RuntimeError("run_on_h200 runs in the container")
    import sys

    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import demo_container

    result = demo_container.run(Path(SOURCE_ROOT), payload)
    artifacts.commit()
    return _plain(result)


def _plain(value):
    """Return a JSON-only copy of *value*.

    Modal pickles the return value with the local interpreter, and objects such
    as ``torch.__version__`` are torch-defined ``str`` subclasses whose unpickling
    needs torch. A JSON round trip makes the round trip independent of that.
    """
    return json.loads(json.dumps(value, default=str))


TASKS_DIR = HERE / "tasks"
DEFAULT_TASK = "tasks/square.json"


def load_and_validate_task(task: str) -> dict:
    """Resolve and validate one task spec, before any remote call.

    Every structural, path, command and resource check happens here. A spec that fails is a
    local error, so a rejected task costs no image build and no GPU.

    Args:
        task: The requested path, relative to the demo root.

    Returns:
        The effective spec.

    Raises:
        SystemExit: If the path leaves ``tasks/`` or the spec does not validate.
    """
    requested = Path(task)
    if requested.is_absolute():
        resolved = requested.resolve()
    else:
        resolved = (HERE / requested).resolve()
    tasks_root = TASKS_DIR.resolve()
    if not resolved.is_relative_to(tasks_root):
        raise SystemExit(
            f"task {task!r} resolves outside {tasks_root}; task specs live in tasks/"
        )
    try:
        return validate_task(load_task(resolved))
    except TaskValidationError as error:
        raise SystemExit(f"task {task} is not acceptable: {error}") from None


@app.local_entrypoint()
def main(
    cpu_only: bool = False,
    run_id: str = "",
    tilefoundry_repo: str = str(DEFAULT_TILEFOUNDRY_REPO),
    task: str = DEFAULT_TASK,
) -> None:
    """Freeze the local source, run the container stages, and report their evidence."""
    global STAGING, SNAPSHOT, SNAPSHOT_FILES
    # Validate the task first: a rejected spec must cost nothing.
    effective_task = load_and_validate_task(task)
    needs_gpu = any(
        stage["type"] in {"gpu_test", "gpu_benchmark"} for stage in effective_task["stages"]
    )
    repo = Path(tilefoundry_repo)
    if not (repo / "pyproject.toml").is_file():
        raise SystemExit(f"not a TileFoundry checkout: {repo}")
    if Path(tilefoundry_repo) != DEFAULT_TILEFOUNDRY_REPO:
        raise SystemExit(
            "set TILEFOUNDRY_REPO in the environment to point the demo at another checkout"
        )

    # A caller-supplied id is taken as given and must be safe and free; a generated id is
    # re-suffixed until it is free. Either way the namespace is claimed here, before
    # `preflight.remote` and long before `run_on_h200.remote`, so a rejection costs no GPU.
    results_root = HERE / "results"
    generated = not run_id
    if generated:
        run_id = resolve_run_id(results_root, _timestamp())
    else:
        validate_run_id(run_id)
    # Claim the namespace now, before any remote work, so a second process cannot take it.
    # `mkdir(exist_ok=False)` is the authority: it also refuses a planted symlink.
    try:
        reserve_directory(results_root, run_id)
    except FileExistsError:
        hint = "drop --run-id to let one be generated" if not generated else "wait and retry"
        raise SystemExit(
            f"run id {run_id!r} already has evidence under {results_root}; {hint}"
        ) from None
    payload = {
        "run_id": run_id,
        "snapshot": SNAPSHOT,
        "inventory": SNAPSHOT_FILES,
        "build_inputs": SNAPSHOT.get("build_inputs"),
        "cpu_only": cpu_only,
        # The effective spec, its digest, and the runner digest travel with the run so the
        # evidence names what was requested and which whitelist ran it.
        "task": effective_task,
        "task_spec_sha": task_spec_sha(effective_task),
        "runner_sha": runner_sha(),
        "runner_inventory": runner_inventory(),
        "completion_scope": "cpu_only" if cpu_only else "full",
    }
    print(
        f"source: {SNAPSHOT['commit'][:12]}"
        f"{' (dirty)' if SNAPSHOT['dirty'] else ''}"
        f" files={SNAPSHOT['snapshot_file_count']}"
        f" digest={SNAPSHOT['snapshot_digest'][:16]}"
        f" cutlass={(SNAPSHOT['cutlass_sha'] or '')[:12]}"
    )
    print(f"run_id: {run_id}")

    try:
        environment = preflight.remote(payload)
    except BaseException:
        cleanup(SNAPSHOT)
        raise
    print(json.dumps(environment, indent=2, sort_keys=True))
    verdict = evaluate(
        {"environment": environment, "preflight_returncode": environment.get("preflight_returncode")}
    )
    write_json(HERE / "results" / run_id / "preflight.json", environment)
    if verdict["environment_status"] != "PASS":
        print(json.dumps(verdict["stages"]["environment"], indent=2, sort_keys=True))
        cleanup(SNAPSHOT)
        raise SystemExit("no-GPU preflight failed; no GPU was allocated")

    if cpu_only:
        # A CPU-only run proves the CPU stages and the archive. It cannot claim the task
        # completed when the task needs a device.
        scope = "NOT_RUN" if needs_gpu else "COMPLETE"
        print(
            json.dumps(
                {
                    "completion_scope": "cpu_only",
                    "task_status": scope,
                    "task_id": effective_task["task_id"],
                    "task_spec_sha": payload["task_spec_sha"],
                    "note": (
                        "the requested task has GPU stages; no GPU was allocated and no "
                        "GPU verdict is claimed"
                        if needs_gpu
                        else "the requested task is CPU-only and completed"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        cleanup(SNAPSHOT)
        return

    try:
        result = run_on_h200.remote(payload)
    finally:
        cleanup(SNAPSHOT)
    write_json(HERE / "results" / run_id / "remote-result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nDownload the full evidence bundle:")
    print(
        f"  {download_command(VOLUME_NAME, run_id, f'./results/{run_id}/tilefoundry-demo-{run_id}.tar.gz')}"
    )
    print(f"local result: {HERE / 'results' / run_id / 'remote-result.json'}")
    if result.get("overall_status") != "PASS":
        raise SystemExit(f"overall_status={result.get('overall_status')}: {result.get('overall_reason')}")


def _timestamp() -> str:
    """Return a sortable UTC run identifier with microsecond resolution.

    Microseconds keep two runs launched within the same second from colliding by default;
    ``resolve_run_id`` still re-suffixes if they somehow do.
    """
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

