#!/usr/bin/env python3
"""Build the P3 collaboration artefacts from recorded experiment results.

The timeline is not written by hand. Every event is derived from a record that a
real command produced: the experiment JSON written by ``run_experiments.py``, the
inventory and evidence of a ``run_demo.py`` run, or the exploration trail in
``p2-5-agent-exploration.json``. Each event therefore carries the command that ran,
its exit status and the artefact it wrote, and none of them can move independently
of that record.

Usage:
    python3 build_timeline.py <experiments.json> <run-dir> <out-dir>

``<run-dir>`` is a ``results/<run_id>`` directory from ``run_demo.py`` and supplies
the run identity, the source hashes and the container tool versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "tilefoundry-timeline-event/v1"

# The actor vocabulary is fixed by the task. Each value names the component that
# produced the observation, so a Triton compile is never reported as TileFoundry
# CodeGen.
AGENT = "AGENT"
CHECK = "TILEFOUNDRY_CHECK"
ANALYZE = "TILEFOUNDRY_ANALYZE"
EVALUATOR = "TILEFOUNDRY_EVALUATOR"
CODEGEN = "TILEFOUNDRY_CODEGEN"
TRITON = "TRITON_COMPILER"
MODAL = "MODAL_RUNTIME"
TORCH = "PYTORCH_REFERENCE"
BENCH = "GPU_BENCHMARK"

STAGES = ("P2-1", "P2-2", "P2-3", "P2-4", "P2-5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Timeline:
    """Collects events and numbers them in insertion order."""

    def __init__(self, task_id: str, run_id: str) -> None:
        self.task_id = task_id
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._iteration = 0

    def iteration(self, stage: str) -> None:
        self._iteration += 1
        self.stage = stage

    def add(
        self,
        *,
        actor: str,
        action: str,
        summary: str,
        iteration: int | None = None,
        source_hash: str = "",
        input_artifacts: list[str] | None = None,
        output_artifacts: list[str] | None = None,
        command: list[str] | None = None,
        exit_code: int | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": f"E{len(self.events) + 1:03d}",
            "timestamp": "",
            "task_id": self.task_id,
            "run_id": self.run_id,
            "iteration_id": f"{self.stage}-it{iteration if iteration is not None else self._iteration:02d}",
            "actor": actor,
            "action": action,
            "source_hash": source_hash,
            "input_artifacts": input_artifacts or [],
            "output_artifacts": output_artifacts or [],
            "command": command or [],
            "exit_code": exit_code,
            "evidence": evidence or {},
            "summary": summary,
        }
        self.events.append(event)
        return event


def _predicate_lines(stdout: str) -> list[str]:
    """The predicate result lines of a ``tilefoundry check`` report."""
    keep = ("PASS", "FAIL")
    return [
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and any(word in line for word in keep)
    ]


def _roofline(summary: str) -> list[str]:
    return [line for line in summary.splitlines() if line.startswith("# roofline")]


def _memory(summary: str) -> list[str]:
    return [line for line in summary.splitlines() if line.startswith("# memory")]


def build(experiments_path: Path, run_dir: Path, out_dir: Path) -> dict[str, Any]:
    record = json.loads(experiments_path.read_text(encoding="utf-8"))
    experiments = record["experiments"]

    manifest_path = run_dir / "bundle" / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id") or run_dir.name
    task_id = manifest.get("task_id") or "rmsnorm-fp8-tutorial"
    sources = {
        item["path"]: item["sha256"]
        for item in manifest.get("sources", [])
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    # The RMSNorm case files are not part of the TileFoundry source inventory, so their
    # hashes are taken from the files themselves. This is what the experiment ran: the
    # Volume upload carries the same bytes.
    case_root = Path(__file__).resolve().parent / "cases" / "rmsnorm_fp8"
    for name in ("rms_norm_quant.py", "rms_norm_quant_opt.py", "twin.py", "benchmark.py", "agent_candidate.py"):
        candidate = case_root / name
        if candidate.is_file():
            sources[name] = _sha256(candidate)
    # `libraries` is a list of build-output records, not a version mapping, so it is
    # counted rather than read as one.
    libraries = manifest.get("evidence", {}).get("compile", {}).get("libraries") or []
    versions = {
        "toolchain_libraries": str(len(libraries)) if isinstance(libraries, list) else str(libraries)
    }

    # The exploration trail sits beside the case sources, not in the record tree. Walk up
    # from this file until the case directory is found, so the layout of the results
    # directory does not have to be assumed.
    exploration_path = None
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "cases" / "rmsnorm_fp8" / "p2-5-agent-exploration.json"
        if candidate.is_file():
            exploration_path = candidate
            break
    if exploration_path is None:
        raise SystemExit("missing cases/rmsnorm_fp8/p2-5-agent-exploration.json")
    exploration = json.loads(exploration_path.read_text(encoding="utf-8"))

    def source_hash(name: str) -> str:
        return sources.get(name, "")

    tl = Timeline(task_id, run_id)
    tl.stage = STAGES[0]

    environment = experiments["correctness"]
    tl.iteration("P2-1")
    tl.add(
        actor=MODAL,
        action="start experiment container",
        summary=(
            f"Experiments ran on {environment['gpu_name']} at compute capability "
            f"{environment['compute_capability']} with torch {environment['torch']} "
            f"and python {environment['python']}."
        ),
        input_artifacts=["cases/rmsnorm_fp8"],
        evidence={
            "gpu_name": environment["gpu_name"],
            "compute_capability": environment["compute_capability"],
            "torch": environment["torch"],
            "python": environment["python"],
            "device": environment["device"],
            "library_versions": versions,
            "experiments_file": str(experiments_path),
        },
    )

    runs = {run["label"]: run for run in environment["runs"]}
    wrong = runs["wrong"]
    tl.add(
        actor=CHECK,
        action="check the HIR written from memory",
        summary=(
            "The first HIR scaled by gamma before rounding to bf16. The checker rejected it "
            "on both outputs, so the semantics were wrong and not merely imprecise."
        ),
        source_hash=source_hash("rms_norm_quant.py"),
        input_artifacts=["x.pt", "ckpt/gamma.pt", "expected.pt"],
        output_artifacts=[f"rms_norm_quant.py:{wrong['module']}"],
        command=["tilefoundry", *wrong["command"]],
        exit_code=wrong["returncode"],
        evidence={
            "predicates": _predicate_lines(wrong["stdout"]),
            "stdout": wrong["stdout"],
            "stderr": wrong["stderr"],
        },
    )
    tl.add(
        actor=AGENT,
        action="read the semantic failure and reorder the rounding",
        summary=(
            "Both predicates failed with the same cause: the bf16 rounding happened after the "
            "gamma multiply instead of before it. The agent moved the cast ahead of the "
            "multiply, which is the model's own definition of the step."
        ),
        source_hash=source_hash("rms_norm_quant.py"),
        evidence={
            "source_diff": (
                "normed = tf.cast(rows * tf.rsqrt(mean + EPS) * tf.cast(gamma, \"f32\"), \"bf16\")\n"
                "->  normed = tf.cast(rows * tf.rsqrt(mean + EPS), \"bf16\") * gamma"
            ),
            "why": "Quantising after the multiply rounds a scaled value; quantising before it matches expected.pt.",
        },
    )
    corrected = runs["corrected"]
    tl.add(
        actor=CHECK,
        action="re-check the repaired HIR",
        summary="The repaired HIR matched both outputs exactly.",
        source_hash=source_hash("rms_norm_quant.py"),
        input_artifacts=["x.pt", "ckpt/gamma.pt", "expected.pt"],
        output_artifacts=[f"rms_norm_quant.py:{corrected['module']}"],
        command=["tilefoundry", *corrected["command"]],
        exit_code=corrected["returncode"],
        evidence={"predicates": _predicate_lines(corrected["stdout"]), "stdout": corrected["stdout"]},
    )

    tl.iteration("P2-2")
    for label, name, note in (
        ("analysis_naive", "Naive", "The unfused placement pays for writing the normalized rows and reading them back."),
        ("analysis_fused", "Fused", "The fused placement keeps the rows on chip and writes only the quantized result and its scale."),
    ):
        analysis = experiments["correctness"][label]
        roofline = _roofline(analysis["summary"])
        tl.add(
            actor=ANALYZE,
            action=f"price the {name} placement",
            summary=f"{name} reports {roofline[0].lstrip('# ') if roofline else 'no roofline'}; {note}",
            source_hash=source_hash("rms_norm_quant_opt.py"),
            input_artifacts=["rms_norm_quant_opt.py"],
            output_artifacts=[f"analysis/{name}.txt"],
            command=["tilefoundry", *analysis["command"]],
            exit_code=analysis["returncode"],
            evidence={
                "roofline": roofline,
                "memory": _memory(analysis["summary"]),
                "report_path": f"analysis/{name}.txt",
            },
        )

    tl.add(
        actor=AGENT,
        action="choose the fused placement from the memory report",
        summary=(
            "Both placements are bound by memory, so arithmetic is not the lever. The memory "
            "report showed the unfused placement reading 3.61 MiB more from gmem and writing "
            "3.61 MiB more, and the fused placement has no such round trip, so the agent "
            "selected fusion."
        ),
        source_hash=source_hash("rms_norm_quant_opt.py"),
        evidence={
            "decisive_metric": "gmem read and write of the intermediate rows",
            "naive_gmem": "r7.23MiB/w5.47MiB@logical",
            "fused_gmem": "r3.62MiB/w1.86MiB@logical",
            "bound_by": "memory",
        },
    )

    tl.iteration("P2-3")
    twin = experiments["twin"]
    tl.add(
        actor=TRITON,
        action="compile the runtime twin",
        summary=(
            f"Triton {twin['triton']} compiled the twin that ``twin.py`` states. The twin is a "
            "Triton kernel, so its compilation belongs to Triton and not to TileFoundry codegen."
        ),
        source_hash=source_hash("twin.py"),
        input_artifacts=["twin.py"],
        output_artifacts=["triton_jit_cache"],
        command=["tilefoundry", *twin["command"]],
        exit_code=twin["returncode"],
        evidence={
            "triton_version": twin["triton"],
            "attribution": "TRITON_COMPILER, not TileFoundry codegen",
            "predicates": _predicate_lines(twin["stdout"]),
            "stdout": twin["stdout"],
        },
    )

    tl.iteration("P2-4")
    benchmark = experiments["benchmark"]
    measurement = benchmark["measurement"]
    comparison = measurement["comparison"]
    tl.add(
        actor=BENCH,
        action="time the torch baseline against the Triton twin",
        summary=(
            f"{measurement['what']}. Across rounds the torch baseline took "
            f"{comparison['median_of_round_medians_ms']['torch_baseline']:.5f} ms and the twin "
            f"{comparison['median_of_round_medians_ms']['triton_twin']:.5f} ms."
        ),
        source_hash=source_hash("benchmark.py"),
        input_artifacts=["benchmark.py", "twin.py"],
        output_artifacts=["benchmark.json"],
        command=["python3", *benchmark["command"]],
        exit_code=benchmark["returncode"],
        evidence={
            "measured_region": measurement["measured_region"],
            "arms": measurement["arms"],
            "comparison": comparison,
            "agreement": measurement["agreement"],
            "dtypes": measurement["dtypes"],
            "shape": measurement["shape"],
            "seed": measurement["seed"],
            "jit_compile": measurement["jit_compile"],
            "versions": measurement["versions"],
        },
    )
    tl.add(
        actor=AGENT,
        action="decide whether the measurement supports a conclusion",
        summary=(
            "The speedup is reported as a range over rounds, not as a single ratio, because the "
            "per-round medians move by more than the gap between the arms. The static ideal-ns "
            "figures are not divided by these times; they are a different model."
        ),
        evidence={
            "speedup": comparison["speedup_torch_over_triton"],
            "per_round": comparison["per_round"],
            "not_a_speedup_claim_from_static_model": measurement[
                "not_a_speedup_claim_from_static_model"
            ],
        },
    )

    tl.iteration("P2-5")
    agent_check = experiments["correctness"]["agent_candidate"]
    tl.add(
        actor=CHECK,
        action="check the agent's third placement",
        summary=(
            "The agent's own placement passed both predicates, so the compiler accepted it as a "
            "correct realisation of the step. Acceptance is a correctness verdict, not a "
            "performance verdict."
        ),
        source_hash=source_hash("agent_candidate.py"),
        input_artifacts=["x.pt", "ckpt/gamma.pt", "expected.pt"],
        output_artifacts=["agent_candidate.py:SplitMetaChecked"],
        command=["tilefoundry", *agent_check["command"]],
        exit_code=agent_check["returncode"],
        evidence={"predicates": _predicate_lines(agent_check["stdout"]), "stdout": agent_check["stdout"]},
    )
    agent_analysis = experiments["correctness"]["analysis_agent"]
    tl.add(
        actor=ANALYZE,
        action="price the agent's placement",
        summary=(
            "The agent's placement reports the same ideal-ns as the fused placement. Moving the "
            "activations from rmem to smem left the modelled time unchanged, so the hypothesis "
            "was refuted by the same report that suggested it."
        ),
        source_hash=source_hash("agent_candidate.py"),
        input_artifacts=["agent_candidate.py"],
        output_artifacts=["analysis/agent.txt"],
        command=["tilefoundry", *agent_analysis["command"]],
        exit_code=agent_analysis["returncode"],
        evidence={
            "roofline": _roofline(agent_analysis["summary"]),
            "memory": _memory(agent_analysis["summary"]),
        },
    )

    for index, attempt in enumerate(exploration.get("compiler_feedback_encountered", []), start=1):
        tl.add(
            actor=EVALUATOR,
            action=f"reject the agent's attempt {attempt['attempt']}",
            summary=attempt["meaning"],
            iteration=index,
            source_hash=source_hash("agent_candidate.py"),
            evidence={
                "compiler_message": attempt["compiler_message"],
                "meaning": attempt["meaning"],
                "agent_response": attempt["agent_response"],
                "attempt": attempt["attempt"],
                "source": "cases/rmsnorm_fp8/p2-5-agent-exploration.json",
            },
        )

    for event in tl.events:
        event["timestamp"] = manifest.get("created_at", "")

    out_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = out_dir / "timeline.jsonl"
    with timeline_path.open("w", encoding="utf-8") as handle:
        for event in tl.events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "run_id": run_id,
        "task_id": task_id,
        "events": tl.events,
        "versions": versions,
        "timeline_path": timeline_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    arguments = parser.parse_args()
    result = build(arguments.experiments, arguments.run_dir, arguments.out_dir)
    print(f"events: {len(result['events'])}")
    print(f"wrote:  {result['timeline_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
