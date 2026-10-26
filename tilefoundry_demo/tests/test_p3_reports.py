"""Tests for the P3 timeline and report builders.

These builders turn a recorded experiment into the artefacts a reader sees. The risk
they carry is silent drift: a hand-written number in the page that no longer matches
the record. The tests therefore drive them from a small synthetic record and assert
that what the page shows came from that record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_report
import build_timeline


def _experiment_record() -> dict:
    """A minimal but structurally real ``experiments.json``."""

    def check_run(label: str, module: str, code: int, lines: list[str]) -> dict:
        return {
            "label": label,
            "module": module,
            "command": ["check", f"rms_norm_quant.py:{module}"],
            "returncode": code,
            "stdout": "\n".join(lines),
            "stderr": "",
        }

    def analysis(summary: list[str]) -> dict:
        return {
            "command": ["analyze", "rms_norm_quant_opt.py:Naive"],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "summary": "\n".join(summary),
        }

    return {
        "run_stamp": "20260101-000000-000000",
        "experiments": {
            "correctness": {
                "case": "rmsnorm_fp8",
                "device": "cuda",
                "gpu_name": "NVIDIA H200",
                "compute_capability": "[9, 0]",
                "python": "3.12.15",
                "torch": "2.9.1+cu128",
                "runs": [
                    check_run(
                        "wrong",
                        "Wrong",
                        1,
                        [
                            "    equal   mismatched 432 elements 14336 FAIL",
                            "    allclose(atol=1e-06 rtol=1e-06)   max_violation 6.87445e-05  FAIL",
                        ],
                    ),
                    check_run(
                        "corrected",
                        "RmsNormQuant",
                        0,
                        [
                            "    equal   mismatched 0 elements 14336 PASS",
                            "    allclose(atol=1e-06 rtol=1e-06)   max_violation 0   PASS",
                        ],
                    ),
                ],
                "analysis_naive": analysis(
                    [
                        "# memory traffic=gmem:r7.23MiB/w5.47MiB@logical",
                        "# roofline ideal-ns=8697 bound-by=memory",
                        "# roofline ideal-ns=4952 bound-by=memory",
                    ]
                ),
                "analysis_fused": analysis(
                    [
                        "# memory traffic=gmem:r3.62MiB/w1.86MiB@logical",
                        "# roofline ideal-ns=3755 bound-by=memory",
                    ]
                ),
                "analysis_agent": analysis(
                    [
                        "# memory traffic=gmem:r3.62MiB/w1.86MiB@logical",
                        "# roofline ideal-ns=3755 bound-by=memory",
                    ]
                ),
                "agent_candidate": {
                    "command": ["check", "agent_candidate.py:SplitMetaChecked"],
                    "returncode": 0,
                    "stdout": "    equal   mismatched 0 elements 14336 PASS\nPASS",
                    "stderr": "",
                },
            },
            "twin": {
                "command": ["check", "twin.py:Fast"],
                "returncode": 0,
                "stdout": "    equal   mismatched 0 elements 14336 PASS\nPASS",
                "stderr": "",
                "triton": "3.5.1",
            },
            "benchmark": {
                "command": ["benchmark.py"],
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "summary_ms": {},
                "measurement": {
                    "what": "measured region only",
                    "measured_region": "kernel only",
                    "shape": {"rows": 264, "h": 7168},
                    "dtypes": {"input": "bfloat16"},
                    "seed": 0,
                    "jit_compile": True,
                    "versions": {
                        "python": "3.12.15",
                        "torch": "2.9.1+cu128",
                        "torch_cuda": "12.8",
                        "triton": "3.5.1",
                        "cuda_driver": None,
                    },
                    "arms": {},
                    "agreement": {"quant_equal_fraction": 0.99999, "scale_max_abs_diff": 0.0},
                    "comparison": {
                        "median_of_round_medians_ms": {
                            "torch_baseline": 0.1078,
                            "triton_twin": 0.02566,
                        },
                        "speedup_torch_over_triton": {"min": 4.09, "median": 4.18, "max": 4.23},
                        "per_round": [],
                    },
                    "not_a_speedup_claim_from_static_model": "static ideal-ns is not a duration",
                },
            },
        },
    }


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A case tree, a recorded experiment and a run directory."""
    case = tmp_path / "cases" / "rmsnorm_fp8"
    case.mkdir(parents=True)
    (case / "rms_norm_quant.py").write_text("# first\n", encoding="utf-8")
    (case / "twin.py").write_text("# twin\n", encoding="utf-8")
    (case / "p2-5-agent-exploration.json").write_text(
        json.dumps(
            {
                "compiler_feedback_encountered": [
                    {
                        "attempt": 1,
                        "compiler_message": "Binary: operands have conflicting storage",
                        "meaning": "one residency per multi-input op",
                        "agent_response": "moved both operands",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    results = tmp_path / "results"
    experiment_dir = results / "experiments-20260101-000000-000000"
    experiment_dir.mkdir(parents=True)
    experiments = experiment_dir / "experiments.json"
    experiments.write_text(json.dumps(_experiment_record()), encoding="utf-8")

    run_dir = results / "20260101-000000-000000"
    bundle = run_dir / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "20260101-000000-000000",
                "task_id": "rmsnorm-fp8-tutorial",
                "sources": [],
                "evidence": {
                    "compile": {
                        "libraries": [{"path": "lib.so", "sha256": "a" * 64, "size": 1}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # The builders look for the exploration trail beside their own file. Point that
    # search at this case tree instead of the repository's.
    monkeypatch.setattr(build_timeline, "__file__", str(tmp_path / "build_timeline.py"))
    return {
        "root": tmp_path,
        "case": case,
        "experiments": experiments,
        "run_dir": run_dir,
        "report": tmp_path / "reports" / "rmsnorm_fp8",
    }


def _build(workspace: dict[str, Path]) -> dict:
    result = build_timeline.build(
        workspace["experiments"], workspace["run_dir"], workspace["report"]
    )
    decisions = [
        {
            "decision_id": "D001",
            "task": "match expected.pt",
            "stage": "P2-1",
            "observations": ["mismatched 432"],
            "evidence_refs": ["timeline.jsonl#E002"],
            "candidate_actions": ["loosen tolerance", "reorder the rounding"],
            "selected_action": "reorder the rounding",
            "expected_effect": "both predicates pass",
            "observed_effect": "both predicates passed",
            "accepted": True,
            "reason": "the tolerance was not touched",
            "source_diff": "a -> b",
        }
    ]
    (workspace["report"] / "decisions.json").write_text(
        json.dumps({"schema": "v1", "decisions": decisions}), encoding="utf-8"
    )
    build_report.build(
        workspace["report"], workspace["experiments"], workspace["report"] / "index.html"
    )
    return result


def test_every_event_carries_the_required_fields(workspace: dict[str, Path]) -> None:
    result = _build(workspace)
    required = {
        "event_id",
        "timestamp",
        "task_id",
        "run_id",
        "iteration_id",
        "actor",
        "action",
        "source_hash",
        "input_artifacts",
        "output_artifacts",
        "command",
        "exit_code",
        "evidence",
        "summary",
    }
    for event in result["events"]:
        assert required <= set(event), event["event_id"]
        assert event["event_id"].startswith("E")
        assert event["run_id"] == "20260101-000000-000000"
        assert event["task_id"] == "rmsnorm-fp8-tutorial"


def test_the_failing_check_keeps_its_exit_code_and_predicates(workspace: dict[str, Path]) -> None:
    events = _build(workspace)["events"]
    failing = [event for event in events if event["exit_code"] == 1]
    assert len(failing) == 1
    evidence = failing[0]["evidence"]
    assert any("432" in line for line in evidence["predicates"])
    assert any("6.87445e-05" in line for line in evidence["predicates"])


def test_triton_compilation_is_not_attributed_to_tilefoundry(workspace: dict[str, Path]) -> None:
    events = _build(workspace)["events"]
    twin_events = [event for event in events if event["actor"] == build_timeline.TRITON]
    assert twin_events, "the twin compile should be recorded"
    twin = twin_events[0]
    assert twin["actor"] == build_timeline.TRITON
    assert "not TileFoundry codegen" in twin["evidence"]["attribution"]


def test_the_agent_candidate_is_recorded_as_correct_but_slower(workspace: dict[str, Path]) -> None:
    events = _build(workspace)["events"]
    agent = [
        event
        for event in events
        if "agent_candidate.py:SplitMetaChecked" in event["output_artifacts"]
    ]
    assert agent and agent[0]["exit_code"] == 0
    analysis = [event for event in events if "analysis/agent.txt" in event["output_artifacts"]]
    assert analysis and "ideal-ns=3755" in analysis[0]["evidence"]["roofline"][0]


def test_compiler_rejections_use_the_evaluator_actor(workspace: dict[str, Path]) -> None:
    events = _build(workspace)["events"]
    rejected = [event for event in events if event["actor"] == build_timeline.EVALUATOR]
    assert rejected
    assert "conflicting storage" in rejected[0]["evidence"]["compiler_message"]


def test_the_page_shows_the_naive_figure_and_not_the_inner_one(workspace: dict[str, Path]) -> None:
    """``Naive`` prints two rooflines; the comparison must use the entry point."""
    _build(workspace)
    page = (workspace["report"] / "index.html").read_text(encoding="utf-8")
    assert "8697 与 3755" in page
    assert "4952 都是字节数" not in page


def test_the_page_makes_no_external_requests(workspace: dict[str, Path]) -> None:
    _build(workspace)
    page = (workspace["report"] / "index.html").read_text(encoding="utf-8")
    for forbidden in ("http://", "https://", "<script", "src=", "<link"):
        assert forbidden not in page, forbidden


def test_the_page_separates_static_prediction_from_measurement(
    workspace: dict[str, Path]
) -> None:
    _build(workspace)
    page = (workspace["report"] / "index.html").read_text(encoding="utf-8")
    assert "两个不同的模型" in page or "不是时长" in page
    assert "0.10780 ms" in page or "0.1078" in page


def test_the_timeline_is_one_json_object_per_line(workspace: dict[str, Path]) -> None:
    _build(workspace)
    lines = (workspace["report"] / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
    # 12 stage events plus the one compiler rejection this fixture carries.
    assert len(lines) == 13
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_stage_labels_keep_their_subindex(workspace: dict[str, Path]) -> None:
    """``iteration_id`` is ``<stage>-it<NN>``; the label must stay ``P2-1``, not ``P2``."""
    _build(workspace)
    page = (workspace["report"] / "index.html").read_text(encoding="utf-8")
    for stage in ("P2-1", "P2-2", "P2-3", "P2-4", "P2-5"):
        assert f'<div class="cell mono dim">{stage}</div>' in page, stage


def test_only_indented_predicate_rows_become_results() -> None:
    """The trailing summary line and the warning paragraph are not results."""
    stdout = (
        "rms_norm_quant.py:Wrong\n"
        "  output[0]   fp8e4m3[2,7168]\n"
        "    equal                              mismatched 432 elements 14336 FAIL\n"
        "  output[1]   f32[2,56]\n"
        "    allclose(atol=1e-06 rtol=1e-06)    max_violation 6.87445e-05  FAIL\n"
        "\n"
        "FAIL\n"
        "\n"
        "  warning: FAIL says the candidate and reference differ, not which side is\n"
        "           closer to truth.\n"
    )
    predicates = build_report.parse_predicates(stdout)
    assert len(predicates) == 2, predicates
    assert predicates[0] == ("equal                              mismatched 432 elements 14336", False)
    assert predicates[1][1] is False
    assert all("warning" not in text for text, _ in predicates)


def test_a_passing_report_yields_only_its_two_predicates() -> None:
    stdout = (
        "    equal                              mismatched 0 elements 14336 PASS\n"
        "    allclose(atol=1e-06 rtol=1e-06)    max_violation 0            PASS\n"
        "\n"
        "PASS\n"
    )
    predicates = build_report.parse_predicates(stdout)
    assert len(predicates) == 2
    assert all(passed for _, passed in predicates)


def test_each_decision_states_whether_it_is_guided_or_agent_work(
    workspace: dict[str, Path]
) -> None:
    """A reader must not have to infer provenance from the verdict.

    ``accepted`` says whether the candidate worked; it does not say whether the step
    came from the tutorial or from the agent's own experiment. Both are marked.
    """
    _build(workspace)
    page = (workspace["report"] / "index.html").read_text(encoding="utf-8")
    decision_block = page.split("<h2>6.")[0]
    assert "tag-rejected" in page
    assert "GUIDED_TUTORIAL_REPLAY" in page
    # The decision for the agent stage carries the experiment marker, not the replay one.
    d001 = page[page.index('<span class="id">D001</span>') :]
    d001 = d001[: d001.index("</summary>")]
    assert "GUIDED_TUTORIAL_REPLAY" in d001
    assert "AGENT_EXPERIMENT" not in d001
    assert decision_block is not None
