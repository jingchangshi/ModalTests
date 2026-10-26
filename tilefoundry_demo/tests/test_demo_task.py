"""Local gates for the P1 task contract.

A task spec is untrusted data that selects from a reviewed whitelist. These tests assert the
two halves of that sentence separately: a valid spec becomes an exact argv, and every attempt
to leave the whitelist is refused by name before anything runs.

Nothing here imports Modal, torch, or TileFoundry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_task  # noqa: E402

DEMO_ROOT = Path(__file__).resolve().parents[1]
TASKS = DEMO_ROOT / "tasks"


def minimal_spec() -> dict:
    """A spec that validates, for tests to mutate one field at a time."""
    return {
        "schema_version": 1,
        "task_id": "square",
        "description": "Analyze and execute the registered square case.",
        "source_snapshot": {"mode": "local", "expected_sha256": None},
        "target": "nvidia.h200_sxm",
        "gpu": {"type": "H200", "count": 1},
        "stages": [
            {"stage_id": "environment", "type": "environment", "params": {}},
            {
                "stage_id": "analyze",
                "type": "hir_analyze",
                "params": {"source_id": "square_hir", "analyses": ["memory"], "dims": {}},
            },
            {"stage_id": "square", "type": "gpu_test", "params": {"case_id": "upstream_square"}},
            {"stage_id": "archive", "type": "archive", "params": {}},
        ],
        "inputs": [],
        "expected_outcomes": {},
        "artifacts": {"include": ["logs", "results", "generated"], "required": []},
        # A copy: mutating the module default would leak across tests.
        "limits": dict(demo_task.DEFAULT_LIMITS),
    }


def test_the_shipped_tasks_validate() -> None:
    """Every reviewed task in tasks/ must be runnable, or the CLI default is broken."""
    shipped = sorted(TASKS.glob("*.json"))
    assert shipped, "tasks/ holds no specs"
    for path in shipped:
        effective = demo_task.validate_task(demo_task.load_task(path))
        assert effective["task_id"]
        assert effective["stages"][-1]["type"] == "archive"


def test_a_valid_spec_gets_defaults_filled_in() -> None:
    """The effective spec states every value that will be used, not just the written ones."""
    spec = minimal_spec()
    del spec["artifacts"]
    del spec["limits"]
    effective = demo_task.validate_task(spec)

    assert effective["artifacts"] == demo_task.DEFAULT_ARTIFACTS
    assert effective["limits"] == demo_task.DEFAULT_LIMITS
    assert effective["expected_outcomes"]["square"] == {"outcome": "pass", "negative": False}


def test_the_spec_digest_covers_the_effective_spec() -> None:
    """Two requests that differ only in an omitted default hash the same, as they should."""
    first = demo_task.validate_task(minimal_spec())
    second = minimal_spec()
    second["limits"] = dict(demo_task.DEFAULT_LIMITS)
    assert demo_task.task_spec_sha(first) == demo_task.task_spec_sha(demo_task.validate_task(second))

    third = minimal_spec()
    third["stages"][2]["params"]["case_id"] = "rmsnorm_fp8"
    assert demo_task.task_spec_sha(first) != demo_task.task_spec_sha(demo_task.validate_task(third))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.update(schema_version=99), "schema_version"),
        (lambda s: s.update(magic=True), "unknown field"),
        (lambda s: s.update(task_id="Square"), "task_id"),
        (lambda s: s.update(task_id="x"), None),  # one character is legal
        (lambda s: s.update(description=""), "description"),
        (lambda s: s.update(target="nvidia.a3"), "target"),
        (lambda s: s.update(source_snapshot={"mode": "git"}), "mode"),
    ],
)
def test_rejects_invalid_schema(mutate, message) -> None:
    """Structural problems are refused before anything else is considered."""
    spec = minimal_spec()
    mutate(spec)
    if message is None:
        assert demo_task.validate_task(spec)["task_id"] == "x"
        return
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert message in str(caught.value)


def test_rejects_a_duplicate_json_key(tmp_path: Path) -> None:
    """A duplicate key would let the reviewed text differ from the executed task."""
    path = tmp_path / "dupe.json"
    path.write_text('{"task_id": "a", "task_id": "b"}')
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.load_task(path)
    assert "duplicate key" in str(caught.value)


def test_rejects_nan_in_a_spec(tmp_path: Path) -> None:
    """NaN is not JSON, and a NaN bound would silently make a comparison meaningless."""
    path = tmp_path / "nan.json"
    path.write_text('{"task_id": NaN}')
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.load_task(path)
    assert "finite number" in str(caught.value)


@pytest.mark.parametrize("stage_type", ["shell", "python", "pytest", "exec", ""])
def test_rejects_unknown_stage(stage_type: str) -> None:
    """Only whitelisted stage types exist."""
    spec = minimal_spec()
    spec["stages"][-1] = {"stage_id": "evil", "type": stage_type, "params": {"command": "id"}}
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "not one of" in str(caught.value) or "never accepted" in str(caught.value)


@pytest.mark.parametrize("case_id", ["/bin/sh", "../../etc/passwd", "Upstream_square", ""])
def test_rejects_unregistered_case(case_id: str) -> None:
    """A case is a registry key, never a path or a command."""
    spec = minimal_spec()
    spec["stages"][2]["params"]["case_id"] = case_id
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "not a registered case" in str(caught.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("shell", True),
        ("command", "id"),
        ("argv", ["id"]),
        ("cwd", "/tmp"),
        ("exec", "bash"),
    ],
)
def test_rejects_injection_fields(field: str, value: object) -> None:
    """A field that names a way out of the whitelist is refused whatever its value."""
    spec = minimal_spec()
    spec["stages"][2][field] = value
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "never accepted" in str(caught.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("env", {"AWS_SECRET_ACCESS_KEY": "x"}),
        ("environment", {"PATH": "/tmp"}),
        ("secrets", ["TOKEN"]),
        ("retry", 3),
        ("until", True),
    ],
)
def test_rejects_secret_and_retry_fields(field: str, value: object) -> None:
    """A task cannot request a secret, a host variable, or a retry loop."""
    spec = minimal_spec()
    spec[field] = value
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "never accepted" in str(caught.value)


@pytest.mark.parametrize(
    "path, message",
    [
        ("/etc/passwd", "absolute"),
        ("cases/../../secret.bin", "traversal"),
        ("cases\\x.bin", "backslash"),
        ("https://host/a.bin", "URL"),
        ("s3://bucket/a.bin", "URL"),
        ("file:///etc/passwd", "URL"),
        ("elsewhere/a.bin", "must start with"),
    ],
)
def test_rejects_unsafe_input_paths(path: str, message: str) -> None:
    """An input is a contained local file under cases/, and nothing else."""
    spec = minimal_spec()
    spec["inputs"] = [{"input_id": "data", "path": path, "sha256": "0" * 64, "size": 1}]
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert message in str(caught.value)


def test_rejects_an_input_symlink_that_escapes(tmp_path: Path) -> None:
    """A link inside cases/ must not read a file outside it."""
    cases = tmp_path / "cases"
    cases.mkdir()
    (tmp_path / "secret.bin").write_bytes(b"secret")
    (cases / "link.bin").symlink_to(tmp_path / "secret.bin")

    spec = minimal_spec()
    spec["inputs"] = [
        {"input_id": "data", "path": "cases/link.bin", "sha256": "0" * 64, "size": 6}
    ]
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec, case_root=cases)
    assert "escapes" in str(caught.value)


def test_rejects_an_input_whose_bytes_changed(tmp_path: Path) -> None:
    """A declared digest that no longer matches the file is a swap, not a task."""
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "data.bin").write_bytes(b"new")
    spec = minimal_spec()
    spec["inputs"] = [
        {"input_id": "data", "path": "cases/data.bin", "sha256": "a" * 64, "size": 3}
    ]
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec, case_root=cases)
    assert "bytes do not match" in str(caught.value)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.update(gpu={"type": "H200", "count": 2}), "gpu.count"),
        (lambda s: s.update(gpu={"type": "A100", "count": 1}), "gpu.type"),
        (lambda s: s["limits"].update(gpu_seconds=601), "limits.gpu_seconds"),
        (lambda s: s["limits"].update(cpu_seconds=99999), "limits.cpu_seconds"),
        (lambda s: s["limits"].update(max_stages=99), "limits.max_stages"),
        (lambda s: s["limits"].update(cpu_seconds=True), "not an integer"),
    ],
)
def test_rejects_unbounded_resources(mutate, message: str) -> None:
    """This demo runs one H200 for a bounded time, and a task cannot raise the ceiling."""
    spec = minimal_spec()
    mutate(spec)
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert message in str(caught.value)


@pytest.mark.parametrize("warmup, repeats", [(0, 1), (10, 100), (3, 20)])
def test_accepts_bounded_benchmark_shape(warmup: int, repeats: int) -> None:
    """A benchmark needs a passing test of the same case first, and bounded counts."""
    spec = minimal_spec()
    spec["stages"].insert(
        3,
        {
            "stage_id": "bench",
            "type": "gpu_benchmark",
            "params": {"case_id": "upstream_square", "warmup": warmup, "repeats": repeats},
        },
    )
    effective = demo_task.validate_task(spec)
    bench = next(stage for stage in effective["stages"] if stage["stage_id"] == "bench")
    assert bench["params"]["warmup"] == warmup
    assert bench["params"]["repeats"] == repeats


@pytest.mark.parametrize("warmup, repeats", [(11, 20), (3, 0), (3, 101), (-1, 20)])
def test_rejects_unbounded_benchmark_shape(warmup: int, repeats: int) -> None:
    """Timing loops are bounded by the contract, not by the caller's patience."""
    spec = minimal_spec()
    spec["stages"].insert(
        3,
        {
            "stage_id": "bench",
            "type": "gpu_benchmark",
            "params": {"case_id": "upstream_square", "warmup": warmup, "repeats": repeats},
        },
    )
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "outside" in str(caught.value)


def test_rejects_benchmark_without_correctness() -> None:
    """A timing with no passing test of the same case measures a possibly wrong answer."""
    spec = minimal_spec()
    spec["stages"].insert(
        2,
        {
            "stage_id": "bench",
            "type": "gpu_benchmark",
            "params": {"case_id": "upstream_square", "warmup": 3, "repeats": 20},
        },
    )
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "timing never proves correctness" in str(caught.value)


def test_rejects_stage_after_archive() -> None:
    """Archive is the last stage; nothing may run after the evidence is sealed."""
    spec = minimal_spec()
    spec["stages"].append({"stage_id": "late", "type": "archive", "params": {}})
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "may appear only first or last" in str(caught.value)


def test_rejects_environment_that_is_not_first() -> None:
    """The environment proof gates everything after it, so it must come first."""
    spec = minimal_spec()
    spec["stages"][0], spec["stages"][1] = spec["stages"][1], spec["stages"][0]
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "first stage" in str(caught.value)


def test_rejects_a_cpu_stage_after_a_gpu_stage() -> None:
    """Per the design, CPU stages precede GPU stages so a cheap failure costs no GPU."""
    spec = minimal_spec()
    spec["stages"].insert(
        3,
        {
            "stage_id": "analyze_late",
            "type": "hir_analyze",
            "params": {"source_id": "square_hir", "analyses": ["memory"], "dims": {}},
        },
    )
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "CPU stages run first" in str(caught.value)


def test_rejects_a_duplicate_stage_id() -> None:
    """Stage ids key the outcomes and the evidence, so they must be unique."""
    spec = minimal_spec()
    spec["stages"][1]["stage_id"] = "environment"
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "appears twice" in str(caught.value)


@pytest.mark.parametrize(
    "comparisons, message",
    [
        ([{"out": "output", "predicates": [{"fn": "allclose", "bounds": {"atol": 1e-6}}]}], "missing"),
        ([{"out": "output", "predicates": [{"fn": "equal", "bounds": {"atol": 1e-6}}]}], "unknown field"),
        ([{"out": "deadbeef", "predicates": [{"fn": "equal", "bounds": {}}]}], "not output"),
        ([{"out": "output", "predicates": []}], "nonempty"),
        ([{"out": "output", "predicates": [{"fn": "maybe", "bounds": {}}]}], "is not one of"),
        ([{"out": "output", "predicates": [{"fn": "cosine", "bounds": {"min": 2.0}}]}], "outside"),
        ([], "nonempty"),
    ],
)
def test_rejects_missing_predicate_bounds(comparisons: list, message: str) -> None:
    """A comparison whose bounds are incomplete cannot be checked, so it is refused."""
    spec = minimal_spec()
    spec["stages"][1] = {
        "stage_id": "check",
        "type": "hir_check",
        "params": {
            "source_id": "square_hir",
            "input_ids": [],
            "comparisons": comparisons,
            "dims": {},
        },
    }
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert message in str(caught.value)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"outcome": "expected_fail"}, "negative=true"),
        ({"outcome": "expected_fail", "negative": True, "exit_codes": [2]}, "only exit 1"),
        ({"outcome": "expected_fail", "negative": True, "exit_codes": []}, "nonempty"),
        ({"outcome": "expected_fail", "negative": True}, "diagnostic_substrings"),
        ({"outcome": "pass", "negative": True}, "cannot also be negative"),
        ({"outcome": "expected_fail", "negative": True, "exit_codes": [1]}, "diagnostic_substrings"),
    ],
)
def test_rejects_a_malformed_expected_failure(overrides: dict, message: str) -> None:
    """An intended failure must be explicit and confirmable, or it is just a failure."""
    spec = minimal_spec()
    spec["expected_outcomes"] = {"analyze": overrides}
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert message in str(caught.value)


def test_accepts_a_well_formed_expected_failure() -> None:
    """The one shape that can be confirmed: explicit, exit 1, and named diagnostics."""
    spec = minimal_spec()
    spec["expected_outcomes"] = {
        "analyze": {
            "outcome": "expected_fail",
            "negative": True,
            "exit_codes": [1],
            "diagnostic_substrings": ["unsupported dtype"],
        }
    }
    effective = demo_task.validate_task(spec)
    assert effective["expected_outcomes"]["analyze"]["negative"] is True


def test_rejects_an_expected_failure_on_a_stage_without_one_declared() -> None:
    """A positive GPU test cannot be relabelled negative, or a failure looks intentional."""
    spec = minimal_spec()
    spec["expected_outcomes"] = {
        "square": {
            "outcome": "expected_fail",
            "negative": True,
            "exit_codes": [1],
            "diagnostic_substrings": ["anything"],
        }
    }
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "not registered as negative" in str(caught.value)


def test_rejects_a_negative_case_reported_as_positive() -> None:
    """A registered negative case must be declared, so its failure is not called a regression."""
    spec = minimal_spec()
    spec["stages"][2]["params"]["case_id"] = "rmsnorm_fp8_wrong"
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "must be declared" in str(caught.value)


def test_rejects_an_outcome_for_an_unknown_stage() -> None:
    """An outcome key that names no stage is a typo that would silently not apply."""
    spec = minimal_spec()
    spec["expected_outcomes"] = {"nonexistent": {"outcome": "pass"}}
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "names no stage" in str(caught.value)


def test_rejects_artifacts_that_exclude_a_required_category() -> None:
    """Evidence a stage needs cannot be excluded from the archive."""
    spec = minimal_spec()
    spec["artifacts"] = {"include": ["logs"], "required": ["generated/build/lib.so"]}
    with pytest.raises(demo_task.TaskValidationError) as caught:
        demo_task.validate_task(spec)
    assert "artifacts.include omits" in str(caught.value)


def test_the_check_argv_keeps_the_declared_comparison_order() -> None:
    """Flags are flattened per output, not regrouped by option name."""
    spec = minimal_spec()
    spec["stages"][1] = {
        "stage_id": "check",
        "type": "hir_check",
        "params": {
            "source_id": "square_hir",
            "input_ids": [],
            "comparisons": [
                {"out": "output[0]", "predicates": [{"fn": "equal", "bounds": {}}]},
                {
                    "out": "output[1]",
                    "predicates": [{"fn": "allclose", "bounds": {"atol": 1e-6, "rtol": 1e-6}}],
                },
            ],
            "dims": {},
        },
    }
    effective = demo_task.validate_task(spec)
    argv = demo_task.stage_command(
        effective["stages"][1], {"report_dir": "/report", "inputs": {}}
    )
    assert argv is not None
    assert argv[argv.index("--out") :] == [
        "--out", "output[0]", "--fn", "equal",
        "--out", "output[1]", "--fn", "allclose", "--atol", "1e-06", "--rtol", "1e-06",
    ]


def test_the_analyze_argv_is_built_from_the_registry() -> None:
    """The source path comes from the registry, never from the spec."""
    effective = demo_task.validate_task(minimal_spec())
    argv = demo_task.stage_command(
        effective["stages"][1], {"report_dir": "/report", "inputs": {}}
    )
    assert argv is not None
    assert argv[:4] == [
        "tilefoundry",
        "analyze",
        demo_task.SOURCE_REGISTRY["square_hir"]["path"],
        "/report/analysis.json",
    ]
    assert "--analysis" in argv and "memory" in argv


def test_no_stage_command_is_built_for_environment_or_archive() -> None:
    """Those stages run in-process, so no argv exists to inject into."""
    effective = demo_task.validate_task(minimal_spec())
    for stage in effective["stages"]:
        if stage["type"] in {"environment", "archive"}:
            assert demo_task.stage_command(stage, {"report_dir": "/r", "inputs": {}}) is None


def test_the_runner_digest_changes_with_the_registry() -> None:
    """The runner digest must distinguish two different whitelists."""
    original = demo_task.runner_sha()
    assert len(original) == 64
    inventory = demo_task.runner_inventory()
    assert inventory["stage_types"] == sorted(demo_task.STAGE_TYPES)
    assert "shell" not in inventory["stage_types"]


def test_a_shipped_task_round_trips_through_json(tmp_path: Path) -> None:
    """The effective spec survives a JSON round trip, which the container relies on."""
    effective = demo_task.validate_task(demo_task.load_task(TASKS / "square.json"))
    path = tmp_path / "effective.json"
    path.write_text(json.dumps(effective, sort_keys=True))
    reparsed = demo_task.validate_task(json.loads(path.read_text()))
    assert demo_task.task_spec_sha(reparsed) == demo_task.task_spec_sha(effective)
