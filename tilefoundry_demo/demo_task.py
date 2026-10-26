"""The versioned task contract: what a run may ask the demo to do, and nothing else.

A task spec is data, not code. It names a workload, a target, and an ordered list of stages
drawn from a fixed whitelist. Everything a caller could use to reach outside that
whitelist -- a shell flag, an extra environment variable, an absolute path, a URL, an
unregistered case name -- is rejected by name, before any GPU is allocated.

The whitelist is a reviewed Python constant in this module rather than a field in the task,
so "which commands can this demo run" is answered by reading source, not by trusting input.
Task JSON can select from the registry and supply bounded numeric parameters; it cannot add
to it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from pathlib import PurePosixPath

SPEC_SCHEMA_VERSION = 1
MAX_STAGES = 16
MIN_STAGES = 2
MAX_IDENTIFIER_LENGTH = 48
TASK_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,47}\Z")
DIMENSION_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,47}\Z")
OUTPUT_NAME_PATTERN = re.compile(r"output(?:\[[0-9]{1,3}\])*\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

#: Fields a task may never carry, whatever their value. Each names a way out of the sandbox.
FORBIDDEN_FIELDS = frozenset(
    {
        "shell",
        "command",
        "commands",
        "exec",
        "argv",
        "cwd",
        "env",
        "environment",
        "secrets",
        "python",
        "module",
        "script",
        "runner",
        "url",
        "uri",
        "retry",
        "retries",
        "loop",
        "until",
        "repeat_stage",
    }
)

#: The reviewed stage types. A task selects from these; it cannot define one.
STAGE_TYPES = frozenset(
    {"environment", "hir_analyze", "hir_check", "gpu_test", "gpu_benchmark", "archive"}
)

#: Stage types that need a device. Everything else is CPU.
GPU_STAGE_TYPES = frozenset({"gpu_test", "gpu_benchmark"})

#: Stage types that can be declared an intentional negative check.
NEGATIVE_CAPABLE = frozenset({"hir_analyze", "hir_check", "gpu_test"})

ANALYSES = frozenset({"compute-cost", "memory", "roofline"})

#: Registered sources. Each entry is a repository-relative selector plus its capabilities.
#: A task names a key; it cannot supply a module path.
SOURCE_REGISTRY: dict[str, dict] = {
    "h200_model": {
        "path": "tests/nvidia/cuda/fixtures/cli/products/h200_sxm.py",
        "cpu_capable": True,
        "target": "nvidia.h200_sxm",
    },
    "square_hir": {
        "path": "tests/nvidia/cuda/fixtures/tir/square.py",
        "cpu_capable": True,
        "target": "nvidia.h200_sxm",
    },
}

#: Registered cases. ``node_id`` is the fixed test the demo runs; there is no way to pass
#: pytest options through a task.
CASE_REGISTRY: dict[str, dict] = {
    "upstream_square": {
        "node_id": "tests/nvidia/cuda/integration/test_square.py::test_square",
        "negative": False,
        "dimensions": {"n": 1024},
    },
    "rmsnorm_fp8": {
        "node_id": "cases/rmsnorm_fp8/rms_norm_quant.py::RmsNormQuant",
        "negative": False,
        "dimensions": {},
    },
    "rmsnorm_fp8_wrong": {
        "node_id": "cases/rmsnorm_fp8/rms_norm_quant.py::Wrong",
        "negative": True,
        "dimensions": {},
        "diagnostic_substrings": ["mismatched"],
    },
}

#: Predicate bounds each comparison function requires and permits.
PREDICATE_BOUNDS = {
    "equal": frozenset(),
    "nan_inf": frozenset(),
    "allclose": frozenset({"atol", "rtol"}),
    "cosine": frozenset({"min"}),
    "max_abs": frozenset({"max"}),
    "max_rel": frozenset({"max"}),
}

DEFAULT_LIMITS = {
    "cpu_seconds": 1800,
    "gpu_seconds": 600,
    "stage_seconds": 300,
    "max_input_bytes": 16 * 1024 * 1024,
    "max_artifact_bytes": 1024 * 1024 * 1024,
    "max_stages": MAX_STAGES,
}

#: Hard ceilings no task may raise. The GPU ceiling accounts for the fixed archive reserve.
CEILINGS = {
    "cpu_seconds": 1800,
    "gpu_seconds": 600,
    "stage_seconds": 600,
    "max_input_bytes": 64 * 1024 * 1024,
    "max_artifact_bytes": 4 * 1024 * 1024 * 1024,
    "max_stages": MAX_STAGES,
}

#: Seconds held back from the GPU invocation for final evidence collection and archiving.
ARCHIVE_RESERVE_SECONDS = 30

ARTIFACT_CATEGORIES = frozenset({"logs", "results", "generated"})

DEFAULT_ARTIFACTS = {"include": ["logs", "results", "generated"], "required": []}


class TaskValidationError(ValueError):
    """A task spec is structurally or semantically unacceptable.

    The message names the JSON field so a caller can fix the spec without reading source.
    """


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """Build one object, refusing a key that appears twice.

    ``json.loads`` keeps the last duplicate silently, so ``{"task_id": "a", "task_id": "b"}``
    would run a task different from the one a reviewer read.
    """
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise TaskValidationError(f"duplicate key {key!r} in one JSON object")
        seen[key] = value
    return seen


def _reject_constant(name: str) -> object:
    """Refuse the non-JSON literals ``json`` accepts by default."""
    raise TaskValidationError(f"{name} is not a JSON number; write a finite number")


def load_task(path: Path) -> dict:
    """Read one task spec, refusing duplicate keys and non-JSON numbers.

    Args:
        path: The spec file.

    Returns:
        The parsed object, unvalidated: call :func:`validate_task` next.

    Raises:
        TaskValidationError: If the file is absent, is not UTF-8 JSON, or is not an object.
    """
    path = Path(path)
    if not path.is_file():
        raise TaskValidationError(f"task spec not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise TaskValidationError(f"task spec {path} is not UTF-8: {error}") from None
    try:
        spec = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise TaskValidationError(f"task spec {path} is not valid JSON: {error}") from None
    if not isinstance(spec, dict):
        raise TaskValidationError(f"task spec {path} must be a JSON object")
    return spec


def _require_keys(obj: dict, allowed: frozenset[str], where: str) -> None:
    """Reject unknown fields and forbidden fields, naming the first offender."""
    for key in obj:
        if key in FORBIDDEN_FIELDS:
            raise TaskValidationError(f"{where}: the field {key!r} is never accepted")
        if key not in allowed:
            raise TaskValidationError(f"{where}: unknown field {key!r}")


def _require_identifier(value: object, where: str) -> str:
    """Require a short lowercase identifier."""
    if not isinstance(value, str) or not TASK_ID_PATTERN.match(value):
        raise TaskValidationError(
            f"{where}: {value!r} is not an identifier matching [a-z][a-z0-9_-]{{0,47}}"
        )
    return value


def _require_int(value: object, where: str, *, low: int, high: int) -> int:
    """Require an integer in range. Booleans are refused, though Python treats them as ints."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskValidationError(f"{where}: {value!r} is not an integer")
    if not low <= value <= high:
        raise TaskValidationError(f"{where}: {value!r} is outside [{low}, {high}]")
    return value


def _require_number(value: object, where: str, *, low: float, high: float) -> float:
    """Require a finite number in range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskValidationError(f"{where}: {value!r} is not a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise TaskValidationError(f"{where}: {value!r} is not finite")
    if not low <= number <= high:
        raise TaskValidationError(f"{where}: {value!r} is outside [{low}, {high}]")
    return number


def _require_relative_path(value: object, where: str, *, base: str) -> str:
    """Require a contained POSIX relative path with no scheme and no traversal."""
    if not isinstance(value, str) or not value:
        raise TaskValidationError(f"{where}: {value!r} is not a non-empty path")
    if "\\" in value:
        raise TaskValidationError(f"{where}: {value!r} contains a backslash")
    if "://" in value or value.split(":", 1)[0] in {"http", "https", "s3", "gs", "file", "ftp"}:
        raise TaskValidationError(f"{where}: {value!r} is a URL; only contained paths are accepted")
    if value.startswith("/"):
        raise TaskValidationError(f"{where}: {value!r} is absolute")
    parts = PurePosixPath(value).parts
    if any(part in {"..", "."} for part in parts):
        raise TaskValidationError(f"{where}: {value!r} contains a traversal component")
    if not parts or parts[0] != base:
        raise TaskValidationError(f"{where}: {value!r} must start with {base!r}")
    return value


def _validate_comparisons(value: object, where: str) -> list[dict]:
    """Validate the ordered comparison list, including every required bound."""
    if not isinstance(value, list) or not value:
        raise TaskValidationError(f"{where}: comparisons must be a nonempty list")
    seen_outputs: set[str] = set()
    checked: list[dict] = []
    for index, item in enumerate(value):
        spot = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise TaskValidationError(f"{spot}: must be an object")
        _require_keys(item, frozenset({"out", "predicates"}), spot)
        out = item.get("out")
        if not isinstance(out, str) or not OUTPUT_NAME_PATTERN.match(out):
            raise TaskValidationError(f"{spot}.out: {out!r} is not output or output[N]")
        if out in seen_outputs:
            raise TaskValidationError(f"{spot}.out: {out!r} appears twice")
        seen_outputs.add(out)
        predicates = item.get("predicates")
        if not isinstance(predicates, list) or not predicates:
            raise TaskValidationError(f"{spot}.predicates: must be a nonempty list")
        seen_functions: set[str] = set()
        checked_predicates = []
        for p_index, predicate in enumerate(predicates):
            p_spot = f"{spot}.predicates[{p_index}]"
            if not isinstance(predicate, dict):
                raise TaskValidationError(f"{p_spot}: must be an object")
            _require_keys(predicate, frozenset({"fn", "bounds"}), p_spot)
            fn = predicate.get("fn")
            if fn not in PREDICATE_BOUNDS:
                raise TaskValidationError(
                    f"{p_spot}.fn: {fn!r} is not one of {sorted(PREDICATE_BOUNDS)}"
                )
            if fn in seen_functions:
                raise TaskValidationError(f"{p_spot}.fn: {fn!r} appears twice for {out}")
            seen_functions.add(fn)
            bounds = predicate.get("bounds", {})
            if not isinstance(bounds, dict):
                raise TaskValidationError(f"{p_spot}.bounds: must be an object")
            _require_keys(bounds, PREDICATE_BOUNDS[fn], f"{p_spot}.bounds")
            if set(bounds) != PREDICATE_BOUNDS[fn]:
                missing = sorted(PREDICATE_BOUNDS[fn] - set(bounds))
                raise TaskValidationError(f"{p_spot}.bounds: missing {missing}")
            for name, bound in bounds.items():
                if fn == "cosine" and name == "min":
                    _require_number(bound, f"{p_spot}.bounds.{name}", low=0.0, high=1.0)
                else:
                    _require_number(bound, f"{p_spot}.bounds.{name}", low=0.0, high=1e12)
            checked_predicates.append({"fn": fn, "bounds": dict(bounds)})
        checked.append({"out": out, "predicates": checked_predicates})
    return checked


def _validate_dimensions(value: object, where: str, registry_dims: dict) -> dict:
    """Validate dimension overrides against the registry's shape."""
    if value is None:
        return dict(registry_dims)
    if not isinstance(value, dict):
        raise TaskValidationError(f"{where}: dims must be an object")
    if len(value) > 8:
        raise TaskValidationError(f"{where}: dims holds {len(value)} entries, at most 8 are allowed")
    checked: dict[str, int] = {}
    for name, size in value.items():
        if not DIMENSION_NAME_PATTERN.match(name):
            raise TaskValidationError(f"{where}.{name}: not a dimension name")
        checked[name] = _require_int(size, f"{where}.{name}", low=1, high=1048576)
    if registry_dims and set(checked) != set(registry_dims):
        raise TaskValidationError(
            f"{where}: dims names {sorted(checked)} but the registered case has {sorted(registry_dims)}"
        )
    return checked


def _validate_stage_params(stage_type: str, params: object, where: str, spec: dict) -> dict:
    """Validate one stage's parameters against its type's exact shape."""
    if not isinstance(params, dict):
        raise TaskValidationError(f"{where}.params: must be an object")

    if stage_type in {"environment", "archive"}:
        _require_keys(params, frozenset(), f"{where}.params")
        return {}

    if stage_type == "hir_analyze":
        _require_keys(params, frozenset({"source_id", "analyses", "dims"}), f"{where}.params")
        source_id = params.get("source_id")
        if source_id not in SOURCE_REGISTRY:
            raise TaskValidationError(
                f"{where}.params.source_id: {source_id!r} is not a registered source"
            )
        analyses = params.get("analyses")
        if not isinstance(analyses, list) or not analyses:
            raise TaskValidationError(f"{where}.params.analyses: must be a nonempty list")
        for index, analysis in enumerate(analyses):
            if analysis not in ANALYSES:
                raise TaskValidationError(
                    f"{where}.params.analyses[{index}]: {analysis!r} is not one of {sorted(ANALYSES)}"
                )
        if len(set(analyses)) != len(analyses):
            raise TaskValidationError(f"{where}.params.analyses: duplicate analysis")
        dims = _validate_dimensions(params.get("dims"), f"{where}.params.dims", {})
        return {"source_id": source_id, "analyses": list(analyses), "dims": dims}

    if stage_type == "hir_check":
        _require_keys(
            params,
            frozenset({"source_id", "input_ids", "comparisons", "dims"}),
            f"{where}.params",
        )
        source_id = params.get("source_id")
        if source_id not in SOURCE_REGISTRY:
            raise TaskValidationError(
                f"{where}.params.source_id: {source_id!r} is not a registered source"
            )
        if not SOURCE_REGISTRY[source_id]["cpu_capable"]:
            raise TaskValidationError(
                f"{where}.params.source_id: {source_id!r} is not registered as CPU-capable"
            )
        input_ids = params.get("input_ids", [])
        if not isinstance(input_ids, list):
            raise TaskValidationError(f"{where}.params.input_ids: must be a list")
        if len(input_ids) > 16:
            raise TaskValidationError(f"{where}.params.input_ids: at most 16 entries are allowed")
        declared = {record["input_id"] for record in spec.get("inputs", [])}
        for index, input_id in enumerate(input_ids):
            _require_identifier(input_id, f"{where}.params.input_ids[{index}]")
            if input_id not in declared:
                raise TaskValidationError(
                    f"{where}.params.input_ids[{index}]: {input_id!r} is not declared in inputs"
                )
        if len(set(input_ids)) != len(input_ids):
            raise TaskValidationError(f"{where}.params.input_ids: duplicate input id")
        comparisons = _validate_comparisons(
            params.get("comparisons"), f"{where}.params.comparisons"
        )
        dims = _validate_dimensions(params.get("dims"), f"{where}.params.dims", {})
        return {
            "source_id": source_id,
            "input_ids": list(input_ids),
            "comparisons": comparisons,
            "dims": dims,
        }

    if stage_type == "gpu_test":
        _require_keys(params, frozenset({"case_id"}), f"{where}.params")
        case_id = params.get("case_id")
        if case_id not in CASE_REGISTRY:
            raise TaskValidationError(
                f"{where}.params.case_id: {case_id!r} is not a registered case"
            )
        return {"case_id": case_id}

    if stage_type == "gpu_benchmark":
        _require_keys(
            params, frozenset({"case_id", "warmup", "repeats"}), f"{where}.params"
        )
        case_id = params.get("case_id")
        if case_id not in CASE_REGISTRY:
            raise TaskValidationError(
                f"{where}.params.case_id: {case_id!r} is not a registered case"
            )
        warmup = _require_int(params.get("warmup", 3), f"{where}.params.warmup", low=0, high=10)
        repeats = _require_int(params.get("repeats", 20), f"{where}.params.repeats", low=1, high=100)
        return {"case_id": case_id, "warmup": warmup, "repeats": repeats}

    raise TaskValidationError(f"{where}.type: {stage_type!r} is not a registered stage type")


def _validate_expected_outcome(stage_id: str, value: object, stage_type: str, where: str) -> dict:
    """Validate one expected-outcome record, including the negative-failure shape."""
    if not isinstance(value, dict):
        raise TaskValidationError(f"{where}: must be an object")
    _require_keys(
        value,
        frozenset({"outcome", "negative", "exit_codes", "diagnostic_substrings"}),
        where,
    )
    outcome = value.get("outcome")
    negative = value.get("negative", False)
    if not isinstance(negative, bool):
        raise TaskValidationError(f"{where}.negative: must be true or false")

    if outcome == "pass":
        if negative:
            raise TaskValidationError(f"{where}: outcome 'pass' cannot also be negative")
        for extra in ("exit_codes", "diagnostic_substrings"):
            if extra in value:
                raise TaskValidationError(f"{where}: {extra!r} is only meaningful for expected_fail")
        return {"outcome": "pass", "negative": False}

    if outcome != "expected_fail":
        raise TaskValidationError(f"{where}.outcome: {outcome!r} is not 'pass' or 'expected_fail'")

    if not negative:
        raise TaskValidationError(
            f"{where}: expected_fail requires negative=true, so the intent is explicit"
        )
    if stage_type not in NEGATIVE_CAPABLE:
        raise TaskValidationError(
            f"{where}: a {stage_type} stage cannot be declared an intended failure"
        )
    exit_codes = value.get("exit_codes", [1])
    if not isinstance(exit_codes, list) or not exit_codes:
        raise TaskValidationError(f"{where}.exit_codes: must be a nonempty list")
    if any(isinstance(code, bool) or not isinstance(code, int) for code in exit_codes):
        raise TaskValidationError(f"{where}.exit_codes: every entry must be an integer")
    if set(exit_codes) != {1}:
        raise TaskValidationError(
            f"{where}.exit_codes: {exit_codes!r} is not an acceptable intended failure; "
            "only exit 1 can confirm one, because 2 is a usage error and a signal is not a verdict"
        )

    substrings = value.get("diagnostic_substrings")
    if not isinstance(substrings, list) or not substrings:
        raise TaskValidationError(f"{where}.diagnostic_substrings: must be a nonempty list")
    if len(substrings) > 8:
        raise TaskValidationError(f"{where}.diagnostic_substrings: at most 8 entries are allowed")
    for index, substring in enumerate(substrings):
        if not isinstance(substring, str) or not substring:
            raise TaskValidationError(
                f"{where}.diagnostic_substrings[{index}]: must be a nonempty string"
            )
        if len(substring) > 256:
            raise TaskValidationError(
                f"{where}.diagnostic_substrings[{index}]: longer than 256 characters"
            )
    if len(set(substrings)) != len(substrings):
        raise TaskValidationError(f"{where}.diagnostic_substrings: duplicate substring")

    return {
        "outcome": "expected_fail",
        "negative": True,
        "exit_codes": list(exit_codes),
        "diagnostic_substrings": list(substrings),
    }


def _validate_limits(value: object) -> dict:
    """Validate resource limits against the fixed ceilings."""
    if value is None:
        return dict(DEFAULT_LIMITS)
    if not isinstance(value, dict):
        raise TaskValidationError("limits: must be an object")
    _require_keys(value, frozenset(CEILINGS), "limits")
    limits = dict(DEFAULT_LIMITS)
    for name, ceiling in CEILINGS.items():
        if name in value:
            limits[name] = _require_int(value[name], f"limits.{name}", low=1, high=ceiling)
    if limits["max_stages"] > MAX_STAGES:
        raise TaskValidationError(f"limits.max_stages: above the fixed maximum {MAX_STAGES}")
    return limits


def _validate_inputs(value: object, case_root: Path) -> list[dict]:
    """Validate declared inputs, which must live under ``cases/``."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TaskValidationError("inputs: must be a list")
    seen: set[str] = set()
    checked: list[dict] = []
    total = 0
    for index, record in enumerate(value):
        where = f"inputs[{index}]"
        if not isinstance(record, dict):
            raise TaskValidationError(f"{where}: must be an object")
        _require_keys(record, frozenset({"input_id", "path", "sha256", "size"}), where)
        input_id = _require_identifier(record.get("input_id"), f"{where}.input_id")
        if input_id in seen:
            raise TaskValidationError(f"{where}.input_id: {input_id!r} appears twice")
        seen.add(input_id)
        relative = _require_relative_path(record.get("path"), f"{where}.path", base="cases")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.match(digest):
            raise TaskValidationError(f"{where}.sha256: {digest!r} is not 64 lowercase hex")
        size = _require_int(record.get("size"), f"{where}.size", low=0, high=CEILINGS["max_input_bytes"])
        # A declared input must resolve inside cases/ and match the declared digest, so a
        # task cannot point at a file that changes after review.
        resolved = (case_root.parent / relative).resolve()
        if not resolved.is_relative_to(case_root.resolve()):
            raise TaskValidationError(f"{where}.path: {relative!r} escapes {case_root}")
        if not resolved.is_file():
            raise TaskValidationError(f"{where}.path: {relative!r} is not a regular file")
        from demo_report import sha256_file

        if resolved.stat().st_size != size:
            raise TaskValidationError(
                f"{where}.size: declared {size}, file has {resolved.stat().st_size}"
            )
        if sha256_file(resolved) != digest:
            raise TaskValidationError(f"{where}.sha256: the file's bytes do not match")
        total += size
        checked.append({"input_id": input_id, "path": relative, "sha256": digest, "size": size})
    return checked


def validate_task(spec: dict, *, case_root: Path | None = None) -> dict:
    """Validate a task spec and return the effective spec with defaults filled in.

    Args:
        spec: The parsed task object.
        case_root: The ``cases/`` directory, used to check declared inputs. Defaults to
            ``cases`` beside this module.

    Returns:
        The effective spec. Its digest is :func:`task_spec_sha`.

    Raises:
        TaskValidationError: Naming the offending JSON field.
    """
    if not isinstance(spec, dict):
        raise TaskValidationError("a task spec must be a JSON object")
    _require_keys(
        spec,
        frozenset(
            {
                "schema_version",
                "task_id",
                "description",
                "source_snapshot",
                "target",
                "gpu",
                "stages",
                "inputs",
                "expected_outcomes",
                "artifacts",
                "limits",
            }
        ),
        "task",
    )

    version = spec.get("schema_version")
    if version != SPEC_SCHEMA_VERSION or isinstance(version, bool):
        raise TaskValidationError(
            f"schema_version: {version!r} is not supported; this build reads {SPEC_SCHEMA_VERSION}"
        )

    task_id = _require_identifier(spec.get("task_id"), "task_id")

    description = spec.get("description", "")
    if not isinstance(description, str) or not 1 <= len(description) <= 512:
        raise TaskValidationError("description: must be a string of 1 to 512 characters")

    snapshot = spec.get("source_snapshot", {"mode": "local", "expected_sha256": None})
    if not isinstance(snapshot, dict):
        raise TaskValidationError("source_snapshot: must be an object")
    _require_keys(snapshot, frozenset({"mode", "expected_sha256"}), "source_snapshot")
    if snapshot.get("mode") != "local":
        raise TaskValidationError(
            f"source_snapshot.mode: {snapshot.get('mode')!r} is not supported; only 'local' is"
        )
    expected_digest = snapshot.get("expected_sha256")
    if expected_digest is not None and (
        not isinstance(expected_digest, str) or not SHA256_PATTERN.match(expected_digest)
    ):
        raise TaskValidationError(
            f"source_snapshot.expected_sha256: {expected_digest!r} is not null or 64 lowercase hex"
        )

    target = spec.get("target", "nvidia.h200_sxm")
    if target != "nvidia.h200_sxm":
        raise TaskValidationError(
            f"target: {target!r} is not accepted; this demo runs nvidia.h200_sxm"
        )

    gpu = spec.get("gpu", {"type": "H200", "count": 1})
    if gpu is None:
        gpu = {"type": "H200", "count": 1}
    if not isinstance(gpu, dict):
        raise TaskValidationError("gpu: must be an object")
    _require_keys(gpu, frozenset({"type", "count"}), "gpu")
    if gpu.get("type") != "H200":
        raise TaskValidationError(f"gpu.type: {gpu.get('type')!r} is not 'H200'")
    gpu_count = _require_int(gpu.get("count", 1), "gpu.count", low=0, high=1)

    stages = spec.get("stages")
    if not isinstance(stages, list):
        raise TaskValidationError("stages: must be a list")
    if not MIN_STAGES <= len(stages) <= MAX_STAGES:
        raise TaskValidationError(
            f"stages: {len(stages)} entries, expected between {MIN_STAGES} and {MAX_STAGES}"
        )

    case_root = Path(case_root) if case_root is not None else Path(__file__).resolve().parent / "cases"
    inputs = _validate_inputs(spec.get("inputs"), case_root)
    # Stage validation can read the declared input ids, so pass the checked list on.
    stage_context = dict(spec)
    stage_context["inputs"] = inputs

    checked_stages: list[dict] = []
    seen_stage_ids: set[str] = set()
    seen_gpu = False
    for index, stage in enumerate(stages):
        where = f"stages[{index}]"
        if not isinstance(stage, dict):
            raise TaskValidationError(f"{where}: must be an object")
        _require_keys(stage, frozenset({"stage_id", "type", "params"}), where)
        stage_id = _require_identifier(stage.get("stage_id"), f"{where}.stage_id")
        if stage_id in seen_stage_ids:
            raise TaskValidationError(f"{where}.stage_id: {stage_id!r} appears twice")
        seen_stage_ids.add(stage_id)
        stage_type = stage.get("type")
        if stage_type not in STAGE_TYPES:
            raise TaskValidationError(
                f"{where}.type: {stage_type!r} is not one of {sorted(STAGE_TYPES)}"
            )
        if stage_type in GPU_STAGE_TYPES:
            seen_gpu = True
        elif seen_gpu and stage_type != "archive":
            # `archive` is the mandatory last stage and must run after a failure, so it is
            # the one CPU stage that legitimately follows a GPU stage.
            raise TaskValidationError(
                f"{where}: a {stage_type} stage follows a GPU stage; CPU stages run first"
            )
        if index == 0 and stage_type != "environment":
            raise TaskValidationError(f"{where}: the first stage must be 'environment'")
        if index == len(stages) - 1 and stage_type != "archive":
            raise TaskValidationError(f"{where}: the last stage must be 'archive'")
        if 0 < index < len(stages) - 1 and stage_type in {"environment", "archive"}:
            raise TaskValidationError(
                f"{where}: {stage_type!r} may appear only first or last"
            )
        params = _validate_stage_params(stage_type, stage.get("params", {}), where, stage_context)
        checked_stages.append({"stage_id": stage_id, "type": stage_type, "params": params})

    if seen_gpu and gpu_count != 1:
        raise TaskValidationError(
            f"gpu.count: this task has GPU stages, so the count must be 1, not {gpu_count}"
        )
    if not seen_gpu and gpu_count != 0:
        raise TaskValidationError(
            f"gpu.count: this task has no GPU stage, so the count must be 0, not {gpu_count}"
        )

    limits = _validate_limits(spec.get("limits"))

    outcomes = spec.get("expected_outcomes", {})
    if not isinstance(outcomes, dict):
        raise TaskValidationError("expected_outcomes: must be an object")
    checked_outcomes: dict[str, dict] = {}
    for stage_id, value in outcomes.items():
        if stage_id not in seen_stage_ids:
            raise TaskValidationError(
                f"expected_outcomes: {stage_id!r} names no stage in this task"
            )
    by_id = {stage["stage_id"]: stage for stage in checked_stages}
    for stage in checked_stages:
        stage_id = stage["stage_id"]
        if stage_id in outcomes:
            checked_outcomes[stage_id] = _validate_expected_outcome(
                stage_id, outcomes[stage_id], stage["type"], f"expected_outcomes.{stage_id}"
            )
        else:
            checked_outcomes[stage_id] = {"outcome": "pass", "negative": False}

    # A negative GPU test must be declared on a registry entry that is marked negative, so a
    # task cannot relabel an ordinary failing kernel as an intended failure.
    for stage in checked_stages:
        if stage["type"] != "gpu_test":
            continue
        expected = checked_outcomes[stage["stage_id"]]
        case = CASE_REGISTRY[stage["params"]["case_id"]]
        if expected["negative"] and not case["negative"]:
            raise TaskValidationError(
                f"expected_outcomes.{stage['stage_id']}: case "
                f"{stage['params']['case_id']!r} is not registered as negative"
            )
        if case["negative"] and not expected["negative"]:
            raise TaskValidationError(
                f"expected_outcomes.{stage['stage_id']}: case "
                f"{stage['params']['case_id']!r} is registered as negative and must be declared "
                "expected_fail, so a failure is not reported as a broken kernel"
            )
        if expected["negative"] and case.get("diagnostic_substrings"):
            if expected["diagnostic_substrings"] != case["diagnostic_substrings"]:
                raise TaskValidationError(
                    f"expected_outcomes.{stage['stage_id']}.diagnostic_substrings: must equal the "
                    "registered expectation for this negative case"
                )

    # A benchmark measures timing; a test must already have established that the kernel is
    # right, or the timing describes a wrong answer.
    for index, stage in enumerate(checked_stages):
        if stage["type"] != "gpu_benchmark":
            continue
        case_id = stage["params"]["case_id"]
        earlier = [
            other
            for other in checked_stages[:index]
            if other["type"] == "gpu_test"
            and other["params"]["case_id"] == case_id
            and not checked_outcomes[other["stage_id"]]["negative"]
        ]
        if not earlier:
            raise TaskValidationError(
                f"stages[{index}]: gpu_benchmark of {case_id!r} has no earlier passing gpu_test "
                "of the same case; timing never proves correctness"
            )

    artifacts = spec.get("artifacts", DEFAULT_ARTIFACTS)
    if not isinstance(artifacts, dict):
        raise TaskValidationError("artifacts: must be an object")
    _require_keys(artifacts, frozenset({"include", "required"}), "artifacts")
    include = artifacts.get("include", DEFAULT_ARTIFACTS["include"])
    if not isinstance(include, list) or not include:
        raise TaskValidationError("artifacts.include: must be a nonempty list")
    for index, category in enumerate(include):
        if category not in ARTIFACT_CATEGORIES:
            raise TaskValidationError(
                f"artifacts.include[{index}]: {category!r} is not one of {sorted(ARTIFACT_CATEGORIES)}"
            )
    if len(set(include)) != len(include):
        raise TaskValidationError("artifacts.include: duplicate category")
    required = artifacts.get("required", [])
    if not isinstance(required, list):
        raise TaskValidationError("artifacts.required: must be a list")
    for index, relative in enumerate(required):
        where = f"artifacts.required[{index}]"
        if not isinstance(relative, str) or not relative:
            raise TaskValidationError(f"{where}: must be a nonempty path")
        if relative.startswith("/") or ".." in PurePosixPath(relative).parts or "\\" in relative:
            raise TaskValidationError(f"{where}: {relative!r} is not a contained relative path")
        category = PurePosixPath(relative).parts[0]
        if category not in include:
            raise TaskValidationError(
                f"{where}: {relative!r} is under {category!r}, which artifacts.include omits"
            )

    return {
        "schema_version": SPEC_SCHEMA_VERSION,
        "task_id": task_id,
        "description": description,
        "source_snapshot": {"mode": "local", "expected_sha256": expected_digest},
        "target": target,
        "gpu": {"type": "H200", "count": gpu_count},
        "stages": checked_stages,
        "inputs": inputs,
        "expected_outcomes": checked_outcomes,
        "artifacts": {"include": list(include), "required": list(required)},
        "limits": limits,
    }


def task_spec_sha(spec: dict) -> str:
    """Hash the effective spec, so two different requests cannot share a digest.

    Defaults are already inserted by :func:`validate_task`, so the digest covers what will
    actually run rather than what the caller happened to write down.
    """
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stage_command(stage: dict, context: dict) -> list[str] | None:
    """Return the argv for one stage, or None if the stage has no subprocess.

    The argv is assembled here, from the registry and the validated parameters. No string
    from the spec reaches a shell, and no field can add an option.

    Args:
        stage: One validated stage record.
        context: Paths the stage needs: ``source_root`` and ``report_dir``.

    Returns:
        An argv list for ``subprocess.run(..., shell=False)``, or None.
    """
    stage_type = stage["type"]
    params = stage["params"]
    report_dir = Path(context["report_dir"])

    if stage_type in {"environment", "archive"}:
        return None

    if stage_type == "hir_analyze":
        source = SOURCE_REGISTRY[params["source_id"]]["path"]
        argv = ["tilefoundry", "analyze", source, str(report_dir / "analysis.json"), "--json"]
        for analysis in params["analyses"]:
            argv.extend(["--analysis", analysis])
        for name, size in sorted(params["dims"].items()):
            argv.extend(["--dim", f"{name}={size}"])
        return argv

    if stage_type == "hir_check":
        source = SOURCE_REGISTRY[params["source_id"]]["path"]
        argv = ["tilefoundry", "check", source, "--device", "cpu"]
        if params["input_ids"]:
            paths = [context["inputs"][input_id] for input_id in params["input_ids"]]
            argv.extend(["--inputs", "files:" + ",".join(paths)])
        else:
            argv.extend(["--inputs", "random"])
        argv.extend(["--json", str(report_dir / "check.json")])
        # Flatten in the declared order: each --out, then each --fn with its own bounds.
        for comparison in params["comparisons"]:
            argv.extend(["--out", comparison["out"]])
            for predicate in comparison["predicates"]:
                argv.extend(["--fn", predicate["fn"]])
                for name, value in predicate["bounds"].items():
                    argv.extend([f"--{name}", f"{value}"])
        for name, size in sorted(params["dims"].items()):
            argv.extend(["--dim", f"{name}={size}"])
        return argv

    raise TaskValidationError(
        f"stage {stage['stage_id']!r}: a {stage_type} stage is run by the orchestrator, not by argv"
    )


def runner_inventory() -> dict:
    """Return the registry and whitelist that decide what a task can run.

    This is hashed into the evidence so a reader can tell which runner rules produced a
    verdict, not merely which task was requested.
    """
    return {
        "stage_types": sorted(STAGE_TYPES),
        "gpu_stage_types": sorted(GPU_STAGE_TYPES),
        "negative_capable": sorted(NEGATIVE_CAPABLE),
        "analyses": sorted(ANALYSES),
        "predicates": {name: sorted(bounds) for name, bounds in sorted(PREDICATE_BOUNDS.items())},
        "sources": {name: dict(entry) for name, entry in sorted(SOURCE_REGISTRY.items())},
        "cases": {name: dict(entry) for name, entry in sorted(CASE_REGISTRY.items())},
        "ceilings": dict(CEILINGS),
        "archive_reserve_seconds": ARCHIVE_RESERVE_SECONDS,
    }


def runner_sha() -> str:
    """Hash :func:`runner_inventory`, plus this module's own bytes."""
    inventory = runner_inventory()
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()
