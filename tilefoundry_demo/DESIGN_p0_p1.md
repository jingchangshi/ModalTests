# P0/P1 design contract

Design only; no builds or existing-file changes. Intended artifact: `tilefoundry_demo/DESIGN_p0_p1.md`.

## Scope, evidence, and decisions

Requirements originate in the task request: P0-1–6 are integrity gates; P1 validation and cost limits are pre-allocation gates; stage records and archive correspondence are evidence gates. Applied `engineering-task-planning/SKILL.md` to structure the contract; implementation remains stdlib-only plus existing Modal integration.

Observed source references, relative to `tilefoundry_demo/`:

| Reference | Observed behavior requiring change |
|---|---|
| `demo_source.py:selected_paths/build_snapshot/prepare` | Unvalidated git paths; `copy2` follows links; commit/porcelain comparison misses content races. |
| `run_demo.py:modal.is_local()` image block | Dockerfile/context use live repository; Python upload uses staging. |
| `run_demo.py:main/_timestamp` | Unvalidated second-resolution identifiers; no reservation before preflight/GPU. |
| `demo_container.py:collect_environment/preflight/_run` | Prefix containment; unchecked PEP 610 URLs; versions only compared to each other; misleading inventory phase names. |
| `demo_container.py:_run/_bundle` | Artifact inventory precedes `stage-status.json`; files are rewritten after tar creation; collision discovered during archive publication. |
| `demo_report.py:evaluate_source_integrity/_worst` | Schema 1 and old audit fields; no expected-failure status. |
| `verify_bundle.py:extract/verify` | Safe tar data filter exists; member presence substitutes for content verification. |
| `demo_pytest_plugin.py:_capture_build/_RuntimeProxy` | Library hashes and runtime counters exist; explicit identity links between compile/runtime/test must be added. |
| `README.md:Local tests` | Hardcoded 84 tests; Lead reports 86. Replace count with command/coverage description during implementation. |
| `TileFoundry/Dockerfile` | CUTLASS fetched using build argument; core/target installed later. |
| `TileFoundry/src/tilefoundry/cli/check.py:Ordered/add_arguments` | Ordered comparison options; JSON takes a path; check failure returns 1. |
| `TileFoundry/src/tilefoundry/cli/analyze.py` | Analysis target comes from source; no `--target` injection. |

Assumptions/design choices: Linux/Python 3.12 execution; repository owners and reviewed Dockerfile are trusted; task JSON is untrusted configuration, not a Python sandbox. Hardlink isolation and hostile writers restoring bytes/metadata between every observation are out of scope. Dependency downloads remain permitted through the reviewed image recipe; task-supplied URLs are forbidden. P1 acceptance criteria 1–9 below explicitly enumerate the requested capabilities because no separate numbered list was supplied.

Execution chain remains `run_demo → demo_source → Modal image → CPU preflight → demo_container → observing plugin → demo_report → archive → verify_bundle`. Add one small `demo_task.py` for JSON validation/command construction; no framework.

## Part A — P0 contract

### P0-1 — selected files, containment, frozen bytes

**Invariant:** every uploaded regular file has exactly one `{path,size,sha256}` source-inventory record; no selected path can read outside the repository. Snapshot digest is `demo_report.digest_records(records)` over **staged bytes**.

Functions:

```python
# demo_source.py
def validate_source_path(repo: Path, relative: str) -> Path: ...
def select_sources(repo: Path) -> tuple[list[str], list[dict]]: ...
def selected_paths(repo: Path) -> list[str]: ...  # compatibility wrapper
def copy_checked(repo: Path, relative: str, staging: Path) -> dict: ...
def build_snapshot(repo: Path, staging: Path) -> list[dict]: ...
def prepare(repo: Path, root: Path | None = None) -> tuple[Path, dict, list[dict]]: ...
```

Selection and copying contract:

- Validate **all** raw git names before applying exclusions: nonempty POSIX relative path; reject absolute paths, `..` components, backslashes, NUL/newline, empty components and `.` components.
- Resolve repository root and each candidate strictly; require resolved path to be below root using `Path.is_relative_to`, never string prefix.
- Internal symlinks may be materialized as regular files at their selected names; record `{path,resolved_relative}`. Escaping/dangling symlinks fail the snapshot, including links through parent directories.
- Open resolved targets through root-relative directory descriptors using `os.open`, `dir_fd`, `O_NOFOLLOW`, and directory flags per component. Require regular-file `fstat`; resolve original spelling again after copying and require the same target.
- Copy bytes from that descriptor; record `fstat` before/after, hash while copying, then independently hash staged file. Reject device/inode/size/mtime_ns/ctime_ns changes or unequal hashes.
- First fingerprint pass hashes all selected source contents and records resolution/stat identity. Second pass after copying repeats selection, git identity and fingerprints. Require first = copied = second hashes/sizes and stable selected set/resolution/stat identity.
- No atomic multi-file snapshot is claimed. These checks detect observed per-file edits and stable-porcelain changes; they do not promise detection of every malicious transient write.
- Enumerate staging with `inventory(staging, skip=frozenset())`; upload that entire tree with `ignore=[]`. Apply skips during selection, never silently during final inventory.

Explicit untracked rule: include git-nonignored untracked regular files with suffix `.py`, `.pyi`, `.toml`, `.json`, `.yaml`, `.yml`, `.txt`, `.md`, `.rst`, `.cu`, `.cuh`, `.h`, `.hpp`, `.c`, `.cc`, `.cpp`, `.cmake`, or basename `Dockerfile`, `CMakeLists.txt`, `MANIFEST.in`, `LICENSE`, `NOTICE`; preserve existing `examples/` and CUTLASS gitlink exclusions. Exclude scratch components from `demo_report.SKIP_PARTS` for tracked and untracked paths.

Record every enumerated omission as `{path, tracked, reason}` with reasons `excluded_prefix`, `gitlink`, `runtime_scratch`, `untracked_not_allowlisted`, `tracked_missing`. An unexpected special file is an error. Store include rule, sorted skip list and materialized links in source identity; do not silently broaden inclusion. Required Dockerfile/core/target metadata missing after selection is fatal.

Failure: `ValueError("source <path>: <field>: ...")` for unsafe paths; `RuntimeError` for drift; delete partial staging; no image/GPU call.

| Negative test | Exact input/mutation |
|---|---|
| `tests/test_demo_source.py::test_rejects_absolute_git_path` | Stub git name `/etc/passwd\0`. |
| `tests/test_demo_source.py::test_rejects_parent_git_path` | `src/../../secret.py\0`. |
| `tests/test_demo_source.py::test_rejects_external_symlink` | `src/leak.py -> /etc/passwd`; also parent `src/link -> /etc`. |
| `tests/test_demo_source.py::test_rejects_link_swap_during_copy` | Replace `src/a.py` with link to `/etc/passwd` after selection. |
| `tests/test_demo_source.py::test_detects_same_porcelain_edit` | Untracked `new.py`: change `b"old"` to `b"new"` between passes; porcelain remains `?? new.py`. |
| `tests/test_demo_source.py::test_detects_copy_race` | Rewrite tracked `a.py` during read with same byte length. |
| `tests/test_demo_source.py::test_records_untracked_skips` | `notes.py` included; `weights.bin` skipped with `untracked_not_allowlisted`. |
| `tests/test_demo_source.py::test_no_hidden_uploaded_files` | Selected `.pytest_cache/leak.py` must be skipped, recorded and absent from upload inventory. |

### P0-2 — one frozen build source

**Invariant:** Dockerfile, context, CUTLASS SHA, core/target package source and source digest refer to the same prepared snapshot.

```python
# run_demo.py
def frozen_image_inputs(staging: Path, snapshot: dict, files: list[dict]) -> dict: ...
def build_image(staging: Path, snapshot: dict, files: list[dict]): ...
```

Required construction:

```python
modal.Image.from_dockerfile(
    str(STAGING / "Dockerfile"),
    context_dir=str(STAGING),
    build_args={"CUTLASS_SHA": SNAPSHOT["cutlass_sha"]},
    ignore=[],
)
# Preserve existing pip/env/install chain:
# .add_local_dir(str(STAGING), "/opt/tilefoundry", copy=True, ignore=[])
# pip install --no-deps --no-build-isolation -e . -e targets/nvidia
```

- `cutlass_sha` must be a 40-character lowercase hex gitlink SHA captured during `prepare`; missing SHA fails instead of passing `""`.
- `frozen_image_inputs` verifies whole staged inventory and requires `Dockerfile`, `pyproject.toml`, and `targets/nvidia/pyproject.toml`.
- Record `build_inputs.dockerfile={path:"Dockerfile",size,sha256}`, context/source digest, CUTLASS SHA and existing bootstrap pip requirement list.
- Hash exact `STAGING/Dockerfile` immediately before handing its path to Modal; assert equality with its source-inventory record. Recheck staging after image inputs have been materialized and on CPU preflight.
- Keep staging private, read-only and alive until image consumption/run completes; never reopen the live repository for build inputs.
- Assert manifest Dockerfile hash = validated source record = pre-Modal hash = container `/opt/tilefoundry/Dockerfile` hash.
- Core and NVIDIA packages come only from staged editable paths. Existing third-party bootstrap requirements remain explicit image configuration, recorded in `build_inputs`; this contract does not claim hermetic package-index resolution.

Failure: source/build-input error before GPU; CPU mismatch blocks invocation. No fallback to live files.

| Negative test | Exact input/mutation |
|---|---|
| `tests/test_run_demo.py::test_image_uses_staged_dockerfile_and_context` | After freezing, replace live Dockerfile with `FROM scratch\n`; mocked Modal must receive only staging paths. |
| `tests/test_run_demo.py::test_rejects_changed_staged_dockerfile` | Append `RUN false\n` after inventory. |
| `tests/test_run_demo.py::test_rejects_missing_cutlass_pin` | Snapshot `cutlass_sha=None`. |
| `tests/test_demo_container.py::test_rejects_dockerfile_digest_mismatch` | Container Dockerfile SHA differs from `build_inputs.dockerfile.sha256`. |

### P0-3 — identifiers and reservation before GPU

**Invariant:** a run names one immutable local/Volume namespace; no overwrites.

```python
# demo_report.py
def validate_run_id(run_id: str) -> str: ...
# run_demo.py
def _timestamp() -> str: ...
def reserve_local_run(results: Path, candidate: str) -> Path: ...
# demo_container.py — CPU function, existing mounted Volume
def reserve_volume_run(volume_root: Path, run_id: str, token: str) -> dict: ...
```

- Full-match ASCII regex: `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`; maximum 64 characters. Validate at local boundary and every container entry.
- Generated base: UTC `%Y%m%d-%H%M%S-%f`; candidate suffixes `-001` through `-999`, truncating base to `64-len(suffix)` before suffixing. Bound attempts to 1,000.
- Explicit `--run-id`: collision fails; generated IDs retry on collision at **either** location.
- Before preflight writes and before `run_on_h200.remote`, exclusively create local `results/<run_id>` and CPU-reserve Volume `runs/<run_id>`; existing directory, file or symlink is a collision, even if empty.
- CPU reservation reloads Volume, checks path with `lstat`, creates directory exclusively and writes reservation token with exclusive create; commit before returning. GPU verifies the same token.
- Reservation function uses Modal `max_containers=1` and `@modal.concurrent(max_inputs=1)` to serialize this app’s writers; failed Volume inspection fails closed.
- Other applications must not write this namespace. Modal Volume is not claimed to provide a distributed cross-application transaction.
- Roll back only this attempt’s empty local reservation when remote collision occurs. Retain consumed/failed reservations with failure metadata; archive publication uses exclusive creation as final defense.

| Negative test | Exact input/state |
|---|---|
| `tests/test_run_demo.py::test_rejects_unsafe_run_id` | Parametrize `../x`, `/tmp/x`, `a/b`, `a b`, `a;id`, `""`, `"x"*65`; reject explicit empty after distinguishing omitted default. |
| `tests/test_run_demo.py::test_local_collision_before_gpu` | Existing `results/demo`; explicit `demo`; GPU mock call count 0. |
| `tests/test_run_demo.py::test_volume_collision_before_gpu` | Existing `runs/demo` but no tarball; GPU count 0. |
| `tests/test_run_demo.py::test_generated_collision_suffix` | Both locations collide for base and `base-001`; choose `base-002`. |
| `tests/test_run_demo.py::test_volume_inspection_failure_blocks_gpu` | Reservation raises `OSError("unavailable")`; GPU count 0. |

### P0-4 — installed provenance, fail closed

```python
# demo_container.py
def collect_environment(source_root: Path, log: Path) -> dict: ...
# demo_report.py
def evaluate_source_integrity(environment: dict) -> tuple[str, str]: ...
```

All conditions required:

| Observation | Required value |
|---|---|
| Core PEP 610 URL | Exactly `file:///opt/tilefoundry` |
| Target PEP 610 URL | Exactly `file:///opt/tilefoundry/targets/nvidia` |
| Both records | Objects; `dir_info.editable is True`; no `vcs_info` or `archive_info`. |
| Core import | Resolved existing regular file beneath `/opt/tilefoundry/src/tilefoundry`. |
| Target import | Resolved existing regular file beneath `/opt/tilefoundry/targets/nvidia/tilefoundry_nvidia`. |
| Versions | Both metadata versions equal `snapshot.package_version`; record expected version in environment. |
| Inventories | Nonempty expected inventory; expected digest recomputes to snapshot digest; installed tree equals expected path/size/hash inventory at required audit phases. |

Strict URL equality rejects hosts, queries, fragments, encoded alternatives, traversal and trailing slash; no normalization that broadens acceptance. Replace `origin_inside_snapshot` prefix claim with per-distribution resolved-path observations; reducer derives containment itself. Use `inventory(..., skip=frozenset())`; remove installation metadata during existing image installation cleanup and prevent runtime writes to source.

Missing/malformed observations → `ERROR`; observed mismatch → `FAIL`; either blocks GPU. Runtime revalidation must gate analysis/test, replacing `_run`’s current imports-only gate.

| Negative test | Exact input |
|---|---|
| `tests/test_demo_report.py::test_rejects_wrong_direct_url` | Core URL `file:///opt/tilefoundry-evil`, `https://pypi.org/tilefoundry`, `file://host/opt/tilefoundry`, or missing URL. |
| `tests/test_demo_report.py::test_rejects_wrong_target_origin` | Target URL `file:///opt/tilefoundry`; core URL otherwise correct. |
| `tests/test_demo_report.py::test_rejects_import_escape` | Core import `/opt/tilefoundry-evil/src/tilefoundry/__init__.py`; or link resolving to `/tmp/core.py`. |
| `tests/test_demo_report.py::test_rejects_equal_but_wrong_versions` | Both observed versions `"9.9"`, expected `"0.0.0+demo.123456789abc"`. |
| `tests/test_demo_report.py::test_rejects_inventory_size_or_extra_file` | Same SHA but altered recorded size; alternatively unexpected `src/injected.py`. |
| `tests/test_demo_container.py::test_integrity_failure_stops_test` | Imports true but URL wrong; analysis/test mocks never called. |

### P0-5 — bundle bytes and evidence correspondence

```python
# verify_bundle.py
def extract(archive: Path, destination: Path) -> Path: ...
def verify_members(bundle: Path, manifest: dict) -> list[str]: ...
def verify_correspondence(bundle: Path, manifest: dict, result: dict) -> list[str]: ...
def verify(archive: Path, result: dict, workdir: Path) -> list[str]: ...
```

Verification order and rules:

1. Validate external result schema, archive SHA-256 and byte length **before extraction**; reuse `demo_report.sha256_file`.
2. Extract into a fresh private directory. Preinspect all tar headers: one expected root; normalized relative paths; reject absolute/`..`/backslash names, duplicates, symlinks, hardlinks and special files. Require resolved destination containment; retain `filter="data"`. Reject nonempty extraction destination and existing symlink components.
3. Parse manifest with duplicate-key rejection. Validate record types, unique normalized paths, lowercase 64-hex digests and nonnegative integer sizes.
4. Actual regular members must equal `manifest.artifacts` plus `manifest.json`. Recompute every member size and SHA; report missing, extra and modified files. Include `results/stage-status.json` in inventory.
5. `manifest.json` is the single explicit inventory exception, avoiding a self-hash cycle. External tar hash covers its bytes. All other files, including task/source records, are inventoried.
6. `.so` path set for captured runtime libraries must equal `evidence.compile.libraries` and plugin `compile.build_capture.libraries`; recomputed size/hash must equal both records and artifact inventory. Validate runtime-derived basename.
7. Change `compile.cu_files` to full bundle-relative `{path,size,sha256}` records. Require correspondence to compile dump paths and captured build copies; compare captured `.cu` bytes with dump sources. Build-file `.so` copies outside canonical capture remain inventoried and explicitly identified as build intermediates.
8. Reparse JUnit and analysis/check JSON; compare complete normalized summaries with evidence, not just counts. Plugin collected node IDs/reports must match configured case and JUnit cases. P0 square requires exactly one passed case, no skip/xfail, exit 0.
9. Add plugin `compile_id` and `runtime_id` to compile/runtime/sync/comparison records, with node ID. Verify one compile → captured CUDA/library → runtime call → synchronization → successful comparison for that same case. Counters must agree with event records.
10. Require `run_id`, `task_id`, `source_sha`, `source_commit`, expected/observed package versions and, in P1, `task_spec_sha` equality across result, manifest and identity-bearing evidence/stage records. Verify embedded source inventory digest and archived task-spec digest. Root directory must equal `tilefoundry-demo-<run_id>`.
11. Reuse pure reducers for execution verdict and `combine_with_archive` for delivery verdict. P1 evidence requirements are stage-specific; a CPU-only/negative task does not require a GPU library.

Archive producer changes: finalize evidence and executed-stage verdict → write archived `stage-status.json` → inventory all payload bytes → write manifest once → create tar once → publish → return external delivery envelope. Never mutate archived files after inventory. The archive stage’s own final record belongs in the external envelope.

Errors name file and field, e.g. `generated/build/build/libTirSquare.so: compile.libraries[0].sha256: expected ..., observed ...`. Catch parse/extraction failures and return discrepancies/CLI exit 1; never crash as an apparent success. Hashes establish consistency against the supplied external result, not cryptographic authenticity of the producer.

| Negative test | Exact malicious change |
|---|---|
| `tests/test_verify_bundle.py::test_rejects_length_mismatch` | External `archive_bytes += 1`. |
| `tests/test_verify_bundle.py::test_rehashes_every_member` | Change `logs/analysis.log` from `b"ok"` to `b"no"`; rebuild tar and update outer hash/length only. |
| `tests/test_verify_bundle.py::test_rejects_missing_extra_members` | Remove `results/stage-status.json`; add `results/hidden.json`. |
| `tests/test_verify_bundle.py::test_checks_library_against_compile_hash` | Change only `compile.libraries[0].sha256` to `"0"*64`, preserving actual `.so` hash. |
| `tests/test_verify_bundle.py::test_checks_cuda_correspondence` | Capture CUDA contains `x+1`; compile dump contains `x*x`. |
| `tests/test_verify_bundle.py::test_checks_junit_execution_correspondence` | JUnit names `test_other`; plugin names square; or runtime event references `compile-2` while only `compile-1` exists. |
| `tests/test_verify_bundle.py::test_checks_identity_and_version` | Change only test evidence `task_id="other"` or observed package version `"9.9"`. |
| `tests/test_verify_bundle.py::test_rejects_tar_escapes_and_links` | Headers `../escape`, `/tmp/escape`, root symlink → `/tmp`, duplicate `manifest.json`, hardlink → `../secret`. |
| `tests/test_verify_bundle.py::test_reports_malformed_evidence` | `results/test-results.json` bytes `b"{"`; diagnostic names file and JSON field. |

### P0-6 — honest audit phase names and schema evolution

```python
# demo_report.py
SCHEMA_VERSION = 2
def validate_manifest_version(manifest: dict, *, allow_legacy: bool = False) -> dict: ...
# demo_container.py
def audit_source(source_root: Path, expected: list[dict], phase: str) -> dict: ...
```

Replace `source_check.before_install/after_install` with `source_audit`:

| Execution point | Required phase |
|---|---|
| CPU preflight entry, before environment collection | `preflight_entry` |
| CPU preflight exit, after environment collection and CPU stages | `preflight_exit` |
| GPU function entry, before analysis/test | `execution_entry` |
| GPU function exit/finally, after test | `execution_exit` |

Each audit: `{phase,observed_at,expected_digest,observed_digest,missing,unexpected,modified}`. CPU payload requires first two; GPU manifest carries all four. No claim that runtime observations bracket image-build installation. Partial failure records available audits and cannot pass.

Schema-2 writers use the shared constant; add `expected_package_version`, `build_inputs`, `source_inventory`, `source_sha`, and identity keys. Keep `source_snapshot_sha256` as a schema-2 compatibility alias and require equality with `source_sha`.

Back compatibility: default verification rejects schema 1/missing schema with `schema_version: legacy manifest lacks required integrity evidence`. Explicit `--allow-legacy` permits inspection using old interpretation but returns exit 2 and `LEGACY_UNVERIFIED`, never schema-2 verification success; do not rename old phases into new evidence. Unknown future versions fail.

| Negative test | Exact input |
|---|---|
| `tests/test_demo_report.py::test_v2_rejects_legacy_audit_fields` | Schema 2 containing only `before_install/after_install`. |
| `tests/test_verify_bundle.py::test_v1_requires_explicit_inspection` | `schema_version=1`; default exit 1, legacy inspection exit 2. |
| `tests/test_demo_container.py::test_audits_bracket_actual_work` | Mutate `src/a.py` inside environment collector/test stub; corresponding exit audit must detect it. |

## Part B — P1 Task Spec

### Versioned JSON contract

Task spec version and manifest version are independent. Strict JSON: duplicate keys, unknown fields, NaN/Infinity, wrong types and booleans used as integers fail. No interpolation, environment expansion, executable expressions or arbitrary command fields.

```json
{
  "schema_version": 1,
  "task_id": "square",
  "description": "Analyze H200 HIR and execute the upstream square case",
  "source_snapshot": {"mode": "local", "expected_sha256": null},
  "target": "nvidia.h200_sxm",
  "gpu": {"type": "H200", "count": 1},
  "stages": [
    {"stage_id": "environment", "type": "environment", "params": {}},
    {"stage_id": "analyze", "type": "hir_analyze",
     "params": {"source_id": "h200_model", "analyses": ["compute-cost", "memory"], "dims": {}}},
    {"stage_id": "square", "type": "gpu_test",
     "params": {"case_id": "upstream_square"}},
    {"stage_id": "archive", "type": "archive", "params": {}}
  ],
  "inputs": [],
  "expected_outcomes": {
    "environment": {"outcome": "pass", "negative": false},
    "analyze": {"outcome": "pass", "negative": false},
    "square": {"outcome": "pass", "negative": false},
    "archive": {"outcome": "pass", "negative": false}
  },
  "artifacts": {"include": ["logs", "results", "generated"], "required": []},
  "limits": {
    "cpu_seconds": 1800, "gpu_seconds": 600, "stage_seconds": 300,
    "max_input_bytes": 16777216, "max_artifact_bytes": 1073741824,
    "max_stages": 16
  }
}
```

`task_id`, `stage_id`, input IDs: `[a-z][a-z0-9_-]{0,47}`; unique in their scope. Description is 1–512 characters. Stage arrays have 2–16 entries; exactly one first environment and one last archive. CPU types precede GPU types; no backward dependencies or implicit loops.

`source_snapshot.mode` is only `local`; `expected_sha256` is null or 64 lowercase hex. Inputs are objects `{input_id,path,sha256,size}` relative to `cases/`, regular contained files, each and aggregate bounded by limits. Copy/hash them into frozen per-run inputs before remote execution. `artifacts.include` is a subset of the three fixed categories; `required` contains normalized relative artifact paths; mandatory stage evidence cannot be excluded.

Add stdlib-only interfaces:

```python
# demo_task.py
def load_task(path: Path, task_root: Path, case_root: Path) -> dict: ...
def validate_task(spec: dict, snapshot: dict, registry: dict) -> dict: ...
def task_spec_sha(spec: dict) -> str: ...
def stage_command(stage: dict, context: dict) -> list[str] | None: ...
# demo_container.py
def run_stage(stage: dict, context: dict) -> dict: ...
# demo_report.py
def evaluate_stage(record: dict, expected: dict) -> tuple[str, str]: ...
```

`task_spec_sha` hashes UTF-8 `json.dumps(spec,sort_keys=True,separators=(",",":"),allow_nan=False)` with no newline, after defined defaults are inserted. Freeze the effective spec, not merely its original pathname.

### Stage whitelist and exact params

P1 source/case registry is a small reviewed Python constant: names map to fixed repository source selectors, test node IDs or built-in runners. Task authors cannot specify Python modules, executable paths or pytest options. Registry contents and demo orchestration code are captured in `runner_inventory` and hashed as `runner_sha` in the external result and manifest.

| Type | Exact `params` shape | Runner/evidence |
|---|---|---|
| `environment` | `{}` | Existing environment/provenance/audit collector; CPU. |
| `hir_analyze` | `{source_id:str, analyses:list[str], dims:dict[str,int]}` | Registry selector; `analyze SOURCE REPORT --json` and explicit analysis flags. Analyses nonempty unique subset `compute-cost,memory,roofline`. CPU. |
| `hir_check` | `{source_id:str, input_ids:list[str], comparisons:list[object], dims:dict[str,int]}` | `check SOURCE --device cpu --inputs random` or `files:<frozen paths>`; `--json REPORT`. Registry marks source CPU-capable. CPU. |
| `gpu_test` | `{case_id:str}` | Fixed registered node ID through existing pytest/plugin chain; one invocation. |
| `gpu_benchmark` | `{case_id:str, warmup:int, repeats:int}` | Fixed reviewed benchmark runner; synchronized device timing; correctness gate required first. |
| `archive` | `{}` | Existing `_bundle`, strengthened P0 checks; always attempted after execution failure. |

`dims` keys match `[A-Za-z_][A-Za-z0-9_]{0,47}`, positive integers ≤1,048,576; maximum 8 dimensions; registry supplies tighter case-specific shapes/memory caps. `input_ids` has no duplicates, at most 16, and must match registered positional input arity.

Each comparison is `{out:str, predicates:[{fn:str,bounds:object}]}`. `out` full-matches `output(?:\[[0-9]{1,3}\])*`; output names unique. Predicate bounds:

| `fn` | Exact bounds |
|---|---|
| `equal`, `nan_inf` | `{}` |
| `allclose` | `{atol: finite nonnegative number, rtol: finite nonnegative number}` |
| `cosine` | `{min: finite number in [-1,1]}` |
| `max_abs`, `max_rel` | `{max: finite nonnegative number}` |

At least one predicate per output; reject duplicate predicates and uncovered registered outputs. Flatten arrays in order: each `--out`, then each `--fn`, then its required bounds. Do not group flags by option name. More upstream predicates can be added explicitly later.

Benchmark `warmup` is 0–10; `repeats` 1–100; limits are integers, not booleans. Emit individual samples plus median in milliseconds, synchronization policy, correctness-stage reference and device identity. Timing never proves correctness. A benchmark requires an earlier passing `gpu_test` for the same registered case.

### Validation gate and rejection matrix

All **spec/path/command/resource validation**, source/input hashes, runner registry resolution, output-parent creation, run reservation and CPU environment/check/analysis gates complete before the first GPU invocation. Runtime device/evidence checks necessarily occur after allocation. No GPU retries; one GPU function executes all validated GPU stages sequentially.

| Input/condition | Rejection |
|---|---|
| Unknown top-level/stage/params fields or type | Local `TaskValidationError` naming JSON field; GPU calls 0. |
| `shell:true`, `shell`, `command`, `exec`, `argv`, `cwd` fields | Unknown/forbidden fields; subprocess always `shell=False`, argv built by fixed code. |
| `case_id="/bin/sh"` or unknown registry ID | Reject before importing case code. |
| `gpu.count=2`, non-H200 GPU, timeout above ceilings | Reject; count is 0 or 1. GPU stages require 1; CPU-only specs may use 0. |
| Any task `env`/`secrets` field, including `AWS_SECRET_ACCESS_KEY` | Reject entire field; task cannot request secret injection. Runner uses controlled environment, no forwarding arbitrary host variables. |
| `https://...`, `s3://...`, `file://...` in path/source fields | Reject; registry IDs and contained local paths only. |
| Absolute, traversal or escaping symlink task/input/output paths | Reject using shared normalized-path/containment logic. |
| Unknown stage type, duplicate stage ID, CPU stage after GPU | Reject entire spec, including late stages, before any stage executes. |
| Missing/zero/negative/oversized repeats or recursive stage references | Reject; no repeat/until/retry facility. |
| Unbounded limits, aggregate GPU time >600 seconds | Reject; GPU stage budgets plus archive reserve must fit invocation deadline. |
| CPU gate returns FAIL/ERROR/BLOCKED or invalid evidence | Record result; GPU calls 0. |
| GPU benchmark lacks passing matching correctness case | Reject plan locally, or block benchmark if required test fails at runtime. |

Enforce per-stage subprocess timeout, process-group termination on timeout, bounded log/output files and total input/artifact bytes. Reserve 30 seconds within GPU invocation for final evidence/archive; do not launch another stage when remaining budget is insufficient. Fixed Modal timeout is the outer ceiling. Timeout is `ERROR`, never expected failure.

### Expected negative outcomes

Exact additional expected outcome shape:

```json
{
  "outcome": "expected_fail",
  "negative": true,
  "exit_codes": [1],
  "diagnostic_substrings": ["unsupported dtype"]
}
```

Allowed only for `hir_analyze`, `hir_check`, and explicitly negative registered `gpu_test` cases. `exit_codes` is a nonempty unique subset of `[1]`; parse/usage exit 2, signals, crashes and timeouts cannot confirm an intended failure. Diagnostic strings: 1–8 unique literal nonempty substrings, each ≤256 characters; all must occur in the recorded stage log. Registry records the intended negative condition and requires matching diagnostic expectations.

`EXPECTED_FAIL_CONFIRMED` requires explicit negative flag, allowed observed exit code, all diagnostic substrings and required structured negative evidence. For `hir_check`, require parseable report showing the intended failed predicate/output; for pytest, require the intended case failed in its call phase, not setup/collection.

Exit 0 for an expected failure → `FAIL` (“negative condition did not fail”). Exit 1 with wrong/missing diagnostic → `FAIL`. A positive stage exiting 1 remains `FAIL` even if the log resembles another task’s expected diagnostic. Missing evidence → `ERROR`.

Only confirmed intended failures satisfy their own expectation; the result retains `negative:true` and never describes that stage as a successful kernel. Required stages are satisfied by `PASS` or their explicitly authorized `EXPECTED_FAIL_CONFIRMED`; any unexpected failure keeps overall `FAIL`/`ERROR`. Archive must independently pass.

### CLI and disk layout

```text
modal run run_demo.py                         # tasks/square.json default
modal run run_demo.py --task tasks/square.json
modal run run_demo.py --task tasks/square.json --cpu-only
modal run run_demo.py --task tasks/square.json --run-id square-a
```

Extend `main(..., task: str = "tasks/square.json")`; preserve existing flags. Resolve task relative to demo root and require containment in `tasks/`. `--cpu-only` validates the **entire** spec, executes CPU stages and archives CPU evidence; GPU stages are `NOT_RUN` with reason `cpu_only_requested`. Return CPU workflow success only when CPU/archive gates pass; report `completion_scope:"cpu_only"` and `task_status:"NOT_RUN"` for a task requiring GPU. Do not claim a complete task PASS.

Existing import-time staging/image preparation can remain for default source compatibility, but it must not allocate GPU. Task loading/resolution and reservation occur before remote GPU calls; archive/preflight failures clean up staging in `finally`.

```text
tasks/                         # reviewed versioned JSON requests; square.json default
cases/<case_id>/                # retained existing fixtures; data/config + reviewed runners
experiments/<experiment_id>/    # optional local notes and task variants; not auto-executed
reports/<task_id>/<run_id>/     # derived human-readable report, links to immutable evidence
results/<run_id>/               # existing local raw results, frozen task/inputs, reservation
Volume: runs/<run_id>/          # reservation + immutable tar + external delivery record
```

Experiments must copy a validated task into `tasks/` to execute; no directory scanning/importing. Reports are derived from verified evidence. Existing `cases/rmsnorm_fp8/` remains; no redesign of its implementation is required.

Identity linkage:

- `task_id`: logical workload; `run_id`: reserved invocation.
- `source_sha`: frozen TileFoundry inventory digest; keep `source_commit` separately.
- `task_spec_sha`: effective task JSON digest; `runner_sha`: orchestration/registry bytes.
- `stage_id`: unique within task; include all preceding identity keys in stage records.
- `input_sha`: `digest_records` over exact consumed source/task/input records.
- `output_sha`: `digest_records` over stage-owned output records; exclude stage record itself.
- Copy identities into structured test/check/benchmark evidence and external envelope. No mutable pathname alone identifies an input.

### P1 acceptance criteria and negative tests

| AC | Required capability | Proposed negative tests with exact input/state |
|---|---|---|
| 1 | Strict versioned spec | `tests/test_demo_task.py::test_rejects_invalid_schema`: `schema_version=99`, duplicate `"task_id"` keys, unknown `"magic":true`. |
| 2 | Stage whitelist/typed params | `tests/test_demo_task.py::test_rejects_unknown_stage`: final stage `{"type":"shell","params":{"command":"id"}}`; `::test_rejects_unregistered_case`: `case_id="/bin/sh"`. |
| 3 | Entire validation before GPU | `tests/test_run_demo.py::test_late_invalid_stage_allocates_no_gpu`: valid GPU stage followed by unknown type; `::test_cpu_gate_failure_allocates_no_gpu`: PEP 610 URL wrong. |
| 4 | No command/secret/network/path injection | `tests/test_demo_task.py::test_rejects_injection`: `shell=true`, `env={"AWS_SECRET_ACCESS_KEY":"x"}`, input `../../secret`, input `https://host/a`, input symlink → `/etc/passwd`. |
| 5 | Bounded single-GPU execution | `tests/test_demo_task.py::test_rejects_unbounded_resources`: count 2, repeats 0/101, warmup 11, GPU seconds 601, dimensions exceeding registry memory bound. |
| 6 | Confirm intended negative failures only | `tests/test_demo_report.py::test_expected_fail_requires_all_evidence`: expected exit 1 + substring `"unsupported dtype"`; observe exit 0, exit 2, timeout, or exit 1/log `"ImportError: missing torch"`; each must not confirm. |
| 7 | CLI compatibility and CPU scope | `tests/test_run_demo.py::test_cpu_only_validates_gpu_params`: `--cpu-only` with GPU count 2 still rejects; `::test_cpu_only_never_claims_full_gpu_pass`: valid square task yields no GPU calls/task NOT_RUN. Bare CLI resolves the same default task in companion positive test. |
| 8 | Artifact/identity reproducibility | `tests/test_verify_bundle.py::test_rejects_task_or_input_swap`: change archived task `case_id`, or input bytes `b"old"`→`b"new"` while retaining old `task_spec_sha`/`input_sha`; `::test_rejects_cross_run_stage`: stage run ID `"other"`. |
| 9 | Complete stage evidence and meaningful check/benchmark | `tests/test_demo_task.py::test_rejects_missing_predicate_bounds`: allclose missing `rtol`; `::test_rejects_benchmark_without_correctness`; `tests/test_demo_report.py::test_rejects_missing_stage_log_or_digest`: exit 0 but missing raw log/output; `tests/test_demo_container.py::test_preserves_comparison_order`: two outputs with different predicates must not regroup flags. |

## Part C — exact stage record schema

All keys present; null only under rules below. File paths are bundle-relative, normalized and contained.

```json
{
  "schema_version": 1,
  "stage_id": "square",
  "task_id": "square",
  "run_id": "20261026-120000-123456",
  "source_sha": "<64 lowercase hex>",
  "task_spec_sha": "<64 lowercase hex>",
  "runner_sha": "<64 lowercase hex>",
  "stage_type": "gpu_test",
  "command": ["/usr/bin/python3.12", "-m", "pytest", "<registered node id>"],
  "runner": null,
  "input_sha": "<64 lowercase hex>",
  "input_records": [{"path": "<logical input path>", "size": 1, "sha256": "<64 hex>"}],
  "started_at": "2026-10-26T12:00:00.123456Z",
  "ended_at": "2026-10-26T12:00:01.123456Z",
  "elapsed_seconds": 1.0,
  "exit_code": 0,
  "raw_log_path": "logs/square.log",
  "output_sha": "<64 lowercase hex>",
  "output_records": [{"path": "results/square/junit.xml", "size": 1, "sha256": "<64 hex>"}],
  "evidence_paths": ["results/square/test-results.json", "results/square/junit.xml"],
  "status": "PASS",
  "reason": "registered case and required evidence passed",
  "negative": false,
  "expected_outcome": {"outcome": "pass", "negative": false}
}
```

Rules:

- Exactly one of `command` or `runner` is non-null. Built-in runner IDs: `environment_v1`, `benchmark_v1`, `archive_v1`; no import strings.
- UTC timestamps include microseconds; elapsed time uses monotonic clock, finite and nonnegative. Wall-clock adjustment does not invalidate monotonic elapsed time.
- `exit_code` is actual process status for commands; null for built-ins and stages never launched. Do not invent exit 0 for an in-process collector.
- `raw_log_path` required for started stages, including built-ins; missing log prevents PASS/confirmation.
- `output_records` includes raw log and evidence files; `output_sha=digest_records(output_records)`. Empty output set is legal only for a stage never launched.
- `NOT_RUN`/`BLOCKED` before launch: timestamps, elapsed, exit code and raw log are null; output records empty with defined empty-record digest; reason required.
- Started exceptions/timeouts retain timestamps/log/partial outputs and become `ERROR`.
- Status enum: `NOT_RUN`, `PASS`, `FAIL`, `ERROR`, `BLOCKED`, `EXPECTED_FAIL_CONFIRMED`.
- Extend `_worst` severity with confirmed expected failure at success severity; overall success still requires per-stage expectation validation. Never silently convert ordinary FAIL into confirmation.
- Archive record uses payload inventory as input and `{path,size,sha256}` tar record as output; it lives in the delivery envelope, avoiding self-reference.

## Bounded implementation and validation plan

| Order | Changes | Acceptance gate |
|---|---|---|
| 1 | `demo_source.py` path policy, checked copying, two-pass fingerprints; source tests. | P0-1; no Modal/torch imports. |
| 2 | `run_demo.py` frozen image inputs and ID reservations; `demo_container.py` CPU reservation. | P0-2/3 using Modal stub; GPU call-count assertions. |
| 3 | Schema 2, provenance reducer, honest audits, runtime integrity gate. | P0-4/6 and migrated existing reducer/container fixtures. |
| 4 | Finalize-once archive producer, per-member verifier and plugin correspondence. | P0-5 synthetic real tarballs; diagnostic path/field assertions. |
| 5 | `demo_task.py`, default task and stage records; fixed registry and dispatch through existing chain. | P1 AC1–9, ordered CLI argv tests, CPU-only scope. |
| 6 | Update README for schema/verification/tasks; remove stale test count. | Documentation matches actual implemented CLI. |

Future validation: `python3 -m pytest tests -q` on stdlib+pytest system Python, with fake modules/commands and synthetic archives. No local TileFoundry/torch/triton dependency. Remote CPU smoke and one bounded H200 square run are later integration gates; neither was run for this design. Modal image-input materialization and Volume reservation persistence require those integration checks before claiming end-to-end integrity.