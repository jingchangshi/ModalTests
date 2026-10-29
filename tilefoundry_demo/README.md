# TileFoundry on a Modal H200 — minimal end-to-end demo

Runs a TileFoundry checkout on a remote NVIDIA H200 through Modal: HIR static
analysis, TIR to CUDA codegen, an NVCC/CMake build, a real kernel execution, a
PyTorch comparison, and a downloadable evidence bundle. A second driver runs the
RMSNorm + FP8 tutorial experiments on the same image.

**This README is written for a fresh Linux server.** Any Linux machine that can
reach the network and log in to Modal can run it. There is no dependency on the
workstation this demo was developed on: no user home directory, no conda
environment, and no particular Modal profile name appears in the committed code
or in the commands below.

## 1 · What the host must have

Only these five things:

| Requirement | Detail |
|---|---|
| Git | to clone this repository and the TileFoundry checkout beside it |
| Python 3.11+ | for the Modal client interpreter (a venv is enough) |
| the `modal` client | `python -m pip install modal` in that venv |
| network access | to the Modal API and to PyPI for the client |
| a valid Modal account | logged in, with H200 capacity available |

And plainly, what it must **not** have:

- **No local GPU** and no `nvidia-smi`. Every device operation happens in the
  remote H200 container.
- **No local CUDA** and no local NVCC toolchain.
- **No local PyTorch**, no local `transformers`, no local CUTLASS.
- **No local TileFoundry** install. The checkout is uploaded into the image, not
  installed on this machine.

The Modal profile name is arbitrary. Only `modal profile current` has to
succeed; a fresh server that runs `modal token new` (or `modal setup`) and
creates a profile under any name is ready. Nothing in this repository reads
`MODAL_PROFILE` or names a profile.

## 2 · Quick start

```bash
mkdir -p "$HOME/workspace" && cd "$HOME/workspace"
git clone https://github.com/jingchangshi/TileFoundry.git
git clone https://github.com/jingchangshi/ModalTests.git

python3 -m venv "$HOME/.venvs/tilefoundry-modal"
source "$HOME/.venvs/tilefoundry-modal/bin/activate"
python -m pip install modal
modal profile current          # must succeed; the profile name does not matter

export TILEFOUNDRY_REPO="$HOME/workspace/TileFoundry"
cd "$HOME/workspace/ModalTests/tilefoundry_demo"

modal run run_demo.py --cpu-only     # no-GPU preflight: cheapest way to prove the image
modal run run_demo.py                # full run on one H200
modal run run_experiments.py --experiment all
```

Every command above is home-relative, so the same block works on any server
whose `$HOME` differs. Running the two clones as siblings under
`$HOME/workspace` is also what makes the checkout auto-discovery fallback in
section 4 work, so `TILEFOUNDRY_REPO` could be omitted there; it is exported
above to make the choice explicit and to survive a different layout.

## 3 · Cost control

- `--cpu-only` stops after the preflight and never allocates a GPU. Always run
  it after changing dependencies: a broken image then costs CPU time only.
- Every remote call is one bounded invocation with a fixed timeout: `600` s for
  the H200 function in `run_demo.py`, `1800` s for its CPU preflight, `900` s for
  `run_experiments.py`'s CPU generator and `1800` s for each of its GPU
  experiment functions. No function is given retries, and stages and experiments
  are called one after another, never in parallel.
- The Dockerfile and pip layers are cached. The source is added by the last
  local layer, so a `.py` change rebuilds only that layer plus the three cheap
  commands after it (the editable install, the `*.egg-info` cleanup, and the
  import check) — not the CUDA/Torch layers.
- Image builds, CPU time, and Volume storage are billed separately from GPU time.

## 4 · Which TileFoundry checkout gets used

The demo freezes a local checkout into the image, digests it, and records the
digest in the manifest. Resolution order is exactly this, and nothing else
(`run_demo.resolve_tilefoundry_repo()`):

1. `--tilefoundry-repo PATH` — the command-line argument, when passed.
2. `TILEFOUNDRY_REPO` — the environment variable, when set and non-empty.
3. **Sibling auto-discovery** — a directory named `TileFoundry` beside this
   repository: literally `<workspace>/TileFoundry` next to
   `<workspace>/ModalTests`. That is exactly the layout section 2 clones, so
   the `export` there is a belt-and-braces measure rather than a requirement.
   The candidate is derived from `run_demo.py`'s own location (two levels up
   from `tilefoundry_demo/`), so running `modal run` from another working
   directory cannot change which checkout is frozen; no candidate is ever
   resolved from the current working directory.

A candidate counts only when it contains `pyproject.toml`. If none does, the
run stops **before any image is built** — no GPU, no image build, no upload —
with:

```text
no TileFoundry checkout was found, so no Modal image was built.
Looked for a directory holding 'pyproject.toml', in this order:
  ...
Fix it by cloning TileFoundry and naming the checkout explicitly:
  git clone https://github.com/jingchangshi/TileFoundry.git "$HOME/workspace/TileFoundry"
  export TILEFOUNDRY_REPO="$HOME/workspace/TileFoundry"
or pass --tilefoundry-repo PATH. The checkout must be a git clone, because the
snapshot records its commit and its pinned CUTLASS gitlink.
```

Three details worth knowing:

- Prefer `TILEFOUNDRY_REPO` over `--tilefoundry-repo`. The source is frozen when
  the module is imported, so a path that differs from that frozen snapshot is
  refused; the environment variable is read at import time and always agrees.
- The path must be a real git clone: the manifest records its commit, branch,
  dirty state, and the pinned CUTLASS submodule SHA.
- Nothing here reads the current working directory.

## 5 · Tasks and stages

A run executes a task spec from `tasks/`. The spec, its digest, and the runner
digest travel with the run, so the evidence names what was requested and which
runner executed it.

`tasks/square.json` (`task_id: square`, one H200) requests four stages:

| stage_id | type | What it observes |
|---|---|---|
| `environment` | `environment` | the image installs the uploaded snapshot; both distributions import, resolve inside that tree as editable installs, and match their digested inventory; `nvidia.h200_sxm` is registered; nvcc/CMake/ninja/g++ are present |
| `analyze` | `hir_analyze` | `tilefoundry analyze` exits 0 and writes JSON naming `compute-cost` and `memory` |
| `square` | `gpu_test` | the upstream TIR square case is compiled and executed on the H200, and its own `torch.testing.assert_close` comparison passed |
| `archive` | `archive` | the evidence tarball reached the Volume and its digest and size are recorded |

`tasks/square_cpu.json` (`task_id: square_cpu`, no GPU) swaps the device stage
for `check`, an `hir_check` stage that compares the analyzed module on CPU.
Its stages are `environment`, `analyze`, `check`, `archive`.

The registry is reviewed: a task selects from the stage types `environment`,
`hir_analyze`, `hir_check`, `gpu_test`, `gpu_benchmark` and `archive`, and
cannot define a new one. `demo_task.assert_runner_support()` runs at load time,
so a stage type this runner has no executor for — or a GPU case registered to a
different runner — is refused as a local `SystemExit` before any image is built
or any GPU allocated, instead of being executed and reported as run.

```bash
modal run run_demo.py --task tasks/square_cpu.json    # CPU-only task
modal run run_demo.py --task tasks/square.json        # the H200 task (default)
modal run run_demo.py --run-id my-run-1               # name the run instead of generating an id
```

## 6 · Status vocabulary

Every stage is reduced independently and gets exactly one of these statuses:

| Status | Meaning |
|---|---|
| `PASS` | the stage's own recorded evidence supports it |
| `FAIL` | the stage ran and its evidence contradicts the claim |
| `ERROR` | the stage could not be judged: it ran without producing a usable observation, or it raised |
| `BLOCKED` | a precondition it cannot satisfy was absent, e.g. no CUDA device |
| `NOT_RUN` | the stage never executed; `overall_reason` says which one and why |
| `EXPECTED_FAIL_CONFIRMED` | a stage failed exactly as its task declared it would, with the stated evidence. It is **not** `PASS`: it never claims the kernel worked |

`overall_status` is the worst status among the stages that have a verdict, with
`PASS` and `EXPECTED_FAIL_CONFIRMED` ranked equally best. A partial run
therefore cannot be reported as `PASS`: a stage that did not run stays `NOT_RUN`
with a reason, and a stage that ran without a judgement is `ERROR` rather than an
assumed success.

The `archive` stage is judged too, at the very end: it becomes `PASS` with a
reason like `archived 4024663 bytes as runs/<run-id>/...tar.gz` once the tarball
has actually reached the Volume with a recorded digest and size. One subtlety
follows from the ordering — the `manifest.json` copied *inside* the archive was
written before that archive existed, so it necessarily records the pre-archive
`overall_status`. That is why `verify_bundle.py` re-derives the delivered verdict
from the printed result envelope for this one stage instead of trusting the
archived copy.

### How stages are named in the evidence

The per-stage entries in `manifest.json` — under `stage_details` (and the
run's `results/stage-status.json`) — are keyed by **the task's own stage ids**,
not by a fixed set of names. `tasks/square.json` reports
`environment`/`analyze`/`square`/`archive`; `tasks/square_cpu.json` reports
`environment`/`analyze`/`check`/`archive`. `manifest.json` also carries
`task_stages`, one record per stage in execution order with its `stage_id`,
`type`, `command`, and whether it `executed`.

It also carries the **whole validated task spec** under `task`, next to `task_id`,
`task_spec_sha`, `task_stages` and `completion_scope`. That is what makes the
archive self-describing: `verify_bundle.py` re-derives the verdict from the
downloaded archive alone — it scopes the evidence to the stages `task` actually
declares, rather than to a fixed list baked into the checker. A bundle therefore
cannot be re-interpreted as a different task after the fact.

The five legacy keys (`environment_status`, `analysis_status`,
`compile_status`, `execution_status`, `correctness_status`) are still present
for compatibility, but they are now **derived from the task's stages**. A
CPU-only task therefore reports the device-less keys as `NOT_RUN` instead of
inheriting a judgement about work it never requested.

`completion_scope` is separate from the stage statuses and says what this
invocation claimed to cover: `full` for a device run, `cpu_only` for a
preflight. A `--cpu-only` run stamps `completion_scope=cpu_only`, and every
declared GPU stage becomes `NOT_RUN` with the reason `this invocation does not
allocate a GPU, so a device stage was not run`; `task_status` is `COMPLETE` for
a CPU-only task and `NOT_RUN` for a task with GPU stages.

## 7 · The RMSNorm + FP8 experiment driver

`run_experiments.py` reuses `run_demo.py`'s frozen image, so the experiments see
exactly the environment the gates above verified:

```bash
modal run run_experiments.py --experiment all
modal run run_experiments.py --experiment correctness
modal run run_experiments.py --experiment twin
modal run run_experiments.py --experiment benchmark
```

The RMSNorm inputs (`x.pt`, `expected.pt`, `model.safetensors`) are **not
committed** (see `cases/rmsnorm_fp8/.gitignore`) and are **regenerated inside a
Modal CPU container on every run**. The sequence is fixed, and generation runs
before any device is requested:

1. Only the case *source* travels, to this run's own Volume prefix
   `experiments/<run-id>/cases/rmsnorm_fp8/`: `config.json`,
   `make_reference.py`, `rms_norm_quant.py`, `rms_norm_quant_opt.py`, `twin.py`,
   `benchmark.py`, and `agent_candidate.py` when it exists. Nothing a previous
   machine produced is read or copied — the three reference files are absent
   from that list on purpose.
2. A Modal CPU Function (no `gpu=` argument) runs `make_reference.py` in a
   private `/tmp` working directory and validates the result against the case
   contract: `x.pt` `[2, 7168]` bf16, `expected.pt` a 2-element
   `[quant (2, 7168) float8_e4m3fn, scale (2, 56) float32]`, and
   `model.safetensors` key `gamma` `[1, 7168]` bf16. It records the SHA-256 of
   each file, the digests of `make_reference.py` and `config.json`, the
   provenance the script wrote, and the installed torch/transformers/safetensors
   versions.
3. Only after that succeeds does a GPU Function run. It reloads the run's
   manifest, copies the case into a private directory, and re-checks every
   reference digest, so it can never benchmark against another run's data.

A failed generation is recorded as an error, and the host then stops with a
`SystemExit` whose text contains `reference generation failed` and
`No GPU was allocated and no experiment was run.` A GPU Function is never
reached; `tests/test_run_experiments.py` asserts that the only remote call made
in that case is the generator itself. A fresh clone therefore needs no local
`torch` and no reference data copied from another machine.

`--run-id` names the run explicitly; without it one is generated, and the local
record lands in `results/experiments-<run-id>/experiments.json`.

### What a recorded run looks like

Numbers below come from the accepted run `20261029-082303-087224`, so they are an
example of the record, not a promise about a future run:

| File | Size | SHA-256 (prefix) |
|---|---|---|
| `x.pt` | 30015 | `4d3206a53565d4ce…` |
| `expected.pt` | 16685 | `7adfcdf1207747c9…` |
| `model.safetensors` | 14416 | `4d3e4270115742c0…` |

Generators as digested: `make_reference.py` `207b6379beef8f9e…`, `config.json`
`67ba5386a7d28983…`. Container versions at generation time: torch
`2.9.1+cu128`, transformers `5.14.1`, safetensors `0.8.0`. Note that the
Triton in this container is **3.5.1** — the `3.8.0` in the `triton/` subproject
section is a different environment and does not apply here.

The benchmark arm records both what it measured and what it compared against
(100 repeats, 10 warmup, `torch.cuda.Event` timing): torch baseline median
`0.130160 ms` (p10 `0.127296` / p90 `0.138496`) versus the Triton twin median
`0.030288 ms` (p10 `0.028864` / p90 `0.032928`). Alongside the timings it records
agreement with the reference: `quant_equal_fraction` `0.9999995` and
`scale_max_abs_diff` `0.0`. Nothing in the record predicts performance — it
states what the device that ran actually measured and what it was measured
against.

## 8 · Download and verify the evidence

The run prints the exact command:

```bash
modal volume get tilefoundry-artifacts \\
  runs/<run-id>/tilefoundry-demo-<run-id>.tar.gz \\
  ./results/<run-id>/tilefoundry-demo-<run-id>.tar.gz
```

Then check the downloaded bundle against the result the run printed:

```bash
python verify_bundle.py \\
  results/<run-id>/tilefoundry-demo-<run-id>.tar.gz \\
  --result results/<run-id>/remote-result.json
```

`VERIFY OK` means: the archive digest matches what the container printed, the
archived manifest agrees with the printed result field by field, re-deriving the
verdict from the archived evidence reproduces it, the JUnit report shows the one
expected test, and the built library and generated CUDA source are present.

## 9 · Bundle layout

```text
tilefoundry-demo-<run-id>/
  manifest.json              # identity, versions, task_stages, per-stage stage_details, digests
  results/
    analysis.json            # the HIR report the CLI wrote (hir_analyze stage)
    check.json               # the comparison report (hir_check stage only)
    junit.xml                # pytest's own record (gpu_test stage)
    test-results.json        # what the observing plugin saw
    stage-status.json        # the reduced per-stage verdict
  logs/
    environment.log
    analysis.log
    check.log                # hir_check stages only
    compile-and-test.log     # full pytest output
    exception.txt            # only when a stage raised
  generated/
    <test_file>/<test>/NN_<pass>/{before,after}.txt   # per-pass IR, before/after each pass
    <test_file>/<test>/build/                         # CMakeLists.txt, module.device_0.cu, cmake logs
    build/                                            # device_0.cu, host.cpp, CMakeLists.txt
      build/                                          # the CMake build tree: lib<TirSquare>.so lives here
  build/                                              # the stage build root: TMPDIR plus cache/ (TILEFOUNDRY_CACHE_DIR)
```

On the host, the run writes `results/<run-id>/preflight.json` and
`results/<run-id>/remote-result.json` next to the downloaded archive. The
library's relative path is recorded in the manifest under
`evidence.compile.libraries` (for the square case:
`generated/build/build/libTirSquare.so`), with its size and SHA-256.

## 10 · Container-internal paths are not host dependencies

These absolute paths appear in the code and in container commands. They exist
**only inside the Modal container**, so they are not path dependencies of the
machine that starts the run:

| Path | What it is |
|---|---|
| `/opt/tilefoundry` | where the image installs the uploaded TileFoundry snapshot |
| `/opt/cutlass/include` | CUTLASS headers baked into the image (`TILEFOUNDRY_CUTLASS_INCLUDE`) |
| `/artifacts` | the Modal Volume mount point inside the container |
| `/tmp` | container-local scratch space |
| `/tmp/tilefoundry-demo` | the per-run evidence directory the container function creates |
| `/cases` | not a mount point: the `cases` segment of the run-scoped Volume prefix `experiments/<run-id>/cases/<case>`, whose container view is under `/artifacts`. It is listed here because the token appears in the code; it is never a host path |

The host reaches the archived bytes through `modal volume get`, never through
those paths. Nothing outside the container reads or writes them, and no command
in this README uses them.

## 11 · Historical provenance records

Two files are records of **past** runs and are deliberately not rewritten:

- `cases/rmsnorm_fp8/tutorial-provenance.json` — the TileFoundry commit, its
  tutorial sources, and the CLI surface observed at that freeze. It names the
  checkout path it was produced from; that path is history, not a dependency.
- `cases/rmsnorm_fp8/reference-provenance.json` — the generation parameters
  (shape, dtype, seed, torch version) of the reference data a previous run
  produced. It describes data that is **not** in git; the numbers are a record,
  not an input.

A new run writes its own `reference-provenance.json` beside the data it
generates, and `run_experiments.py` copies that fresh record into
`experiments.json` under `reference.provenance`. These two tracked files are
never presented as the current run's provenance, and treating either as an
input would defeat the point of regenerating per run.

## 12 · Failure triage

| Symptom | Where to look |
|---|---|
| `no TileFoundry checkout was found` | section 4: set `TILEFOUNDRY_REPO` or clone beside this repo |
| `no-GPU preflight failed` | `results/<run-id>/preflight.json`, then `logs/environment.log` |
| `analysis_status` is `FAIL` | `logs/analysis.log`; `results/analysis.json` may not exist |
| `compile_status` is `ERROR` | `logs/compile-and-test.log`, then the `cmake-*.stderr` dumps under `generated/` |
| `correctness_status` is `FAIL` | `results/test-results.json` → `correctness.comparisons` |
| a skip reported as `FAIL` | the device was not really a usable H200; check `gpu` in the manifest |
| `overall_status` is `NOT_RUN` | a stage never executed: read `overall_reason` for the first missing one |
| `reference generation failed` | no GPU was allocated and no record was written; the message quotes the generator error, and `results/experiments-<run-id>/` is empty because the run stopped before the record |

## 13 · Local tests (no GPU, no torch, no Modal client)

```bash
cd "$(git rev-parse --show-toplevel)/tilefoundry_demo"
python -m pytest tests -q
```

The suite needs no GPU, no Modal client, no network, and no TileFoundry install.

One prerequisite is real, because `run_demo.py` freezes its source snapshot at
import time and the tests that import it do so at module scope: a TileFoundry
checkout must be resolvable (section 4). The files that need one are
`tests/test_run_demo.py`, `tests/test_checkout_resolution.py`, and
`tests/test_task_runner_match.py`. Either export `TILEFOUNDRY_REPO`, or clone
TileFoundry beside this repository as section 4 describes. Without one those
three files error at import with the `no TileFoundry checkout was found`
message; every other test file passes regardless. Nothing is installed from
that checkout — it is only read and hashed:

```bash
export TILEFOUNDRY_REPO="$HOME/workspace/TileFoundry"   # or use the repo-internal sibling
cd "$(git rev-parse --show-toplevel)/tilefoundry_demo"
python -m pytest tests -q
```

Among other properties the suite asserts, one file audits this portability directly:
`tests/test_no_machine_paths.py` scans the tracked runtime files and refuses any
absolute home path, any conda path or activation, any named Modal profile
requirement, and any absolute `cd` in a documented command, while asserting the
container-internal paths of section 10 and the two historical provenance files
of section 11 as explicitly permitted. It also refuses a tracked reference-data
binary, since a fresh clone regenerates those on a Modal CPU container.

## 14 · Reproducing exactly what a run used

`manifest.json` records the git commit, branch, dirty state, the CUTLASS
submodule SHA, the injected package version, and `source_snapshot_sha256` — the
digest of the exact file set uploaded. `source_check` in the same file compares
that inventory against the tree the container installed and ran, both before and
after installation.

The package version is `0.0.0+demo.<digest prefix>`: derived from the snapshot,
never from the current date, so the same tree always installs the same version
and two different trees never collide.

## 15 · What a PASS cannot survive

Each of these is checked by a local test, so the claim is executable rather than
asserted:

- a skipped or `xfail`ed test, or a JUnit report with more than one case;
- a `torch.testing.assert_close` that did not run, or one with non-zero tolerance;
- a compile stage with no CUDA source, unclean CMake logs, no library, or a
  library with no digest or the wrong name;
- codegen for an architecture other than `sm_90`, or no recorded architecture;
- a device that is not an H200 at compute capability 9.0;
- a package that resolves outside the uploaded tree, was installed from an index,
  or sits in a tree that drifted from the digested inventory;
- core and target versions that differ;
- an archive that never reached the Volume, or whose digest does not match the
  printed value;
- a stage that was declared but never executed, reported as anything better than
  `NOT_RUN`.
