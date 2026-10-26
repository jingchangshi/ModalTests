# TileFoundry on a Modal H200 — minimal end-to-end demo

Runs the **local** TileFoundry checkout on a remote NVIDIA H200 through Modal:
HIR static analysis, TIR → CUDA codegen, NVCC/CMake build, a real kernel
execution, a PyTorch comparison, and a downloadable evidence bundle.

The machine that starts the run needs no GPU, no CUDA, and no TileFoundry
install. It needs the `modal` client and a Modal account with H200 access.

## 1 · What it proves, and what it does not

Covered:

| Stage | Evidence it produces |
|---|---|
| environment | Image installs the uploaded snapshot; both distributions resolve inside it; `nvidia.h200_sxm` is registered; nvcc/CMake/ninja/g++ present |
| gpu | `torch.cuda.get_device_properties` reports an H200 at compute capability 9.0 |
| analysis | `tilefoundry analyze` exits 0 and writes JSON naming `compute-cost` and `memory` |
| compile | One `tilefoundry.compile` call returns `TirSquare.square_host`; CUDA source, CMake logs, and the built `.so` are captured |
| execution | The compiled runtime is called and the device synchronized |
| correctness | The upstream test's own `torch.testing.assert_close` comparison passed |
| pytest | Exactly one test ran and passed, exit code 0 |
| archive | The evidence tarball reached the Volume and its digest/size are recorded |

Explicitly **not** covered: operator generation or optimisation, model weights,
benchmarks, multi-GPU, other GPUs, the full TileFoundry test suite, and any
claim that a *skipped* or *xfail* test is a success. HIR analysis is reported as
its own stage and is never treated as proof that a kernel ran.

## 2 · Architecture and data flow

```text
local machine                                    Modal H200 container
─────────────                                    ────────────────────
git ls-files ─┐
              ├─► staging dir ──► SHA-256 ─┐
untracked ────┘                             │
                                            ▼
TileFoundry/Dockerfile ──► from_dockerfile ──► CUDA 12.9 devel + python3.12
                              + pip: setuptools-scm, ortools, transformers
                              + add_local_dir(staging, copy=True)
                              + pip install -e . -e targets/nvidia
                                            │
local_entrypoint ──── preflight.remote ────►│ environment + origin check  (no GPU)
                                            │
                    run_on_h200.remote ────►│ analyze ─┐
                                            │          ├─ DumpScope → generated/
                                            │ pytest ──┘  (CUDA source, CMake logs)
                                            │ compile → execute → compare
                                            │ tar.gz ──► Volume tilefoundry-artifacts
                    modal volume get ◄──────┘
```

Two properties matter:

- **The uploaded bytes are the hashed bytes.** One staging directory is both
  digested and uploaded; the container re-inventories the installed tree and
  compares it with the recorded inventory (`source_check`).
- **A partial run cannot be reported as `PASS`.** Every stage is reduced
  independently; archiving is judged separately, because a tarball cannot state
  the outcome of writing itself.

## 3 · Prerequisites

- `modal` client (verified with 1.5.5) and a working Modal profile.
- Python 3.11+ for the client. No local torch, CUDA, or TileFoundry.
- A TileFoundry checkout at `/home/shijingchang/workspace/TileFoundry`, or set
  `TILEFOUNDRY_REPO`.
- A Modal account with H200 capacity.

Check the profile without printing secrets:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate triton
modal profile current
```

## 4 · Run the demo

```bash
cd /home/shijingchang/workspace/modal/tilefoundry_demo
modal run run_demo.py
```

The first run builds the image (several minutes, no GPU). Later runs reuse the
Dockerfile and pip layers and rebuild only the source layer.

### No-GPU preflight only

```bash
modal run run_demo.py --cpu-only
```

Proves the image installs the snapshot, both packages import from it, and the
H200 target is registered. **Always run this after changing dependencies** — it
costs no GPU time.

## 5 · Download and verify the evidence

The run prints the exact command:

```bash
modal volume get tilefoundry-artifacts \
  runs/<run-id>/tilefoundry-demo-<run-id>.tar.gz \
  ./results/<run-id>/tilefoundry-demo-<run-id>.tar.gz
```

Then check the downloaded bundle against the printed result:

```bash
python verify_bundle.py \
  results/<run-id>/tilefoundry-demo-<run-id>.tar.gz \
  --result results/<run-id>/remote-result.json
```

`VERIFY OK` means: the archive digest matches what the container printed, the
archived manifest agrees with the printed result field by field, re-deriving the
verdict from the archived evidence reproduces it, the JUnit report shows one
passing test, and the built library and generated CUDA source are present.

## 6 · Bundle layout

```text
tilefoundry-demo-<run-id>/
  manifest.json              # identity, versions, per-stage status, artifact digests
  results/
    analysis.json            # the HIR report the CLI wrote
    junit.xml                # pytest's own record of the single test
    test-results.json        # what the observing plugin saw: compile call, runtime call, comparison
    stage-status.json        # the reduced verdict
  logs/
    environment.log
    analysis.log
    compile-and-test.log     # full pytest output
    exception.txt            # only when a stage raised
  generated/
    <test_file>/<test>/NN_<pass>/{before,after}.txt   # per-pass IR
    <test_file>/<test>/build/                          # CMakeLists.txt, module.device_0.cu, cmake logs
    build/build/libTirSquare.so                        # the linked library, copied right after compile
  build/tilefoundry_build_.../                          # the raw build tree
```

## 7 · Failure triage

| Symptom | Where to look |
|---|---|
| `no-GPU preflight failed` | `results/<run-id>/preflight.json`, then `logs/environment.log` |
| `analysis_status: FAIL` | `logs/analysis.log`; `results/analysis.json` may not exist |
| `compile_status: ERROR` | `logs/compile-and-test.log`, then the `cmake-*.stderr` dumps under `generated/` |
| `compile_status: FAIL` on the library name | `evidence.compile.libraries` versus `runtime_name` |
| `correctness_status: FAIL` | `results/test-results.json` → `correctness.comparisons` |
| `pytest` reported as `FAIL` for a skip | the device was not really a usable H200; check `gpu_name` |
| `overall_status: NOT_RUN` | a stage never executed: read `overall_reason` for the first missing one |

`overall_status` is one of `PASS`, `FAIL`, `ERROR`, `BLOCKED`, `NOT_RUN`.

## 8 · Cost control

- The Dockerfile and pip layers are cached; only the source layer changes when a
  `.py` file changes.
- Preflight is a CPU function: a broken image costs no GPU time.
- One GPU function per command, `timeout=1800`, no retries, no concurrency.
- A typical passing run is under a minute of H200 time (observed ~52 s of
  container work, ~38 s of it in pytest).
- Image builds, CPU time, and Volume storage are billed separately from GPU time.

## 9 · Local tests (no GPU, no torch)

```bash
cd /home/shijingchang/workspace/modal/tilefoundry_demo
python -m pytest tests -q
```

73 tests cover the evidence reducer, the inventory/snapshot rules, the plugin's
build capture, the orchestration path with stubbed stages, and the bundle
verifier. They assert one property repeatedly: **no missing, partial, skipped,
or contradictory observation can be reported as `PASS`.**

## 10 · Reproducing exactly what a run used

`manifest.json` records the git commit, branch, dirty state, the CUTLASS
submodule SHA, the injected package version, and `source_snapshot_sha256` — the
digest of the exact file set uploaded. `source_check` in the same file compares
that inventory against the tree the container installed and ran, both before and
after installation.

The package version is `0.0.0+demo.<digest prefix>`: derived from the snapshot,
never from the current date, so the same tree always installs the same version
and two different trees never collide.
