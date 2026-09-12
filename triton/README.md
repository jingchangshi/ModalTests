# Triton GPU compiler studies on Modal

> 供自动化 agent 阅读的工作区说明（环境、运行命令、注意事项、已验证基线）见仓库根的 [AGENTS.md](../AGENTS.md)。

This directory has two intentionally separate workflows:

- `run_test.py`: correctness/regression execution and stage artifact capture.
- `study_compiler.py`: compiler research runs with structured metadata, pipeline inspection, timing, and immutable artifact bundles.

## Compiler study

Stage-level study (recommended default):

```bash
cd triton
modal run study_compiler.py --test-path tests/test_add.py --detail stage
```

Pass-level study (much larger output):

```bash
modal run study_compiler.py --test-path tests/test_add.py --detail pass
```

The command prints the exact `modal volume get` command for the resulting archive.

## Bundle layout

Each archive contains roughly:

```text
triton-study-<run-id>/
├── manifest.json
├── dumps/
│   └── <triton-cache-key>/
│       ├── *.ttir
│       ├── *.ttgir
│       ├── *.llir
│       ├── *.ptx
│       ├── *.cubin
│       └── *.sass
├── results/
│   ├── environment.json
│   ├── pipeline.json
│   ├── kernel_metrics.jsonl
│   └── pytest_wall_times.jsonl
└── logs/
    ├── compiler.log
    └── mlir-pass-ir.log       # pass detail only, when supported
```

## Timing semantics

`cold_jit_and_first_run_ms` is deliberately not called pure compile time. It is the wall-clock interval around the first kernel invocation followed by `torch.cuda.synchronize()`, so it includes Triton JIT compilation, CUDA module loading, launch overhead, and first execution.

`warm_runtime_ms` is collected with `triton.testing.do_bench` after the kernel is compiled.

`MLIR_ENABLE_TIMING=1` and `LLVM_ENABLE_TIMING=1` preserve compiler pass-manager timing in `logs/compiler.log`. Use these together with cold/warm measurements rather than subtracting warm runtime from cold time and treating the result as exact compilation time.

## Detail modes

`stage` keeps the normal Triton stage artifacts and compiler timing. It is suitable for most experiments and comparisons.

`pass` additionally enables full MLIR pass-boundary dumping and LLVM IR dumping. It can generate large logs, so use it for focused kernel investigations rather than broad regression suites.

## Storage model

Triton writes compiler scratch data under `/tmp`, not directly into the Modal Volume. After the subprocess exits, the runner inventories stable files, builds an immutable `tar.gz`, copies that single archive into `triton-artifacts`, and commits the Volume. This avoids persisting/downloading transient `tmp.pid_*` files.
