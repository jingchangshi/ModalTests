import os
import time

import torch
import triton

from kernels.add import add_kernel


N_ELEMENTS = 98432
BLOCK_SIZE = 256


def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda
    assert y.is_cuda
    assert x.shape == y.shape

    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def _record_study_metrics(**metrics):
    results_dir = os.environ.get("TRITON_STUDY_RESULTS_DIR")
    if not results_dir:
        return

    import json
    from pathlib import Path

    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "kernel_metrics.jsonl").open("a") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")


def test_add():
    x = torch.randn(N_ELEMENTS, device="cuda", dtype=torch.float32)
    y = torch.randn(N_ELEMENTS, device="cuda", dtype=torch.float32)

    study_mode = bool(os.environ.get("TRITON_STUDY_RESULTS_DIR"))

    if study_mode:
        torch.cuda.synchronize()
        cold_start = time.perf_counter_ns()

    actual = triton_add(x, y)

    if study_mode:
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000.0
    else:
        cold_ms = None

    expected = x + y
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    max_error = (actual - expected).abs().max().item()

    if study_mode:
        # The JITFunction has already produced its in-process compiled kernel.
        # Repeated calls reuse it; do_bench measures the warm launch/execution path.
        warm_ms = float(triton.testing.do_bench(lambda: triton_add(x, y)))
        _record_study_metrics(
            kernel="add_kernel",
            n_elements=N_ELEMENTS,
            dtype=str(x.dtype),
            block_size=BLOCK_SIZE,
            cold_jit_and_first_run_ms=cold_ms,
            warm_runtime_ms=warm_ms,
            max_error=max_error,
        )
        print(f"cold JIT + first run: {cold_ms:.3f} ms")
        print(f"warm runtime: {warm_ms:.6f} ms")

    print(f"max error: {max_error}")
