import os
import time

import torch
import triton

from kernels.softmax import softmax_kernel


N_ROWS = 4096
N_COLS = 256
BLOCK_SIZE = 256


def triton_softmax(x: torch.Tensor):
    assert x.is_cuda
    assert x.dim() == 2
    assert x.shape[1] == N_COLS, f"expected last dim {N_COLS}, got {x.shape[1]}"

    output = torch.empty_like(x)
    n_rows = x.shape[0]
    grid = (n_rows,)

    softmax_kernel[grid](
        x,
        output,
        N_COLS=N_COLS,
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


def test_softmax():
    x = torch.randn(N_ROWS, N_COLS, device="cuda", dtype=torch.float32)
    # Exercise the numerically interesting regime: large magnitudes where a
    # naive exp() would overflow and the row-max shift is load-bearing.
    x[0, :] = 1000.0

    study_mode = bool(os.environ.get("TRITON_STUDY_RESULTS_DIR"))

    if study_mode:
        torch.cuda.synchronize()
        cold_start = time.perf_counter_ns()

    actual = triton_softmax(x)

    if study_mode:
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000.0
    else:
        cold_ms = None

    expected = torch.softmax(x, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    # Every output row must sum to 1 and stay finite.
    assert torch.isfinite(actual).all(), "softmax produced non-finite values"
    row_sums = actual.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
    max_error = (actual - expected).abs().max().item()

    if study_mode:
        # The JITFunction has already produced its in-process compiled kernel.
        # Repeated calls reuse it; do_bench measures the warm launch/execution path.
        warm_ms = float(triton.testing.do_bench(lambda: triton_softmax(x)))
        _record_study_metrics(
            kernel="softmax_kernel",
            libdevice_functions=["exp", "fma", "div_rn"],
            n_rows=N_ROWS,
            n_cols=N_COLS,
            dtype=str(x.dtype),
            block_size=BLOCK_SIZE,
            cold_jit_and_first_run_ms=cold_ms,
            warm_runtime_ms=warm_ms,
            max_error=max_error,
        )
        print(f"cold JIT + first run: {cold_ms:.3f} ms")
        print(f"warm runtime: {warm_ms:.6f} ms")

    print(f"max error: {max_error}")
