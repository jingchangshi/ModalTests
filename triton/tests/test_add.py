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


def test_add(triton_study_metrics):
    # Input construction is intentionally outside the cold-JIT interval.
    x = torch.randn(N_ELEMENTS, device="cuda", dtype=torch.float32)
    y = torch.randn(N_ELEMENTS, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    cold_start = time.perf_counter_ns()
    actual = triton_add(x, y)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000.0

    expected = x + y
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    # The kernel is already compiled here. do_bench therefore measures warm
    # launch/execution rather than the JIT path.
    warm_ms = triton.testing.do_bench(lambda: triton_add(x, y))
    max_error = (actual - expected).abs().max().item()

    triton_study_metrics(
        kernel="add_kernel",
        n_elements=N_ELEMENTS,
        dtype=str(x.dtype),
        block_size=BLOCK_SIZE,
        cold_jit_and_first_run_ms=cold_ms,
        warm_runtime_ms=float(warm_ms),
        max_error=max_error,
    )

    print(f"cold JIT + first run: {cold_ms:.3f} ms")
    print(f"warm runtime: {warm_ms:.6f} ms")
    print(f"max error: {max_error}")
