import torch
import triton

from kernels.add import add_kernel


def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda
    assert y.is_cuda
    assert x.shape == y.shape

    output = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )

    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=256,
    )

    return output


def test_add():
    n = 98432

    x = torch.randn(
        n,
        device="cuda",
        dtype=torch.float32,
    )

    y = torch.randn(
        n,
        device="cuda",
        dtype=torch.float32,
    )

    actual = triton_add(x, y)

    expected = x + y

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )

    print(
        "max error:",
        (actual - expected).abs().max().item(),
    )

