#!/usr/bin/env python3
"""The runtime twin of the fused step, as one Triton kernel.

Extracted from ``docs/tutorial/optimize.md`` at the frozen TileFoundry SHA. The only
change from the tutorial text is the import: the tutorial's twin reads its constants and
its ``Fused`` reference from ``rms_norm_quant``, and in this case that module is
``rms_norm_quant_opt`` so the two placement variants and this twin share one definition of
the step.

The kernel is compiled by the **Triton** compiler. ``@runtime_module(Fused)`` only tells
TileFoundry which authored HIR this twin claims to implement; it is the reference the
checker compares against, not a code generator for this kernel.

The row count comes from the tensors the checker passes in, not from a module constant.
The authored HIR in ``rms_norm_quant_opt`` fixes ``ROWS = PAIR * CTAS`` for the analysis
experiment, which describes 264 rows; the checked inputs here carry 2. Hard-coding the
analysis row count made the twin reject its own inputs with a shape mismatch, so the grid
and the buffers follow ``a.shape[0]``.
"""

import torch
import triton
import triton.language as tl
from rms_norm_quant_opt import BLOCK, BLOCKS, EPS, FP8_MAX, H, Fused

from tilefoundry.runtime import runtime_func, runtime_module

_TILE = triton.next_power_of_2(BLOCKS)


@triton.jit
def _rms_norm_quant(
    a_ptr, gamma_ptr, q_ptr, s_ptr, eps, fp8_max,
    H: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr, TILE: tl.constexpr,
):
    """One row per program: normalize it, then quantize it block by block."""
    row = tl.program_id(0)
    blocks = tl.arange(0, TILE)[:, None]
    lanes = tl.arange(0, BLOCK)[None, :]
    offsets = blocks * BLOCK + lanes
    live = blocks < BLOCKS

    a = tl.load(a_ptr + row * H + offsets, mask=live, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offsets, mask=live, other=0.0)
    normed = (a * tl.rsqrt(tl.sum(a * a) / H + eps)).to(tl.bfloat16) * gamma

    held = normed.to(tl.float32)
    scale = tl.max(tl.abs(held), axis=1, keep_dims=True) * (1.0 / fp8_max)
    quant = tl.clamp(held / scale, -fp8_max, fp8_max).to(tl.float8e4nv)

    tl.store(q_ptr + row * H + offsets, quant, mask=live)
    tl.store(s_ptr + row * BLOCKS + blocks, scale, mask=live)


@runtime_module(Fused)
class Fast:
    @runtime_func
    def rms_norm_quant(self, a, gamma):
        rows = a.shape[0]
        quant = torch.empty((rows, H), dtype=torch.float8_e4m3fn, device=a.device)
        scale = torch.empty((rows, BLOCKS), dtype=torch.float32, device=a.device)
        _rms_norm_quant[(rows,)](
            a, gamma, quant, scale, EPS, FP8_MAX,
            H=H, BLOCK=BLOCK, BLOCKS=BLOCKS, TILE=_TILE,
        )
        return quant, scale
