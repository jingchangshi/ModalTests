import triton
import triton.language as tl
import triton.language.extra.libdevice as tld


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-wise softmax over a (num_rows, N_COLS) fp32 tensor.

    Numeric strategy (matches standard softmax):
      1. row max (for exp overflow stability)
      2. exp(x - max) computed with libdevice exp (__nv_expf)
      3. sum of exponentials with libdevice fma (__nv_fmaf)
      4. multiply by reciprocal sum with libdevice div_rn (__nv_fdiv_rn)

    The grid is 1D; BLOCK_SIZE covers the row length, so one program
    handles one whole row and never needs a loop or cross-program sync.
    """
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    offsets = pid * N_COLS + cols
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))

    row_max = tl.max(x, axis=0)
    centered = x - row_max
    exp_x = tld.exp(centered)

    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # fma(exp_x, 1.0, acc) == acc + exp_x, but goes through libdevice __nv_fmaf.
    # Using fma instead of tl.sum's internal adds keeps the reduction on the
    # libdevice path (the sum itself is still a tl reduction).
    exp_sum = tl.sum(tld.fma(exp_x, 1.0, zero), axis=0)

    inv_sum = tld.div_rn(1.0, exp_sum)
    output = exp_x * inv_sum

    tl.store(output_ptr + offsets, output, mask=mask)
