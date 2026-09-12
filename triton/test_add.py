import torch
import triton
import triton.language as tl

@triton.jit
def triton_add(in_ptr0, in_ptr1, out_ptr0,
               stride: tl.constexpr,
               XS: tl.constexpr, XSS: tl.constexpr):
    pid = tl.program_id(0)
    poff = pid * XS
    xbase = tl.arange(0, XSS)
    num_loops = (XS + XSS - 1) // XSS
    for loop_idx in range(num_loops):
        xoff = poff + loop_idx * XSS
        xidx = xoff + xbase
        tmp0 = tl.load(in_ptr0 + xidx * stride)
        tmp1 = tl.load(in_ptr1 + xidx * stride)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + xidx, tmp2)

def triton_func(buf0, buf1, stride):
    N0, N1 = buf1.size()
    buf2 = torch.empty_like(buf0)
    # grid = (48, 1, 1)
    grid = (1, 1, 1)
    XSS = N1
    # XS = N0 // 48 * XSS
    XS = N0 // 1 * XSS
    triton_add[grid](buf0, buf1, buf2, stride, XS, XSS)
    return buf2

DEV = 'cuda'
# N0 = 480
# N1 = 4096
N0 = 32
N1 = 32
stride = 1
buf0 = torch.rand((N0, N1 * stride), dtype=torch.float32, device=DEV)
buf1 = torch.rand((N0, N1), dtype=torch.float32, device=DEV)

cal = triton_func(buf0, buf1, stride)
print(cal)

