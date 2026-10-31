#!/usr/bin/env python3
"""The same RMSNorm + FP8 quantize step, written so 'schedule finalize' can lower it.

'rms_norm_quant_opt.py' is the authored HIR and is left untouched. It cannot be
finalized as written, for two independent structural reasons:

  1. Instruction selection. Every tf.reshard has six registered TIR carriers
     (T.copy, T.copy_async, T.copy_async_tensor, T.copy_reduce_async_tensor,
     T.ldmatrix, T.stmatrix), so sole_candidate refuses to pick one and lowering
     stops with "no automatic selection". Naming the instruction fixes it.
  2. Mesh scope of a reduction. T.reduce's operands are declared against
     _ONE_MESH = MeshPattern(("thread",), ...) in ir/tir/reduce.py, and
     MeshPattern requires the picked topology set to equal the pattern set
     (ir/pattern/match.py). A ShardLayout naming both cta and thread can
     therefore never match. The cta index has to live in the pointer slice, not
     in the rmem layout: hence the row window a[m0:m0 + PAIR, :, :].

Because the mesh-crossing tf.reshape (2-D rows -> 3-D blocks) is also not
affine-lowerable, the parameters are declared here at (ROWS, BLOCKS, BLOCK):
that is the same row-major buffer as the original (ROWS, H), and the original
HIR already builds exactly this view with
tf.reshape(tf.cast(held, "f32"), (ROWS, BLOCKS, BLOCK)).

Nothing else changes: same constants, dtypes, operator order, and reduction
semantics (MEAN over all of H, ABS_MAX over the 128-wide quantization block,
times 1/448.0, clamped to +-448, cast to fp8e4m3).
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

PAIR = 2  # rows one CTA normalizes
CTAS = 132  # one CTA per SM of the H200, so the grid is one wave
ROWS = PAIR * CTAS
H = 7168  # this example's row width, not a field of any model
BLOCK = 128  # config.json: quantization_config.weight_block_size[1]
BLOCKS = H // BLOCK
FP8_MAX = 448.0  # the largest finite fp8e4m3, because fmt says e4m3
INV_FP8_MAX = 1.0 / FP8_MAX  # hoisted: a BinaryKind.DIV is not a shape expression
EPS = 1e-6  # config.json: rms_norm_eps

K = BLOCKS // 4  # the register block each m.y lane holds
VEC = BLOCK // 32  # elements per m.t lane inside one block

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", CTAS)
_THREAD = Topology("thread", PAIR * 4 * 32)


@module(entry="rms_norm_quant", target=_H200, topologies=(_CTA, _THREAD))
class FusedScheduled:
    """One func: the normalized rows never leave a register."""

    @func
    def rms_norm_quant(
        a: Tensor[(ROWS, BLOCKS, BLOCK), "bf16"],
        gamma: ConstTensor[(1, BLOCKS, BLOCK), "bf16"],
    ):
        with Mesh(("cta",), (CTAS,), ("tile",)) as cta:
            m0 = cta.tile * PAIR
            with Mesh(("thread",), (PAIR, 4, 32), ("x", "y", "t")) as m:
                held_dst = tf.empty(
                    Tensor[
                        (PAIR, BLOCKS, BLOCK),
                        "bf16",
                        (PAIR @ m.x, 4 @ m.y, K, 32 @ m.t, VEC),
                        "rmem",
                    ]
                )
                held = tf.schedule((a[m0:m0 + PAIR, :, :], held_dst), op=T.copy())
                scaling_dst = tf.empty(
                    Tensor[
                        (1, BLOCKS, BLOCK),
                        "bf16",
                        (1, 4 @ m.y, K, 32 @ m.t, VEC),
                        "rmem",
                    ]
                )
                scaling = tf.schedule((gamma, scaling_dst), op=T.copy())
                rows = tf.cast(held, "f32")
                mean_dst = tf.empty(
                    Tensor[(PAIR, 1, 1), "f32", (PAIR @ m.x, 1, 1, 1, 1), "rmem"]
                )
                mean = tf.schedule(
                    (tf.square(rows), mean_dst),
                    op=T.reduce(axes=(1, 2), keepdim=True, kind=ReduceKind.MEAN),
                )
                normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * scaling
                blocks = tf.cast(normed, "f32")
                scale_dst = tf.empty(
                    Tensor[
                        (PAIR, BLOCKS, 1),
                        "f32",
                        (PAIR @ m.x, 4 @ m.y, K, 1, 1),
                        "rmem",
                    ]
                )
                scale = (
                    tf.schedule(
                        (blocks, scale_dst),
                        op=T.reduce(axes=(2,), keepdim=True, kind=ReduceKind.ABS_MAX),
                    )
                    * INV_FP8_MAX
                )
                quant = tf.cast(
                    tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3"
                )
                return (
                    tf.insert_slice(
                        tf.zeros(Tensor[(ROWS, BLOCKS, BLOCK), "fp8e4m3"]),
                        quant,
                        (m0, 0, 0),
                    ),
                    tf.insert_slice(
                        tf.zeros(Tensor[(ROWS, BLOCKS, 1), "f32"]),
                        scale,
                        (m0, 0, 0),
                    ),
                )
