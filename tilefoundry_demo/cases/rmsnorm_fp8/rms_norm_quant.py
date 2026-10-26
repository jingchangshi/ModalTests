#!/usr/bin/env python3
"""One published step, authored as HIR: RMS norm, then blockwise FP8 quantization.

The case is DeepSeek-V4 Flash's published step. ``EPS``, ``BLOCK`` and the fp8 format are
fields of the model's own config, not constants to remember; ``H`` is this example's row
width. Everything else is the operation, written as HIR so ``tilefoundry check`` and
``tilefoundry analyze`` can read it.

``Wrong`` is deliberately wrong, and it is wrong in one place: it multiplies by ``gamma``
while the values are still f32 and rounds to bf16 afterwards. The reference rounds to bf16
first and scales second. The two orders differ by one bf16 ulp on most elements, which is
enough to move some of them across an fp8 code boundary.
"""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

ROWS = 2
H = 7168  # this example's row width, not a field of the model
BLOCK = 128  # config.json: quantization_config.weight_block_size[1]
BLOCKS = H // BLOCK
FP8_MAX = 448.0  # the largest finite fp8e4m3, because fmt says e4m3
EPS = 1e-6  # config.json: rms_norm_eps


@module(
    entry="rms_norm_quant",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class Wrong:
    """The step as I remember it: scale by gamma, then land in bf16."""

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        rows = tf.cast(a, "f32")
        mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
        normed = tf.cast(rows * tf.rsqrt(mean + EPS) * tf.cast(gamma, "f32"), "bf16")
        blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
        scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
        quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
        return tf.reshape(quant, (ROWS, H)), tf.reshape(scale, (ROWS, BLOCKS))


@module(
    entry="rms_norm_quant",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class RmsNormQuant:
    """The step as the model defines it: land in bf16, then scale by gamma."""

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        rows = tf.cast(a, "f32")
        mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
        normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * gamma
        blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
        scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
        quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
        return tf.reshape(quant, (ROWS, H)), tf.reshape(scale, (ROWS, BLOCKS))
