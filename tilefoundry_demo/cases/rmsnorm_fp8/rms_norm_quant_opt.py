#!/usr/bin/env python3
"""The step from the migrate page, placed two ways: with a boundary, and without."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

PAIR = 2  # rows one CTA normalizes
CTAS = 132  # one CTA per SM of the H200, so the grid is one wave
ROWS = PAIR * CTAS
H = 7168  # this example's row width, not a field of any model
BLOCK = 128  # config.json: quantization_config.weight_block_size[1]
BLOCKS = H // BLOCK
FP8_MAX = 448.0  # the largest finite fp8e4m3, because fmt says e4m3
EPS = 1e-6  # config.json: rms_norm_eps

_H200 = CudaTarget("nvidia.h200_sxm")
_CTA = Topology("cta", CTAS)
_THREAD = Topology("thread", PAIR * 4 * 32)


@module(entry="rms_norm_quant", target=_H200, topologies=(_CTA, _THREAD))
class Naive:
    """Normalize, hand the rows on, quantize them: two funcs, one boundary."""

    @func
    def norm(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        with Mesh(("cta",), (CTAS,), ("tile",)) as cta:
            with Mesh(("thread",), (PAIR, 4, 32), ("x", "y", "t")) as m:
                held = tf.reshard(a, (ROWS @ (cta.tile, m.x), BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                scaling = tf.reshard(gamma, (1, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                rows = tf.cast(held, "f32")
                mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
                normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * scaling
                return tf.reshard(normed, (ROWS, H), "gmem")

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        with Mesh(("cta",), (CTAS,), ("tile",)) as cta:
            normed = norm(a, gamma)  # noqa: F821
            with Mesh(("thread",), (PAIR, 4, 32), ("x", "y", "t")) as m:
                held = tf.reshard(normed, (ROWS @ (cta.tile, m.x), BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                blocks = tf.reshape(tf.cast(held, "f32"), (ROWS, BLOCKS, BLOCK))
                scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
                quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
                return (
                    tf.reshard(tf.reshape(quant, (ROWS, H)), (ROWS, H), "gmem"),
                    tf.reshard(tf.reshape(scale, (ROWS, BLOCKS)), (ROWS, BLOCKS), "gmem"),
                )


@module(entry="rms_norm_quant", target=_H200, topologies=(_CTA, _THREAD))
class Fused:
    """One func: the normalized rows never leave a register."""

    @func
    def rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"]):
        with Mesh(("cta",), (CTAS,), ("tile",)) as cta:
            with Mesh(("thread",), (PAIR, 4, 32), ("x", "y", "t")) as m:
                held = tf.reshard(a, (ROWS @ (cta.tile, m.x), BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                scaling = tf.reshard(gamma, (1, BLOCKS @ m.y, BLOCK @ m.t), "rmem")
                rows = tf.cast(held, "f32")
                mean = tf.reduce(tf.square(rows), (-1,), True, ReduceKind.MEAN)
                normed = tf.cast(rows * tf.rsqrt(mean + EPS), "bf16") * scaling
                blocks = tf.reshape(tf.cast(normed, "f32"), (ROWS, BLOCKS, BLOCK))
                scale = tf.reduce(blocks, (-1,), True, ReduceKind.ABS_MAX) * (1.0 / FP8_MAX)
                quant = tf.cast(tf.clamp(blocks / scale, -FP8_MAX, FP8_MAX), "fp8e4m3")
                return (
                    tf.reshard(tf.reshape(quant, (ROWS, H)), (ROWS, H), "gmem"),
                    tf.reshard(tf.reshape(scale, (ROWS, BLOCKS)), (ROWS, BLOCKS), "gmem"),
                )
