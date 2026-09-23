# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Standalone integration of chunk_delta_h_kda.py.
# Repository-local dependencies from utils.py, op.py, and fla_utils.py are
# inlined below so this file can be moved outside the kda_kernel source tree.

import functools
import inspect
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


# -----------------------------------------------------------------------------
# Inlined from fla_utils.py: only the pieces required by this file.
# -----------------------------------------------------------------------------
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}


def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    last_args: tuple | None = None
    last_kwargs: dict | None = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if FLA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args, strict=False)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


# -----------------------------------------------------------------------------
# Inlined from utils.py: only the pieces required by this file.
# -----------------------------------------------------------------------------
@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


# -----------------------------------------------------------------------------
# Inlined from op.py: exp/exp2 required by the Triton kernels below.
# -----------------------------------------------------------------------------
if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    @triton.jit
    def exp(x):
        return tldevice.fast_expf(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tldevice.exp2(x.to(tl.float32))
else:
    @triton.jit
    def exp(x):
        return tl.exp(x.to(tl.float32))

    @triton.jit
    def exp2(x):
        return tl.math.exp2(x.to(tl.float32))


# -----------------------------------------------------------------------------
# Original chunk_delta_h_kda.py implementation.
# -----------------------------------------------------------------------------
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV})
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, K], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
            if K > 64:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if K > 128:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if K > 192:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        if TRANSPOSE_STATE:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, K), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v = tl.dot(b_w, tl.trans(b_h.to(b_w.dtype)))
        else:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            if TRANSPOSE_STATE:
                b_h *= b_g_last
            else:
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

        if USE_GK:
            if TRANSPOSE_STATE:
                o_k = tl.arange(0, K)
                b_gk_last = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k, mask=(o_k < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h *= exp2(b_gk_last)[None, :]
                else:
                    b_h *= exp(b_gk_last)[None, :]
            else:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
                if K > 64:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
                if K > 128:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
                if K > 192:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        if TRANSPOSE_STATE:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (K, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h += tl.dot(tl.trans(b_v), tl.trans(b_k))
        else:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h1 += tl.dot(b_k, b_v)
            if K > 64:
                p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h2 += tl.dot(b_k, b_v)
            if K > 128:
                p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h3 += tl.dot(b_k, b_v)
            if K > 192:
                p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV})
        for BV in [16, 32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2', 'TRANSPOSE_STATE', 'COMPUTE_O', 'STORE_H'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_h_blockdim64_fused(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    q,
    o,
    A,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    COMPUTE_O: tl.constexpr,
    STORE_H: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_3 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if TRANSPOSE_STATE:
                p_h0_4 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    if COMPUTE_O:
        q += (bos * H + i_h).to(tl.int64) * K
        o += (bos * H + i_h).to(tl.int64) * V
        A += (bos * H + i_h).to(tl.int64) * BT
        m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        if STORE_H:
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            if STORE_H:
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_h3 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            if STORE_H:
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_h4 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            if STORE_H:
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        if COMPUTE_O:
            b_v_i16 = b_v.to(k.dtype.element_ty)
            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            p_gq1 = tl.make_block_ptr(gk, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1))
            b_gq1 = tl.load(p_gq1, boundary_check=(0, 1)).to(tl.float32)
            if USE_EXP2:
                b_qg1 = (b_q1 * exp2(b_gq1)).to(b_q1.dtype)
            else:
                b_qg1 = (b_q1 * exp(b_gq1)).to(b_q1.dtype)
            if TRANSPOSE_STATE:
                b_o = tl.dot(b_qg1, tl.trans(b_h1).to(b_qg1.dtype)) * scale
            else:
                b_o = tl.dot(b_qg1, b_h1.to(b_qg1.dtype)) * scale
            if K > 64:
                p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                p_gq2 = tl.make_block_ptr(gk, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_q2 = tl.load(p_q2, boundary_check=(0, 1))
                b_gq2 = tl.load(p_gq2, boundary_check=(0, 1)).to(tl.float32)
                if USE_EXP2:
                    b_qg2 = (b_q2 * exp2(b_gq2)).to(b_q2.dtype)
                else:
                    b_qg2 = (b_q2 * exp(b_gq2)).to(b_q2.dtype)
                if TRANSPOSE_STATE:
                    b_o += tl.dot(b_qg2, tl.trans(b_h2).to(b_qg2.dtype)) * scale
                else:
                    b_o += tl.dot(b_qg2, b_h2.to(b_qg2.dtype)) * scale
            if K > 128:
                p_q3 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                p_gq3 = tl.make_block_ptr(gk, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_q3 = tl.load(p_q3, boundary_check=(0, 1))
                b_gq3 = tl.load(p_gq3, boundary_check=(0, 1)).to(tl.float32)
                if USE_EXP2:
                    b_qg3 = (b_q3 * exp2(b_gq3)).to(b_q3.dtype)
                else:
                    b_qg3 = (b_q3 * exp(b_gq3)).to(b_q3.dtype)
                if TRANSPOSE_STATE:
                    b_o += tl.dot(b_qg3, tl.trans(b_h3).to(b_qg3.dtype)) * scale
                else:
                    b_o += tl.dot(b_qg3, b_h3.to(b_qg3.dtype)) * scale
            if K > 192:
                p_q4 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                p_gq4 = tl.make_block_ptr(gk, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_q4 = tl.load(p_q4, boundary_check=(0, 1))
                b_gq4 = tl.load(p_gq4, boundary_check=(0, 1)).to(tl.float32)
                if USE_EXP2:
                    b_qg4 = (b_q4 * exp2(b_gq4)).to(b_q4.dtype)
                else:
                    b_qg4 = (b_q4 * exp(b_gq4)).to(b_q4.dtype)
                if TRANSPOSE_STATE:
                    b_o += tl.dot(b_qg4, tl.trans(b_h4).to(b_qg4.dtype)) * scale
                else:
                    b_o += tl.dot(b_qg4, b_h4.to(b_qg4.dtype)) * scale
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_A = tl.where(m_s, b_A, 0.).to(b_v_i16.dtype)
            b_o += tl.dot(b_A, b_v_i16)
            p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * H + last_idx * H + i_h).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g + (bos * H + i_h).to(tl.int64), (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if TRANSPOSE_STATE:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
            else:
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[:, None]
                else:
                    b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                else:
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[:, None]
                    else:
                        b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                else:
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[:, None]
                    else:
                        b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                if TRANSPOSE_STATE:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
                else:
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[:, None]
                    else:
                        b_h4 *= exp(b_gk_last4)[:, None]

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if TRANSPOSE_STATE:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if TRANSPOSE_STATE:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV})
        for BV in [16, 32, 64, 128]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2', 'TRANSPOSE_STATE', 'COMPUTE_O', 'STORE_H'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_h_blockdim128_fused(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    q,
    o,
    A,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    COMPUTE_O: tl.constexpr,
    STORE_H: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([128, BV], dtype=tl.float32)
    if K > 128:
        b_h2 = tl.zeros([128, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (1, K), (0, i_v * BV), (128, BV), (0, 1))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (128, BV), (1, 0))
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if TRANSPOSE_STATE:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (1, K), (128, i_v * BV), (128, BV), (0, 1))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            else:
                p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (128, BV), (1, 0))
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)

    # precompute g/gk base addresses to reduce scalar ops in loop
    if USE_G:
        g += (bos * H + i_h).to(tl.int64)
    if USE_GK:
        gk += (bos * H*K + i_h * K).to(tl.int64)

    if COMPUTE_O:
        q += (bos * H + i_h).to(tl.int64) * K
        o += (bos * H + i_h).to(tl.int64) * V
        A += (bos * H + i_h).to(tl.int64) * BT
        m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        b_h1_i16 = b_h1.to(k.dtype.element_ty)
        if TRANSPOSE_STATE:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (1, K), (0, i_v * BV), (128, BV), (0, 1))
        else:
            p_h1 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (0, i_v * BV), (128, BV), (1, 0))
        if STORE_H:
            tl.store(p_h1, b_h1_i16, boundary_check=(0, 1))
        if K > 128:
            b_h2_i16 = b_h2.to(k.dtype.element_ty)
            if TRANSPOSE_STATE:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (1, K), (128, i_v * BV), (128, BV), (0, 1))
            else:
                p_h2 = tl.make_block_ptr(h + i_t_int64 * H*K*V, (K, V), (V, 1), (128, i_v * BV), (128, BV), (1, 0))
            if STORE_H:
                tl.store(p_h2, b_h2_i16, boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 128), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1_i16)
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 128), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2_i16)
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        b_v_i16 = b_v.to(k.dtype.element_ty)

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v_i16, boundary_check=(0, 1))

        if COMPUTE_O:
            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 128), (1, 0))
            p_gq1 = tl.make_block_ptr(gk, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 128), (1, 0))
            b_q1 = tl.load(p_q1, boundary_check=(0, 1))
            b_gq1 = tl.load(p_gq1, boundary_check=(0, 1)).to(tl.float32)
            if USE_EXP2:
                b_qg1 = (b_q1 * exp2(b_gq1)).to(b_q1.dtype)
            else:
                b_qg1 = (b_q1 * exp(b_gq1)).to(b_q1.dtype)
            b_o = tl.dot(b_qg1, b_h1_i16) * scale
            if K > 128:
                p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 128), (1, 0))
                p_gq2 = tl.make_block_ptr(gk, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 128), (1, 0))
                b_q2 = tl.load(p_q2, boundary_check=(0, 1))
                b_gq2 = tl.load(p_gq2, boundary_check=(0, 1)).to(tl.float32)
                if USE_EXP2:
                    b_qg2 = (b_q2 * exp2(b_gq2)).to(b_q2.dtype)
                else:
                    b_qg2 = (b_q2 * exp(b_gq2)).to(b_q2.dtype)
                b_o += tl.dot(b_qg2, b_h2_i16) * scale
            p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_A = tl.where(m_s, b_A, 0.).to(b_v_i16.dtype)
            b_o += tl.dot(b_A, b_v_i16)
            p_o = tl.make_block_ptr(o, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (last_idx * H).to(tl.int64)).to(tl.float32)
            p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_v_i16 = b_v.to(k.dtype.element_ty)

        if USE_GK:
            o_k1 = tl.arange(0, 128)
            b_gk_last1 = tl.load(gk + (last_idx * H*K).to(tl.int64) + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            if USE_EXP2:
                b_gk_exp1 = exp2(b_gk_last1)
            else:
                b_gk_exp1 = exp(b_gk_last1)
            if K > 128:
                o_k2 = 128 + o_k1
                b_gk_last2 = tl.load(gk + (last_idx * H*K).to(tl.int64) + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                if USE_EXP2:
                    b_gk_exp2 = exp2(b_gk_last2)
                else:
                    b_gk_exp2 = exp(b_gk_last2)

        if USE_G:
            if USE_GK:
                b_h1 *= (b_g_last * b_gk_exp1)[:, None]
                if K > 128:
                    b_h2 *= (b_g_last * b_gk_exp2)[:, None]
            else:
                b_h1 *= b_g_last
                if K > 128:
                    b_h2 *= b_g_last
        elif USE_GK:
            b_h1 *= b_gk_exp1[:, None]
            if K > 128:
                b_h2 *= b_gk_exp2[:, None]

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v_i16)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (128, i_t * BT), (128, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v_i16)

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = tl.make_block_ptr(ht, (K, V), (1, K), (0, i_v * BV), (128, BV), (0, 1))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (128, BV), (1, 0))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if TRANSPOSE_STATE:
                p_ht = tl.make_block_ptr(ht, (K, V), (1, K), (128, i_v * BV), (128, BV), (0, 1))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            else:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (128, BV), (1, 0))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV})
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_G', 'USE_EXP2', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if TRANSPOSE_STATE:
        b_dh = tl.zeros([BV, K], dtype=tl.float32)
    else:
        b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    do += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * H + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht = tl.make_block_ptr(dht, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            b_dh += tl.load(p_dht, boundary_check=(0, 1))
        else:
            p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
            if K > 64:
                p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
            if K > 128:
                p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
            if K > 192:
                p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = exp(bg_last)
                b_g_exp = exp(b_g)

        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        if TRANSPOSE_STATE:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, K), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k = tl.arange(0, K)
                b_gk_last = tl.load(gk + last_idx * H*K + o_k, mask=(o_k < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, tl.trans(b_dh.to(b_k.dtype)))
        else:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))
            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k2 = 64 + o_k1
                    b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k3 = 128 + o_k1
                    b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    o_k4 = 192 + o_k1
                    b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            if USE_EXP2:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            else:
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        # Preload w/q for dh update (before dv2 store for MTE2/MTE3 overlap)
        if TRANSPOSE_STATE:
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (0, i_t * BT), (K, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (K, BT), (0, 1))
        else:
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        # Store dh (moved after dv2 store for maximum MTE2/MTE3 overlap)
        if TRANSPOSE_STATE:
            p_dh = tl.make_block_ptr(dh + i_t_int64*H*K*V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_dh1 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_dh2 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_dh3 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_dh4 = tl.make_block_ptr(dh + i_t_int64*H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        # Update dh
        if TRANSPOSE_STATE:
            if USE_G:
                b_dh *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh *= exp2(b_gk_last)[None, :]
                else:
                    b_dh *= exp(b_gk_last)[None, :]
            b_dh += tl.dot(tl.trans(b_do), tl.trans(b_q)) * scale - tl.dot(tl.trans(b_dv.to(b_w.dtype)), tl.trans(b_w))
        else:
            if USE_G:
                b_dh1 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh1 *= exp2(b_gk_last1[:, None])
                else:
                    b_dh1 *= exp(b_gk_last1[:, None])
            b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh2 *= exp2(b_gk_last2[:, None])
                    else:
                        b_dh2 *= exp(b_gk_last2[:, None])
                b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh3 *= exp2(b_gk_last3[:, None])
                    else:
                        b_dh3 *= exp(b_gk_last3[:, None])
                b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                    b_q = b_q * b_g_exp[None, :]
                if USE_GK:
                    if USE_EXP2:
                        b_dh4 *= exp2(b_gk_last4[:, None])
                    else:
                        b_dh4 *= exp(b_gk_last4[:, None])
                b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_dh0 = tl.make_block_ptr(dh0, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0))
            tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        else:
            p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, H, V, K)
        final_state = k.new_zeros(N, H, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, H, K, V)
        final_state = k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=state_v_first,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_fwd_h_o_fused(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
    Aqk: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    store_h: bool = True,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_v_first:
        h = k.new_empty(B, NT, H, V, K) if store_h else k.new_empty(1)
        final_state = k.new_zeros(N, H, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, H, K, V) if store_h else k.new_empty(1)
        final_state = k.new_zeros(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    o = torch.zeros_like(u)
    if K >= 128:
        fused_kernel = chunk_gated_delta_rule_fwd_h_blockdim128_fused
    else:
        fused_kernel = chunk_gated_delta_rule_fwd_h_blockdim64_fused
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    fused_kernel[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        q=q,
        o=o,
        A=Aqk,
        scale=scale,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=state_v_first,
        COMPUTE_O=True,
        STORE_H=store_h,
    )
    return o, h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, H, V, K)
    else:
        dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
        TRANSPOSE_STATE=state_v_first,
    )
    return dh, dh0, dv2

