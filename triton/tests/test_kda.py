import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from kernels.chunk_delta_h_kda import (
    chunk_gated_delta_rule_fwd_h_blockdim64_fused,
    chunk_gated_delta_rule_fwd_h_o_fused,
)


# Reduced standalone case derived from the original perf case:
#   B1-T8192-H96-HV96-D128-bfloat16-chunk_size64
#
# The original D=128 / chunk_size=64 shape dispatches to the blockdim128
# kernel and its autotuner may compile a BV=128 variant. On GPUs with about
# 99 KiB shared memory per block this can fail before launch with e.g.
#   OutOfResources: Required: 161792, Hardware limit: 101376
#
# Merely reducing T or H does not materially change per-program shared-memory
# usage. The two important reductions are:
#   HEAD_DIM:   128 -> 64   (dispatch to blockdim64 fused kernel)
#   CHUNK_SIZE:  64 -> 32   (smaller BT-dependent tiles, especially Aqk)
# T/H/HV are also reduced to make allocation/JIT smoke testing lightweight.
BATCH = 1
N_CTX = 1024
N_HEADS = 8
N_VALUE_HEADS = 8
HEAD_DIM = 64
VALUE_DIM = 64
MASK_P = 0.0
DTYPE = torch.bfloat16
USE_GATE_IN_KERNEL = True
SAFE_GATE = True
DISABLE_RECOMPUTE = False
CHUNK_SIZE = 32
LOWER_BOUND = -5.0
SCALE = HEAD_DIM ** -0.5
RCP_LN2 = 1.0 / math.log(2.0)

# Preserve the original state-layout semantics while reducing only tensor/tile
# sizes.
STATE_V_FIRST = True
OUTPUT_FINAL_STATE = True

# Keep the default study run reasonably small. Override from the environment
# when longer timing is wanted.
WARMUP_ITERS = int(os.environ.get("KDA_WARMUP_ITERS", "3"))
BENCH_ITERS = int(os.environ.get("KDA_BENCH_ITERS", "10"))


def _device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("This Triton KDA test requires an NPU (preferred) or CUDA device.")


def _synchronize(device: str) -> None:
    if device == "npu":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def _record_study_metrics(**metrics) -> None:
    results_dir = os.environ.get("TRITON_STUDY_RESULTS_DIR")
    if not results_dir:
        return

    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "kernel_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")


def _bench_ms(fn, device: str) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    _synchronize(device)

    start = time.perf_counter_ns()
    for _ in range(BENCH_ITERS):
        fn()
    _synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return elapsed_ms / BENCH_ITERS


def _chunk_local_cumsum(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Torch equivalent of the fixed-length chunk-local cumsum used by KDA."""
    B, T, H, D = x.shape
    if T % chunk_size != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size} for this fixed test.")

    x = x.view(B, T // chunk_size, chunk_size, H, D)
    x = torch.cumsum(x.float(), dim=2) * RCP_LN2
    return x.to(DTYPE).view(B, T, H, D)


def _make_safe_gate(device: str) -> torch.Tensor:
    """
    Reproduce the semantics relevant to the selected full-KDA case:
      use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.

    The original full pipeline performs the gate activation in a Triton kernel
    and then a chunk-local cumsum. Here both are expressed in PyTorch so the
    standalone test has no dependency on gate.py/cumsum_kda.py.
    """
    raw_g = torch.randn(
        BATCH,
        N_CTX,
        N_VALUE_HEADS,
        HEAD_DIM,
        dtype=DTYPE,
        device=device,
    )
    A_log = torch.log(
        torch.empty(N_VALUE_HEADS, dtype=torch.float32, device=device).uniform_(1.0, 16.0)
    )
    dt_bias = torch.randn(
        N_VALUE_HEADS,
        HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )

    gate = LOWER_BOUND * torch.sigmoid(
        A_log.exp().view(1, 1, N_VALUE_HEADS, 1)
        * (raw_g.float() + dt_bias.view(1, 1, N_VALUE_HEADS, HEAD_DIM))
    )
    del raw_g, A_log, dt_bias
    return _chunk_local_cumsum(gate, CHUNK_SIZE)


def _build_inputs(device: str) -> dict[str, torch.Tensor]:
    """
    Build self-contained inputs for chunk_gated_delta_rule_fwd_h_o_fused.

    In the full chunk_kda pipeline, w/u/kg/Aqk are produced by earlier KDA
    kernels. To keep this file independent of kda_kernel, we construct a
    coherent synthetic set of those intermediates here.
    """
    q = torch.randn(
        BATCH, N_CTX, N_HEADS, HEAD_DIM, dtype=DTYPE, device=device
    )
    q = F.normalize(q.float(), p=2, dim=-1).to(DTYPE)

    k = torch.randn(
        BATCH, N_CTX, N_HEADS, HEAD_DIM, dtype=torch.float32, device=device
    )
    k = F.normalize(k, p=2, dim=-1).to(DTYPE)

    v = torch.randn(
        BATCH, N_CTX, N_VALUE_HEADS, VALUE_DIM, dtype=DTYPE, device=device
    )

    # use_beta_sigmoid_in_kernel=True in the original full test. The standalone
    # delta-H stage receives the already activated beta through w/u, so create
    # the post-sigmoid value here.
    beta = torch.randn(
        BATCH, N_CTX, N_VALUE_HEADS, dtype=DTYPE, device=device
    ).float().sigmoid().to(DTYPE)

    gk = _make_safe_gate(device)

    # Reconstruct the kg transform used by recompute_w_u_fwd:
    #   kg[t] = k[t] * exp2(g_last_in_chunk - gk[t])
    gk_chunks = gk.view(
        BATCH,
        N_CTX // CHUNK_SIZE,
        CHUNK_SIZE,
        N_VALUE_HEADS,
        HEAD_DIM,
    )
    k_chunks = k.view(
        BATCH,
        N_CTX // CHUNK_SIZE,
        CHUNK_SIZE,
        N_HEADS,
        HEAD_DIM,
    )
    g_last = gk_chunks[:, :, -1:, :, :]
    kg = (k_chunks * torch.exp2(g_last - gk_chunks)).view_as(k)

    # Use an identity intra-chunk solve as a simple, numerically stable
    # synthetic predecessor for this isolated stage:
    #   u = v * beta
    #   w = k * beta * exp2(gk)
    beta_v = beta.unsqueeze(-1)
    u = v * beta_v
    w = k * beta_v * torch.exp2(gk)

    # Aqk normally comes from chunk_kda_fwd_intra_fused. Zero is a valid
    # synthetic matrix for isolating the recurrent delta-H path while retaining
    # the exact production shape and memory layout.
    Aqk = torch.zeros(
        BATCH,
        N_CTX,
        N_VALUE_HEADS,
        CHUNK_SIZE,
        dtype=DTYPE,
        device=device,
    )

    return {
        "q": q,
        "kg": kg,
        "w": w,
        "u": u,
        "gk": gk,
        "Aqk": Aqk,
    }


def _run_delta_h(inputs: dict[str, torch.Tensor]):
    # disable_recompute=False in the selected full-KDA case means the forward
    # path does not retain h/v_new for backward. Match that here.
    return chunk_gated_delta_rule_fwd_h_o_fused(
        k=inputs["kg"],
        w=inputs["w"],
        u=inputs["u"],
        q=inputs["q"],
        Aqk=inputs["Aqk"],
        gk=inputs["gk"],
        scale=SCALE,
        initial_state=None,
        output_final_state=OUTPUT_FINAL_STATE,
        chunk_size=CHUNK_SIZE,
        store_h=DISABLE_RECOMPUTE,
        save_new_value=DISABLE_RECOMPUTE,
        use_exp2=True,
        state_v_first=STATE_V_FIRST,
    )


def test_kda_fwd():
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = _device()
    inputs = _build_inputs(device)
    study_mode = bool(os.environ.get("TRITON_STUDY_RESULTS_DIR"))

    _synchronize(device)
    cold_start = time.perf_counter_ns()
    output, h, v_new, final_state = _run_delta_h(inputs)
    _synchronize(device)
    cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000.0

    assert output.shape == (BATCH, N_CTX, N_VALUE_HEADS, VALUE_DIM)
    assert h.numel() == 1, "disable_recompute=False should avoid storing h"
    assert v_new is None, "disable_recompute=False should avoid storing v_new"
    assert final_state is not None
    assert final_state.shape == (BATCH, N_VALUE_HEADS, VALUE_DIM, HEAD_DIM)

    # Sample the output to catch obvious NaN/Inf failures without making the
    # smoke test itself expensive.
    sample = output[:, ::128, ::2, ::8].float()
    assert torch.isfinite(sample).all(), "non-finite values found in sampled output"

    warm_ms = None
    if study_mode:
        warm_ms = _bench_ms(lambda: _run_delta_h(inputs), device)
        n_compiled = len(
            getattr(chunk_gated_delta_rule_fwd_h_blockdim64_fused, "cache", {})
        )

        _record_study_metrics(
            kernel="chunk_gated_delta_rule_fwd_h_blockdim64_fused",
            source="chunk_delta_h_kda.py",
            case=(
                "B1-T1024-H8-HV8-D64-mask_p0-torch.bfloat16-"
                "gateTrue-safe_gateTrue-disable_recomputeFalse-chunk_size32"
            ),
            batch=BATCH,
            n_ctx=N_CTX,
            n_heads=N_HEADS,
            n_value_heads=N_VALUE_HEADS,
            head_dim=HEAD_DIM,
            value_dim=VALUE_DIM,
            dtype=str(DTYPE),
            mask_p=MASK_P,
            use_gate_in_kernel=USE_GATE_IN_KERNEL,
            safe_gate=SAFE_GATE,
            disable_recompute=DISABLE_RECOMPUTE,
            chunk_size=CHUNK_SIZE,
            state_v_first=STATE_V_FIRST,
            cold_jit_and_first_run_ms=cold_ms,
            warm_runtime_ms=warm_ms,
            compiled_configs=n_compiled,
            warmup_iters=WARMUP_ITERS,
            bench_iters=BENCH_ITERS,
        )

        print(f"cold JIT + first run: {cold_ms:.3f} ms")
        print(f"warm runtime: {warm_ms:.6f} ms")
        print(f"compiled configs: {n_compiled}")

    print(f"device: {device}")
    print(f"output shape: {tuple(output.shape)}")
    print(f"final_state shape: {tuple(final_state.shape)}")


if __name__ == "__main__":
    test_kda_fwd()

