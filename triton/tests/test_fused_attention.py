import os
import time

import torch
import triton

from kernels.fused_attention import _attn_fwd, attention


# Single fixed shape so the study run compiles exactly one config of one kernel.
# All four values are powers of two >= 16 for tl.dot (mma) compatibility.
BATCH = 1
N_HEADS = 2
N_CTX = 1024
HEAD_DIM = 64
CAUSAL = True
SM_SCALE = 0.5

# fp16 accumulate-in-fp32 tolerance used by the upstream tutorial test.
RTOL = 0.0
ATOL = 1e-2


def _reference_attention(q, k, v, causal, sm_scale):
    """Plain PyTorch reference: softmax(QK^T * scale) V, in fp32 for the softmax."""
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        M = torch.tril(torch.ones((q.shape[-2], q.shape[-2]), device=q.device))
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


def _record_study_metrics(**metrics):
    results_dir = os.environ.get("TRITON_STUDY_RESULTS_DIR")
    if not results_dir:
        return

    import json
    from pathlib import Path

    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "kernel_metrics.jsonl").open("a") as f:
        f.write(json.dumps(metrics, sort_keys=True) + "\n")


def test_fused_attention_fwd():
    torch.manual_seed(20)
    q = torch.empty((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=torch.float16,
                    device="cuda").normal_(mean=0.0, std=0.5)
    k = torch.empty((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=torch.float16,
                    device="cuda").normal_(mean=0.0, std=0.5)
    v = torch.empty((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=torch.float16,
                    device="cuda").normal_(mean=0.0, std=0.5)

    study_mode = bool(os.environ.get("TRITON_STUDY_RESULTS_DIR"))

    if study_mode:
        torch.cuda.synchronize()
        cold_start = time.perf_counter_ns()

    actual = attention(q, k, v, CAUSAL, SM_SCALE).half()

    if study_mode:
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter_ns() - cold_start) / 1_000_000.0
    else:
        cold_ms = None

    expected = _reference_attention(q, k, v, CAUSAL, SM_SCALE).half()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    max_error = (actual.float() - expected.float()).abs().max().item()

    if study_mode:
        # Warm timing: autotuner has settled, in-process JIT cache is hot.
        warm_ms = float(triton.testing.do_bench(
            lambda: attention(q, k, v, CAUSAL, SM_SCALE)))

        # FLOP model identical to the tutorial benchmark: 2 GEMMs per attention,
        # causal saves the strictly-upper triangle (x0.5).
        flops_per_matmul = 2.0 * BATCH * N_HEADS * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        if CAUSAL:
            total_flops *= 0.5
        tflops = total_flops * 1e-12 / (warm_ms * 1e-3)

        # How many distinct configs did the autotuner compile?
        n_compiled = len(getattr(_attn_fwd, "cache", {}))

        _record_study_metrics(
            kernel="_attn_fwd",
            tutorial="06-fused-attention (flash-attention v2)",
            batch=BATCH,
            n_heads=N_HEADS,
            n_ctx=N_CTX,
            head_dim=HEAD_DIM,
            causal=CAUSAL,
            dtype="torch.float16",
            cold_jit_and_first_run_ms=cold_ms,
            warm_runtime_ms=warm_ms,
            tflops=tflops,
            max_error=max_error,
            atol=ATOL,
        )
        print(f"cold JIT + first run: {cold_ms:.3f} ms")
        print(f"warm runtime: {warm_ms:.6f} ms")
        print(f"achieved: {tflops:.2f} TFLOPS")

    print(f"max error: {max_error}")
