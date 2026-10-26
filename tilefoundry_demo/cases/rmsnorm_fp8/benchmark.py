#!/usr/bin/env python3
"""Measure warm-kernel latency for the Triton twin and a defined PyTorch baseline.

This is the piece the tutorial does not have. ``tilefoundry analyze`` predicts a cost from
a static model; this script measures what the GPU actually does, on the same inputs, with
the same semantics.

Scope, stated so the numbers cannot be over-read:

* Measured regions are the *whole* RMSNorm + FP8 block-quantization step: it consumes
  ``a`` and ``gamma`` and produces ``(quant, scale)``. A partial baseline such as
  ``torch.nn.functional.rms_norm`` alone is not comparable and is not used.
* Tensor allocation is **inside** the timed region for both arms, because both arms
  allocate their outputs. ``torch.empty`` for the Triton outputs and the baseline's own
  intermediates therefore appear in both numbers.
* Python launch overhead is **inside** the timed region for both arms: each repetition is
  one Python-level call. This is a launch-bound step, so hiding it would flatter the
  result.
* JIT compilation is **outside** the timed region. Each arm is warmed up first, and the
  Triton arm's compile happens during warmup.
* Every repetition is bracketed by CUDA events on the current stream, and the device is
  synchronized before the clock is read.

Nothing here decides whether the step is correct; ``tilefoundry check`` does that. This
script asserts only that the measured kernel produced a non-degenerate result before it
times anything, so a kernel that silently wrote nothing cannot post a fast time.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

BLOCK = 128
H = 7168
FP8_MAX = 448.0
EPS = 1e-6
BLOCKS = H // BLOCK
PAIR = 2
CTAS = 132
ROWS = PAIR * CTAS


def make_inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed bf16 activations and gamma this benchmark uses."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((ROWS, H), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    gamma = (1.0 + 0.02 * torch.randn((H,), generator=generator, dtype=torch.float32)).to(torch.bfloat16)
    return a.cuda(), gamma.reshape(1, H).cuda()


def reference_step(a: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The step in PyTorch, in the order the model defines it.

    ``LlamaRMSNorm.forward`` is ``self.weight * hidden_states.to(input_dtype)``: the bf16
    cast happens before the scale. This baseline reproduces that order and the published
    block quantization, so it is an alternative *implementation*, not a different
    algorithm.
    """
    rows = a.float()
    mean = rows.square().mean(-1, keepdim=True)
    normed = (rows * torch.rsqrt(mean + EPS)).to(torch.bfloat16) * gamma
    blocks = normed.float().reshape(ROWS, BLOCKS, BLOCK)
    scale = blocks.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)
    quant = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant.reshape(ROWS, H), scale.reshape(ROWS, BLOCKS)


def time_arm(call, *, warmup: int, repeats: int) -> dict:
    """Time one callable with CUDA events, returning raw samples plus summary stats."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for index in range(repeats):
        starts[index].record()
        call()
        ends[index].record()
    torch.cuda.synchronize()

    samples_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    ordered = sorted(samples_ms)
    return {
        "repeats": repeats,
        "warmup": warmup,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": ordered[0],
        "p10_ms": ordered[max(0, int(round(0.10 * (len(ordered) - 1))))],
        "p90_ms": ordered[max(0, int(round(0.90 * (len(ordered) - 1))))],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    """Run both arms, refuse degenerate output, and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="alternating A/B rounds; the reported comparison is their median",
    )
    arguments = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("benchmark: no CUDA device", file=sys.stderr)
        return 2
    if arguments.warmup < 0 or not 1 <= arguments.repeats <= 1000:
        print("benchmark: need warmup >= 0 and 1 <= repeats <= 1000", file=sys.stderr)
        return 2
    if not 1 <= arguments.rounds <= 21:
        print("benchmark: need 1 <= rounds <= 21", file=sys.stderr)
        return 2

    device = torch.cuda.get_device_properties(0)
    a, gamma = make_inputs(arguments.seed)

    # The Triton arm: the twin, compiled by Triton on first call (during warmup).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from twin import Fast

    twin = Fast()
    # `gamma` is a declared weight of the authored module, so it is bound through a
    # resource rather than passed as a positional argument. Passing it positionally made
    # the runtime raise "weight 'gamma' is not bound; call load(resource) first".
    from tilefoundry.runtime.resource import DictResource

    twin.load(DictResource({"gamma": gamma}))

    def triton_call():
        return twin.rms_norm_quant(a)

    quant_triton, scale_triton = triton_call()
    torch.cuda.synchronize()
    if int(quant_triton.float().abs().sum().item()) == 0 or int(scale_triton.abs().sum().item()) == 0:
        print("benchmark: the twin produced an all-zero result; refusing to time it", file=sys.stderr)
        return 1

    def torch_call():
        return reference_step(a, gamma)

    quant_torch, scale_torch = torch_call()
    torch.cuda.synchronize()
    if int(quant_torch.float().abs().sum().item()) == 0 or int(scale_torch.abs().sum().item()) == 0:
        print("benchmark: the baseline produced an all-zero result; refusing to time it", file=sys.stderr)
        return 1

    agreement = {
        "quant_equal_fraction": float(
            (quant_triton == quant_torch).float().mean().item()
        ),
        "scale_max_abs_diff": float((scale_triton - scale_torch).abs().max().item()),
    }

    # Alternate the arms across rounds and take the median of the per-round medians. Two
    # runs that time one arm after the other see different clocks: on this device the
    # baseline median moved from 0.119 ms to 0.210 ms between runs, which is larger than
    # the difference the comparison is supposed to describe. Alternating does not remove
    # clock variation, but it stops that variation from always landing on one arm.
    rounds = max(1, arguments.rounds)
    triton_medians: list[float] = []
    torch_medians: list[float] = []
    per_round: list[dict] = []
    for index in range(rounds):
        if index % 2 == 0:
            triton_round = time_arm(triton_call, warmup=arguments.warmup, repeats=arguments.repeats)
            torch_round = time_arm(torch_call, warmup=arguments.warmup, repeats=arguments.repeats)
        else:
            torch_round = time_arm(torch_call, warmup=arguments.warmup, repeats=arguments.repeats)
            triton_round = time_arm(triton_call, warmup=arguments.warmup, repeats=arguments.repeats)
        triton_medians.append(triton_round["median_ms"])
        torch_medians.append(torch_round["median_ms"])
        per_round.append(
            {
                "round": index,
                "triton_median_ms": triton_round["median_ms"],
                "torch_median_ms": torch_round["median_ms"],
            }
        )

    triton_timing = time_arm(triton_call, warmup=arguments.warmup, repeats=arguments.repeats)
    torch_timing = time_arm(torch_call, warmup=arguments.warmup, repeats=arguments.repeats)
    round_triton = statistics.median(triton_medians)
    round_torch = statistics.median(torch_medians)
    ratios = [
        entry["torch_median_ms"] / entry["triton_median_ms"]
        for entry in per_round
        if entry["triton_median_ms"] > 0
    ]

    report = {
        "what": "RMSNorm + FP8 block quantization, warm-kernel latency",
        # The headline comparison, stated with its spread so a reader can see how much of
        # the difference is stable and how much is clock variation.
        "comparison": {
            "baseline": "torch_baseline: the same step as unfused PyTorch tensor ops",
            "candidate": "triton_twin: the Triton kernel from the tutorial's twin",
            "rounds": rounds,
            "median_of_round_medians_ms": {
                "triton_twin": round_triton,
                "torch_baseline": round_torch,
            },
            "speedup_torch_over_triton": {
                "min": min(ratios) if ratios else None,
                "median": statistics.median(ratios) if ratios else None,
                "max": max(ratios) if ratios else None,
            },
            "per_round": per_round,
            "what_this_is_not": (
                "not a comparison of the compiler against PyTorch: both arms are "
                "hand-written kernels/ops, and the HIR ideal-ns figures describe a "
                "different program, so they are never divided by these numbers"
            ),
        },
        "measured_region": "whole step: (a, gamma) -> (quant, scale), allocation and Python launch included",
        "jit_compile": "excluded; each arm warmed up before timing",
        "shape": {"rows": ROWS, "h": H, "block": BLOCK, "blocks": BLOCKS},
        "dtypes": {"input": "bfloat16", "quant": "float8_e4m3fn", "scale": "float32"},
        "seed": arguments.seed,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            # torch exposes the driver version through torch.version.cuda only for the
            # CUDA runtime; the driver itself is reported by torch.cuda.driver_version
            # when the build provides it, and stays null otherwise rather than being
            # guessed from a constant.
            "cuda_driver": getattr(torch.cuda, "driver_version", None),
        },
        "device": {
            "name": device.name,
            "compute_capability": [device.major, device.minor],
            "total_memory": device.total_memory,
            "multi_processor_count": device.multi_processor_count,
        },
        "arms": {
            "triton_twin": triton_timing,
            "torch_baseline": torch_timing,
        },
        "agreement": agreement,
        "not_a_speedup_claim_from_static_model": (
            "These are measured latencies. The Naive/Fused roofline ideal-ns values are a "
            "static model of a different (HIR) program and are never divided by these numbers."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
