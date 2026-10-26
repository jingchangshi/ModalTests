#!/usr/bin/env python3
"""Build the reference artefacts for the RMSNorm + FP8 block-quantization case.

This is the tutorial's own reference recipe, not a paraphrase: it imports the real
``transformers`` ``LlamaRMSNorm`` and applies the quantization the model's published
``quantization_config`` describes. The seed is fixed, so every number is reproducible.

It writes three files into the directory it is run from:

    x.pt               the bf16 activations the check reads as ``--inputs files:x.pt``
    expected.pt        ``[quant, scale]`` in return order, read as ``--expected``
    model.safetensors  ``gamma`` under the key ``gamma``, read as ``--weights ckpt:.``

Run it with an interpreter that has torch, transformers and safetensors:

    python make_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def load_published(config: Path) -> dict:
    """Read the published fields the case depends on, and refuse to invent them."""
    published = json.loads(config.read_text(encoding="utf-8"))
    eps = published["rms_norm_eps"]
    block = published["quantization_config"]["weight_block_size"][1]
    fmt = published["quantization_config"]["fmt"]
    if fmt != "e4m3":
        raise SystemExit(f"make_reference: fmt is {fmt!r}, the fp8_max constant assumes e4m3")
    print(f"rms_norm_eps={eps}  weight_block_size={block}  fmt={fmt}")
    return {"eps": eps, "block": block, "fmt": fmt}


def main() -> int:
    """Generate x.pt, expected.pt and model.safetensors in the working directory."""
    here = Path(__file__).resolve().parent
    published = load_published(here / "config.json")
    eps, block = published["eps"], published["block"]

    rows, width, fp8_max = 2, 7168, 448.0
    torch.manual_seed(0)
    a = torch.randn(rows, width, dtype=torch.bfloat16)
    gamma = (1.0 + 0.02 * torch.randn(width)).to(torch.bfloat16)

    # The real class. Its forward is ``self.weight * hidden_states.to(input_dtype)``:
    # the cast to bf16 happens BEFORE the scale, which is the operation order the
    # correct HIR has to reproduce.
    norm = LlamaRMSNorm(width, eps=eps)
    norm.weight = torch.nn.Parameter(gamma.clone())
    with torch.no_grad():
        normed = norm(a)

    blocks = normed.float().reshape(rows, width // block, block)
    scale = blocks.abs().amax(-1, keepdim=True) * (1.0 / fp8_max)
    quant = (blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    torch.save(a, here / "x.pt")
    torch.save([quant.reshape(rows, width), scale.reshape(rows, width // block)], here / "expected.pt")
    save_file({"gamma": gamma.reshape(1, width).contiguous()}, here / "model.safetensors")

    provenance = {
        "rows": rows,
        "h": width,
        "block": block,
        "blocks": width // block,
        "fp8_max": fp8_max,
        "eps": eps,
        "fmt": published["fmt"],
        "seed": 0,
        "torch_version": torch.__version__,
        "normed_dtype": str(normed.dtype),
        "quant_dtype": str(quant.dtype),
        "scale_dtype": str(scale.dtype),
        "reference_class": "transformers.models.llama.modeling_llama.LlamaRMSNorm",
    }
    (here / "reference-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote x.pt expected.pt model.safetensors for {tuple(a.shape)} {a.dtype}")
    print(f"expected: quant {tuple(quant.shape)} {quant.dtype}, scale {tuple(scale.shape)} {scale.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
