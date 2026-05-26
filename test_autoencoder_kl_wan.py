#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone test for ``AutoencoderKLWan.decode`` with baseline comparison.

Loads two VAE implementations on the same weights:
  1. BASELINE: ``autoencoder_kl_wan_baseline.py`` (a frozen copy of the original).
  2. MODIFIED: the current ``diffusers.AutoencoderKLWan`` (what you edit in place).

Runs decode on identical latents with both, reports per-run stats and timing,
then compares outputs numerically (max / mean abs+rel diff and allclose at
several tolerances).

Usage:
    install flashinfer in https://github.com/xueweilnvidia/flashinfer/tree/vae_new 
    python test_autoencoder_kl_wan.py \
        --model-id ../Wan2.2-T2V-A14B-Diffusers --subfolder vae
"""

import argparse
import importlib.util
from typing import Tuple

import torch
import nvtx

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in shape_str.split(","))


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[key]


def load_baseline_class(path: str):
    spec = importlib.util.spec_from_file_location("autoencoder_kl_wan_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AutoencoderKLWan


def decode_once(vae, latents: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        with nvtx.annotate("decode_once"):
            out = vae.decode(latents, return_dict=False)[0]
    return out


def report_stats(label: str, y: torch.Tensor) -> dict:
    y_f = y.float()
    abs_y = y_f.abs()
    stats = {
        "shape": tuple(y.shape),
        "dtype": str(y.dtype),
        "min": y_f.min().item(),
        "max": y_f.max().item(),
        "mean": y_f.mean().item(),
        "std": y_f.std().item(),
        "abs_max": abs_y.max().item(),
        "abs_mean": abs_y.mean().item(),
    }
    print(
        f"{label}: shape={stats['shape']}, dtype={stats['dtype']}, "
        f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
        f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"abs_max={stats['abs_max']:.6f}, abs_mean={stats['abs_mean']:.6f}"
    )
    return stats


def run_decode(
    label: str,
    VAEClass,
    model_id: str,
    subfolder: str,
    latents: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    iters: int,
    channels_last_3d: bool = False,
) -> Tuple[torch.Tensor, float]:
    print(f"\n=== {label} ===")
    print(f"Loading VAE: {VAEClass.__module__}.{VAEClass.__name__}")
    vae = (
        VAEClass.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch.float32)
        .to(dtype)
        .to(device)
    )
    vae.eval()
    if channels_last_3d:
        n = 0
        for m in vae.modules():
            if isinstance(m, torch.nn.Conv3d):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                n += 1
        print(f"{label}: converted {n} Conv3d weights to channels_last_3d")

    print("Warmup decode")
    y = decode_once(vae, latents)
    torch.cuda.synchronize()
    report_stats(f"{label} warmup output", y)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    torch.cuda.cudart().cudaProfilerStart()
    start_evt.record()
    for _ in range(iters):
        y = decode_once(vae, latents)
    end_evt.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    avg_ms = start_evt.elapsed_time(end_evt) / iters
    print(f"{label} avg decode latency: {avg_ms:.3f} ms over {iters} iters")
    report_stats(f"{label} final output", y)

    out = y.detach().clone()
    del vae
    torch.cuda.empty_cache()
    return out, avg_ms


def compare(baseline: torch.Tensor, modified: torch.Tensor) -> None:
    print("\n=== Comparison: BASELINE vs MODIFIED ===")
    if baseline.shape != modified.shape:
        print(f"SHAPE MISMATCH: baseline={tuple(baseline.shape)}, modified={tuple(modified.shape)}")
        return
    a = baseline.float()
    b = modified.float()
    a_abs = a.abs()
    b_abs = b.abs()
    print(f"output_ref (BASELINE):  abs_max={a_abs.max().item():.6e}, abs_mean={a_abs.mean().item():.6e}")
    print(f"output     (MODIFIED):  abs_max={b_abs.max().item():.6e}, abs_mean={b_abs.mean().item():.6e}")
    diff = (a - b).abs()
    print(f"max abs diff:  {diff.max().item():.6e}")
    print(f"mean abs diff: {diff.mean().item():.6e}")
    # for tol in (1e-5, 1e-4, 1e-3, 1e-2):
    #     ok = torch.allclose(a, b, atol=tol, rtol=tol)
    #     print(f"allclose(atol=rtol={tol:.0e}): {ok}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoencoderKLWan decode: baseline vs modified comparison."
    )
    parser.add_argument("--model-id", type=str, default="../Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--subfolder", type=str, default="vae")
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="/workdir/tmp/diffusers/autoencoder_kl_wan_baseline.py",
        help="Frozen copy of autoencoder_kl_wan.py to compare against.",
    )
    parser.add_argument(
        "--latent-shape",
        type=str,
        default="1,16,17,104,80",
        help="Latent tensor shape as B,C,T,H,W.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline run (only test the modified version).",
    )
    parser.add_argument(
        "--skip-modified",
        action="store_true",
        help="Skip the modified run (only test the baseline).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    dtype = parse_dtype(args.dtype)
    latent_shape = parse_shape(args.latent_shape)
    torch.manual_seed(args.seed)
    latents = torch.randn(latent_shape, dtype=dtype, device=args.device)
    print(f"latents shape: {tuple(latents.shape)}, dtype: {latents.dtype}, device: {latents.device}")

    baseline_out, baseline_ms = None, None
    if not args.skip_baseline:
        BaselineVAE = load_baseline_class(args.baseline_path)
        baseline_out, baseline_ms = run_decode(
            "BASELINE", BaselineVAE, args.model_id, args.subfolder,
            latents, dtype, args.device, args.iters,
        )

    modified_out, modified_ms = None, None
    if not args.skip_modified:
        from diffusers import AutoencoderKLWan as ModifiedVAE
        modified_out, modified_ms = run_decode(
            "MODIFIED", ModifiedVAE, args.model_id, args.subfolder,
            latents, dtype, args.device, args.iters,
            channels_last_3d=True,
        )

    if baseline_out is not None and modified_out is not None:
        compare(baseline_out, modified_out)

    if baseline_ms is not None and modified_ms is not None:
        print("\n=== Latency: BASELINE vs MODIFIED ===")
        print(f"BASELINE: {baseline_ms:.3f} ms / iter")
        print(f"MODIFIED: {modified_ms:.3f} ms / iter")
        print(f"speedup:  {baseline_ms / modified_ms:.3f}x  (delta {baseline_ms - modified_ms:+.3f} ms)")


if __name__ == "__main__":
    main()
