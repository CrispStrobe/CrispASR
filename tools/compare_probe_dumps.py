#!/usr/bin/env python
"""Compare CPU vs GPU probe dumps from PROBE_BLOCK1=0 to localise where
the first divergence appears in the s3.fd.db.0.0.b1 causal_block1d chain.

Runs ./build/bin/crispasr twice; matches files by stage name; reports
per-stage shape, finite-count, cosine, max_abs, and the first 5 indices
where the two disagree by more than a threshold.

Stage order in causal_block1d (with PROBE_BLOCK1=0):
  after_im2col  (F32, K*C_in=960, T_out=384)
  after_mul_mat (F32, C_out=256, T_out=384)
  after_conv1d  (F32, T_want=382, C_out=256) — view + cont + bias add
  after_transpose_in (F32, C_out=256, T=382) — transpose for LN
  after_norm    (F32, C=256, T=382)
  after_ln_mul  (F32, C=256, T=382)
  after_ln_bias (F32, C=256, T=382)
  after_transpose_out (F32, T=382, C=256)
  after_mish    (F32, T=382, C=256)
"""
import os
import struct
import sys
from pathlib import Path

import numpy as np

STAGES = [
    "after_im2col",
    "after_mul_mat",
    "after_mul_mat (reshaped)",
    "after_mul_mat (reshaped) (view) (cont)",
    "after_conv1d",
    "after_conv1d (transposed)",
    "after_transpose_in",
    "after_norm",
    "after_ln_mul",
    "after_ln_bias",
    "after_ln_bias (transposed)",
    "after_transpose_out",
    "after_mish",
]


def load(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32)


def cos_sim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def summarise(name, cpu_path, gpu_path):
    if not cpu_path.exists():
        return (name, "NO CPU FILE", None)
    if not gpu_path.exists():
        return (name, "NO GPU FILE", None)
    cpu = load(cpu_path)
    gpu = load(gpu_path)
    if cpu.shape != gpu.shape:
        return (name, f"SHAPE MISMATCH {cpu.shape} vs {gpu.shape}", None)
    nfc = int(np.sum(~np.isfinite(cpu)))
    nfg = int(np.sum(~np.isfinite(gpu)))
    # Replace non-finite in gpu for cosine calc — track separately.
    cpu_f = cpu.copy()
    gpu_f = gpu.copy()
    cpu_f[~np.isfinite(cpu_f)] = 0
    gpu_f[~np.isfinite(gpu_f)] = 0
    cos = cos_sim(cpu_f, gpu_f)
    diff = np.abs(cpu_f - gpu_f)
    max_abs = float(np.max(diff))
    # First column index where any element differs by >1e-4 — try to tease
    # out the "3 contiguous columns" pattern.
    return (name, dict(
        n=int(cpu.size),
        nfc=nfc, nfg=nfg, cos=cos, max_abs=max_abs,
        cpu_min=float(np.min(cpu_f)), cpu_max=float(np.max(cpu_f)),
        gpu_min=float(np.min(gpu_f)), gpu_max=float(np.max(gpu_f)),
        cpu_rms=float(np.sqrt(np.mean(cpu_f**2))),
        gpu_rms=float(np.sqrt(np.mean(gpu_f**2))),
    ), (cpu, gpu))


def main():
    tmp = Path("/tmp")
    cpu_prefix = "cb-unet-dump-cpu-probe-dump_probe_"
    gpu_prefix = "cb-unet-dump-gpu-probe-dump_probe_"
    rows = []
    for stage in STAGES:
        cpu_path = tmp / f"{cpu_prefix}{stage}.bin"
        gpu_path = tmp / f"{gpu_prefix}{stage}.bin"
        rows.append(summarise(stage, cpu_path, gpu_path))

    # Print table
    print(f"{'stage':45s}  {'n':>9s}  {'nfc':>5s}  {'nfg':>5s}  {'cos':>10s}  {'max_abs':>10s}  {'cpu_rms':>10s}  {'gpu_rms':>10s}")
    print("-" * 130)
    for name, info, _ in rows:
        if not isinstance(info, dict):
            print(f"{name:45s}  {info}")
            continue
        print(f"{name:45s}  {info['n']:>9d}  {info['nfc']:>5d}  {info['nfg']:>5d}  "
              f"{info['cos']:>10.6f}  {info['max_abs']:>10.3e}  "
              f"{info['cpu_rms']:>10.3e}  {info['gpu_rms']:>10.3e}")

    # For the first stage where they diverge significantly (cos < 0.99 or
    # any non-finite), drill down: show which contiguous-column regions
    # are wrong.
    print()
    print("=== Per-stage column-level diff for stages with cos<0.99 ===")
    for name, info, raw in rows:
        if not isinstance(info, dict):
            continue
        if info["cos"] > 0.99 and info["nfg"] == 0:
            continue
        cpu, gpu = raw
        print(f"\n--- {name}  (n={cpu.size}, cos={info['cos']:.4f}, nfg={info['nfg']}) ---")
        # Best-guess shape: try to find shape by looking up known shapes.
        # For the conv chain we have (T_out=384 or T_want=382) × (C=256 or 960).
        # Try (X, 256) where X is 380, 382, 384.
        shape_guesses = [(960, 384), (256, 384), (256, 382), (382, 256), (1, 256), (384, 960)]
        for sg in shape_guesses:
            if sg[0] * sg[1] == cpu.size:
                c2 = cpu.reshape(sg)
                g2 = gpu.reshape(sg)
                # find which rows/cols have non-finite or large diff
                print(f"  shape guess: {sg}")
                # Non-finite map
                nf = ~np.isfinite(g2)
                if nf.any():
                    # rows with any nf
                    nf_rows = np.where(nf.any(axis=1))[0]
                    nf_cols = np.where(nf.any(axis=0))[0]
                    print(f"  nf rows ({len(nf_rows)}): "
                          f"{nf_rows[:20].tolist()}{'...' if len(nf_rows)>20 else ''}  "
                          f"contiguous runs (start, len): "
                          f"{contiguous_runs(nf_rows)[:8]}")
                    print(f"  nf cols ({len(nf_cols)}): "
                          f"{nf_cols[:20].tolist()}{'...' if len(nf_cols)>20 else ''}  "
                          f"contiguous runs (start, len): "
                          f"{contiguous_runs(nf_cols)[:8]}")
                # large diff map (>1% of cpu rms)
                thr = max(1e-6, 0.01 * info["cpu_rms"])
                cf = np.where(np.isfinite(c2), c2, 0)
                gf = np.where(np.isfinite(g2), g2, 0)
                diff = np.abs(cf - gf)
                lr = np.where((diff > thr).any(axis=1))[0]
                lc = np.where((diff > thr).any(axis=0))[0]
                if len(lr) > 0:
                    print(f"  diff>{thr:.2e} rows: count={len(lr)}, "
                          f"first={lr[:10].tolist()}, runs={contiguous_runs(lr)[:8]}")
                if len(lc) > 0:
                    print(f"  diff>{thr:.2e} cols: count={len(lc)}, "
                          f"first={lc[:10].tolist()}, runs={contiguous_runs(lc)[:8]}")
                break  # one shape guess is enough


def contiguous_runs(xs):
    if len(xs) == 0:
        return []
    runs = []
    start = xs[0]
    prev = xs[0]
    for v in xs[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((int(start), int(prev - start + 1)))
        start = v
        prev = v
    runs.append((int(start), int(prev - start + 1)))
    return runs


if __name__ == "__main__":
    main()
