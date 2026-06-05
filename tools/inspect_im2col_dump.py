#!/usr/bin/env python
"""Inspect raw bytes of the after_im2col probe dump for CPU and GPU.

Verifies the expected causal-padding pattern (first 2 entries per (iow=0,
iic) tuple are zero, third is x[iic, 0]) and prints differences."""
import numpy as np
from pathlib import Path

tmp = Path("/tmp")
cpu = np.frombuffer(open(tmp / "cb-unet-dump-cpu-probe-dump_probe_after_im2col.bin", "rb").read(),
                    dtype=np.float32).copy()
gpu = np.frombuffer(open(tmp / "cb-unet-dump-gpu-probe-dump_probe_after_im2col.bin", "rb").read(),
                    dtype=np.float32).copy()

# Reshape: (T_out=384, K=3, IC=320) in the layout
# the im2col writes (ne0 = IC*K = 960, ne1 = T_out = 384).
# Actually from the kernel: offset_dst = (iow)*CHW + iic*KW + ikw where CHW = IC*KH*KW = 320*1*3=960
# So linear index = iow*960 + iic*3 + ikw
# Reshape to (T_out=384, IC=320, K=3):
cpu_r = cpu.reshape(384, 320, 3)
gpu_r = gpu.reshape(384, 320, 3)

print("=== iow=0 (first 5 IC) ===")
for iic in range(5):
    print(f"  iic={iic}: CPU={cpu_r[0, iic].tolist()}  GPU={gpu_r[0, iic].tolist()}")

print("\n=== iow=1 (first 5 IC) ===")
for iic in range(5):
    print(f"  iic={iic}: CPU={cpu_r[1, iic].tolist()}  GPU={gpu_r[1, iic].tolist()}")

print("\n=== iow=2 (first 5 IC) ===")
for iic in range(5):
    print(f"  iic={iic}: CPU={cpu_r[2, iic].tolist()}  GPU={gpu_r[2, iic].tolist()}")

# Find rows where CPU has [0, 0, x] pattern (causal zero pad) but GPU doesn't.
zero_pad_ok_cpu = (cpu_r[0, :, 0] == 0).all() and (cpu_r[0, :, 1] == 0).all()
zero_pad_ok_gpu = (gpu_r[0, :, 0] == 0).all() and (gpu_r[0, :, 1] == 0).all()
print(f"\niow=0 zero-pad: CPU first 2 cols all zero = {zero_pad_ok_cpu}  GPU = {zero_pad_ok_gpu}")
zero_pad_ok_cpu = (cpu_r[1, :, 0] == 0).all()
zero_pad_ok_gpu = (gpu_r[1, :, 0] == 0).all()
print(f"iow=1 zero-pad: CPU first col all zero = {zero_pad_ok_cpu}  GPU = {zero_pad_ok_gpu}")

# Check the slot that should be x[iic=0, t=0]:
print(f"\nx[iic=0, t=0]: from CPU im2col = {cpu_r[0, 0, 2]:.6f}, from GPU im2col = {gpu_r[0, 0, 2]:.6f}")
print(f"x[iic=0, t=1]: from CPU im2col @ iow=1, ikw=2 = {cpu_r[1, 0, 2]:.6f}, from GPU = {gpu_r[1, 0, 2]:.6f}")

# Check whole-row sum to see global discrepancy
print(f"\nGlobal: CPU sum={cpu.sum():.3f}  GPU sum={gpu.sum():.3f}")
print(f"Global: CPU rms={np.sqrt((cpu**2).mean()):.6f}  GPU rms={np.sqrt((gpu**2).mean()):.6f}")
print(f"Cosine: {float(np.dot(cpu, gpu) / (np.linalg.norm(cpu)*np.linalg.norm(gpu))):.6f}")
print(f"GPU/CPU max abs ratio at matched indices (non-zero): "
      f"{np.median(np.abs(gpu[cpu != 0]) / np.abs(cpu[cpu != 0])):.3f}")

# Check if GPU is just a permutation/scaled version
print(f"\nGPU range: [{gpu.min():.4f}, {gpu.max():.4f}]")
print(f"CPU range: [{cpu.min():.4f}, {cpu.max():.4f}]")

# How many GPU values exactly equal a CPU value (could be permutation)?
# Use set intersection of float64 hashes.
cpu_set = set(cpu.tolist())
gpu_set = set(gpu.tolist())
print(f"\n|CPU∩GPU| = {len(cpu_set & gpu_set)} / |CPU| = {len(cpu_set)} / |GPU| = {len(gpu_set)}")

# Print the iow=0,iic=0 row pattern for both
print(f"\nCPU iow=0 first 30 floats: {cpu[:30].tolist()}")
print(f"GPU iow=0 first 30 floats: {gpu[:30].tolist()}")
