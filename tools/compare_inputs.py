"""Compare CPU vs GPU unet inputs (post-upload, post-step-0 compute)."""
import numpy as np
from pathlib import Path

tmp = Path("/tmp")
for nm, n_expected in [("unet_input", 382 * 320), ("time_emb", 1024), ("mask", 382)]:
    cpu = np.frombuffer(open(tmp / f"cb-unet-dump-cpu-probe-input_{nm}.bin", "rb").read(), dtype=np.float32)
    gpu = np.frombuffer(open(tmp / f"cb-unet-dump-gpu-probe-input_{nm}.bin", "rb").read(), dtype=np.float32)
    print(f"\n=== {nm} (CPU n={cpu.size}, GPU n={gpu.size}, expected n={n_expected}) ===")
    if cpu.size != gpu.size:
        print(f"  SIZE MISMATCH")
        continue
    nf_cpu = (~np.isfinite(cpu)).sum()
    nf_gpu = (~np.isfinite(gpu)).sum()
    cpu_f = cpu.copy(); cpu_f[~np.isfinite(cpu_f)] = 0
    gpu_f = gpu.copy(); gpu_f[~np.isfinite(gpu_f)] = 0
    bit_identical = bool(np.array_equal(cpu, gpu))
    cos = float(np.dot(cpu_f, gpu_f) / (np.linalg.norm(cpu_f) * np.linalg.norm(gpu_f) + 1e-30))
    max_abs = float(np.max(np.abs(cpu_f - gpu_f)))
    print(f"  nf_cpu={nf_cpu}, nf_gpu={nf_gpu}")
    print(f"  bit_identical={bit_identical}, cos={cos:.6f}, max_abs_diff={max_abs:.6e}")
    print(f"  CPU first 8: {cpu[:8].tolist()}")
    print(f"  GPU first 8: {gpu[:8].tolist()}")
    if nm == "unet_input":
        # Reshape: ne=(T_mel=382, C=320) ggml layout means
        # data[ch * T_mel + t]: channel-first, T is fast axis.
        # First 80 ch are x (noise), next 80 ch are mu, next 80 ch are spk_emb (broadcast over T),
        # last 80 ch are cond.
        cpu2 = cpu.reshape(320, 382)
        gpu2 = gpu.reshape(320, 382)
        print(f"  CPU x[:5, t=0] (noise): {cpu2[:5, 0].tolist()}")
        print(f"  GPU x[:5, t=0]        : {gpu2[:5, 0].tolist()}")
        print(f"  CPU mu[:5, t=0] (ch 80..84): {cpu2[80:85, 0].tolist()}")
        print(f"  GPU mu[:5, t=0]            : {gpu2[80:85, 0].tolist()}")
        print(f"  CPU spk_emb[:5] (ch 160..164, all t same): {cpu2[160:165, 0].tolist()}")
        print(f"  GPU spk_emb[:5]                          : {gpu2[160:165, 0].tolist()}")
        print(f"  CPU cond[:5, t=0] (ch 240..244): {cpu2[240:245, 0].tolist()}")
        print(f"  GPU cond[:5, t=0]              : {gpu2[240:245, 0].tolist()}")
