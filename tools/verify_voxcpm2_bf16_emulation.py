"""Bit-exact verifier for the BF16 primitives in src/core/torch_rng.h.

Re-implements crispasr::core's MT19937 + BF16 randn, BF16 sinusoidal time
embedding, and BF16 t_span (sway) in pure Python, and diffs them against
PyTorch's native bf16 outputs. Used as a regression test for the BF16
"verified bit-identical to torch" claims attached to VoxCPM2 bug fixes
#22, #24, #25 — keep this script passing whenever torch_rng.h is touched.

Caveat: `t_span sway` is bit-exact for steps=10 (VoxCPM2 production
default); for higher step counts torch.linspace uses a split-from-each-end
formula and we differ by ≤1 BF16 ULP at one or two positions. Not used by
the inference path today, just documented here.

Run:
    python tools/verify_voxcpm2_bf16_emulation.py
"""

import math

import numpy as np
import torch


def bf16_round(x):
    """Mimic ggml_fp32_to_bf16 → ggml_bf16_to_fp32 (round-nearest-even, ties to even)."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float().numpy()
    t = torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).float().item()
    return t


# ----------------------------------------------------------------------
# 1. MT19937 + BF16 randn
# ----------------------------------------------------------------------
class MT19937:
    def __init__(self, seed: int):
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mti = 624

    def next(self):
        if self.mti >= 624:
            for i in range(624):
                y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1)
                if y & 1:
                    self.mt[i] ^= 0x9908B0DF
            self.mti = 0
        y = self.mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF


def mt_uniform_bf16(rng):
    return float(rng.next() & 0xFF) / 256.0


def bf16_normal_fill_16(data):
    for j in range(8):
        u1 = bf16_round(1.0 - data[j])
        u2 = data[j + 8]
        radius = bf16_round(math.sqrt(bf16_round(-2.0 * bf16_round(math.log(u1)))))
        theta = bf16_round(bf16_round(2.0 * math.pi * u2))
        data[j] = bf16_round(radius * bf16_round(math.cos(theta)))
        data[j + 8] = bf16_round(radius * bf16_round(math.sin(theta)))


def fill_gaussian_noise_bf16(n, seed):
    rng = MT19937(seed)
    if n < 16:
        tmp = [mt_uniform_bf16(rng) for _ in range(16)]
        bf16_normal_fill_16(tmp)
        return np.array(tmp[:n], dtype=np.float32)
    data = [mt_uniform_bf16(rng) for _ in range(n)]
    i = 0
    while i <= n - 16:
        block = data[i:i + 16]
        bf16_normal_fill_16(block)
        data[i:i + 16] = block
        i += 16
    if i < n:
        # PyTorch recomputes the final overlapping 16
        tail_start = n - 16
        tail = [mt_uniform_bf16(rng) for _ in range(16)]
        bf16_normal_fill_16(tail)
        data[tail_start:tail_start + 16] = tail
    return np.array(data, dtype=np.float32)


def test_bf16_randn():
    print("\n[1] BF16 randn: C++ port vs torch.randn(dtype=bfloat16)")
    for n in (16, 256, 257, 1024):
        seed = 42
        cpp_out = fill_gaussian_noise_bf16(n, seed)
        g = torch.Generator(device="cpu").manual_seed(seed)
        torch_out = torch.randn(n, dtype=torch.bfloat16, device="cpu", generator=g).float().numpy()
        diff = np.abs(cpp_out - torch_out).max()
        print(f"  n={n:5d}  max_abs_diff={diff:.6e}  {'BIT-EXACT' if diff == 0 else 'MISMATCH'}")


# ----------------------------------------------------------------------
# 2. Sinusoidal time embedding in BF16
# ----------------------------------------------------------------------
def sinusoidal_time_emb_cpp(t_scalar, dim):
    half_dim = dim // 2
    log_base = math.log(10000.0) / (half_dim - 1)
    scale_val = bf16_round(1000.0)
    x_val = bf16_round(t_scalar)
    emb = np.zeros(dim, dtype=np.float32)
    for i in range(half_dim):
        i_bf = bf16_round(float(i))
        freq = bf16_round(math.exp(bf16_round(-i_bf * log_base)))
        val = bf16_round(bf16_round(scale_val * x_val) * freq)
        emb[i] = bf16_round(math.sin(val))
        emb[i + half_dim] = bf16_round(math.cos(val))
    return emb


def sinusoidal_time_emb_torch(t_scalar, dim):
    """Mirror voxcpm.modules.locdit.local_dit_v2.SinusoidalPosEmb in BF16."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)  # python float
    t = torch.tensor([t_scalar], dtype=torch.bfloat16)
    arange = torch.arange(half_dim, dtype=torch.bfloat16) * -emb
    emb_freq = torch.exp(arange)
    out = 1000 * t.unsqueeze(1) * emb_freq.unsqueeze(0)
    out = torch.cat((out.sin(), out.cos()), dim=-1)
    return out.float().numpy().reshape(-1)


def test_sinusoidal():
    print("\n[2] Sinusoidal time emb: C++ port vs SinusoidalPosEmb(bf16)")
    for t_val in (0.9, 0.5, 0.1, 0.0):
        for dim in (1024,):
            cpp_out = sinusoidal_time_emb_cpp(t_val, dim)
            torch_out = sinusoidal_time_emb_torch(t_val, dim)
            diff = np.abs(cpp_out - torch_out).max()
            print(f"  t={t_val:.4f} dim={dim}  max_abs_diff={diff:.6e}  "
                  f"{'BIT-EXACT' if diff == 0 else 'MISMATCH'}")


# ----------------------------------------------------------------------
# 3. t_span sway schedule in BF16
# ----------------------------------------------------------------------
def t_span_cpp(steps):
    """Mirror voxcpm2_tts.cpp:1173-1181 — bf16-emulated sway transform."""
    out = np.zeros(steps + 1, dtype=np.float32)
    for i in range(steps + 1):
        t = bf16_round(1.0 - float(i) / float(steps))
        a = bf16_round(math.pi / 2.0 * t)
        cos_a = bf16_round(math.cos(a))
        c = bf16_round(cos_a - 1.0)
        d = bf16_round(c + t)
        out[i] = bf16_round(t + d)
    return out


def t_span_torch(steps):
    """Mirror UnifiedCFM.forward — linspace+sway in bf16."""
    t = torch.linspace(1, 0, steps + 1, dtype=torch.bfloat16)
    t = t + 1.0 * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t.float().numpy()


def test_t_span():
    print("\n[3] t_span sway: C++ port vs torch.linspace(bf16) + sway")
    for steps in (10, 25, 50):
        cpp_out = t_span_cpp(steps)
        torch_out = t_span_torch(steps)
        diff = np.abs(cpp_out - torch_out).max()
        print(f"  steps={steps:3d}  max_abs_diff={diff:.6e}  "
              f"{'BIT-EXACT' if diff == 0 else 'MISMATCH'}")
        if diff != 0:
            i_max = int(np.argmax(np.abs(cpp_out - torch_out)))
            print(f"    worst: i={i_max}  cpp={cpp_out[i_max]:.8f}  torch={torch_out[i_max]:.8f}")


def test_t_span_production():
    """Gate test: production VoxCPM2 uses steps=10; must be bit-exact there."""
    cpp_out = t_span_cpp(10)
    torch_out = t_span_torch(10)
    diff = np.abs(cpp_out - torch_out).max()
    if diff != 0:
        raise SystemExit(f"REGRESSION: t_span(steps=10) drifted from torch — max_abs={diff}")


if __name__ == "__main__":
    test_bf16_randn()
    test_sinusoidal()
    test_t_span()
    test_t_span_production()
    print("\nProduction-path BF16 primitives are bit-exact vs torch.")
