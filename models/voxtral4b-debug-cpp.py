#!/usr/bin/env python3
"""Compare C++ intermediate results with Python reference for voxtral4b."""

import numpy as np
import sys

# ============================================================
# 1. Compare t_cond: C++ vs Python
# ============================================================
print("=" * 60)
print("1. Compare t_cond")
print("=" * 60)

# C++ t_cond: run a small test binary that dumps it
# For now, compute the Python reference
import math  # noqa: E402

dim = 3072
delay_tokens = 6
theta = 10000.0

inv_freq = np.exp(-math.log(theta) * np.arange(dim // 2, dtype=np.float32) / (dim // 2))
emb = delay_tokens * inv_freq
t_cond_py = np.concatenate([np.cos(emb), np.sin(emb)])

print(f"  Python t_cond[:8]: {t_cond_py[:8].tolist()}")
print(f"  Python t_cond stats: min={t_cond_py.min():.6f} max={t_cond_py.max():.6f}")

# ============================================================
# 2. Load ada_norm weights from GGUF and compute scale
# ============================================================
print()
print("=" * 60)
print("2. Ada-norm from GGUF")
print("=" * 60)

try:
    import gguf

    reader = gguf.GGUFReader(
        "/mnt/akademie_storage/test_cohere/voxtral-mini-4b-realtime.gguf"
    )

    # Find ada_norm tensors for layer 0
    ada_down = ada_up = None
    for tensor in reader.tensors:
        if tensor.name == "blk.0.ada_norm_down.weight":
            ada_down = tensor.data.reshape(tensor.shape).astype(np.float32)
        elif tensor.name == "blk.0.ada_norm_up.weight":
            ada_up = tensor.data.reshape(tensor.shape).astype(np.float32)

    if ada_down is not None and ada_up is not None:
        print(f"  ada_down shape: {ada_down.shape}")  # should be (32, 3072)
        print(f"  ada_up shape: {ada_up.shape}")  # should be (3072, 32)

        # hidden = ada_down @ t_cond
        hidden = ada_down @ t_cond_py
        print(f"  hidden pre-gelu[:8]: {hidden[:8].tolist()}")

        # GELU
        from scipy.special import erf

        hidden_gelu = 0.5 * hidden * (1.0 + erf(hidden / math.sqrt(2.0)))
        print(f"  hidden post-gelu[:8]: {hidden_gelu[:8].tolist()}")

        # scale = ada_up @ hidden
        scale = ada_up @ hidden_gelu
        one_plus_scale = 1.0 + scale
        print(f"  scale[:8]: {scale[:8].tolist()}")
        print(f"  (1+scale)[:8]: {one_plus_scale[:8].tolist()}")
        print(
            f"  scale stats: min={scale.min():.6f} max={scale.max():.6f} mean={scale.mean():.6f}"
        )

        np.save("/tmp/v4b-ref-ada_scale_gguf_l0.npy", one_plus_scale)
    else:
        print("  Ada-norm tensors not found in GGUF")
except Exception as e:
    import traceback

    print(f"  ERROR: {e}")
    traceback.print_exc()

# ============================================================
# 3. Compute CORRECT mel and compare with fixed max
# ============================================================
print()
print("=" * 60)
print("3. Mel with global_log_mel_max=1.5 (VoxtralRealtime style)")
print("=" * 60)

try:
    import scipy.io.wavfile as wavfile
    from scipy.signal import get_window
    from scipy.fft import rfft

    sr, data = wavfile.read("samples/jfk.wav")
    audio = (
        data.astype(np.float32) / 32768.0
        if data.dtype == np.int16
        else data.astype(np.float32)
    )

    n_fft, hop, n_mels, n_freqs = 400, 160, 128, 201
    hann = get_window("hann", n_fft, fftbins=True).astype(np.float32)

    try:
        from librosa.filters import mel as librosa_mel

        mel_filters = librosa_mel(sr=16000, n_fft=n_fft, n_mels=n_mels).T
    except ImportError:
        print("  librosa not available, skipping mel check")
        sys.exit(0)

    pad = n_fft // 2
    padded = np.pad(audio, (pad, pad), mode="constant")
    T = (len(padded) - n_fft) // hop + 1 - 1

    power = np.zeros((n_freqs, T), dtype=np.float32)
    for t in range(T):
        frame = padded[t * hop : t * hop + n_fft] * hann
        spectrum = rfft(frame)
        power[:, t] = np.abs(spectrum) ** 2

    mel = np.zeros((n_mels, T), dtype=np.float32)
    for m in range(n_mels):
        for t in range(T):
            mel[m, t] = np.log10(max(np.dot(mel_filters[:, m], power[:, t]), 1e-10))

    # VoxtralRealtime normalization with FIXED max
    global_log_mel_max = 1.5
    floor_v = global_log_mel_max - 8.0  # -6.5
    mel_fixed = np.clip(mel, floor_v, None)
    mel_fixed = (mel_fixed + 4.0) / 4.0

    # Pad to 3000
    T_out = 3000
    mel_padded = np.full((n_mels, T_out), (floor_v + 4.0) / 4.0, dtype=np.float32)
    mel_padded[:, : min(T, T_out)] = mel_fixed[:, : min(T, T_out)]

    print(f"  mel (fixed max) shape: {mel_padded.shape}")
    print(
        f"  mel stats: min={mel_padded.min():.4f} max={mel_padded.max():.4f} mean={mel_padded.mean():.4f}"
    )
    print(f"  mel[0,:5]: {mel_padded[0,:5].tolist()}")
    print(f"  mel[64,:5]: {mel_padded[64,:5].tolist()}")

    np.save("/tmp/v4b-ref-mel-fixed.npy", mel_padded)
    print("  saved to /tmp/v4b-ref-mel-fixed.npy")

except Exception as e:
    import traceback

    print(f"  ERROR: {e}")
    traceback.print_exc()

print("\nDone.")
