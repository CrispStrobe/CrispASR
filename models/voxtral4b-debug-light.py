#!/usr/bin/env python3
"""
Lightweight Voxtral 4B debug — runs on 2GB RAM by NOT loading the full model.
Checks: mel features, t_cond, ada_norm scales, and individual tensor loads.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

model_dir = Path("/mnt/akademie_storage/voxtral-4b-realtime")
audio_path = "samples/jfk.wav"

# ============================================================
# 1. Time embedding (t_cond) — pure math, no model needed
# ============================================================
print("=" * 60)
print("1. Time embedding (t_cond) for delay_tokens=6")
print("=" * 60)

dim = 3072
theta = 10000.0
delay_tokens = 6

inv_freq = np.exp(-math.log(theta) * np.arange(dim // 2, dtype=np.float32) / (dim // 2))
emb = delay_tokens * inv_freq
t_cond = np.concatenate([np.cos(emb), np.sin(emb)])

print(f"  dim={dim}, delay_tokens={delay_tokens}")
print(f"  inv_freq[:8] = {inv_freq[:8].tolist()}")
print(f"  t_cond[:8]   = {t_cond[:8].tolist()}")
print(f"  t_cond[1536:1544] = {t_cond[1536:1544].tolist()}")  # sin part
print(
    f"  t_cond stats: min={t_cond.min():.6f} max={t_cond.max():.6f} mean={t_cond.mean():.6f}"
)
np.save("/tmp/v4b-ref-t_cond.npy", t_cond)
print("  saved to /tmp/v4b-ref-t_cond.npy")

# ============================================================
# 2. Ada-norm scale for layer 0 (load just 2 small tensors)
# ============================================================
print()
print("=" * 60)
print("2. Ada-norm scale for layer 0")
print("=" * 60)

try:
    from safetensors import safe_open

    f = safe_open(str(model_dir / "model.safetensors"), framework="numpy")

    ada_down = f.get_tensor("language_model.model.layers.0.ada_rms_norm.linear1.weight")
    ada_up = f.get_tensor("language_model.model.layers.0.ada_rms_norm.linear2.weight")
    print(f"  ada_down shape: {ada_down.shape} dtype: {ada_down.dtype}")
    print(f"  ada_up   shape: {ada_up.shape} dtype: {ada_up.dtype}")

    # Convert from bf16 if needed
    if ada_down.dtype == np.uint16:
        # bf16 stored as uint16 — convert manually
        import torch

        ada_down = (
            torch.from_numpy(ada_down.view(np.int16)).to(torch.bfloat16).float().numpy()
        )
        ada_up = (
            torch.from_numpy(ada_up.view(np.int16)).to(torch.bfloat16).float().numpy()
        )
    elif "bfloat" in str(ada_down.dtype):
        ada_down = ada_down.astype(np.float32)
        ada_up = ada_up.astype(np.float32)

    # hidden = ada_down @ t_cond  (32, 3072) @ (3072,) -> (32,)
    hidden = ada_down @ t_cond
    print(f"  hidden (pre-gelu) [:8] = {hidden[:8].tolist()}")

    # GELU
    from scipy.special import erf

    hidden_gelu = 0.5 * hidden * (1.0 + erf(hidden / math.sqrt(2.0)))
    print(f"  hidden (post-gelu) [:8] = {hidden_gelu[:8].tolist()}")

    # scale = ada_up @ hidden  (3072, 32) @ (32,) -> (3072,)
    scale = ada_up @ hidden_gelu
    print(f"  scale[:8] = {scale[:8].tolist()}")
    print(
        f"  scale stats: min={scale.min():.6f} max={scale.max():.6f} mean={scale.mean():.6f}"
    )
    print(f"  (1+scale)[:8] = {(1+scale)[:8].tolist()}")

    np.save("/tmp/v4b-ref-ada_scale_layer0.npy", scale)
    print("  saved to /tmp/v4b-ref-ada_scale_layer0.npy")
    del f
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================
# 3. Mel features comparison
# ============================================================
print()
print("=" * 60)
print("3. Mel features from HF feature extractor vs our C++ mel")
print("=" * 60)

try:
    # Load audio
    import scipy.io.wavfile as wavfile

    sr, data = wavfile.read(audio_path)
    assert sr == 16000
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)
    print(f"  audio: {len(audio)} samples, {len(audio)/16000:.2f}s")

    # Try to use the HF feature extractor (lightweight, doesn't need model)
    with open(model_dir / "processor_config.json") as pf:
        proc_cfg = json.load(pf)
    fe_cfg = proc_cfg.get("feature_extractor", {})
    print("  HF feature extractor config:")
    for k, v in sorted(fe_cfg.items()):
        print(f"    {k}: {v}")

    # Check if global_log_mel_max is used (this is a key difference!)
    global_log_mel_max = fe_cfg.get("global_log_mel_max", None)
    print(f"\n  *** global_log_mel_max = {global_log_mel_max} ***")
    if global_log_mel_max is not None:
        print("  THIS IS NOT USED IN OUR C++ MEL — likely the bug!")
        print(
            "  The HF extractor clips log-mel to [-global_log_mel_max, global_log_mel_max]"
        )
        print("  and normalizes differently than Whisper's (v-floor)/(4) formula")

    # Try loading the actual HF feature extractor
    try:
        sys.path.insert(
            0, str(Path(__file__).parent.parent / "ref" / "voxtral_realtime")
        )
        # Can't easily import VoxtralRealtimeFeatureExtractor without full transformers 5
        # Let's just compute mel manually with the HF algorithm
        pass
    except Exception:
        pass

    # Compute mel the Whisper way (what our C++ does)
    n_fft = 400
    hop = 160
    n_mels = 128
    n_freqs = n_fft // 2 + 1

    # Load mel filterbank from GGUF
    import gguf

    reader = gguf.GGUFReader(
        str(Path("test_cohere") / "voxtral-mini-4b-realtime.gguf").replace(
            "test_cohere", "/mnt/akademie_storage/test_cohere"
        )
    )
    # Actually let's just compute from scratch
    from scipy.signal import get_window
    from scipy.fft import rfft

    # Hann window
    hann = get_window("hann", n_fft, fftbins=True).astype(np.float32)

    # Mel filterbank (same as Whisper)
    try:
        from librosa.filters import mel as librosa_mel

        mel_filters = librosa_mel(
            sr=16000, n_fft=n_fft, n_mels=n_mels
        ).T  # (n_freqs, n_mels)
    except ImportError:
        print("  WARNING: librosa not available, using scipy mel")
        mel_filters = None

    if mel_filters is not None:
        # Center-pad
        pad = n_fft // 2
        padded = np.pad(audio, (pad, pad), mode="constant")

        T_full = (len(padded) - n_fft) // hop + 1
        T = T_full - 1  # Whisper drops last frame

        # STFT -> power spectrum
        power = np.zeros((n_freqs, T), dtype=np.float32)
        for t in range(T):
            frame = padded[t * hop : t * hop + n_fft] * hann
            spectrum = rfft(frame)
            power[:, t] = np.abs(spectrum) ** 2

        # Mel projection + log10
        mel = np.zeros((n_mels, T), dtype=np.float32)
        for m in range(n_mels):
            for t in range(T):
                mel[m, t] = np.log10(max(np.dot(mel_filters[:, m], power[:, t]), 1e-10))

        mel_max = mel.max()
        floor_v = mel_max - 8.0
        mel_clipped = np.clip(mel, floor_v, None)
        mel_whisper = (mel_clipped + 4.0) / 4.0

        print(f"\n  Whisper-style mel (our C++): shape={mel_whisper.shape}")
        print(
            f"    min={mel_whisper.min():.4f} max={mel_whisper.max():.4f} mean={mel_whisper.mean():.4f}"
        )
        print(f"    mel_max={mel_max:.4f} floor={floor_v:.4f}")

        # Now compute mel the VoxtralRealtime way (if global_log_mel_max is set)
        if global_log_mel_max is not None:
            # The HF feature extractor uses a different normalization:
            # From feature_extraction_voxtral_realtime.py:
            # log_spec = np.log(np.clip(magnitudes, a_min=1e-10, a_max=None))
            # log_spec = np.clip(log_spec / global_log_mel_max, a_min=-1.0, a_max=1.0)
            mel_ln = np.zeros((n_mels, T), dtype=np.float32)
            for m in range(n_mels):
                for t in range(T):
                    val = max(np.dot(mel_filters[:, m], power[:, t]), 1e-10)
                    mel_ln[m, t] = np.log(val)  # natural log, NOT log10!

            mel_voxtral = np.clip(mel_ln / global_log_mel_max, -1.0, 1.0)

            print(f"\n  VoxtralRealtime-style mel: shape={mel_voxtral.shape}")
            print(
                f"    min={mel_voxtral.min():.4f} max={mel_voxtral.max():.4f} mean={mel_voxtral.mean():.4f}"
            )
            print(f"    mel_ln max={mel_ln.max():.4f}")

            # Compare
            diff = np.abs(
                mel_whisper[:, : min(T, 3000)] - mel_voxtral[:, : min(T, 3000)]
            )
            print("\n  *** DIFFERENCE between Whisper and VoxtralRealtime mel ***")
            print(f"    max diff: {diff.max():.4f}")
            print(f"    mean diff: {diff.mean():.4f}")
            print(f"    Whisper[0,:5]: {mel_whisper[0,:5].tolist()}")
            print(f"    Voxtral[0,:5]: {mel_voxtral[0,:5].tolist()}")

            np.save("/tmp/v4b-ref-mel-whisper.npy", mel_whisper)
            np.save("/tmp/v4b-ref-mel-voxtral.npy", mel_voxtral)
            print("\n  Saved both mel variants to /tmp/v4b-ref-mel-*.npy")
except Exception as e:
    import traceback

    print(f"  ERROR: {e}")
    traceback.print_exc()

# ============================================================
# 4. Check feature extractor source for exact mel formula
# ============================================================
print()
print("=" * 60)
print("4. Feature extractor analysis")
print("=" * 60)

fe_path = (
    Path(__file__).parent.parent
    / "ref"
    / "voxtral_realtime"
    / "feature_extraction_voxtral_realtime.py"
)
if fe_path.exists():
    with open(fe_path) as f:
        src = f.read()
    # Find the mel computation
    for line_no, line in enumerate(src.split("\n"), 1):
        if any(
            kw in line
            for kw in [
                "log_spec",
                "global_log_mel",
                "magnitudes",
                "mel_filters",
                "np.log",
                "clip",
                "normalize",
            ]
        ):
            print(f"  {line_no:4d}: {line.rstrip()}")
else:
    print(f"  {fe_path} not found")

print("\nDone.")
