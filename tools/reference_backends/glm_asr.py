"""GLM-ASR GGUF-direct reference backend.

Loads weights from the same GGUF as the C++ runtime and reproduces the
full audio encoder forward pass in NumPy. No PyTorch model required.

Architecture (nano): 128-mel → conv stem → 32-layer transformer encoder
with partial NeoX RoPE (50%) → 4-frame stacking projector → (T_proj, 2048).

Stages:
  raw_audio       (N,)           input PCM
  mel_spectrogram (n_mels, T_mel) MelsTime layout matching glm_asr_compute_mel
  encoder_output  (T_proj, 2048) projector output matching glm_asr_run_encoder

The mel and encoder outputs match the C++ implementation at cos_min=1.000
(modulo float32/float64 rounding) when both use the same F32 GGUF file.

Weight layout note: GGUFReader returns t.data with axes REVERSED from the
GGUF shape (numpy convention). After np.ascontiguousarray(data.T) the array
has the same axis order as the GGUF shape, so for a 2D weight w[i0, i1] in
Python corresponds to ggml element (i0, i1). For linear layers this means
"X @ W" in Python equals ggml_mul_mat(W, X).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "mel_spectrogram", "encoder_output"]


# ---------------------------------------------------------------------------
# GGUF weight loader
# ---------------------------------------------------------------------------

def _load_gguf_weights(gguf_path: Path) -> Dict[str, np.ndarray]:
    from gguf import GGUFReader
    r = GGUFReader(str(gguf_path))
    out: Dict[str, np.ndarray] = {}
    for t in r.tensors:
        data = np.array(t.data, dtype=np.float32)
        # GGUFReader reverses axes: GGUF shape [ne0, ne1] → t.data.shape = (ne1, ne0).
        # Transpose so out[t.name][i0, i1] == ggml element (i0, i1).
        out[t.name] = np.ascontiguousarray(data.T)
    return out


# ---------------------------------------------------------------------------
# Activation and norm helpers
# ---------------------------------------------------------------------------

def _gelu_erf(x: np.ndarray) -> np.ndarray:
    """Exact GELU: x * 0.5 * (1 + erf(x/sqrt(2))). Matches ggml_gelu_erf."""
    from scipy.special import erf as _erf
    return (x * 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))).astype(np.float32)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                eps: float = 1e-5) -> np.ndarray:
    """Pre-norm LayerNorm with bias. x: (..., d). w, b: (d,)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * w + b).astype(np.float32)


# ---------------------------------------------------------------------------
# Mel spectrogram (matches glm_asr_compute_mel)
# ---------------------------------------------------------------------------

def _compute_mel(audio: np.ndarray,
                 mel_filters: np.ndarray,
                 mel_window: np.ndarray) -> np.ndarray:
    """Compute GLM-ASR mel spectrogram.

    Params match the C++ core_mel call:
      n_fft=400, hop=160, win=400, center_pad, log10, GlobalClipMax norm,
      MelsTime layout. FFT zero-padded to 512 (next power-of-2 >= 400).

    mel_filters: (n_mels=128, n_freqs=201) float32 — after GGUF .T.
    mel_window: (400,) float32.
    Returns: (n_mels=128, T_mel) float32.
    """
    n_fft = 400
    hop = 160
    fft_size = 512  # next power-of-2 >= 400
    n_freqs = n_fft // 2 + 1  # 201

    # Centre-pad: pad by n_fft//2 on each side (reflect mode)
    pad = n_fft // 2
    audio_d = audio.astype(np.float64)
    padded = np.pad(audio_d, (pad, pad), mode="reflect")

    # Frame
    n_padded = len(padded)
    n_frames = (n_padded - n_fft) // hop + 1
    window_d = mel_window.astype(np.float64)

    power = np.empty((n_freqs, n_frames), dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        frame = padded[s : s + n_fft] * window_d
        # Zero-pad to fft_size then take rfft
        buf = np.zeros(fft_size, dtype=np.float64)
        buf[:n_fft] = frame
        spec = np.fft.rfft(buf)[:n_freqs]
        power[:, i] = spec.real ** 2 + spec.imag ** 2

    # mel_filters: (128, 201). mel_spec = mel_filters @ power → (128, T_mel)
    mel_spec = mel_filters.astype(np.float64) @ power

    # log10 with eps=1e-10
    mel_spec = np.log10(np.maximum(mel_spec, 1e-10))

    # GlobalClipMax normalization (Whisper-style):
    #   clip below (max - 8), then affine to [-1, 1] approx range
    max_val = mel_spec.max()
    mel_spec = np.maximum(mel_spec, max_val - 8.0)
    mel_spec = (mel_spec + 4.0) / 4.0

    return mel_spec.astype(np.float32)  # (128, T_mel)


# ---------------------------------------------------------------------------
# 1D convolution
# ---------------------------------------------------------------------------

def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray,
            stride: int, pad: int) -> np.ndarray:
    """Causal-padded conv1d. x: (T, in_ch). w: (k, in_ch, out_ch). b: (out_ch,).
    Returns (T_out, out_ch) float32."""
    if pad > 0:
        x = np.pad(x, ((pad, pad), (0, 0)), mode="constant")
    T_in, _ = x.shape
    k, _, out_ch = w.shape
    T_out = (T_in - k) // stride + 1
    out = np.zeros((T_out, out_ch), dtype=np.float32)
    for ki in range(k):
        idx = np.arange(T_out) * stride + ki
        out += x[idx] @ w[ki]  # (T_out, in_ch) @ (in_ch, out_ch)
    out += b
    return out


# ---------------------------------------------------------------------------
# NeoX RoPE (partial)
# ---------------------------------------------------------------------------

def _rope_neox(x: np.ndarray, positions: np.ndarray,
               rope_dim: int, base: float = 10000.0) -> np.ndarray:
    """Apply NeoX RoPE to the first `rope_dim` dims of x.

    x: (T, n_heads, hd). positions: (T,) int/float.
    NeoX pairs element i with element i+rope_dim//2:
      x_new[i]              = x[i]*cos - x[i+half]*sin
      x_new[i + half]       = x[i]*sin + x[i+half]*cos
    for i in [0, half), where half = rope_dim // 2.
    """
    half = rope_dim // 2
    i_vals = np.arange(half, dtype=np.float32)
    freqs = 1.0 / (base ** (2.0 * i_vals / rope_dim))  # (half,)

    pos_f = positions[:, None].astype(np.float32)  # (T, 1)
    theta = pos_f * freqs[None, :]                  # (T, half)
    cos = np.cos(theta)[:, None, :]  # (T, 1, half)
    sin = np.sin(theta)[:, None, :]  # (T, 1, half)

    x_out = x.copy()
    x1 = x[..., :half]            # (T, n_heads, half)
    x2 = x[..., half:rope_dim]    # (T, n_heads, half)
    x_out[..., :half]          = x1 * cos - x2 * sin
    x_out[..., half:rope_dim]  = x1 * sin + x2 * cos
    return x_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Full encoder forward (matches glm_asr_run_encoder output)
# ---------------------------------------------------------------------------

def _run_encoder(audio: np.ndarray, wts: Dict[str, np.ndarray]) -> np.ndarray:
    """Run the GLM-ASR audio encoder. Returns (T_proj, 2048) float32."""
    n_layers = 32
    n_heads = 20
    hd = 64          # head dim
    d = 1280         # enc_hidden
    rope_dim = 32    # partial_rotary_factor=0.5 → hd*0.5=32
    scale = 1.0 / np.sqrt(hd)

    # --- Mel ---
    mel = _compute_mel(audio, wts["audio.mel_filters"], wts["audio.mel_window"])
    # mel: (128, T_mel). For conv, transpose to (T_mel, 128) = (T, C_in).
    x = mel.T.astype(np.float32)  # (T_mel, 128)

    # --- Conv stem ---
    # conv1.weight GGUF shape [3, 128, 1280] → after .T: (3, 128, 1280) = (k, in_ch, out_ch)
    x = _conv1d(x, wts["audio.conv1.weight"], wts["audio.conv1.bias"], stride=1, pad=1)
    x = _gelu_erf(x)   # (T_mel, 1280)
    x = _conv1d(x, wts["audio.conv2.weight"], wts["audio.conv2.bias"], stride=2, pad=1)
    x = _gelu_erf(x)   # (T_enc, 1280)
    T_enc = x.shape[0]

    # --- 32 × Transformer blocks ---
    positions = np.arange(T_enc, dtype=np.float32)

    for il in range(n_layers):
        pfx = f"audio.blk.{il}"

        # Pre-attention LayerNorm
        xn = _layer_norm(x, wts[f"{pfx}.attn_norm.weight"], wts[f"{pfx}.attn_norm.bias"])

        # Q, K, V projections. Weights: (d, d) after .T → matmul as X @ W.
        Q = xn @ wts[f"{pfx}.attn_q.weight"] + wts[f"{pfx}.attn_q.bias"]  # (T, d)
        K = xn @ wts[f"{pfx}.attn_k.weight"]                               # (T, d) no bias
        V = xn @ wts[f"{pfx}.attn_v.weight"] + wts[f"{pfx}.attn_v.bias"]  # (T, d)

        Q = Q.reshape(T_enc, n_heads, hd)
        K = K.reshape(T_enc, n_heads, hd)
        V = V.reshape(T_enc, n_heads, hd)

        # Partial NeoX RoPE on first rope_dim=32 dims
        Q = _rope_neox(Q, positions, rope_dim)
        K = _rope_neox(K, positions, rope_dim)

        # Bidirectional attention
        Q = Q.transpose(1, 0, 2)  # (n_heads, T, hd)
        K = K.transpose(1, 0, 2)
        V = V.transpose(1, 0, 2)

        attn_w = (Q @ K.transpose(0, 2, 1)) * scale  # (n_heads, T, T)
        attn_w = attn_w - attn_w.max(axis=-1, keepdims=True)  # numerics
        attn_w = np.exp(attn_w)
        attn_w = attn_w / attn_w.sum(axis=-1, keepdims=True)

        attn_out = (attn_w @ V).transpose(1, 0, 2).reshape(T_enc, d)  # (T, d)

        # Output projection + residual
        attn_out = attn_out @ wts[f"{pfx}.attn_out.weight"] + wts[f"{pfx}.attn_out.bias"]
        x = x + attn_out

        # FFN: pre-norm LN → up(GELU) → down + residual
        residual = x
        xn = _layer_norm(x, wts[f"{pfx}.ffn_norm.weight"], wts[f"{pfx}.ffn_norm.bias"])
        xn = xn @ wts[f"{pfx}.ffn.up.weight"] + wts[f"{pfx}.ffn.up.bias"]  # (T, 5120)
        xn = _gelu_erf(xn)
        xn = xn @ wts[f"{pfx}.ffn.down.weight"] + wts[f"{pfx}.ffn.down.bias"]  # (T, d)
        x = residual + xn

    # --- Final LayerNorm ---
    x = _layer_norm(x, wts["audio.norm.weight"], wts["audio.norm.bias"])

    # --- Projector: 4-frame stacking ---
    T_proj = T_enc // 4
    T_pack = T_proj * 4
    x = x[:T_pack].reshape(T_proj, 4 * d)  # (T_proj, 5120)

    x = x @ wts["proj.linear_1.weight"] + wts["proj.linear_1.bias"]  # (T_proj, 4096)
    x = _gelu_erf(x)
    x = x @ wts["proj.linear_2.weight"] + wts["proj.linear_2.bias"]  # (T_proj, 2048)

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Public dump() entry point
# ---------------------------------------------------------------------------

def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run GGUF-direct GLM-ASR encoder reference forward pass.

    model_dir: path to a GLM-ASR GGUF file (F32 recommended for exact match)
               or a directory containing glm-asr-nano.gguf.
    audio:     16 kHz mono float32 PCM.
    """
    gguf_path = Path(model_dir)
    if gguf_path.is_dir():
        candidates = list(gguf_path.glob("glm-asr*.gguf"))
        if not candidates:
            raise SystemExit(f"no glm-asr-*.gguf found in {gguf_path}")
        gguf_path = candidates[0]
    if not gguf_path.exists():
        raise SystemExit(f"GGUF not found: {gguf_path}")

    print(f"  loading GLM-ASR weights from {gguf_path.name}")
    wts = _load_gguf_weights(gguf_path)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    mel = _compute_mel(audio, wts["audio.mel_filters"], wts["audio.mel_window"])
    if "mel_spectrogram" in stages:
        out["mel_spectrogram"] = mel  # (128, T_mel)

    if "encoder_output" in stages:
        print("  running GLM-ASR encoder (32 layers + projector)…")
        out["encoder_output"] = _run_encoder(audio, wts)

    return out
