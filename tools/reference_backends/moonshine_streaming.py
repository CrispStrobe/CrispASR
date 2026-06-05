"""Moonshine-Streaming reference dump backend.

Runs a Python-native forward pass using the **same GGUF weights** that the
C++ runtime loads.  This guarantees that any cosine gap reflects only
algorithmic differences between the C++ and Python implementations, not
weight-version drift.

Architecture (tiny model: 6 enc layers, enc_hidden=320, dec_hidden=320):

  Audio frontend (audio_frontend_cpu equivalent):
    1. Frame + CMVN: reshape PCM → (T, 80) frames; per-frame mean-center + RMS-norm
    2. Asinh compression: asinh(exp(log_k) * x)
    3. Linear(80 → enc_hidden) + SiLU — weight shape [enc_hidden, 80] in ggml
    4. CausalConv1d(enc_hidden → 2*enc_hidden, k=5, s=2) + SiLU
    5. CausalConv1d(2*enc_hidden → enc_hidden, k=5, s=2)

  Transformer encoder:
    Sliding-window multi-head attention with pre-norm (unit-offset LayerNorm)
    + FFN (GELU erf). Final layer norm applied to output.

Stages:

  raw_audio        (N,)        input PCM at 16 kHz
  encoder_output   (T, D)      encoder output, matching moonshine_streaming_encode()
                               in src/moonshine_streaming.h: (T_enc, enc_hidden)
                               row-major float32.

Usage:

  python tools/dump_reference.py --backend moonshine-streaming \\
      --model-dir /mnt/storage/models/moonshine-streaming-tiny.gguf \\
      --audio samples/jfk.wav \\
      --output /mnt/storage/ref/moonshine-streaming-tiny-ref.gguf

`model_dir` may also be a HuggingFace repo id or local directory containing
a GGUF file named `moonshine-streaming-tiny.gguf` (or
`moonshine-streaming-*.gguf`).  If `model_dir` itself is a `.gguf` file the
path is used directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Optional

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "encoder_output",
]

# ---------------------------------------------------------------------------
# GGUF weight loader
# ---------------------------------------------------------------------------

def _load_gguf_weights(gguf_path: Path) -> Dict[str, np.ndarray]:
    """Load all F32/F16 tensors from a GGUF file as float32 numpy arrays."""
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise SystemExit("gguf package required. Install: pip install gguf\n" + str(e))

    r = GGUFReader(str(gguf_path))
    out = {}
    for t in r.tensors:
        data = np.array(t.data, dtype=np.float32)
        # GGUFReader reverses axes: GGUF shape [ne0, ne1] comes back as numpy (ne1, ne0).
        # Transpose to restore ggml layout: out[t.name][i0, i1] == ggml element (i0, i1),
        # so matmul "X @ W" in Python gives the same result as ggml_mul_mat(W_ggml, X_ggml).
        out[t.name] = np.ascontiguousarray(data.T)
    return out


def _find_gguf(model_dir: Path) -> Path:
    """Locate the GGUF file given a directory or direct GGUF path."""
    if model_dir.suffix == ".gguf" and model_dir.is_file():
        return model_dir
    for candidate in sorted(model_dir.glob("moonshine-streaming-*.gguf")):
        return candidate
    for candidate in sorted(model_dir.glob("*.gguf")):
        return candidate
    raise FileNotFoundError(
        f"No GGUF file found in {model_dir}. "
        "Pass the path to the .gguf file directly via --model-dir.")


def _read_scalar(w: Dict[str, np.ndarray], key: str) -> float:
    return float(w[key].flat[0])


# ---------------------------------------------------------------------------
# Audio frontend (mirrors audio_frontend_cpu in moonshine_streaming.cpp)
# ---------------------------------------------------------------------------

def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _gelu_erf(x: np.ndarray) -> np.ndarray:
    """Exact GELU with error function — matches ggml_gelu_erf used in the encoder FFN.
    Formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    try:
        from scipy.special import erf as _scipy_erf
        return x * 0.5 * (1.0 + _scipy_erf(x / np.sqrt(2.0)))
    except ImportError:
        import math
        _erf_f = np.frompyfunc(math.erf, 1, 1)
        return x * 0.5 * (1.0 + _erf_f(x / np.sqrt(2.0)).astype(np.float32))


def _causal_conv1d(x: np.ndarray, weight_ggml: np.ndarray, bias: np.ndarray,
                   kernel_size: int, stride: int) -> np.ndarray:
    """Causal Conv1d on (T, C_in) input → (T_out, C_out).

    weight_ggml has ggml shape (kernel_size, C_in, C_out), stored as
    data[kk + ic * k + oc * k * C_in] = PyTorch weight[oc, ic, kk].
    """
    T, C_in = x.shape
    C_out = bias.shape[0]
    left_pad = kernel_size - 1

    padded = np.zeros((left_pad + T, C_in), dtype=np.float32)
    padded[left_pad:] = x

    T_out = (left_pad + T - kernel_size) // stride + 1
    out = np.empty((T_out, C_out), dtype=np.float32)

    # weight_ggml shape: (kernel_size, C_in, C_out) in ggml layout
    # w[kk, ic, oc] = pytorch_weight[oc, ic, kk]
    for t in range(T_out):
        t_in = t * stride
        window = padded[t_in:t_in + kernel_size, :]  # (k, C_in)
        # output[oc] = bias[oc] + sum_{kk, ic}(window[kk, ic] * weight_ggml[kk, ic, oc])
        out[t] = bias + np.tensordot(window, weight_ggml, axes=([0, 1], [0, 1]))
    return out


def _audio_frontend(pcm: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Run the Moonshine-Streaming audio frontend (mirrors audio_frontend_cpu).

    Returns (T_enc, enc_hidden) float32 array.
    """
    frame_size = 80
    n_samples = len(pcm)
    n_frames = n_samples // frame_size
    if n_frames < 1:
        return np.zeros((0, 0), dtype=np.float32)

    frames = pcm[:n_frames * frame_size].reshape(n_frames, frame_size)

    # --- CMVN (per-frame mean-center + RMS-norm) ---
    mean = frames.mean(axis=-1, keepdims=True)
    centered = frames - mean
    rms = np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-8)
    cmvn_out = centered / rms

    # --- Asinh compression ---
    log_k = _read_scalar(w, "encoder.embedder.log_k")
    k = np.exp(log_k)
    asinh_out = np.arcsinh(k * cmvn_out)  # (T, 80)

    # --- Linear(80 → enc_hidden) + SiLU ---
    # GGUF shape [80, enc_hidden] in ggml (ne0=80 fastest).
    # Data: data[i + o * 80] = lin_w[i, o] = pytorch_weight[o, i]
    # Output[t, o] = sum_i(lin_w[i, o] * asinh_out[t, i])
    #             = (asinh_out @ lin_w_ggml)[t, o]
    lin_w_ggml = w["encoder.embedder.linear.weight"]   # shape (80, enc_hidden)
    lin_out = asinh_out @ lin_w_ggml                   # (T, enc_hidden)
    lin_silu = _silu(lin_out)                          # (T, enc_hidden)

    # --- CausalConv1d 1: enc_hidden → 2*enc_hidden, k=5, s=2, + SiLU ---
    conv1_w = w["encoder.embedder.conv1.weight"]   # ggml (5, enc_hidden, 2*enc_hidden)
    conv1_b = w["encoder.embedder.conv1.bias"]     # (2*enc_hidden,)
    conv1_out = _causal_conv1d(lin_silu, conv1_w, conv1_b, kernel_size=5, stride=2)
    conv1_silu = _silu(conv1_out)                  # (T_conv1, 2*enc_hidden)

    # --- CausalConv1d 2: 2*enc_hidden → enc_hidden, k=5, s=2, no activation ---
    conv2_w = w["encoder.embedder.conv2.weight"]   # ggml (5, 2*enc_hidden, enc_hidden)
    conv2_b = w["encoder.embedder.conv2.bias"]     # (enc_hidden,)
    frontend_out = _causal_conv1d(conv1_silu, conv2_w, conv2_b, kernel_size=5, stride=2)
    # (T_enc, enc_hidden)

    return frontend_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Transformer encoder (mirrors run_encoder in moonshine_streaming.cpp)
# ---------------------------------------------------------------------------

def _run_encoder(frontend_out: np.ndarray, w: Dict[str, np.ndarray],
                 n_layers: int, n_heads: int, head_dim: int,
                 sliding_windows: list) -> np.ndarray:
    """Run the sliding-window transformer encoder.

    Input: (T_enc, enc_hidden). Output: (T_enc, enc_hidden).
    """
    T, D = frontend_out.shape
    ln_eps = 1e-5

    cur = frontend_out.copy()  # (T, D)

    for li in range(n_layers):
        residual = cur

        # Pre-norm (unit-offset: weight = gamma + 1, baked at convert time)
        attn_norm_w = w[f"encoder.layers.{li}.attn_norm.weight"]  # (D,)
        # ggml norm: (x - mean) / sqrt(var + eps)  * weight (no bias)
        mu = cur.mean(axis=-1, keepdims=True)
        diff = cur - mu
        var = (diff ** 2).mean(axis=-1, keepdims=True)
        normed = diff / np.sqrt(var + ln_eps) * attn_norm_w  # (T, D)

        # Q/K/V projections
        # ggml weight shape (D, D) = ne[0]=D, ne[1]=D
        # data[di + dj * D] = w[di][dj] = pytorch_weight[dj, di]
        # output[t, dj] = sum_di(normed[t, di] * w_ggml[di, dj]) = normed @ w_ggml
        q_w = w[f"encoder.layers.{li}.attn.q.weight"]  # (D, D) ggml
        k_w = w[f"encoder.layers.{li}.attn.k.weight"]  # (D, D) ggml
        v_w = w[f"encoder.layers.{li}.attn.v.weight"]  # (D, D) ggml
        o_w = w[f"encoder.layers.{li}.attn.o.weight"]  # (D, D) ggml

        Q = normed @ q_w  # (T, D)
        K = normed @ k_w  # (T, D)
        V = normed @ v_w  # (T, D)

        # Reshape to multi-head: (T, n_heads, head_dim)
        Q = Q.reshape(T, n_heads, head_dim)
        K = K.reshape(T, n_heads, head_dim)
        V = V.reshape(T, n_heads, head_dim)

        # Sliding-window attention per head
        wl, wr = sliding_windows[li]
        scale = 1.0 / np.sqrt(head_dim)
        attn_out = np.zeros((T, n_heads, head_dim), dtype=np.float32)
        for h in range(n_heads):
            Qh = Q[:, h, :]  # (T, head_dim)
            Kh = K[:, h, :]
            Vh = V[:, h, :]
            # Attention scores (T, T)
            scores = (Qh @ Kh.T) * scale
            # Apply sliding window mask (causal + window)
            for tq in range(T):
                for tk in range(T):
                    if not (tk >= tq - wl and tk <= tq + wr):
                        scores[tq, tk] = -np.inf
            # Softmax
            scores -= scores.max(axis=-1, keepdims=True)
            attn_w = np.exp(scores)
            attn_w /= attn_w.sum(axis=-1, keepdims=True) + 1e-10
            attn_out[:, h, :] = attn_w @ Vh

        attn_out = attn_out.reshape(T, D)
        attn_out = attn_out @ o_w   # project output (T, D)
        cur = residual + attn_out

        # FFN: pre-norm + fc1 + SiLU + fc2 + residual
        residual = cur
        ffn_norm_w = w[f"encoder.layers.{li}.ffn_norm.weight"]
        mu = cur.mean(axis=-1, keepdims=True)
        diff = cur - mu
        var = (diff ** 2).mean(axis=-1, keepdims=True)
        normed = diff / np.sqrt(var + ln_eps) * ffn_norm_w

        fc1_w = w[f"encoder.layers.{li}.ffn.fc1.weight"]  # (D, intermediate) ggml
        fc2_w = w[f"encoder.layers.{li}.ffn.fc2.weight"]  # (intermediate, D) ggml
        fc1_b_key = f"encoder.layers.{li}.ffn.fc1.bias"
        fc2_b_key = f"encoder.layers.{li}.ffn.fc2.bias"

        fn = normed @ fc1_w
        if fc1_b_key in w:
            fn += w[fc1_b_key]
        fn = _gelu_erf(fn)
        fn = fn @ fc2_w
        if fc2_b_key in w:
            fn += w[fc2_b_key]
        cur = fn + residual

    # Final output norm
    out_norm_w = w["encoder.output_norm.weight"]
    mu = cur.mean(axis=-1, keepdims=True)
    diff = cur - mu
    var = (diff ** 2).mean(axis=-1, keepdims=True)
    enc_out = diff / np.sqrt(var + ln_eps) * out_norm_w

    return enc_out.astype(np.float32)


# ---------------------------------------------------------------------------
# GGUF hyperparameter reader
# ---------------------------------------------------------------------------

def _read_hparams(gguf_path: Path) -> Dict:
    try:
        from gguf import GGUFReader
    except ImportError as e:
        raise SystemExit("gguf required: pip install gguf\n" + str(e))

    r = GGUFReader(str(gguf_path))
    fields = {}
    for k, v in r.fields.items():
        try:
            part = v.parts[v.data[0]]
            if hasattr(part, '__len__') and len(part) == 1:
                fields[k] = part[0]
            else:
                fields[k] = part
        except Exception:
            pass

    prefix = "moonshine_streaming."

    def g_u32(key, default=0):
        return int(fields.get(prefix + key, default))

    def g_f32(key, default=0.0):
        return float(fields.get(prefix + key, default))

    enc_n_layers = g_u32("encoder.block_count")
    enc_n_heads = g_u32("encoder.attention.head_count")
    enc_head_dim = g_u32("encoder.attention.head_dim",
                         g_u32("encoder.embedding_length") // max(1, enc_n_heads))

    # Per-layer sliding windows
    sliding_windows = []
    for i in range(enc_n_layers):
        wl = g_u32(f"encoder.layers.{i}.window_left", 65535)
        wr = g_u32(f"encoder.layers.{i}.window_right", 65535)
        sliding_windows.append((wl, wr))

    return {
        "enc_n_layers": enc_n_layers,
        "enc_n_heads": enc_n_heads,
        "enc_head_dim": enc_head_dim,
        "sliding_windows": sliding_windows,
    }


# ---------------------------------------------------------------------------
# FFN weight key discovery
# ---------------------------------------------------------------------------

def _fix_ffn_keys(w: Dict[str, np.ndarray], n_layers: int) -> None:
    """Remap ffn weight keys to the expected names if the model uses
    different naming (e.g. ffn_fc1_w vs ffn.fc1.weight)."""
    for li in range(n_layers):
        # Try alternate naming patterns and add aliases
        for expected, alternates in [
            (f"encoder.layers.{li}.ffn.fc1.weight",
             [f"encoder.layers.{li}.ffn_fc1_w",
              f"encoder.layers.{li}.ffn_up.weight"]),
            (f"encoder.layers.{li}.ffn.fc2.weight",
             [f"encoder.layers.{li}.ffn_fc2_w",
              f"encoder.layers.{li}.ffn_down.weight"]),
            (f"encoder.layers.{li}.ffn_norm.weight",
             [f"encoder.layers.{li}.ffn_norm_w"]),
            (f"encoder.layers.{li}.attn_norm.weight",
             [f"encoder.layers.{li}.attn_norm_w"]),
            (f"encoder.layers.{li}.attn.q.weight",
             [f"encoder.layers.{li}.attn_q_w"]),
            (f"encoder.layers.{li}.attn.k.weight",
             [f"encoder.layers.{li}.attn_k_w"]),
            (f"encoder.layers.{li}.attn.v.weight",
             [f"encoder.layers.{li}.attn_v_w"]),
            (f"encoder.layers.{li}.attn.o.weight",
             [f"encoder.layers.{li}.attn_o_w"]),
        ]:
            if expected not in w:
                for alt in alternates:
                    if alt in w:
                        w[expected] = w[alt]
                        break


# ---------------------------------------------------------------------------
# Main dump function
# ---------------------------------------------------------------------------

def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Moonshine-Streaming forward from GGUF weights and return stage captures.

    `model_dir` may be a .gguf file path or a directory containing one.
    This backend loads weights directly from the GGUF to guarantee that the
    reference exactly matches the C++ runtime (same weights, equivalent math).
    """
    model_dir = Path(model_dir)
    gguf_path = _find_gguf(model_dir)
    print(f"  loading Moonshine-Streaming weights from {gguf_path}")

    w = _load_gguf_weights(gguf_path)
    hp = _read_hparams(gguf_path)
    _fix_ffn_keys(w, hp["enc_n_layers"])

    print(f"  enc={hp['enc_n_layers']}L, heads={hp['enc_n_heads']}, head_dim={hp['enc_head_dim']}")

    out: Dict[str, np.ndarray] = {}
    audio_f32 = audio.astype(np.float32)

    if "raw_audio" in stages:
        out["raw_audio"] = audio_f32

    if "encoder_output" in stages:
        # Audio frontend
        frontend = _audio_frontend(audio_f32, w)
        T_enc = frontend.shape[0]
        print(f"  audio frontend: {len(audio_f32)} samples → {T_enc} frames")

        # Transformer encoder
        enc = _run_encoder(
            frontend, w,
            n_layers=hp["enc_n_layers"],
            n_heads=hp["enc_n_heads"],
            head_dim=hp["enc_head_dim"],
            sliding_windows=hp["sliding_windows"],
        )
        print(f"  encoder output: {enc.shape}")
        out["encoder_output"] = enc

    return out
