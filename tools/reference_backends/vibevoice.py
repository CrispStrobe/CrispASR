"""VibeVoice-ASR-7B reference dump backend.

Captures every stage needed to validate the C++ ggml pipeline:

  audio_norm          (N,)          normalized PCM at 24kHz (-25 dBFS)
  at_enc_mean         (T', 64)      acoustic encoder VAE mean (pre-sampling)
  at_tokens           (T', 64)      acoustic tokens (= mean; noise skipped for reproducibility)
  st_enc_mean         (T', 128)     semantic encoder output (mean, no sampling)
  at_conn_out         (T', D)       after acoustic SpeechConnector
  st_conn_out         (T', D)       after semantic SpeechConnector
  speech_features     (T', D)       combined (at_conn + st_conn)
  llm_argmax          (T_gen,)      greedy generated token IDs (requires full model load)
  generated_text      str           decoded transcript (requires full model load)

Key facts (7B ASR model):
  - Sample rate: 24 kHz (NOT 16 kHz like most ASR models)
  - Audio normalization: -25 dBFS before encoding
  - pad_mode = 'constant' (zero padding, NOT reflect)
  - vae_tok_len = ceil(samples / 3200)  [product of encoder ratios]
  - Acoustic: std_dist_type='gaussian', fix_std=0.5, vae_dim=64
  - Semantic:  std_dist_type='none' (mean only), vae_dim=128
  - SpeechConnector: Linear(vae_dim→D) + RMSNorm + Linear(D→D)
  - Combined = acoustic_connector(at_mean) + semantic_connector(st_mean)

Memory note:
  The encoder + connector stages load only ~4 GB (shards 6+7) and run on CPU/MPS.
  The LLM (llm_argmax, generated_text) requires the full 14 GB F16 model and is
  skipped by default. Pass --stages llm_argmax or generated_text to include.

Usage:
  python tools/dump_reference.py --backend vibevoice \\
      --model-dir /path/to/microsoft/VibeVoice-ASR/snapshots/<hash> \\
      --audio samples/jfk.wav \\
      --output /tmp/vibevoice-ref.gguf
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "audio_norm",
    "at_enc_mean",
    "at_tokens",
    "st_enc_mean",
    "at_conn_out",
    "st_conn_out",
    "speech_features",
]

_SR = 24000
_COMPRESS_RATIO = 3200   # product of encoder ratios [8,5,5,4,2,2]


def _normalize_audio(audio: np.ndarray, target_dB_FS: float = -25.0, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    audio = audio * (10 ** (target_dB_FS / 20) / (rms + eps))
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / (max_val + eps)
    return audio.astype(np.float32)


def _resample_to_24k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample from src_sr to 24 kHz. Uses librosa if available, else linear."""
    if src_sr == _SR:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_sr, target_sr=_SR).astype(np.float32)
    except ImportError:
        new_len = int(len(audio) * _SR / src_sr)
        return np.interp(np.linspace(0, len(audio) - 1, new_len),
                         np.arange(len(audio)), audio).astype(np.float32)


# ── Pure-PyTorch encoder / connector forward (no VibeVoice package needed) ────
# Implements the exact same operations as vibevoice.cpp so the diff is meaningful.
# All tensors are F32; operations match the C++ graph step by step.

def _rms_norm(x: "torch.Tensor", weight: "torch.Tensor", eps: float = 1e-5) -> "torch.Tensor":
    """x: [B, C, T] or [B, T, C]; weight: [C]. Normalizes over C."""
    # C is always the last dimension of weight, and we normalize over C.
    # For [B, C, T]: normalise over dim 1 (C).
    rms = x.pow(2).mean(dim=1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight.unsqueeze(0).unsqueeze(-1)


def _causal_conv1d(x: "torch.Tensor", weight: "torch.Tensor", bias=None, stride: int = 1) -> "torch.Tensor":
    """x: [B, C_in, T]; weight: [C_out, C_in, K]. Zero causal padding (pad_mode='constant')."""
    import torch.nn.functional as F
    K = weight.shape[2]
    pad_left = (K - 1) - (stride - 1)
    if pad_left < 0:
        pad_left = 0
    T_in = x.shape[-1]
    pad_right = 0
    if stride > 1:
        n_frames = (T_in - K + pad_left) / stride + 1.0
        ideal_length = (math.ceil(n_frames) - 1) * stride + (K - pad_left)
        pad_right = max(0, ideal_length - T_in)
    if pad_left > 0 or pad_right > 0:
        x = F.pad(x, (pad_left, pad_right), mode="constant", value=0)
    return F.conv1d(x, weight, bias, stride=stride)


def _causal_dw_conv1d(x: "torch.Tensor", weight: "torch.Tensor", bias=None) -> "torch.Tensor":
    """x: [B, C, T]; weight: [C, 1, K]. Depthwise causal conv. Zero padding."""
    import torch.nn.functional as F
    K = weight.shape[2]
    pad_left = K - 1
    if pad_left > 0:
        x = F.pad(x, (pad_left, 0), mode="constant", value=0)
    return F.conv1d(x, weight, bias, stride=1, groups=weight.shape[0])


def _block1d(x: "torch.Tensor", w: dict) -> "torch.Tensor":
    """One ConvNeXt Block1D: mixer (RMSNorm→dw_conv→gamma→+res) + FFN (same).

    x: [B, C, T] channels-first.
    FFN uses pointwise Linear (not causal conv1d).
    """
    import torch
    import torch.nn.functional as F

    # Mixer path
    res = x
    h = _rms_norm(x, w["norm"])                      # [B, C, T]
    h = _causal_dw_conv1d(h, w["dw_w"], w["dw_b"])   # [B, C, T]
    if w["gamma"] is not None:
        h = h * w["gamma"].unsqueeze(0).unsqueeze(-1)
    x = res + h

    # FFN path: Linear is applied over C dim per time-step.
    # x: [B, C, T] → transpose to [B, T, C] for F.linear → back to [B, C, T]
    res = x
    h = _rms_norm(x, w["ffn_norm"])                  # [B, C, T]
    h = h.permute(0, 2, 1)                            # [B, T, C]
    h = F.linear(h, w["ffn_up_w"], w["ffn_up_b"])    # [B, T, C_ffn]
    h = F.silu(h)
    h = F.linear(h, w["ffn_down_w"], w["ffn_down_b"])# [B, T, C]
    h = h.permute(0, 2, 1)                            # [B, C, T]
    if w["ffn_gamma"] is not None:
        h = h * w["ffn_gamma"].unsqueeze(0).unsqueeze(-1)
    return res + h


def _load_encoder_weights(state_dict: dict, prefix: str) -> dict:
    """Load ConvNeXt encoder weights from the safetensors state_dict.

    Safetensors key pattern (acoustic example):
      model.acoustic_tokenizer.encoder.downsample_layers.{i}.0.conv.conv.weight
      model.acoustic_tokenizer.encoder.stages.{si}.{bi}.norm.weight
      model.acoustic_tokenizer.encoder.stages.{si}.{bi}.mixer.conv.conv.conv.weight
      model.acoustic_tokenizer.encoder.stages.{si}.{bi}.ffn.linear1.weight
      model.acoustic_tokenizer.encoder.stages.{si}.{bi}.ffn.linear2.weight
      model.acoustic_tokenizer.encoder.head.conv.conv.weight
    """

    def g(key):
        t = state_dict.get(f"model.{prefix}.encoder.{key}")
        return t.float() if t is not None else None

    weights: dict = {"ds": {}, "stages": {}, "norm": None, "head": None}

    # Downsample conv layers: index 0 = stem (stride-1), 1..6 = strided
    for i in range(8):
        dw = g(f"downsample_layers.{i}.0.conv.conv.weight")
        db = g(f"downsample_layers.{i}.0.conv.conv.bias")
        if dw is None:
            break
        weights["ds"][i] = (dw, db)

    # ConvNeXt blocks: stages.{si}.{bi}.*
    si = 0
    while True:
        bi = 0
        found_stage = False
        while True:
            base = f"stages.{si}.{bi}"
            norm_w = g(f"{base}.norm.weight")
            if norm_w is None:
                break
            found_stage = True

            dw_conv_w = g(f"{base}.mixer.conv.conv.conv.weight")
            dw_conv_b = g(f"{base}.mixer.conv.conv.conv.bias")
            gamma     = g(f"{base}.gamma")
            ffn_norm  = g(f"{base}.ffn_norm.weight")
            ffn_up_w  = g(f"{base}.ffn.linear1.weight")   # [C_ffn, C]
            ffn_up_b  = g(f"{base}.ffn.linear1.bias")
            ffn_down_w = g(f"{base}.ffn.linear2.weight")  # [C, C_ffn]
            ffn_down_b = g(f"{base}.ffn.linear2.bias")
            ffn_gamma  = g(f"{base}.ffn_gamma")

            if si not in weights["stages"]:
                weights["stages"][si] = []
            weights["stages"][si].append({
                "norm":       norm_w,
                "dw_w":       dw_conv_w,   # [C, 1, K] depthwise
                "dw_b":       dw_conv_b,
                "gamma":      gamma,
                "ffn_norm":   ffn_norm,
                "ffn_up_w":   ffn_up_w,    # [C_ffn, C]  — used with F.linear
                "ffn_up_b":   ffn_up_b,
                "ffn_down_w": ffn_down_w,  # [C, C_ffn]
                "ffn_down_b": ffn_down_b,
                "ffn_gamma":  ffn_gamma,
            })
            bi += 1
        if not found_stage:
            break
        si += 1

    weights["norm"] = g("norm.weight")   # None if disable_last_norm=True
    weights["head"] = (g("head.conv.conv.weight"), g("head.conv.conv.bias"))

    return weights


def _run_encoder(audio24: "torch.Tensor", weights: dict, config: dict) -> "torch.Tensor":
    """Run one σ-VAE encoder.

    audio24: [1, T] float32 mono 24kHz PCM.
    Returns: [1, T', vae_dim] mean latents.

    Architecture: downsample_layers[0] = stem (stride 1),
                  downsample_layers[1..6] use enc_ratios = reversed([8,5,5,4,2,2]).
    """
    ratios = config["encoder_ratios"]  # [8,5,5,4,2,2] — decoder order
    enc_ratios = list(reversed(ratios))  # [2,2,4,5,5,8] — encoder order

    x = audio24.unsqueeze(1)  # [1, 1, T]

    n_stages = len(weights["stages"])
    for si in range(n_stages):
        # Downsample layer (ds[0] = stem with stride 1; ds[1..] have strided convs)
        if si in weights["ds"]:
            ds_w, ds_b = weights["ds"][si]
            stride = enc_ratios[si - 1] if si > 0 else 1
            x = _causal_conv1d(x, ds_w, ds_b, stride=stride)

        # ConvNeXt blocks for this stage
        for bw in weights["stages"][si]:
            x = _block1d(x, bw)

    # Optional final norm (disable_last_norm=True in ASR model → skipped)
    if weights["norm"] is not None:
        x = _rms_norm(x, weights["norm"])

    # Head conv → vae_dim channels
    head_w, head_b = weights["head"]
    x = _causal_conv1d(x, head_w, head_b, stride=1)

    # [B, vae_dim, T'] → [B, T', vae_dim]
    return x.permute(0, 2, 1)


def _run_connector(latents: "torch.Tensor", state_dict: dict, prefix: str) -> "torch.Tensor":
    """SpeechConnector: FC1 → RMSNorm → FC2.

    latents: [1, T', vae_dim].
    Returns: [1, T', d_lm].
    """
    import torch

    def g(key):
        t = state_dict.get(f"model.{prefix}.{key}")
        return t.float() if t is not None else None

    fc1_w = g("fc1.weight")   # [d_lm, vae_dim]
    fc1_b = g("fc1.bias")
    norm_w = g("norm.weight")  # [d_lm]
    fc2_w = g("fc2.weight")   # [d_lm, d_lm]
    fc2_b = g("fc2.bias")

    h = torch.nn.functional.linear(latents, fc1_w, fc1_b)

    # RMSNorm over last dim (d_lm)
    eps = 1e-6
    rms = h.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    h = (h / rms) * norm_w

    h = torch.nn.functional.linear(h, fc2_w, fc2_b)
    return h


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run VibeVoice-ASR encoder/connector forward and return stage captures.

    `audio` arrives as 16 kHz mono float32 from the shared loader in
    dump_reference.py. We resample to 24 kHz internally.

    Only encoder+connector stages are computed by default (< 4 GB RAM needed).
    LLM stages (llm_argmax, generated_text) require the full 14 GB model.
    """
    import torch
    from safetensors.torch import load_file

    # ── Resample 16kHz → 24kHz and normalize ─────────────────────────────────
    audio24 = _resample_to_24k(audio, src_sr=16000)
    norm_audio = _normalize_audio(audio24)

    out: Dict[str, np.ndarray] = {}
    if "audio_norm" in stages:
        out["audio_norm"] = norm_audio

    enc_stages = {"at_enc_mean", "at_tokens", "st_enc_mean",
                  "at_conn_out", "st_conn_out", "speech_features"}
    llm_stages = {"llm_argmax", "generated_text"}

    need_enc  = bool(stages & enc_stages)
    need_llm  = bool(stages & llm_stages)

    if need_enc:
        # ── Load only encoder/connector shards (shards 6+7, ~4 GB) ──────────
        index_path = model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)["weight_map"]

        enc_prefixes = ("model.acoustic_tokenizer.", "model.semantic_tokenizer.",
                        "model.acoustic_connector.", "model.semantic_connector.")

        # Collect which shards we need
        needed_shards: set = set()
        for key, shard in index.items():
            if any(key.startswith(p) for p in enc_prefixes):
                needed_shards.add(shard)

        print(f"  loading encoder/connector shards: {sorted(needed_shards)}")
        state_dict: dict = {}
        for shard_name in sorted(needed_shards):
            shard_path = model_dir / shard_name
            sd = load_file(str(shard_path))
            for k, v in sd.items():
                if any(k.startswith(p) for p in enc_prefixes):
                    state_dict[k] = v
        print(f"  loaded {len(state_dict)} tensors")

        # ── Parse model config ───────────────────────────────────────────────
        cfg = json.loads((model_dir / "config.json").read_text())
        at_cfg = cfg.get("acoustic_tokenizer_config", cfg.get("acoustic_tokenizer", {}))
        st_cfg = cfg.get("semantic_tokenizer_config", cfg.get("semantic_tokenizer", {}))

        at_cfg["encoder_ratios"] = at_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])
        st_cfg["encoder_ratios"] = st_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])

        # ── Load encoder weights ─────────────────────────────────────────────
        at_weights = _load_encoder_weights(state_dict, "acoustic_tokenizer")
        st_weights = _load_encoder_weights(state_dict, "semantic_tokenizer")

        speech_t = torch.tensor(norm_audio, dtype=torch.float32).unsqueeze(0)  # [1, T]

        with torch.no_grad():
            # Acoustic encoder
            at_mean = _run_encoder(speech_t, at_weights, at_cfg)  # [1, T', 64]
            if "at_enc_mean" in stages:
                out["at_enc_mean"] = at_mean.squeeze(0).numpy()   # (T', 64)
            at_tokens = at_mean   # skip gaussian noise for reproducible diffs
            if "at_tokens" in stages:
                out["at_tokens"] = at_tokens.squeeze(0).numpy()

            # Semantic encoder
            st_mean = _run_encoder(speech_t, st_weights, st_cfg)  # [1, T', 128]
            if "st_enc_mean" in stages:
                out["st_enc_mean"] = st_mean.squeeze(0).numpy()   # (T', 128)

            # Acoustic connector
            at_feat = _run_connector(at_mean, state_dict, "acoustic_connector")
            if "at_conn_out" in stages:
                out["at_conn_out"] = at_feat.squeeze(0).numpy()

            # Semantic connector
            st_feat = _run_connector(st_mean, state_dict, "semantic_connector")
            if "st_conn_out" in stages:
                out["st_conn_out"] = st_feat.squeeze(0).numpy()

            combined = at_feat + st_feat
            if "speech_features" in stages:
                out["speech_features"] = combined.squeeze(0).numpy()

    if need_llm:
        raise NotImplementedError(
            "LLM stages (llm_argmax, generated_text) require loading the full 14 GB model.\n"
            "The 7B VibeVoice-ASR model checkpoint uses 'model_type: vibevoice' which is not\n"
            "supported by transformers 5.x without the original vibevoice package.\n"
            "Workaround: downgrade transformers, or use the crispasr CLI to generate text."
        )

    return out
