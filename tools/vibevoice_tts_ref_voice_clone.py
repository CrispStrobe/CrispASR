#!/usr/bin/env python3
"""Dump VibeVoice-1.5B voice-clone reference features (acoustic + semantic).

Runs the official VibeVoice pipeline on a reference WAV and dumps the
intermediate encoder/connector outputs so they can be compared with the
C++ implementation via crispasr-diff or manual cosine comparison.

Dumped stages (all little-endian float32):

  voice_at_enc_mean.bin     [T_at, vae_dim_acoustic]   acoustic encoder mean
  voice_at_scaled.bin       [T_at, vae_dim_acoustic]   after (x + bias) * scale
  voice_at_conn.bin         [T_at, d_lm]               acoustic connector output
  voice_st_enc_mean.bin     [T_st, vae_dim_semantic]    semantic encoder mean
  voice_st_conn.bin         [T_st, d_lm]               semantic connector output
  voice_combined.bin        [T, d_lm]                   acoustic + semantic (final)
  tts_voice_embeds.bin      [T, d_lm]                   same as voice_combined (for diff harness compat)

Usage:
  python tools/vibevoice_tts_ref_voice_clone.py \
      --model microsoft/VibeVoice-1.5B \
      --voice samples/jfk.wav \
      --output-dir /tmp/vv_voice_ref
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import wave
from pathlib import Path

import numpy as np

# Patch transformers AutoModel.register to allow re-registration (vibevoice package conflict)
def _patch_automodel_register():
    from transformers.models.auto import auto_factory
    _orig = auto_factory._LazyAutoMapping.register
    def _patched(self, key, value, exist_ok=False):
        return _orig(self, key, value, exist_ok=True)
    auto_factory._LazyAutoMapping.register = _patched
_patch_automodel_register()


def dump_f32(path: Path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(str(path))
    print(f"  wrote {path.name}: {arr.shape} float32 ({arr.nbytes} bytes)")


def load_wav_mono_24k(path: str) -> np.ndarray:
    """Load a WAV file and return mono float32 PCM at 24 kHz."""
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    return wav  # [1, N_samples]


def normalize_audio(wav: np.ndarray, target_db: float = -25.0) -> np.ndarray:
    """Normalize audio to target dB FS (matching VibeVoice's AudioNormalizer)."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms < 1e-10:
        return wav
    target_rms = 10 ** (target_db / 20.0)
    scalar = target_rms / rms
    # Clamp to [-1, 1]
    result = wav * scalar
    result = np.clip(result, -1.0, 1.0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--voice", required=True, help="Reference WAV for voice cloning")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip audio normalization")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)

    # ── 1. Load config ──────────────────────────────────────────────────
    cfg_path = model_path / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    at_cfg = cfg.get("acoustic_tokenizer_config", cfg.get("acoustic_tokenizer", {}))
    st_cfg = cfg.get("semantic_tokenizer_config", cfg.get("semantic_tokenizer", {}))

    d_lm = dec_cfg["hidden_size"]
    vae_dim_at = at_cfg.get("vae_dim", 64)
    vae_dim_st = st_cfg.get("vae_dim", 128)

    print(f"config: d_lm={d_lm}, vae_dim_acoustic={vae_dim_at}, vae_dim_semantic={vae_dim_st}")

    # ── 2. Load model ───────────────────────────────────────────────────
    print(f"loading model from {model_path}")

    # Load weights directly from safetensors (avoids vibevoice package version conflicts)
    from safetensors.torch import load_file as load_safetensors

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f2:
            weight_map = json.load(f2)["weight_map"]
        # Only load encoder + connector shards (skip the heavy LM layers)
        needed_prefixes = ("model.acoustic_tokenizer.encoder.",
                          "model.semantic_tokenizer.encoder.",
                          "model.acoustic_connector.", "model.semantic_connector.")
        needed_shards = set()
        for k, shard in weight_map.items():
            if any(k.startswith(p) for p in needed_prefixes) or k in ("model.speech_scaling_factor", "model.speech_bias_factor"):
                needed_shards.add(shard)
        state_dict = {}
        for shard_name in sorted(needed_shards):
            print(f"  loading shard: {shard_name}")
            sd = load_safetensors(str(model_path / shard_name))
            state_dict.update(sd)
    else:
        state_dict = load_safetensors(str(model_path / "model.safetensors"))
    print(f"  {len(state_dict)} tensors loaded")

    # Build a minimal model object with just what we need
    class MinimalModel:
        pass
    model = MinimalModel()
    model_inner = MinimalModel()
    model.model = model_inner

    # Extract scaling factors
    sf_t = state_dict.get("model.speech_scaling_factor")
    bf_t = state_dict.get("model.speech_bias_factor")
    model_inner.speech_scaling_factor = sf_t.float() if sf_t is not None else torch.tensor(0.196)
    model_inner.speech_bias_factor = bf_t.float() if bf_t is not None else torch.tensor(-0.049)

    # Import model classes directly (bypass vibevoice.__init__ which triggers broken imports)
    import importlib.util
    _vv_pkg = "/home/claudeuser/.local/lib/python3.13/site-packages/vibevoice/modular"

    spec_tok = importlib.util.spec_from_file_location(
        "vibevoice.modular.modular_vibevoice_tokenizer", f"{_vv_pkg}/modular_vibevoice_tokenizer.py")
    mod_tok = importlib.util.module_from_spec(spec_tok)
    spec_tok.loader.exec_module(mod_tok)

    spec_cfg = importlib.util.spec_from_file_location(
        "vibevoice.modular.configuration_vibevoice", f"{_vv_pkg}/configuration_vibevoice.py")
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    spec_cfg.loader.exec_module(mod_cfg)

    spec_model = importlib.util.spec_from_file_location(
        "vibevoice.modular.modeling_vibevoice", f"{_vv_pkg}/modeling_vibevoice.py")
    mod_model = importlib.util.module_from_spec(spec_model)
    spec_model.loader.exec_module(mod_model)

    # Build acoustic tokenizer encoder (σ-VAE)
    at_cfg_dict = cfg.get("acoustic_tokenizer_config", {})
    at_config = mod_cfg.VibeVoiceAcousticTokenizerConfig(**at_cfg_dict)
    at_model = mod_tok.VibeVoiceAcousticTokenizerModel(at_config).float().eval()
    at_state = {k.replace("model.acoustic_tokenizer.", ""): v for k, v in state_dict.items()
                if k.startswith("model.acoustic_tokenizer.")}
    at_model.load_state_dict(at_state, strict=False)
    model_inner.acoustic_tokenizer = at_model

    # Build semantic tokenizer encoder
    st_cfg_dict = cfg.get("semantic_tokenizer_config", {})
    st_config = mod_cfg.VibeVoiceSemanticTokenizerConfig(**st_cfg_dict)
    st_model = mod_tok.VibeVoiceSemanticTokenizerModel(st_config).float().eval()
    st_state = {k.replace("model.semantic_tokenizer.", ""): v for k, v in state_dict.items()
                if k.startswith("model.semantic_tokenizer.")}
    st_model.load_state_dict(st_state, strict=False)
    model_inner.semantic_tokenizer = st_model

    # Build connectors (simple FC1 → RMSNorm → FC2)
    SpeechConnector = mod_model.SpeechConnector
    ac_conn = SpeechConnector(vae_dim_at, d_lm).float().eval()
    ac_state = {k.replace("model.acoustic_connector.", ""): v for k, v in state_dict.items()
                if k.startswith("model.acoustic_connector.")}
    ac_conn.load_state_dict(ac_state)
    model_inner.acoustic_connector = ac_conn

    se_conn = SpeechConnector(vae_dim_st, d_lm).float().eval()
    se_state = {k.replace("model.semantic_connector.", ""): v for k, v in state_dict.items()
                if k.startswith("model.semantic_connector.")}
    se_conn.load_state_dict(se_state)
    model_inner.semantic_connector = se_conn
    print("  model components loaded successfully")

    # ── 3. Load and normalize reference audio ────────────────────────────
    print(f"loading voice reference: {args.voice}")
    wav_tensor = load_wav_mono_24k(args.voice)  # [1, N]

    if not args.no_normalize:
        # Apply the same normalization as the C++ code
        wav_np = wav_tensor.squeeze().numpy()
        wav_np = normalize_audio(wav_np, target_db=-25.0)
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)  # [1, N]

    n_samples = wav_tensor.shape[1]
    print(f"  samples: {n_samples} ({n_samples / 24000:.2f}s @ 24kHz)")

    # ── 4. Run acoustic tokenizer encoder ─────────────────────────────────
    print("running acoustic tokenizer encoder...")
    with torch.no_grad():
        at_output = model.model.acoustic_tokenizer.encode(wav_tensor.unsqueeze(1))  # [1, 1, N]
        at_mean = at_output.mean  # [1, T, vae_dim_at]

    at_mean_np = at_mean.squeeze(0).numpy()  # [T, vae_dim_at]
    T_at = at_mean_np.shape[0]
    print(f"  acoustic encoder mean: [{T_at}, {vae_dim_at}]")
    dump_f32(out / "voice_at_enc_mean.bin", at_mean_np)

    # ── 5. Apply scaling (acoustic only) ─────────────────────────────────
    sf = model.model.speech_scaling_factor.item()
    bf = model.model.speech_bias_factor.item()
    print(f"  scaling: sf={sf:.6f}, bf={bf:.6f}")

    at_scaled = (at_mean + bf) * sf
    at_scaled_np = at_scaled.squeeze(0).numpy()
    dump_f32(out / "voice_at_scaled.bin", at_scaled_np)

    # ── 6. Run acoustic connector ─────────────────────────────────────────
    print("running acoustic connector...")
    with torch.no_grad():
        at_conn = model.model.acoustic_connector(at_scaled)  # [1, T, d_lm]

    at_conn_np = at_conn.squeeze(0).numpy()  # [T, d_lm]
    print(f"  acoustic connector: [{at_conn_np.shape[0]}, {at_conn_np.shape[1]}]")
    dump_f32(out / "voice_at_conn.bin", at_conn_np)

    # ── 7. Run semantic tokenizer encoder ─────────────────────────────────
    print("running semantic tokenizer encoder...")
    with torch.no_grad():
        st_output = model.model.semantic_tokenizer.encode(wav_tensor.unsqueeze(1))
        st_mean = st_output.mean  # [1, T, vae_dim_st]

    st_mean_np = st_mean.squeeze(0).numpy()
    T_st = st_mean_np.shape[0]
    print(f"  semantic encoder mean: [{T_st}, {vae_dim_st}]")
    dump_f32(out / "voice_st_enc_mean.bin", st_mean_np)

    # ── 8. Run semantic connector (NO scaling) ────────────────────────────
    print("running semantic connector...")
    with torch.no_grad():
        st_conn = model.model.semantic_connector(st_mean)  # [1, T, d_lm]

    st_conn_np = st_conn.squeeze(0).numpy()
    print(f"  semantic connector: [{st_conn_np.shape[0]}, {st_conn_np.shape[1]}]")
    dump_f32(out / "voice_st_conn.bin", st_conn_np)

    # ── 9. Combine: element-wise sum ──────────────────────────────────────
    if T_at != T_st:
        print(f"  WARNING: frame mismatch T_at={T_at} T_st={T_st}")
        T = min(T_at, T_st)
        at_conn_np = at_conn_np[:T]
        st_conn_np = st_conn_np[:T]
    else:
        T = T_at

    combined = at_conn_np + st_conn_np
    print(f"  combined (acoustic + semantic): [{T}, {d_lm}]")
    dump_f32(out / "voice_combined.bin", combined)
    dump_f32(out / "tts_voice_embeds.bin", combined)  # compat name for diff harness

    # ── Summary ──────────────────────────────────────────────────────────
    at_rms = np.sqrt(np.mean(at_conn_np ** 2))
    st_rms = np.sqrt(np.mean(st_conn_np ** 2))
    cb_rms = np.sqrt(np.mean(combined ** 2))
    print(f"\n  acoustic RMS: {at_rms:.6f}")
    print(f"  semantic RMS: {st_rms:.6f}")
    print(f"  combined RMS: {cb_rms:.6f}")
    print(f"  semantic/acoustic ratio: {st_rms / at_rms:.4f}")
    print(f"\nDone. {T} voice frames dumped to {out}")


if __name__ == "__main__":
    main()
