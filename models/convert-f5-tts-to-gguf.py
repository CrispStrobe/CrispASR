#!/usr/bin/env python3
"""
Convert SWivid/F5-TTS safetensors + Vocos vocoder → GGUF for the CrispASR
`f5-tts` backend.

F5-TTS pipeline:
  - TextEmbedding: char-level embedding (2546, 512) + 4 ConvNeXtV2 blocks
  - InputEmbedding: Linear(712, 1024) + ConvPositionEmbedding (2× Conv1d k=31)
  - TimestepEmbedding: sinusoidal(256) → MLP(256→1024→1024)
  - DiT: 22 DiT blocks (AdaLN-Zero + self-attention + FFN)
    dim=1024, heads=16, dim_head=64, ff_mult=2
  - ODE solver: Euler, 32 steps, CFG strength=2.0
  - Vocos vocoder: Conv1d(100,512,k7) → 8 ConvNeXt blocks → ISTFTHead

Produces ONE GGUF containing all F5-TTS + Vocos weights + vocab + config.

Usage:
    python models/convert-f5-tts-to-gguf.py \\
        --model-dir /mnt/storage/f5-tts \\
        --output /mnt/storage/f5-tts/f5-tts-v1-base-f16.gguf

    # With custom paths:
    python models/convert-f5-tts-to-gguf.py \\
        --safetensors /mnt/storage/f5-tts/F5TTS_v1_Base/model_1250000.safetensors \\
        --vocos-dir /mnt/storage/f5-tts/vocos \\
        --vocab /path/to/vocab.txt \\
        --output /mnt/storage/f5-tts/f5-tts-v1-base-f16.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


# ── Architecture hyperparameters ───────────────────────────────────

F5TTS_HPARAMS = dict(
    arch="f5-tts",
    dim=1024,
    depth=22,
    heads=16,
    dim_head=64,
    ff_mult=2,
    text_dim=512,
    text_num_embeds=2546,   # vocab_size + 1 (0 is filler)
    conv_layers=4,
    mel_dim=100,
    sample_rate=24000,
    hop_length=256,
    win_length=1024,
    n_fft=1024,
    # ODE defaults
    ode_steps=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1.0,
    # ConvPositionEmbedding
    conv_pos_kernel=31,
    conv_pos_groups=16,
)

VOCOS_HPARAMS = dict(
    voc_input_channels=100,
    voc_dim=512,
    voc_intermediate_dim=1536,
    voc_num_layers=8,
    voc_n_fft=1024,
    voc_hop_length=256,
)


# ── Helpers ────────────────────────────────────────────────────────

def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()

def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()

# Quantization support: gguf library can write Q8_0 / Q4_K via raw data
# but it's simpler to mark the type and let gguf handle it.
try:
    from gguf import quants as gguf_quants
    _HAS_GGUF_QUANTS = True
except ImportError:
    _HAS_GGUF_QUANTS = False

def _quantize_to(t: torch.Tensor, qtype: GGMLQuantizationType):
    """Quantize a tensor using gguf's built-in quantization."""
    data_f32 = t.detach().to(torch.float32).numpy()
    if _HAS_GGUF_QUANTS:
        try:
            return gguf_quants.quantize(data_f32, qtype), qtype
        except Exception:
            # Fall back to F16 for tensors that can't be quantized
            # (e.g. 3D conv weights with odd shapes)
            return to_f16(t), GGMLQuantizationType.F16
    else:
        return to_f16(t), GGMLQuantizationType.F16


def choose_dtype(name: str, shape: list, t: torch.Tensor, quant: str = "f16"):
    """Choose tensor precision.

    quant="f16":  F16 for large weights, F32 for small/critical.
    quant="q8_0": Q8_0 for bulk weight matrices (QKV, FFN, Vocos conv),
                  F32 for conditioning pathway (AdaLN, time MLP, norms).
    """
    n = int(np.prod(shape))
    if t.ndim <= 1 or n < 256:
        return to_f32(t), GGMLQuantizationType.F32

    # ── Conditioning pathway: ALWAYS F32 regardless of quant level ──
    # These tensors control timestep conditioning. Errors here compound
    # through 32 ODE steps × 22 DiT layers. Even Q8_0 degrades quality.
    always_f32 = (
        'text_emb' in name or              # embedding table (lookup)
        'freqs_cis' in name or             # position encoding
        'inv_freq' in name or              # rotary frequencies
        'time_' in name or                 # timestep MLP (critical)
        'conv_pos' in name or              # positional conv (critical)
        'input_proj' in name or            # input projection
        'input_embed.proj' in name or      # input projection
        'final_adaln' in name or           # final AdaLN
        'final_proj' in name or            # final output projection
        'adaln' in name or                 # AdaLN modulation (scale/shift/gate)
        'voc.norm' in name or              # Vocos norms
        'voc.final' in name or
        'voc.head' in name or
        '.layer_scale' in name
    )
    if always_f32:
        return to_f32(t), GGMLQuantizationType.F32

    # ── Bulk weight matrices: quantize according to --quant ──
    if quant == "q8_0":
        return _quantize_to(t, GGMLQuantizationType.Q8_0)
    elif quant == "q4_k":
        return _quantize_to(t, GGMLQuantizationType.Q4_K)
    else:  # "f16" or default
        return to_f16(t), GGMLQuantizationType.F16


# ── F5-TTS tensor name remapping ──────────────────────────────────

def map_f5tts_name(hf_name: str) -> str:
    """Map safetensors name → GGUF tensor name.

    Input format:  ema_model.transformer.{submodule}.{param}
    Output format: f5.{submodule}.{param}
    """
    n = hf_name
    # Strip ema_model prefix
    n = n.replace("ema_model.", "")
    # Strip transformer prefix
    n = n.replace("transformer.", "")
    # Map transformer_blocks → blk
    n = n.replace("transformer_blocks.", "blk.")
    # Map attention
    n = n.replace(".attn.to_q.", ".attn_q.")
    n = n.replace(".attn.to_k.", ".attn_k.")
    n = n.replace(".attn.to_v.", ".attn_v.")
    n = n.replace(".attn.to_out.0.", ".attn_o.")
    # Map AdaLN
    n = n.replace(".attn_norm.linear.", ".adaln.")
    # Map FFN: ff.ff.0.0 → ffn_up, ff.ff.2 → ffn_down
    n = n.replace(".ff.ff.0.0.", ".ffn_up.")
    n = n.replace(".ff.ff.2.", ".ffn_down.")
    # Map text encoder
    n = n.replace("text_embed.text_embed.", "text_emb.")
    n = n.replace("text_embed.text_blocks.", "text_blk.")
    n = n.replace(".dwconv.", ".dw.")
    n = n.replace(".pwconv1.", ".pw_up.")
    n = n.replace(".pwconv2.", ".pw_down.")
    n = n.replace(".grn.gamma", ".grn_gamma")
    n = n.replace(".grn.beta", ".grn_beta")
    # Map time embedding
    n = n.replace("time_embed.time_mlp.0.", "time_mlp_0.")
    n = n.replace("time_embed.time_mlp.2.", "time_mlp_1.")
    # Map input embedding
    n = n.replace("input_embed.proj.", "input_proj.")
    n = n.replace("input_embed.conv_pos_embed.conv1d.0.", "conv_pos_0.")
    n = n.replace("input_embed.conv_pos_embed.conv1d.2.", "conv_pos_1.")
    # Map output norm + proj
    n = n.replace("norm_out.linear.", "final_adaln.")
    n = n.replace("proj_out.", "final_proj.")
    # Rotary
    n = n.replace("rotary_embed.inv_freq", "rotary_inv_freq")

    return "f5." + n


def map_vocos_name(pt_name: str) -> str:
    """Map Vocos state_dict name → GGUF tensor name."""
    n = pt_name
    # backbone.embed → voc.embed
    n = n.replace("backbone.embed.", "voc.embed.")
    # backbone.norm → voc.norm (initial LN)
    n = n.replace("backbone.norm.", "voc.norm.")
    # backbone.convnext.N → voc.blk.N
    n = n.replace("backbone.convnext.", "voc.blk.")
    # backbone.final_layer_norm → voc.final_norm
    n = n.replace("backbone.final_layer_norm.", "voc.final_norm.")
    # head.out → voc.head
    n = n.replace("head.out.", "voc.head.")
    # Map ConvNeXt block internals
    n = n.replace(".dwconv.", ".dw.")
    n = n.replace(".pwconv1.", ".pw_up.")
    n = n.replace(".pwconv2.", ".pw_down.")
    # gamma (layer scale)
    n = n.replace(".gamma", ".layer_scale")
    # Skip feature_extractor and head.istft (runtime computes these)
    if n.startswith("feature_extractor.") or n.startswith("head.istft."):
        return None
    return n


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Convert F5-TTS → GGUF")
    p.add_argument("--model-dir", type=Path,
                   help="Directory containing F5TTS_v1_Base/ and vocos/")
    p.add_argument("--safetensors", type=Path, default=None,
                   help="Path to F5-TTS safetensors (overrides --model-dir)")
    p.add_argument("--vocos-dir", type=Path, default=None,
                   help="Path to vocos model dir (overrides --model-dir)")
    p.add_argument("--vocab", type=Path, default=None,
                   help="Path to vocab.txt (default: bundled with f5_tts)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output GGUF path")
    p.add_argument("--quant", choices=["f16", "q8_0", "q4_k"], default="f16",
                   help="Quantization: f16 (default), q8_0, q4_k. "
                        "Only bulk weights (QKV, FFN, conv) are quantized; "
                        "conditioning pathway (AdaLN, time MLP) stays F32.")
    args = p.parse_args()

    # Resolve paths
    if args.safetensors:
        st_path = args.safetensors
    elif args.model_dir:
        st_path = args.model_dir / "F5TTS_v1_Base" / "model_1250000.safetensors"
        if not st_path.exists():
            st_path = args.model_dir / "model_1250000.safetensors"
    else:
        p.error("--model-dir or --safetensors is required")

    if args.vocos_dir:
        vocos_dir = args.vocos_dir
    elif args.model_dir:
        vocos_dir = args.model_dir / "vocos"
    else:
        vocos_dir = None

    assert st_path.exists(), f"Safetensors not found: {st_path}"
    print(f"Loading F5-TTS: {st_path}")

    # ── Load F5-TTS weights ──
    f5_tensors = {}
    with safe_open(str(st_path), framework='pt') as f:
        for k in f.keys():
            if k in ["initted", "step"]:
                continue
            t = f.get_tensor(k)
            # Skip mel_spec buffers
            if "mel_spec." in k or "mel_stft." in k:
                continue
            f5_tensors[k] = t
    print(f"  {len(f5_tensors)} F5-TTS tensors")

    # ── Load Vocos weights ──
    vocos_tensors = {}
    if vocos_dir and (vocos_dir / "pytorch_model.bin").exists():
        voc_sd = torch.load(str(vocos_dir / "pytorch_model.bin"),
                            map_location="cpu", weights_only=True)
        for k, v in voc_sd.items():
            # Skip feature_extractor and istft window
            if k.startswith("feature_extractor.") or k.startswith("head.istft."):
                continue
            vocos_tensors[k] = v
        del voc_sd
        print(f"  {len(vocos_tensors)} Vocos tensors")
    else:
        print("  WARNING: Vocos not found, GGUF will NOT include vocoder")

    # ── Load vocab ──
    if args.vocab:
        vocab_path = args.vocab
    else:
        from importlib.resources import files as pkg_files
        vocab_path = Path(str(pkg_files("f5_tts").joinpath("infer/examples/vocab.txt")))
    assert vocab_path.exists(), f"Vocab not found: {vocab_path}"
    with open(vocab_path, "r", encoding="utf-8") as vf:
        vocab_chars = [line.rstrip("\n") for line in vf]
    print(f"  Vocab: {len(vocab_chars)} chars from {vocab_path}")

    # ── Write GGUF ──
    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(args.output), arch="f5-tts")

    # Hyperparameters
    for k, v in F5TTS_HPARAMS.items():
        if isinstance(v, int):
            w.add_int32(f"f5.{k}", v)
        elif isinstance(v, float):
            w.add_float32(f"f5.{k}", v)
        elif isinstance(v, str):
            w.add_string(f"f5.{k}", v)
    for k, v in VOCOS_HPARAMS.items():
        if isinstance(v, int):
            w.add_int32(f"f5.{k}", v)
        elif isinstance(v, float):
            w.add_float32(f"f5.{k}", v)

    # Vocab as a string array
    w.add_array(f"f5.vocab", vocab_chars)

    quant = args.quant
    if quant != "f16":
        print(f"  Quantization: {quant} (bulk weights only; conditioning stays F32)")

    def _add_tensor(writer, name, t, quant_level):
        shape = list(t.shape)
        data, dtype = choose_dtype(name, shape, t, quant=quant_level)
        if dtype not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            writer.add_tensor(name, data, raw_dtype=dtype)
        else:
            writer.add_tensor(name, data)

    # ── F5-TTS tensors ──
    n_f5 = 0
    for hf_name, t in sorted(f5_tensors.items()):
        gguf_name = map_f5tts_name(hf_name)
        _add_tensor(w, gguf_name, t, quant)
        n_f5 += 1

    # ── Vocos tensors ──
    n_voc = 0
    for pt_name, t in sorted(vocos_tensors.items()):
        gguf_name = map_vocos_name(pt_name)
        if gguf_name is None:
            continue
        _add_tensor(w, gguf_name, t, quant)
        n_voc += 1

    # ── Write ──
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    fsize = args.output.stat().st_size
    print(f"\nWrote {args.output}  ({fsize/1024/1024:.1f} MiB)")
    print(f"  F5-TTS: {n_f5} tensors")
    print(f"  Vocos:  {n_voc} tensors")
    print(f"  Vocab:  {len(vocab_chars)} chars")


if __name__ == "__main__":
    main()
