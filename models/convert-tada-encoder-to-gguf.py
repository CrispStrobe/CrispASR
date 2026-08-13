#!/usr/bin/env python3
"""Convert TADA encoder (WavEncoder + LocalAttentionEncoder + hidden_linear) → GGUF.

The TADA encoder converts 24 kHz audio into 512-d acoustic features at 50 Hz.
It consists of:
  1. WavEncoder: DAC-style strided conv (strides [6,5,4,4] = 480× downsample)
     Conv1d(1→64, k=7, p=3) → 4× EncoderBlock(double channels, stride conv)
     Each EncoderBlock: 3× ResidualUnit(dil=1,3,9) + Snake1d + stride conv
     Final: Snake1d(1024) + WNConv1d(1024→latent_dim, k=3, p=1)
  2. LocalAttentionEncoder: 6-layer transformer with RoPE + segment attention
     Same architecture as the TADA codec decoder's attention layers
  3. hidden_linear: Linear(1024→512) if hidden_dim != embed_dim
  4. pos_emb: Embedding(2, 1024) for token position markers

Weight-norm convolutions are pre-materialized (g*v/||v||) at conversion time.

Usage:
    python models/convert-tada-encoder-to-gguf.py \\
        --input HumeAI/tada-codec \\
        --output tada-encoder.gguf

The output GGUF can be used with the C++ tada_encoder runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_encoder_dir(model_id: str) -> Path:
    """Download or locate the encoder subfolder."""
    p = Path(model_id)
    if (p / "encoder").is_dir():
        return p / "encoder"
    if p.is_dir() and (p / "model.safetensors").exists():
        return p
    # Download from HuggingFace
    print(f"Downloading {model_id}/encoder...")
    st = hf_hub_download(model_id, "encoder/model.safetensors")
    cfg = hf_hub_download(model_id, "encoder/config.json")
    return Path(st).parent


def map_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace tensor names to GGUF names.

    The encoder's state_dict has these prefixes:
      wav_encoder.block.{i}.* — WavEncoder conv/snake layers
      local_attention_encoder.* — attention layers
      hidden_linear.* — Linear(1024→512)
      pos_emb.* — Embedding(2, 1024)
    """
    n = hf_name

    # WavEncoder: wav_encoder.block.{i}.* → enc.wav.{i}.*
    n = n.replace("wav_encoder.block.", "enc.wav.")

    # LocalAttentionEncoder layers
    n = n.replace("local_attention_encoder.input_proj.", "enc.attn.input_proj.")
    n = n.replace("local_attention_encoder.final_norm.", "enc.attn.final_norm.")
    n = n.replace("local_attention_encoder.layers.", "enc.attn.blk.")
    n = n.replace(".self_attn.qkv.", ".attn_qkv.")
    n = n.replace(".self_attn.out_proj.", ".attn_output.")
    n = n.replace(".self_attn.layer_norm.", ".attn_norm.")
    n = n.replace(".self_attn.rope_freqs", ".rope_freqs")
    n = n.replace(".self_attn._precomputed_mask", "._precomputed_mask")
    n = n.replace(".ffn.0.", ".ffn_up.")    # Linear(1024, 4096)
    n = n.replace(".ffn.3.", ".ffn_down.")  # Linear(4096, 1024)
    n = n.replace(".norm.", ".ffn_norm.")

    # hidden_linear
    n = n.replace("hidden_linear.", "enc.hidden_linear.")

    # pos_emb
    n = n.replace("pos_emb.", "enc.pos_emb.")

    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. HumeAI/tada-codec) or local encoder dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                    help="Weight dtype for linear/attention layers (default: f16)")
    args = ap.parse_args()

    encoder_dir = load_encoder_dir(args.input)
    print(f"Encoder dir: {encoder_dir}")

    # Load config
    config_path = encoder_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    # EncoderConfig defaults from tada/modules/encoder.py
    hidden_dim = int(cfg.get("hidden_dim", 1024))
    embed_dim = int(cfg.get("embed_dim", 512))
    strides = cfg.get("strides", [6, 5, 4, 4])
    num_attn_layers = int(cfg.get("num_attn_layers", 6))
    num_attn_heads = int(cfg.get("num_attn_heads", 8))
    attn_dim_ff = int(cfg.get("attn_dim_feedforward", 4096))
    attn_dropout = float(cfg.get("attn_dropout", 0.1))
    dist_type = cfg.get("dist_type", "fixed")
    block_attention = cfg.get("block_attention", "v2")
    std_val = float(cfg.get("std", 0.5))
    acoustic_mean = float(cfg.get("acoustic_mean", 0.0))
    acoustic_std = float(cfg.get("acoustic_std", 1.5))

    print(f"\nTADA Encoder")
    print(f"  WavEncoder:   strides={strides} → {np.prod(strides)}× downsample")
    print(f"  Attention:    hidden={hidden_dim} layers={num_attn_layers} "
          f"heads={num_attn_heads} ff={attn_dim_ff}")
    print(f"  Output:       embed_dim={embed_dim}")
    print(f"  Noise:        dist={dist_type} std={std_val}")
    print(f"  Normalize:    mean={acoustic_mean} std={acoustic_std}")
    print(f"  Block attn:   {block_attention}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    # Load safetensors
    st_files = sorted(encoder_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {encoder_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors: {len(name_to_idx)} tensors")

    # ── Detect and materialize weight-norm convolutions ──
    wn_pairs = {}
    for hf_name in sorted(name_to_idx.keys()):
        if ".parametrizations.weight.original0" in hf_name:
            prefix = hf_name.replace(".parametrizations.weight.original0", "")
            wn_pairs.setdefault(prefix, {})["g"] = hf_name
        elif ".parametrizations.weight.original1" in hf_name:
            prefix = hf_name.replace(".parametrizations.weight.original1", "")
            wn_pairs.setdefault(prefix, {})["v"] = hf_name

    materialized = {}
    for prefix, pair in sorted(wn_pairs.items()):
        if "g" in pair and "v" in pair:
            g = handles[name_to_idx[pair["g"]]].get_tensor(pair["g"]).to(torch.float32)
            v = handles[name_to_idx[pair["v"]]].get_tensor(pair["v"]).to(torch.float32)
            norm_dims = tuple(range(1, v.ndim))
            v_norm = torch.linalg.vector_norm(v, dim=norm_dims, keepdim=True)
            wt = v * (g / (v_norm + 1e-12))
            materialized[prefix] = wt.numpy()
            print(f"  WN: {prefix} g={list(g.shape)} → w={list(wt.shape)}")

    skip_wn_raw = set()
    for prefix in wn_pairs:
        if "g" in wn_pairs[prefix]:
            skip_wn_raw.add(wn_pairs[prefix]["g"])
        if "v" in wn_pairs[prefix]:
            skip_wn_raw.add(wn_pairs[prefix]["v"])

    # ── Write GGUF ──
    out_path = Path(args.output)
    print(f"\nWriting GGUF: {out_path}")
    w = GGUFWriter(str(out_path), arch="tada-encoder", use_temp_file=False)
    w.add_name("tada-encoder")

    # Metadata
    w.add_uint32("tada_encoder.hidden_dim", hidden_dim)
    w.add_uint32("tada_encoder.embed_dim", embed_dim)
    w.add_uint32("tada_encoder.num_attn_layers", num_attn_layers)
    w.add_uint32("tada_encoder.num_attn_heads", num_attn_heads)
    w.add_uint32("tada_encoder.attn_dim_ff", attn_dim_ff)
    w.add_array("tada_encoder.strides", strides)
    w.add_uint32("tada_encoder.sample_rate", 24000)
    w.add_float32("tada_encoder.noise_std", std_val)
    w.add_float32("tada_encoder.acoustic_mean", acoustic_mean)
    w.add_float32("tada_encoder.acoustic_std", acoustic_std)
    w.add_string("tada_encoder.dist_type", dist_type)
    w.add_string("tada_encoder.block_attention", block_attention)

    # ── Map and write tensors ──
    n_mapped = 0
    n_skipped = 0

    # Skip precomputed masks and rope freqs (recomputed at runtime)
    skip_patterns = ["_precomputed_mask", "rope_freqs"]

    for hf_name in sorted(name_to_idx.keys()):
        if hf_name in skip_wn_raw:
            n_skipped += 1
            continue
        if any(p in hf_name for p in skip_patterns):
            n_skipped += 1
            continue

        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 50 == 0:
            print(f"  [{n_mapped:3d}] {gn:55s} {str(list(t.shape)):25s} {t.dtype}")

    # Add materialized weight-norm tensors
    for prefix, wt_arr in sorted(materialized.items()):
        gn = map_tensor_name(prefix + ".weight")
        if gn is None:
            gn = map_tensor_name(prefix + ".")
            if gn:
                gn = gn.rstrip(".") + ".weight"
        if gn is None:
            print(f"  WARN: could not map WN prefix {prefix}", file=sys.stderr)
            continue
        t = np.ascontiguousarray(wt_arr.astype(out_dtype))
        w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        print(f"  [{n_mapped:3d}] {gn:55s} {str(list(wt_arr.shape)):25s} (WN materialized)")

    print(f"\n  Mapped: {n_mapped}, skipped: {n_skipped}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {out_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
