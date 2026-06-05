#!/usr/bin/env python3
"""
Convert the WavTokenizer decoder (OuteAI/wavtokenizer-large-75token-interface)
PyTorch checkpoint -> GGUF for the CrispASR `outetts` backend.

Architecture:
  - Codebook: (1, 4096, 512) embedding table (single VQ codebook)
  - Backbone (VocosBackbone):
      - Conv1d(512, 768, k=7, pad=3)  -- input projection
      - 12x ConvNeXtBlock(dim=768, intermediate=2304) with AdaNorm
      - FinalLayerNorm(768)
  - Head (ISTFTHead):
      - Linear(768, n_fft+2) -> split mag/phase -> iSTFT -> 24kHz PCM

Usage:
    python models/convert-wavtokenizer-to-gguf.py \\
        --input /path/to/wavtokenizer-large-75token-interface \\
        --output /mnt/storage/outetts/wavtokenizer-decoder-f16.gguf
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
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id))


def main():
    ap = argparse.ArgumentParser(description="Convert WavTokenizer decoder to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID or local dir containing decoder/decoder_model.pt + decoder/config.json")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    ap.add_argument("--dump-keys", action="store_true",
                    help="Just print state_dict keys and exit")
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    # Load config
    cfg_path = model_dir / "decoder" / "config.json"
    if not cfg_path.exists():
        sys.exit(f"no decoder/config.json in {model_dir}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    bb_cfg = cfg["backbone_config"]
    head_cfg = cfg["head_config"]

    input_channels = bb_cfg["input_channels"]  # 512
    dim = bb_cfg["dim"]  # 768
    intermediate_dim = bb_cfg["intermediate_dim"]  # 2304
    num_layers = bb_cfg["num_layers"]  # 12
    adanorm_num = bb_cfg.get("adanorm_num_embeddings", 4)
    n_fft = head_cfg["n_fft"]  # 1280
    hop_length = head_cfg["hop_length"]  # 320

    print(f"\nWavTokenizer decoder")
    print(f"  Backbone:  input={input_channels}  dim={dim}  intermediate={intermediate_dim}  "
          f"layers={num_layers}  adanorm_embeddings={adanorm_num}")
    print(f"  ISTFT:     n_fft={n_fft}  hop={hop_length}")

    # Load weights
    pt_path = model_dir / "decoder" / "decoder_model.pt"
    if not pt_path.exists():
        sys.exit(f"no decoder/decoder_model.pt in {model_dir}")
    print(f"  Loading {pt_path}...")
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    # Handle nested checkpoint structure
    if isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]
        # Top-level codebook_weights may also exist
        if "codebook_weights" in raw and "codebook_weights" not in sd:
            sd["codebook_weights"] = raw["codebook_weights"]
    else:
        sd = raw

    if args.dump_keys:
        for k in sorted(sd.keys()):
            v = sd[k]
            if hasattr(v, "shape"):
                print(f"  {k:60s} {list(v.shape)}")
            else:
                print(f"  {k:60s} {type(v)}")
        return

    # Print keys for debugging
    print(f"  State dict: {len(sd)} tensors")
    for k in sorted(sd.keys()):
        v = sd[k]
        if not hasattr(v, "shape"):
            continue
        if len(sd) <= 80 or ".0." in k or ".11." in k or "codebook" in k or "head" in k or "embed" in k or "pos_net.2" in k or "final" in k or "norm" == k.split(".")[-2]:
            print(f"    {k:60s} {list(v.shape)}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="wavtokenizer", use_temp_file=False)

    # Metadata
    w.add_name("wavtokenizer-decoder")

    def u32(k, v): w.add_uint32(k, int(v))

    u32("wavtok.input_channels", input_channels)
    u32("wavtok.backbone_dim", dim)
    u32("wavtok.intermediate_dim", intermediate_dim)
    u32("wavtok.n_layers", num_layers)
    u32("wavtok.adanorm_num_embeddings", adanorm_num)
    u32("wavtok.n_fft", n_fft)
    u32("wavtok.hop_length", hop_length)
    u32("wavtok.sample_rate", 24000)
    u32("wavtok.codebook_size", 4096)
    u32("wavtok.codebook_dim", input_channels)

    # Pre-bake AdaNorm: at inference time bandwidth_id=0 is fixed, so we
    # can fold the norm.scale.weight[0] and norm.shift.weight[0] into
    # per-block scale/shift constants.  But to keep the converter simple
    # and allow future bandwidth_id switching, we store all 4 embeddings.

    # Map tensor names
    def map_name(key: str) -> str | None:
        """Map PyTorch state_dict key to GGUF tensor name."""
        # Codebook
        if key == "codebook_weights":
            return "wavtok.codebook.weight"

        # Backbone conv_pre (embed = input projection Conv1d)
        if key == "backbone.embed.weight":
            return "wavtok.conv_pre.weight"
        if key == "backbone.embed.bias":
            return "wavtok.conv_pre.bias"

        # Backbone-level AdaNorm (for initial norm before ConvNeXt blocks)
        if key == "backbone.norm.scale.weight":
            return "wavtok.backbone_norm.scale.weight"
        if key == "backbone.norm.shift.weight":
            return "wavtok.backbone_norm.shift.weight"

        # Position network blocks
        if key.startswith("backbone.pos_net."):
            rest = key[len("backbone.pos_net."):]
            return f"wavtok.pos_net.{rest}"

        # ConvNeXt blocks
        if key.startswith("backbone.convnext."):
            rest = key[len("backbone.convnext."):]
            rest = rest.replace("dwconv.", "dw_conv.")
            rest = rest.replace("pwconv1.", "pw_up.")
            rest = rest.replace("pwconv2.", "pw_down.")
            rest = rest.replace("norm.scale.", "adanorm_scale.")
            rest = rest.replace("norm.shift.", "adanorm_shift.")
            rest = rest.replace("gamma", "grn_gamma")
            return f"wavtok.block.{rest}"

        # Final layer norm
        if key == "backbone.final_layer_norm.weight":
            return "wavtok.final_norm.weight"
        if key == "backbone.final_layer_norm.bias":
            return "wavtok.final_norm.bias"

        # ISTFT head
        if key == "head.out.weight":
            return "wavtok.istft_head.weight"
        if key == "head.out.bias":
            return "wavtok.istft_head.bias"
        if key == "head.istft.window":
            return "wavtok.istft_window"

        return None

    n_mapped = 0
    n_skipped = 0
    for key in sorted(sd.keys()):
        gn = map_name(key)
        if gn is None:
            print(f"  [SKIP] {key}")
            n_skipped += 1
            continue

        t = sd[key].to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        print(f"  [{n_mapped}] {gn:50s} <- {key:50s}  {t.shape}")

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    print(f"Writing {out_path}...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e6
    print(f"Done: {out_path}  ({sz:.1f} MB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()
