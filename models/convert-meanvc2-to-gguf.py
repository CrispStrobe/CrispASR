#!/usr/bin/env python3
"""
Convert MeanVC2 PyTorch / SafeTensors weights + Vocos + WeNet Fast-U2++ ASR → GGUF for CrispASR.

Architecture (MeanVC2 - ASLP@NPU, Apache-2.0):
  - Streaming ASR (Fast-U2++ Conformer JIT):
      Extracts 256-dim bottleneck features (BNF) from 16kHz speech via sliding window
      (80ms or 160ms chunking).
  - Speaker Encoder (WavLM Large + ECAPA-TDNN):
      Extracts 256-dim static speaker embedding from reference speech.
  - Global Timbre Memory (GTM):
      32 slots, 256-dim memory bank with learnable tanh priors and LayerNorm.
  - Temporal Timbre Encoder (TVT):
      4-head cross-attention (BNF query x GTM KV) to produce phonetic-varying timbre conditioning.
  - Diffusion Transformer (DiT):
      4 ChunkDiT blocks, dim=512, heads=2, head_dim=64, FFN dim=1024.
      Future-Receptive Chunking (FRC) with KV-cache.
      1-NFE Euler mean-flow solve: x_pred = x - u.
  - Vocoder (Vocos):
      8 ConvNeXt blocks (dim=320, intermediate=1536) + ISTFT head (n_fft=640, hop=160).

Usage:
    python models/convert-meanvc2-to-gguf.py \\
        --dit /path/to/meanvc2_120ms_40ms.safetensors \\
        --config /path/to/config_120ms_40ms.json \\
        --asr /path/to/fastu2pp_160ms.pt \\
        --vocos /path/to/vocos.pt \\
        --output /path/to/meanvc2-120ms-40ms-f16.gguf \\
        --dtype f16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

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


def should_preserve_f32(name: str) -> bool:
    """Carve out sensitive weights that MUST stay FP32 for numerical stability."""
    name_l = name.lower()
    if any(k in name_l for k in ["norm", "bias", "prior", "gamma", "rotary", "time_embed"]):
        return True
    if any(k in name_l for k in ["pos_bias", "inv_freq", "embed.conv", "dwconv", "dw."]):
        return True
    return False


def add_tensor_carveout(writer: GGUFWriter, name: str, tensor: torch.Tensor, target_dtype: str):
    """Add tensor with FP32 carve-outs for sensitive projections, norms, and priors."""
    arr = tensor.detach().cpu().float().numpy()

    if should_preserve_f32(name) or target_dtype == "f32":
        writer.add_tensor(name, arr.astype(np.float32))
    elif target_dtype == "f16":
        writer.add_tensor(name, arr.astype(np.float16))
    else:
        # Default to FP32 for carve-outs
        writer.add_tensor(name, arr.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Convert MeanVC2 models to GGUF format for CrispASR")
    parser.add_argument("--dit", type=Path, required=True, help="Path to meanvc2 DiT safetensors")
    parser.add_argument("--config", type=Path, required=True, help="Path to DiT config json")
    parser.add_argument("--asr", type=Path, required=True, help="Path to Fast-U2++ JIT model (.pt)")
    parser.add_argument("--vocos", type=Path, required=True, help="Path to Vocos JIT vocoder (.pt)")
    parser.add_argument("--spk", type=Path, default=None, help="Optional path to WavLM+ECAPA speaker model (.pth)")
    parser.add_argument("--output", type=Path, required=True, help="Path to output GGUF file")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f16", help="Precision format")
    args = parser.parse_args()

    assert args.dit.exists(), f"DiT checkpoint not found: {args.dit}"
    assert args.config.exists(), f"Config file not found: {args.config}"
    assert args.asr.exists(), f"ASR checkpoint not found: {args.asr}"
    assert args.vocos.exists(), f"Vocos checkpoint not found: {args.vocos}"

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dit_cfg = cfg.get("model", cfg)
    dim = dit_cfg.get("dim", 512)
    depth = dit_cfg.get("depth", 4)
    heads = dit_cfg.get("heads", 2)
    ff_mult = dit_cfg.get("ff_mult", 2)
    bn_dim = dit_cfg.get("bn_dim", 256)
    chunk_size = dit_cfg.get("chunk_size", 12)
    block_size = dit_cfg.get("block_size", 4)
    mel_dim = dit_cfg.get("mel_dim", 80)
    head_dim = dim // heads

    print(f"Creating GGUF with architecture 'meanvc2' at {args.output}...")
    writer = GGUFWriter(args.output, arch="meanvc2")

    # ── Metadata ──
    writer.add_string("general.name", "MeanVC2")
    writer.add_string("meanvc2.author", "ASLP@NPU")
    writer.add_string("meanvc2.license", "Apache-2.0")

    writer.add_uint32("meanvc2.dim", dim)
    writer.add_uint32("meanvc2.depth", depth)
    writer.add_uint32("meanvc2.heads", heads)
    writer.add_uint32("meanvc2.head_dim", head_dim)
    writer.add_uint32("meanvc2.ff_mult", ff_mult)
    writer.add_uint32("meanvc2.bn_dim", bn_dim)
    writer.add_uint32("meanvc2.chunk_size", chunk_size)
    writer.add_uint32("meanvc2.block_size", block_size)
    writer.add_uint32("meanvc2.mel_dim", mel_dim)
    writer.add_uint32("meanvc2.speaker_dim", 256)
    writer.add_uint32("meanvc2.memory_slots", 32)
    writer.add_uint32("meanvc2.sample_rate", 16000)

    # Vocos vocoder metadata
    writer.add_uint32("meanvc2.vocos_sample_rate", 16000)
    writer.add_uint32("meanvc2.vocos_dim", 320)
    writer.add_uint32("meanvc2.vocos_intermediate_dim", 1536)
    writer.add_uint32("meanvc2.vocos_num_layers", 8)
    writer.add_uint32("meanvc2.vocos_hop_length", 160)
    writer.add_uint32("meanvc2.vocos_n_fft", 640)

    # Fast-U2++ ASR metadata
    writer.add_uint32("meanvc2.asr_bn_window", 19 if chunk_size == 12 else 11)
    writer.add_uint32("meanvc2.asr_bn_stride", 16 if chunk_size == 12 else 8)
    writer.add_uint32("meanvc2.asr_offset_init", 8 if chunk_size == 12 else 4)
    writer.add_uint32("meanvc2.asr_offset_step", 4 if chunk_size == 12 else 2)
    writer.add_uint32("meanvc2.asr_req_cache", 8 if chunk_size == 12 else 4)

    # ── 1. DiT weights ──
    print(f"Loading DiT weights from {args.dit}...")
    n_dit = 0
    with safe_open(str(args.dit), framework="pt") as f:
        for k in sorted(f.keys()):
            t = f.get_tensor(k)
            gguf_name = f"vc.{k}"
            add_tensor_carveout(writer, gguf_name, t, args.dtype)
            n_dit += 1
    print(f"  Added {n_dit} DiT tensors.")

    # ── 2. Vocos weights ──
    print(f"Loading Vocos weights from {args.vocos}...")
    n_vocos = 0
    vocos_jit = torch.jit.load(str(args.vocos), map_location="cpu")
    voc_sd = vocos_jit.state_dict()
    for k, v in sorted(voc_sd.items()):
        # Skip internal istft buffers / window
        if "istft" in k:
            continue
        gguf_name = f"vocos.{k}"
        add_tensor_carveout(writer, gguf_name, v, args.dtype)
        n_vocos += 1
    print(f"  Added {n_vocos} Vocos tensors.")

    # ── 3. Fast-U2++ ASR weights ──
    print(f"Loading Fast-U2++ ASR weights from {args.asr}...")
    n_asr = 0
    asr_jit = torch.jit.load(str(args.asr), map_location="cpu")
    asr_sd = asr_jit.state_dict()
    for k, v in sorted(asr_sd.items()):
        gguf_name = f"asr.{k}"
        add_tensor_carveout(writer, gguf_name, v, args.dtype)
        n_asr += 1
    print(f"  Added {n_asr} Fast-U2++ ASR tensors.")

    # ── 4. Speaker model (optional) ──
    n_spk = 0
    if args.spk and args.spk.exists():
        print(f"Loading speaker model weights from {args.spk}...")
        spk_ckpt = torch.load(str(args.spk), map_location="cpu", weights_only=True)
        spk_sd = spk_ckpt.get("model", spk_ckpt)
        for k, v in sorted(spk_sd.items()):
            if "wavlm" in k.lower():
                gguf_name = f"speaker_wavlm.{k}"
            else:
                gguf_name = f"speaker_ecapa.{k}"
            add_tensor_carveout(writer, gguf_name, v, args.dtype)
            n_spk += 1
        print(f"  Added {n_spk} speaker model tensors.")

    print(f"Writing GGUF output to {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully converted MeanVC2 to {args.output} (total {n_dit + n_vocos + n_asr + n_spk} tensors)!")


if __name__ == "__main__":
    main()
