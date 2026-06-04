#!/usr/bin/env python
"""
Convert OpenVoice V2 Tone Color Converter (TCC) to GGUF format.

Usage:
    python models/convert-openvoice2-to-gguf.py \
        --ckpt /mnt/storage/openvoice2/converter/checkpoint.pth \
        --config /mnt/storage/openvoice2/converter/config.json \
        --output /mnt/storage/openvoice2-tcc-f16.gguf

The GGUF embeds all hyperparameters. Components:
  - ref_enc: 6 Conv2d layers + GRU + linear -> 256-d speaker embedding
  - enc_q:   16-layer WaveNet posterior encoder (spec -> latent z)
  - flow:    4 ResidualCouplingBlocks (WaveNet-based, mean-only)
  - dec:     HiFi-GAN generator (latent -> waveform)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")


def fuse_weight_norm(sd: dict) -> dict:
    """Fuse weight_norm pairs (weight_g, weight_v) -> weight."""
    processed = set()
    fused_weights = {}

    for key in sd.keys():
        if key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            g_key = base + ".weight_g"
            if g_key in sd:
                v = sd[key]
                g = sd[g_key]
                dims = tuple(range(1, v.ndim))
                v_norm = torch.linalg.vector_norm(v, dim=dims, keepdim=True)
                weight = v * (g / (v_norm + 1e-12))
                fused_weights[base + ".weight"] = weight
                processed.add(key)
                processed.add(g_key)

    result = {}
    for key in sd.keys():
        if key in processed:
            continue
        result[key] = sd[key]
    result.update(fused_weights)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="converter/checkpoint.pth")
    ap.add_argument("--config", required=True, help="converter/config.json")
    ap.add_argument("--output", required=True, help="Output .gguf")
    ap.add_argument("--all-f32", action="store_true",
                    help="Store ALL tensors as F32 (for quantization base)")
    ap.add_argument("--base-speakers", default=None,
                    help="Directory with base_speakers/ses/*.pth (from OpenVoiceV2)")
    args = ap.parse_args()

    # ── Load config ──
    with open(args.config) as f:
        cfg = json.load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # ── Load checkpoint ──
    print("Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    del ckpt

    # ── Fuse weight_norm pairs ──
    sd = fuse_weight_norm(sd)
    print(f"After weight_norm fusion: {len(sd)} tensors")

    # ── Collect tensors ──
    tensors = {}
    for name, param in sd.items():
        if isinstance(param, torch.Tensor):
            arr = param.cpu().numpy()
        else:
            arr = np.array(param)
        tensors[name] = arr

    # ── Count architecture params ──
    # enc_q WaveNet layers
    n_wn_layers_enc_q = 0
    while f"enc_q.enc.in_layers.{n_wn_layers_enc_q}.weight" in tensors:
        n_wn_layers_enc_q += 1

    # Flow blocks (indexed 0, 2, 4, 6 — odd indices are Flip layers)
    n_flow_blocks = 0
    for idx in range(0, 20, 2):
        if f"flow.flows.{idx}.pre.weight" in tensors:
            n_flow_blocks += 1

    # WaveNet layers per flow block
    n_wn_layers_flow = 0
    while f"flow.flows.0.enc.in_layers.{n_wn_layers_flow}.weight" in tensors:
        n_wn_layers_flow += 1

    # HiFi-GAN upsample stages
    n_upsample_stages = 0
    while f"dec.ups.{n_upsample_stages}.weight" in tensors:
        n_upsample_stages += 1

    # ResBlocks
    n_resblocks = 0
    while f"dec.resblocks.{n_resblocks}.convs1.0.weight" in tensors:
        n_resblocks += 1

    # ref_enc conv layers
    n_ref_convs = 0
    while f"ref_enc.convs.{n_ref_convs}.weight" in tensors:
        n_ref_convs += 1

    # Extract config values
    inter_channels = model_cfg.get("inter_channels", 192)
    hidden_channels = model_cfg.get("hidden_channels", 192)
    filter_channels = model_cfg.get("filter_channels", 768)
    gin_channels = model_cfg.get("gin_channels", 256)
    sample_rate = data_cfg.get("sampling_rate", 22050)
    hop_length = data_cfg.get("hop_length", 256)
    win_length = data_cfg.get("win_length", 1024)
    filter_length = data_cfg.get("filter_length", 1024)
    n_mels = 80  # OpenVoice2 uses 80-bin mel for ref_enc
    spec_channels = filter_length // 2 + 1  # 513
    zero_g = model_cfg.get("zero_g", True)

    upsample_rates = model_cfg["upsample_rates"]
    upsample_kernels = model_cfg["upsample_kernel_sizes"]
    resblock_kernels = model_cfg["resblock_kernel_sizes"]
    resblock_dilations = model_cfg["resblock_dilation_sizes"]

    print(f"Architecture: inter={inter_channels}, hidden={hidden_channels}, "
          f"gin={gin_channels}, sr={sample_rate}")
    print(f"  enc_q: {n_wn_layers_enc_q} WaveNet layers")
    print(f"  flow: {n_flow_blocks} blocks x {n_wn_layers_flow} WaveNet layers")
    print(f"  dec: {n_upsample_stages} upsample stages, {n_resblocks} resblocks")
    print(f"  ref_enc: {n_ref_convs} conv2d layers")

    # ── Write GGUF ──
    print(f"\nWriting {args.output}...")
    writer = GGUFWriter(args.output, "openvoice2-tcc")

    # Hyperparameters
    writer.add_uint32("openvoice2.inter_channels", inter_channels)
    writer.add_uint32("openvoice2.hidden_channels", hidden_channels)
    writer.add_uint32("openvoice2.filter_channels", filter_channels)
    writer.add_uint32("openvoice2.gin_channels", gin_channels)
    writer.add_uint32("openvoice2.sample_rate", sample_rate)
    writer.add_uint32("openvoice2.hop_length", hop_length)
    writer.add_uint32("openvoice2.win_length", win_length)
    writer.add_uint32("openvoice2.filter_length", filter_length)
    writer.add_uint32("openvoice2.spec_channels", spec_channels)
    writer.add_uint32("openvoice2.n_mels", n_mels)
    writer.add_bool("openvoice2.zero_g", zero_g)

    writer.add_uint32("openvoice2.n_wn_layers_enc_q", n_wn_layers_enc_q)
    writer.add_uint32("openvoice2.n_flow_blocks", n_flow_blocks)
    writer.add_uint32("openvoice2.n_wn_layers_flow", n_wn_layers_flow)
    writer.add_uint32("openvoice2.n_upsample_stages", n_upsample_stages)
    writer.add_uint32("openvoice2.n_resblocks", n_resblocks)
    writer.add_uint32("openvoice2.n_ref_convs", n_ref_convs)

    writer.add_array("openvoice2.upsample_rates",
                     [int(x) for x in upsample_rates])
    writer.add_array("openvoice2.upsample_kernel_sizes",
                     [int(x) for x in upsample_kernels])
    writer.add_array("openvoice2.resblock_kernel_sizes",
                     [int(x) for x in resblock_kernels])
    # Flatten dilation sizes: [[1,3,5],[1,3,5],[1,3,5]] -> [1,3,5,1,3,5,1,3,5]
    flat_dilations = []
    for d in resblock_dilations:
        flat_dilations.extend(d)
    writer.add_array("openvoice2.resblock_dilation_sizes",
                     [int(x) for x in flat_dilations])

    # Tensors
    n_written = 0
    total_params = 0
    for name, arr in sorted(tensors.items()):
        if arr.ndim == 0:
            continue  # skip scalars

        # Choose dtype — WaveNet flow and enc_q weights are numerically
        # sensitive (16 sequential layers, errors accumulate). Store as F32.
        is_sensitive = (name.startswith("flow.") or name.startswith("enc_q.")
                        or name.startswith("ref_enc."))
        if args.all_f32 or arr.ndim <= 1 or is_sensitive:
            data = arr.astype(np.float32)
            dtype_gguf = 0  # GGUF_TYPE_F32
        else:
            # Decoder weights as F16 (less sensitive)
            data = arr.astype(np.float16)
            dtype_gguf = 1  # GGUF_TYPE_F16

        writer.add_tensor(name, data)
        n_written += 1
        total_params += arr.size

    # Base speaker embeddings (pre-saved source SE for each MeloTTS voice)
    base_speakers_dir = args.base_speakers
    if not base_speakers_dir:
        # Auto-detect: look for base_speakers/ses/ next to checkpoint
        candidate = os.path.join(os.path.dirname(args.ckpt), "..", "base_speakers", "ses")
        if os.path.isdir(candidate):
            base_speakers_dir = candidate

    n_base_speakers = 0
    base_speaker_names = []
    if base_speakers_dir and os.path.isdir(base_speakers_dir):
        for fname in sorted(os.listdir(base_speakers_dir)):
            if not fname.endswith(".pth"):
                continue
            se_path = os.path.join(base_speakers_dir, fname)
            se = torch.load(se_path, map_location="cpu", weights_only=False)
            se_np = se.squeeze().numpy().astype(np.float32)
            if se_np.shape != (gin_channels,):
                print(f"  WARNING: {fname} shape {se_np.shape} != ({gin_channels},), skipping")
                continue
            name = fname.replace(".pth", "")
            tensor_name = f"base_speaker.{name}"
            writer.add_tensor(tensor_name, se_np)
            base_speaker_names.append(name)
            n_base_speakers += 1
            n_written += 1
            total_params += se_np.size
            print(f"  base speaker '{name}': mean={se_np.mean():.4f}")
        print(f"Added {n_base_speakers} base speaker embeddings")
    else:
        print("No base speaker embeddings found (pass --base-speakers dir)")

    writer.add_uint32("openvoice2.n_base_speakers", n_base_speakers)
    if base_speaker_names:
        writer.add_string("openvoice2.base_speaker_names",
                          ",".join(base_speaker_names))

    print(f"Written {n_written} tensors, {total_params:,} params total")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Output: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
