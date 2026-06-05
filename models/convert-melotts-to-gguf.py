#!/usr/bin/env python
"""
Convert MeloTTS PyTorch checkpoint to GGUF format.

Usage:
    python models/convert-melotts-to-gguf.py \
        --ckpt /mnt/storage/melotts-en/checkpoint.pth \
        --config /mnt/storage/melotts-en/config.json \
        --output /mnt/storage/melotts-en-f16.gguf

The GGUF embeds all hyperparameters, the symbol table (for G2P), and the
CMU pronunciation dictionary needed for English text processing.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")


def transpose_conv_weight(w: np.ndarray) -> np.ndarray:
    """No-op: GGUF reverses numpy dims automatically.

    PyTorch Conv1d weight shape: (Cout, Cin, K)
    Numpy stores as-is:          (Cout, Cin, K)
    GGUF/ggml reverses to:       ne[0]=K, ne[1]=Cin, ne[2]=Cout
    Which is exactly what ggml_conv_1d expects.
    """
    return w


def fuse_weight_norm(sd: dict) -> dict:
    """Fuse weight_norm pairs (weight_g, weight_v) -> weight.

    PyTorch weight_norm stores g (scale) and v (direction) separately:
        weight = v * (g / ||v||)
    where ||v|| is the L2 norm over all dims except dim 0 (output channels).
    """
    # First pass: find all weight_norm pairs
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
                v_norm = torch.norm(v, dim=dims, keepdim=True)
                weight = v * (g / (v_norm + 1e-12))
                fused_weights[base + ".weight"] = weight
                processed.add(key)
                processed.add(g_key)

    # Second pass: build output, skip processed keys
    result = {}
    for key in sd.keys():
        if key in processed:
            continue
        result[key] = sd[key]

    # Add fused weights
    result.update(fused_weights)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint.pth")
    ap.add_argument("--config", required=True, help="config.json")
    ap.add_argument("--output", required=True, help="Output .gguf")
    ap.add_argument("--all-f32", action="store_true",
                    help="Store ALL tensors as F32 (for quantization base)")
    ap.add_argument(
        "--cmudict",
        default=None,
        help="Path to cmudict.rep (default: auto-find in melotts-ref)",
    )
    args = ap.parse_args()

    # ── Load config ──
    with open(args.config) as f:
        cfg = json.load(f)

    symbols = cfg["symbols"]
    model_cfg = cfg["model"]

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
    n_layers_enc = 0
    while f"enc_p.encoder.attn_layers.{n_layers_enc}.conv_q.weight" in tensors:
        n_layers_enc += 1

    n_flow_blocks = 0
    for idx in range(0, 20, 2):
        if f"flow.flows.{idx}.pre.weight" in tensors:
            n_flow_blocks += 1

    n_upsample_stages = 0
    while (f"dec.ups.{n_upsample_stages}.weight" in tensors or
           f"dec.ups.{n_upsample_stages}.weight_v" in tensors):
        n_upsample_stages += 1

    n_sdp_flows = 0
    for idx in [3, 5, 7, 9, 11]:
        if f"dp.flows.{idx}.pre.weight" in tensors:
            n_sdp_flows += 1

    n_sdp_dds_layers = 0
    while f"dp.convs.convs_sep.{n_sdp_dds_layers}.weight" in tensors:
        n_sdp_dds_layers += 1

    n_resblocks = 0
    while f"dec.resblocks.{n_resblocks}.convs1.0.weight" in tensors:
        n_resblocks += 1

    hidden = tensors["enc_p.emb.weight"].shape[1]
    inter = hidden  # same for MeloTTS
    filter_ch = model_cfg.get("filter_channels", 768)
    n_heads = model_cfg.get("n_heads", 2)
    head_dim = hidden // n_heads
    gin_channels = model_cfg.get("gin_channels", 256)
    n_speakers = cfg["data"].get("n_speakers", 256)
    sample_rate = cfg["data"].get("sampling_rate", 44100)
    n_layers_trans_flow = model_cfg.get("n_layers_trans_flow", 3)

    upsample_rates = model_cfg["upsample_rates"]
    upsample_kernels = model_cfg["upsample_kernel_sizes"]
    resblock_kernels = model_cfg["resblock_kernel_sizes"]
    resblock_dilations = model_cfg["resblock_dilation_sizes"]

    num_symbols = len(symbols)
    num_tones = cfg.get("num_tones", 11)
    num_languages = cfg.get("num_languages", 3)

    print(f"Architecture: hidden={hidden}, enc_layers={n_layers_enc}, "
          f"flow_blocks={n_flow_blocks}, upsample={n_upsample_stages}, "
          f"symbols={num_symbols}, tones={num_tones}, langs={num_languages}, "
          f"speakers={n_speakers}, sr={sample_rate}")

    # ── Load CMU dictionary ──
    cmudict_json = "{}"
    cmudict_path = args.cmudict
    if not cmudict_path:
        # Try to find it
        for candidate in [
            Path(__file__).parent.parent / "melotts-ref" / "melo" / "text" / "cmudict_cache.pickle",
            Path("/mnt/storage/melotts-ref/melo/text/cmudict_cache.pickle"),
        ]:
            if candidate.exists():
                cmudict_path = str(candidate)
                break

    if cmudict_path and os.path.exists(cmudict_path):
        if cmudict_path.endswith(".pickle"):
            with open(cmudict_path, "rb") as f:
                cmudict = pickle.load(f)
            cmudict_json = json.dumps(cmudict)
            print(f"CMU dict loaded: {len(cmudict)} entries")
        else:
            # Read raw .rep format
            cmudict = {}
            with open(cmudict_path) as f:
                for line_no, line in enumerate(f, 1):
                    if line_no < 49:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("  ", 1)
                    if len(parts) == 2:
                        word = parts[0]
                        syllables = [s.split(" ") for s in parts[1].split(" - ")]
                        cmudict[word] = syllables
            cmudict_json = json.dumps(cmudict)
            print(f"CMU dict loaded: {len(cmudict)} entries")
    else:
        print("WARNING: CMU dict not found, G2P will be limited")

    # ── Write GGUF ──
    writer = GGUFWriter(args.output, arch="melotts")

    # Hyperparameters
    writer.add_uint32("melotts.hidden_channels", hidden)
    writer.add_uint32("melotts.inter_channels", inter)
    writer.add_uint32("melotts.filter_channels", filter_ch)
    writer.add_uint32("melotts.n_heads", n_heads)
    writer.add_uint32("melotts.head_dim", head_dim)
    writer.add_uint32("melotts.n_layers_enc", n_layers_enc)
    writer.add_uint32("melotts.n_layers_trans_flow", n_layers_trans_flow)
    writer.add_uint32("melotts.n_flow_blocks", n_flow_blocks)
    writer.add_uint32("melotts.n_upsample_stages", n_upsample_stages)
    writer.add_uint32("melotts.n_speakers", n_speakers)
    writer.add_uint32("melotts.gin_channels", gin_channels)
    writer.add_uint32("melotts.sample_rate", sample_rate)
    writer.add_uint32("melotts.num_symbols", num_symbols)
    writer.add_uint32("melotts.num_tones", num_tones)
    writer.add_uint32("melotts.num_languages", num_languages)
    writer.add_uint32("melotts.n_sdp_flows", n_sdp_flows)
    writer.add_uint32("melotts.sdp_num_bins", 10)
    writer.add_uint32("melotts.n_sdp_dds_layers", n_sdp_dds_layers)
    writer.add_uint32("melotts.upsample_initial_channel",
                      model_cfg.get("upsample_initial_channel", 512))

    for i, r in enumerate(upsample_rates):
        writer.add_uint32(f"melotts.upsample_rate.{i}", r)
    for i, k in enumerate(upsample_kernels):
        writer.add_uint32(f"melotts.upsample_kernel.{i}", k)
    for i, k in enumerate(resblock_kernels):
        writer.add_uint32(f"melotts.resblock_kernel.{i}", k)
    for i, dils in enumerate(resblock_dilations):
        for j, d in enumerate(dils):
            writer.add_uint32(f"melotts.resblock_dilation.{i}.{j}", d)

    writer.add_float32("melotts.noise_scale", 0.667)
    writer.add_float32("melotts.length_scale", 1.0)
    writer.add_float32("melotts.noise_w", 0.8)
    writer.add_float32("melotts.sdp_ratio", 0.0)  # 0 for disable-bert; 0.2 with BERT

    # Text processing data
    writer.add_string("melotts.symbols_json", json.dumps(symbols))
    writer.add_string("melotts.cmudict_json", cmudict_json)

    # spk2id
    spk2id = cfg["data"].get("spk2id", {})
    writer.add_string("melotts.spk2id_json", json.dumps(spk2id))

    # Neural G2P weights (g2p_en model, ~4 KB base64 JSON)
    g2p_weights_path = Path(__file__).parent.parent / "melotts-ref" / "g2p_weights.json"
    if not g2p_weights_path.exists():
        # Try to generate from g2p_en
        try:
            import base64
            from g2p_en import G2p
            g = G2p()
            g2p_data = {"meta": {"graphemes": g.graphemes, "phonemes": g.phonemes}, "weights": {}}
            for k in sorted(g.variables.keys()):
                arr = g.variables[k].astype(np.float32)
                g2p_data["weights"][k] = {
                    "shape": list(arr.shape),
                    "data": base64.b64encode(arr.tobytes()).decode("ascii"),
                }
            writer.add_string("melotts.g2p_en_json", json.dumps(g2p_data))
            print(f"g2p_en weights embedded ({len(json.dumps(g2p_data))//1024} KB)")
        except ImportError:
            print("WARNING: g2p_en not available, neural G2P weights not embedded")
    else:
        with open(g2p_weights_path) as f:
            writer.add_string("melotts.g2p_en_json", f.read())
        print("g2p_en weights embedded from cache")

    # ── Write tensors ──
    # Skip training-only tensors
    skip_prefixes = ["enc_q.", "dur_disc."]

    n_written = 0
    n_skipped = 0
    for name in sorted(tensors.keys()):
        skip = False
        for sp in skip_prefixes:
            if name.startswith(sp):
                skip = True
                break
        if skip:
            n_skipped += 1
            continue

        arr = tensors[name]

        # Transpose conv weights: PyTorch (Cout, Cin, K) -> ggml (K, Cin, Cout)
        if arr.ndim == 3:
            arr = transpose_conv_weight(arr)

        # For linear weights (2D), transpose: PyTorch (Out, In) -> ggml (In, Out)
        # Actually ggml expects (In, Out) for mul_mat, which is same as PyTorch (Out, In).T
        # But ggml_get_rows uses (vocab, dim) which matches PyTorch embedding shape.
        # For Conv1d 1x1 (stored as 3D with K=1), the transpose_conv_weight handles it.

        # Quantization strategy:
        # - 1D (norms, biases): always F32
        # - Embeddings: F32 (errors amplify through sqrt(hidden) scaling)
        # - SDP spline weights: F32 (rational quadratic splines are
        #   numerically sensitive, F16 causes audible degradation)
        # - Everything else: F16
        is_emb = name.endswith(".weight") and (
            "emb" in name or "emb_g" in name or "emb_rel" in name
        )
        is_sdp = name.startswith("sdp.") or name.startswith("dp.")
        if args.all_f32 or arr.ndim <= 1 or is_emb or is_sdp:
            writer.add_tensor(name, arr.astype(np.float32))
        else:
            writer.add_tensor(name, arr.astype(np.float16))
        n_written += 1

    print(f"Tensors: {n_written} written, {n_skipped} skipped")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output)
    print(f"Wrote {args.output} ({out_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
