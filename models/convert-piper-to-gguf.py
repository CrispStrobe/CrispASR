#!/usr/bin/env python3
"""
Convert rhasspy/piper ONNX + JSON config → GGUF for the CrispASR
`piper` backend.

Piper uses VITS (Variational Inference TTS):
  - TextEncoder: 6-layer relative-position transformer (192-d, 2 heads)
  - StochasticDurationPredictor: flow-based duration model
  - ResidualCouplingFlow: 4 affine coupling blocks with WaveNet
  - HiFi-GAN decoder: 3 upsample stages + 9 resblocks

Produces ONE GGUF containing all weights + JSON config as KV metadata.

Usage:
    python models/convert-piper-to-gguf.py \
        --onnx /mnt/storage/piper/en_US-lessac-medium.onnx \
        --json /mnt/storage/piper/en_US-lessac-medium.onnx.json \
        --output /mnt/storage/piper/piper-en_US-lessac-medium-f16.gguf

    # Auto-detect JSON sidecar:
    python models/convert-piper-to-gguf.py \
        --onnx /mnt/storage/piper/en_US-lessac-medium.onnx \
        --output /mnt/storage/piper/piper-en_US-lessac-medium-f16.gguf
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
    import onnx
    from onnx import numpy_helper
except ImportError:
    sys.exit("pip install onnx")


def load_onnx_tensors(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Extract all initializer tensors from ONNX model."""
    tensors = {}
    for init in model.graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)
    return tensors


def rename_anonymous_flow_convs(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map anonymous onnx::Conv_NNNN weights to proper flow.flows.N.enc names.

    The ONNX export strips names from the WaveNet dilated conv weights in
    the residual coupling flow. There are 4 flow blocks (indices 0,2,4,6),
    each with 4 WaveNet layers, each with an in_layer and a res_skip_layer.
    The anonymous Conv tensors appear in sequential order matching this
    layout.
    """
    anon_convs = sorted(
        [(k, v) for k, v in tensors.items() if k.startswith("onnx::Conv_")],
        key=lambda x: int(x[0].split("_")[-1]),
    )

    if len(anon_convs) != 32:
        print(f"WARNING: expected 32 anonymous Conv tensors, got {len(anon_convs)}")
        return tensors

    # ONNX exports flows in reverse order (6, 4, 2, 0) because VITS runs
    # them reversed during inference. The anonymous Conv weights follow
    # this order, not the forward (0, 2, 4, 6) order.
    flow_block_indices = [6, 4, 2, 0]
    renamed = dict(tensors)

    for i, (old_name, arr) in enumerate(anon_convs):
        block_num = i // 8
        layer_in_block = i % 8
        wn_layer = layer_in_block // 2
        is_res_skip = layer_in_block % 2 == 1

        flow_idx = flow_block_indices[block_num]
        sublayer = "res_skip_layers" if is_res_skip else "in_layers"
        new_name = f"flow.flows.{flow_idx}.enc.{sublayer}.{wn_layer}.weight"

        renamed[new_name] = arr
        del renamed[old_name]

    return renamed


def transpose_conv_weight(arr: np.ndarray) -> np.ndarray:
    """Transpose conv1d weight from PyTorch (Cout, Cin, K) to ggml (K, Cin, Cout)."""
    if arr.ndim == 3:
        return np.ascontiguousarray(arr.transpose(2, 1, 0))
    return arr


def main():
    parser = argparse.ArgumentParser(description="Convert Piper ONNX → GGUF")
    parser.add_argument("--onnx", required=True, help="Path to .onnx model")
    parser.add_argument("--json", default=None, help="Path to .onnx.json config (auto-detected if omitted)")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    json_path = Path(args.json) if args.json else onnx_path.with_suffix(".onnx.json")

    if not json_path.exists():
        # Try appending .json to the full onnx filename
        json_path = Path(str(onnx_path) + ".json")
    if not json_path.exists():
        sys.exit(f"JSON config not found at {json_path}")

    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))
    tensors = load_onnx_tensors(model)

    print(f"Loading config: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        config = json.load(f)

    # Rename anonymous flow conv weights
    tensors = rename_anonymous_flow_convs(tensors)

    # Rename the phoneme embedding (ONNX names it 'sid' due to export quirk)
    if "sid" in tensors:
        tensors["enc_p.emb.weight"] = tensors.pop("sid")

    # ── Write GGUF ──────────────────────────────────────────────────
    writer = GGUFWriter(str(args.output), arch="piper")

    # Hparams from config
    sample_rate = config.get("audio", {}).get("sample_rate", 22050)
    num_symbols = config.get("num_symbols", 256)
    num_speakers = config.get("num_speakers", 1)
    espeak_voice = config.get("espeak", {}).get("voice", "en-us")
    noise_scale = config.get("inference", {}).get("noise_scale", 0.667)
    length_scale = config.get("inference", {}).get("length_scale", 1.0)
    noise_w = config.get("inference", {}).get("noise_w", 0.8)

    # Architecture hparams (derived from tensor shapes)
    emb_weight = tensors.get("enc_p.emb.weight")
    hidden_channels = int(emb_weight.shape[1]) if emb_weight is not None else 192
    proj_weight = tensors.get("enc_p.proj.weight")
    inter_channels = int(proj_weight.shape[0]) // 2 if proj_weight is not None else 192

    ffn0_w = tensors.get("enc_p.encoder.ffn_layers.0.conv_1.weight")
    filter_channels = int(ffn0_w.shape[0]) if ffn0_w is not None else 768

    # Count encoder layers
    n_layers_enc = 0
    while f"enc_p.encoder.attn_layers.{n_layers_enc}.conv_q.weight" in tensors:
        n_layers_enc += 1

    # Count flow blocks
    n_flow_blocks = 0
    for idx in [0, 2, 4, 6, 8, 10]:
        if f"flow.flows.{idx}.pre.weight" in tensors:
            n_flow_blocks += 1

    # Count WaveNet layers per flow block
    n_wn_layers = 0
    while f"flow.flows.0.enc.in_layers.{n_wn_layers}.bias" in tensors:
        n_wn_layers += 1

    # WaveNet kernel size
    wn_w = tensors.get("flow.flows.0.enc.in_layers.0.weight")
    wn_kernel_size = int(wn_w.shape[2]) if wn_w is not None else 5

    # Attention heads: hidden_channels / head_dim
    # emb_rel_k shape is (1, 2*window+1, head_dim) — first dim is always 1
    rel_k = tensors.get("enc_p.encoder.attn_layers.0.emb_rel_k")
    head_dim = int(rel_k.shape[2]) if rel_k is not None else 96
    n_heads = hidden_channels // head_dim

    # HiFi-GAN config from tensor shapes
    n_upsample_stages = 0
    upsample_rates = []
    upsample_kernels = []
    while f"dec.ups.{n_upsample_stages}.weight" in tensors:
        w = tensors[f"dec.ups.{n_upsample_stages}.weight"]
        kernel = int(w.shape[2])
        stride = kernel // 2  # standard VITS convention: kernel = 2*stride
        upsample_rates.append(stride)
        upsample_kernels.append(kernel)
        n_upsample_stages += 1

    upsample_initial_channel = int(tensors["dec.conv_pre.weight"].shape[0]) if "dec.conv_pre.weight" in tensors else 256

    # Resblock kernel sizes
    resblock_kernel_sizes = []
    for i in range(9):
        key = f"dec.resblocks.{i}.convs.0.weight"
        if key in tensors:
            k = int(tensors[key].shape[2])
            if k not in resblock_kernel_sizes:
                resblock_kernel_sizes.append(k)

    # Count SDP flow layers
    n_sdp_flows = 0
    for idx in [3, 5, 7, 9, 11]:
        if f"dp.flows.{idx}.pre.weight" in tensors:
            n_sdp_flows += 1

    # SDP convflow num_bins (from proj output dim: 3*num_bins - 1)
    sdp_proj = tensors.get("dp.flows.3.proj.weight")
    sdp_num_bins = (int(sdp_proj.shape[0]) + 1) // 3 if sdp_proj is not None else 10

    # SDP DDSConv layers per flow
    n_sdp_dds_layers = 0
    while f"dp.flows.3.convs.convs_sep.{n_sdp_dds_layers}.weight" in tensors:
        n_sdp_dds_layers += 1

    # SDP main DDSConv layers
    n_sdp_main_dds_layers = 0
    while f"dp.convs.convs_sep.{n_sdp_main_dds_layers}.weight" in tensors:
        n_sdp_main_dds_layers += 1

    print(f"Architecture: hidden={hidden_channels}, inter={inter_channels}, "
          f"filter={filter_channels}, heads={n_heads}×{head_dim}")
    print(f"Encoder: {n_layers_enc} layers")
    print(f"Flow: {n_flow_blocks} blocks × {n_wn_layers} WN layers (k={wn_kernel_size})")
    print(f"SDP: {n_sdp_flows} ConvFlow layers, num_bins={sdp_num_bins}, "
          f"dds_layers={n_sdp_dds_layers}")
    print(f"Decoder: {n_upsample_stages} upsample stages, rates={upsample_rates}, "
          f"kernels={upsample_kernels}")
    print(f"Resblock kernels: {resblock_kernel_sizes}")
    print(f"Sample rate: {sample_rate}, Speakers: {num_speakers}")

    # Write metadata
    writer.add_uint32("piper.hidden_channels", hidden_channels)
    writer.add_uint32("piper.inter_channels", inter_channels)
    writer.add_uint32("piper.filter_channels", filter_channels)
    writer.add_uint32("piper.n_heads", n_heads)
    writer.add_uint32("piper.head_dim", head_dim)
    writer.add_uint32("piper.n_layers_enc", n_layers_enc)
    writer.add_uint32("piper.n_flow_blocks", n_flow_blocks)
    writer.add_uint32("piper.n_wn_layers", n_wn_layers)
    writer.add_uint32("piper.wn_kernel_size", wn_kernel_size)
    writer.add_uint32("piper.n_upsample_stages", n_upsample_stages)
    writer.add_uint32("piper.upsample_initial_channel", upsample_initial_channel)
    writer.add_uint32("piper.num_symbols", num_symbols)
    writer.add_uint32("piper.num_speakers", num_speakers)
    writer.add_uint32("piper.sample_rate", sample_rate)
    writer.add_uint32("piper.n_sdp_flows", n_sdp_flows)
    writer.add_uint32("piper.sdp_num_bins", sdp_num_bins)
    writer.add_uint32("piper.n_sdp_dds_layers", n_sdp_dds_layers)
    writer.add_uint32("piper.n_sdp_main_dds_layers", n_sdp_main_dds_layers)
    writer.add_float32("piper.noise_scale", noise_scale)
    writer.add_float32("piper.length_scale", length_scale)
    writer.add_float32("piper.noise_w", noise_w)
    writer.add_string("piper.espeak_voice", espeak_voice)

    # Write upsample rates and kernels as individual entries
    for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernels)):
        writer.add_uint32(f"piper.upsample_rate.{i}", rate)
        writer.add_uint32(f"piper.upsample_kernel.{i}", kernel)
    for i, k in enumerate(resblock_kernel_sizes):
        writer.add_uint32(f"piper.resblock_kernel.{i}", k)

    # Write phoneme_id_map as JSON string (runtime parses it)
    phoneme_id_map = config.get("phoneme_id_map", {})
    writer.add_string("piper.phoneme_id_map", json.dumps(phoneme_id_map, ensure_ascii=False))

    # Language info
    lang_info = config.get("language", {})
    if lang_info:
        writer.add_string("piper.language_code", lang_info.get("code", ""))
        writer.add_string("piper.language_family", lang_info.get("family", ""))

    # ── Write tensors ───────────────────────────────────────────────
    n_written = 0
    n_skipped = 0

    # Extract dp.flows.0 ElementwiseAffine logs from ONNX constant
    # The ONNX constant-folds exp(-logs) into /dp/flows.0/Exp_output_0.
    # We recover logs = -log(exp_neg_logs) and store it.
    ea_const_name = "/dp/flows.0/Exp_output_0"
    if ea_const_name in tensors:
        exp_neg_logs = tensors[ea_const_name].astype(np.float32)
        logs = -np.log(exp_neg_logs)
        tensors["dp.flows.0.logs"] = logs
        print(f"Recovered dp.flows.0.logs from ONNX constant: {logs.flatten()}")

    for name, arr in sorted(tensors.items()):
        # Skip ONNX graph constants (not model weights)
        if name.startswith("/") or name.startswith("onnx::") or name.isdigit():
            n_skipped += 1
            continue

        arr = arr.astype(np.float32) if arr.dtype != np.float32 else arr

        # NOTE: Do NOT transpose conv weights. ggml reverses numpy dimension order:
        # numpy (Cout, Cin, K) → ggml ne=(K, Cin, Cout), which is exactly what
        # ggml_conv_1d expects (ne[0]=K kernel width, ne[1]=Cin, ne[2]=Cout).
        # 2D weights (linear layers) are also stored without transpose.

        # Choose storage: 1D tensors (norms, biases) → F32, else → F16
        if arr.ndim <= 1:
            writer.add_tensor(name, arr)  # F32
        else:
            # Convert to F16 numpy before writing (GGUF writer sizes from numpy array)
            arr_f16 = arr.astype(np.float16)
            writer.add_tensor(name, arr_f16)
        n_written += 1

    print(f"Written {n_written} tensors, skipped {n_skipped} graph constants")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_path = Path(args.output)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Output: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
