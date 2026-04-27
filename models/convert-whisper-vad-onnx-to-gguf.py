#!/usr/bin/env python3
"""Convert TransWithAI/Whisper-Vad-EncDec-ASMR-onnx to GGUF for CrispASR.

Architecture: whisper-base encoder (6L, 512d, 8h) + 2-layer TransformerDecoder
+ frame_classifier(512→1). Input: [1, 80, 3000] mel. Output: [1, 1500] logits.
~29.8M params, 113 MB ONNX → ~113 MB F32 GGUF (or ~30 MB Q4_K).

Usage:
    python models/convert-whisper-vad-onnx-to-gguf.py \\
        --input /path/to/model.onnx --output whisper-vad-asmr.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("Error: onnx package not found. Install with: pip install onnx")
    sys.exit(1)

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert Whisper-VAD ONNX to GGUF")
    parser.add_argument("--input", required=True, help="ONNX model path or HF model dir")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        onnx_path = inp / "model.onnx"
        meta_path = inp / "model_metadata.json"
    else:
        onnx_path = inp
        meta_path = inp.parent / "model_metadata.json"

    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found")
        sys.exit(1)

    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))

    # Load metadata if available
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Build tensor dict: name → numpy array
    tensors = {}
    for init in model.graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)

    # Build consumer map: tensor_name → first MatMul node name
    from collections import defaultdict
    consumers = {}
    for node in model.graph.node:
        if node.op_type == "MatMul":
            for inp_name in node.input:
                if inp_name.startswith("val_") and inp_name not in consumers:
                    consumers[inp_name] = node.name

    # ── Map anonymous val_* tensors to named weights ────────────────────
    # Encoder layers 0-5: each has 6 MatMul weights in execution order:
    #   q_proj.weight, k_proj.weight, v_proj.weight,
    #   out_proj.weight, fc1.weight, fc2.weight
    # Plus named biases already in the initializers.

    # Sort val_* by node execution order (node numbering)
    val_matmuls = sorted(
        [(name, node_name) for name, node_name in consumers.items()
         if name.startswith("val_") and name in tensors and tensors[name].ndim == 2 and max(tensors[name].shape) >= 512],
        key=lambda x: int(x[1].split("_")[-1]) if x[1].split("_")[-1].isdigit() else 0
    )

    # Map by position: first 36 are encoder (6 layers × 6 weights), then decoder
    gguf_map = {}  # val_name → gguf_name

    enc_weights_per_layer = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                             "self_attn.v_proj.weight", "self_attn.out_proj.weight",
                             "fc1.weight", "fc2.weight"]

    idx = 0
    for layer in range(6):
        for w_name in enc_weights_per_layer:
            if idx < len(val_matmuls):
                gguf_map[val_matmuls[idx][0]] = f"encoder.layers.{layer}.{w_name}"
                idx += 1

    # After encoder: decoder layer 0 self_attn.in_proj_weight [512,1536]
    # Actually, looking at the shapes:
    # val_415 [512, 1536] → decoder self_attn in_proj (but it's named already)
    # Let me map remaining by shape + position
    dec_weight_names = [
        # Decoder layer 0
        "decoder.layers.0.self_attn.in_proj_weight",  # [512, 1536] — this is val_415
        # (intermediate decoder cross-attn and FFN follow)
        "decoder.layers.0.linear1.weight",  # [512, 2048]
        "decoder.layers.0.linear2.weight",  # [2048, 512]
        "decoder.layers.1.self_attn.in_proj_weight",  # [512, 1536]
        # Decoder layer 1 FFN
        "decoder.layers.1.linear1.weight",
        "decoder.layers.1.linear2.weight",
        # Frame classifier
        "frame_classifier.weight",  # [512, 1]
    ]

    remaining = val_matmuls[idx:]
    dec_idx = 0
    for val_name, node_name in remaining:
        shape = tensors[val_name].shape
        if dec_idx < len(dec_weight_names):
            gguf_map[val_name] = dec_weight_names[dec_idx]
            dec_idx += 1
        else:
            gguf_map[val_name] = f"unmapped.{val_name}"

    # ── Write GGUF ──────────────────────────────────────────────────────
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "whisper_vad_encdec", use_temp_file=True)

    # Metadata
    writer.add_name(meta.get("whisper_model_name", "whisper-vad-encdec-asmr"))
    writer.add_uint32("whisper_vad.encoder_layers", 6)
    writer.add_uint32("whisper_vad.encoder_dim", 512)
    writer.add_uint32("whisper_vad.encoder_heads", 8)
    writer.add_uint32("whisper_vad.encoder_ffn_dim", 2048)
    writer.add_uint32("whisper_vad.decoder_layers", meta.get("decoder_layers", 2))
    writer.add_uint32("whisper_vad.decoder_heads", meta.get("decoder_heads", 8))
    writer.add_uint32("whisper_vad.n_mels", 80)
    writer.add_uint32("whisper_vad.n_frames", 1500)
    writer.add_uint32("whisper_vad.frame_duration_ms", meta.get("frame_duration_ms", 20))

    mapped = 0
    skipped = 0

    for name, arr in tensors.items():
        # Skip tiny scalar/shape constants
        if name.startswith("val_") and (arr.ndim < 2 or max(arr.shape) < 512):
            if name not in ["val_482", "val_757"]:  # keep frame_classifier weight and bias
                skipped += 1
                continue

        # Determine GGUF name
        if name in gguf_map:
            gguf_name = gguf_map[name]
        elif name.startswith("val_"):
            # val_482 [1] is likely a constant, skip
            skipped += 1
            continue
        elif name == "transpose_24":
            gguf_name = "decoder.position_queries"
        elif name == "arange":
            skipped += 1  # not needed in GGUF
            continue
        else:
            gguf_name = name  # already named (biases, norms)

        # Convert to float32 numpy
        data = arr.astype(np.float32) if arr.dtype != np.float32 else arr
        data = np.ascontiguousarray(data)
        writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        mapped += 1

    # Also add mel filterbank + Hann window (same as whisper-base)
    n_fft = 400
    n_mels = 80
    sr = 16000
    n_freqs = n_fft // 2 + 1

    hann = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    writer.add_tensor("mel_window", hann, raw_dtype=GGMLQuantizationType.F32)

    mel_lo = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_pts = 700.0 * (10.0 ** (np.linspace(mel_lo, mel_hi, n_mels + 2) / 2595.0) - 1.0)
    fft_freqs = np.arange(n_freqs) * sr / n_fft
    fb = np.zeros((n_freqs, n_mels), dtype=np.float64)
    for m_idx in range(n_mels):
        lo, ctr, hi = mel_pts[m_idx], mel_pts[m_idx + 1], mel_pts[m_idx + 2]
        if hi > lo:
            enorm = 2.0 / (hi - lo)
            for f in range(n_freqs):
                if lo <= fft_freqs[f] <= ctr and ctr > lo:
                    fb[f, m_idx] = (fft_freqs[f] - lo) / (ctr - lo) * enorm
                elif ctr < fft_freqs[f] <= hi and hi > ctr:
                    fb[f, m_idx] = (hi - fft_freqs[f]) / (hi - ctr) * enorm
    fb = np.ascontiguousarray(fb.astype(np.float32))
    writer.add_tensor("mel_filters", fb, raw_dtype=GGMLQuantizationType.F32)
    mapped += 2

    print(f"\nMapped: {mapped}, Skipped: {skipped}")
    print(f"Writing to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"Done! {outfile} ({size_mb:.1f} MB, {mapped} tensors)")


if __name__ == "__main__":
    main()
