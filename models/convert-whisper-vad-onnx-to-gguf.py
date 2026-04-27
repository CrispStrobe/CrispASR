#!/usr/bin/env python3
"""Convert TransWithAI/Whisper-Vad-EncDec-ASMR-onnx to GGUF for CrispASR.

Architecture: whisper-base encoder (6L, 512d, 8h, fine-tuned) +
2-layer TransformerDecoder (cross-attn to encoder) + frame_classifier(512->1).
Input: [1, 80, 3000] mel. Output: [1, 1500] logits (20ms per frame, 30s).
~29.8M params, 113 MB ONNX -> ~113 MB F32 GGUF (or ~22 MB Q4_K).

Uses ONNX graph topology tracing (bias -> weight association) to correctly
map anonymous val_* tensors to named weights.

Usage:
    python models/convert-whisper-vad-onnx-to-gguf.py \\
        --input /path/to/model.onnx --output whisper-vad-asmr.gguf
    # Or from HF directory:
    python models/convert-whisper-vad-onnx-to-gguf.py \\
        --input TransWithAI/Whisper-Vad-EncDec-ASMR-onnx --output whisper-vad-asmr.gguf
"""

import argparse
import json
import sys
from collections import defaultdict
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


def trace_bias_after_matmul(matmul_output, consumers, tensors):
    """Follow a MatMul output through the graph to find the next named bias."""
    visited = set()
    queue = [matmul_output]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for node in consumers.get(cur, []):
            if node.op_type == "Add":
                for inp in node.input:
                    if inp != cur and inp in tensors and not inp.startswith("val_"):
                        return inp
            if node.op_type in ("Reshape", "Transpose", "Unsqueeze", "Add", "Mul"):
                for o in node.output:
                    queue.append(o)
    return None


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

    # Try downloading from HF if not local
    if not onnx_path.exists() and not inp.is_dir():
        try:
            from huggingface_hub import snapshot_download
            dl_dir = snapshot_download(str(inp), allow_patterns=["*.onnx", "*.json"])
            onnx_path = Path(dl_dir) / "model.onnx"
            meta_path = Path(dl_dir) / "model_metadata.json"
        except Exception as e:
            print(f"Error: {onnx_path} not found and HF download failed: {e}")
            sys.exit(1)

    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found")
        sys.exit(1)

    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(str(onnx_path))

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    # Extract all tensors
    tensors = {}
    for init in model.graph.initializer:
        tensors[init.name] = numpy_helper.to_array(init)

    # Build consumer map
    consumers = defaultdict(list)
    for node in model.graph.node:
        for inp_name in node.input:
            consumers[inp_name].append(node)

    # ── Trace graph topology to map val_* → weight names ──────────────
    # For each MatMul that uses a val_* input, follow the output to find
    # the named bias tensor. The bias name gives us the weight name.
    # k_proj has no bias in whisper → inferred by position (follows q_proj).
    val_to_name = {}
    prev_layer_q = None  # track last q_proj to identify k_proj

    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        for inp_name in node.input:
            if not inp_name.startswith("val_") or inp_name not in tensors:
                continue
            arr = tensors[inp_name]
            if arr.ndim < 2 or max(arr.shape) < 512:
                continue

            bias = trace_bias_after_matmul(node.output[0], consumers, tensors)
            if bias:
                wname = bias.replace(".bias", ".weight")
                # Special case: self_attn.in_proj is both bias and weight
                if "in_proj_bias" in bias:
                    wname = bias.replace("_bias", "_weight")
                # Special case: frame_classifier.1.bias -> frame_classifier.weight
                if "frame_classifier" in bias:
                    wname = "frame_classifier.weight"
                val_to_name[inp_name] = wname
                if "q_proj" in wname:
                    prev_layer_q = wname
            else:
                # No bias = k_proj (whisper has no K bias)
                if prev_layer_q:
                    kname = prev_layer_q.replace("q_proj", "k_proj")
                    val_to_name[inp_name] = kname
                    prev_layer_q = None

    # Print mapping
    for vn in sorted(val_to_name, key=lambda x: int(x.split('_')[1])):
        arr = tensors[vn]
        print(f"  {vn:15s} {str(list(arr.shape)):20s} → {val_to_name[vn]}")

    # ── Write GGUF ──────────────────────────────────────────────────────
    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "whisper_vad_encdec", use_temp_file=True)

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
    for name, arr in tensors.items():
        # Determine GGUF name
        if name in val_to_name:
            gguf_name = val_to_name[name]
        elif name == "transpose_24":
            gguf_name = "decoder.position_queries"
        elif name == "arange":
            continue  # not needed
        elif name.startswith("val_"):
            continue  # small constants
        else:
            gguf_name = name  # already named

        data = arr.astype(np.float32) if arr.dtype != np.float32 else arr
        # ONNX MatMul weights are [in, out] for input@weight. ggml_mul_mat(W, x)
        # needs ne[0]=in, ne[1]=out. Since numpy [in, out] → ggml ne[0]=out, ne[1]=in,
        # we must transpose 2D weight matrices so ggml sees ne[0]=in correctly.
        # Exception: embeddings (embed_positions, position_queries) are lookups, not matmuls.
        if data.ndim == 2 and "weight" in gguf_name and "embed" not in gguf_name and "position" not in gguf_name:
            data = data.T
        data = np.ascontiguousarray(data)
        writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        mapped += 1

    # Add mel filterbank + Hann window
    n_fft, n_mels, sr = 400, 80, 16000
    n_freqs = n_fft // 2 + 1
    hann = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    writer.add_tensor("mel_window", hann, raw_dtype=GGMLQuantizationType.F32)
    mel_lo = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_pts = 700.0 * (10.0 ** (np.linspace(mel_lo, mel_hi, n_mels + 2) / 2595.0) - 1.0)
    fft_freqs = np.arange(n_freqs) * sr / n_fft
    fb = np.zeros((n_freqs, n_mels), dtype=np.float64)
    for mi in range(n_mels):
        lo, ctr, hi = mel_pts[mi], mel_pts[mi + 1], mel_pts[mi + 2]
        if hi > lo:
            enorm = 2.0 / (hi - lo)
            for f in range(n_freqs):
                if lo <= fft_freqs[f] <= ctr and ctr > lo:
                    fb[f, mi] = (fft_freqs[f] - lo) / (ctr - lo) * enorm
                elif ctr < fft_freqs[f] <= hi and hi > ctr:
                    fb[f, mi] = (hi - fft_freqs[f]) / (hi - ctr) * enorm
    writer.add_tensor("mel_filters", fb.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
    mapped += 2

    print(f"\nWriting {mapped} tensors to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024 / 1024
    print(f"Done! {outfile} ({size_mb:.1f} MB, {mapped} tensors)")


if __name__ == "__main__":
    main()
