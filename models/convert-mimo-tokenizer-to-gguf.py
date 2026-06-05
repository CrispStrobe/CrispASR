#!/usr/bin/env python3
"""Convert XiaomiMiMo/MiMo-Audio-Tokenizer encoder to GGUF format.

The tokenizer is a 1.2B model with a 32-layer encoder + 32-layer decoder.
For ASR, we only need the ENCODER (waveform → RVQ tokens). The decoder
is for reconstruction/TTS and is skipped.

The encoder processes audio at 24kHz and produces 8-channel RVQ tokens
at 25 Hz (group_size=4, stride=2, avg_pooler=2 → 24000/240/2/2 = 25 Hz).

Usage:
    python models/convert-mimo-tokenizer-to-gguf.py --input XiaomiMiMo/MiMo-Audio-Tokenizer --output mimo-tokenizer.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. Install with: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub package not found. Install with: pip install huggingface_hub")
    sys.exit(1)


# Skip decoder tensors (not needed for ASR — only encoder → RVQ tokens)
SKIP_PREFIXES = [
    "decoder.",      # decoder layers, conv, norm (TTS reconstruction)
    "vocoder.",      # vocoder (waveform synthesis)
]


def should_skip(name: str) -> bool:
    return any(name.startswith(p) for p in SKIP_PREFIXES)


def map_tensor_name(hf_name: str) -> str | None:
    if should_skip(hf_name):
        return None
    name = hf_name
    # Encoder layers
    name = name.replace("encoder.layers.", "enc.blk.")
    name = name.replace("encoder.layer_norm.", "enc.norm.")
    # Encoder conv stem
    name = name.replace("encoder.econv1.", "enc.conv1.")
    name = name.replace("encoder.econv2.", "enc.conv2.")
    # RVQ quantizer
    name = name.replace("quantizer.", "quant.")
    # Attention
    name = name.replace(".self_attn.q_proj.", ".attn.q.")
    name = name.replace(".self_attn.k_proj.", ".attn.k.")
    name = name.replace(".self_attn.v_proj.", ".attn.v.")
    name = name.replace(".self_attn.out_proj.", ".attn.o.")
    name = name.replace(".self_attn_layer_norm.", ".attn_norm.")
    name = name.replace(".final_layer_norm.", ".ffn_norm.")
    return name


def load_model_dir(model_id: str) -> Path:
    model_path = Path(model_id)
    if model_path.is_dir():
        return model_path
    print(f"Downloading model from HuggingFace: {model_id}")
    path = snapshot_download(model_id,
                             allow_patterns=["*.safetensors", "config.json"])
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description="Convert MiMo-Audio-Tokenizer encoder to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16"],
                        help="Output data type for 2D+ tensors (default: f16)")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    print(f"\nMiMo-Audio-Tokenizer (encoder only)")
    print(f"  Encoder: {config['encoder_layers']}L, d_model={config['d_model']}, "
          f"heads={config['encoder_attention_heads']}, ffn={config['encoder_ffn_dim']}")
    print(f"  RVQ: {config['num_quantizers']} codebooks, sizes={config['codebook_size'][:8]}")
    print(f"  Sample rate: {config['sampling_rate']}, hop={config['hop_length']}")

    if args.outtype == "f16":
        out_dtype = np.float16
        ggml_type = GGMLQuantizationType.F16
    else:
        out_dtype = np.float32
        ggml_type = GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    tensor_names = {}
    for idx, h in enumerate(handles):
        for name in h.keys():
            tensor_names[name] = idx

    total = len(tensor_names)
    skippable = sum(1 for n in tensor_names if should_skip(n))
    print(f"  Safetensors: {total} tensors ({skippable} decoder/vocoder skipped)")

    outfile = Path(args.output)
    writer = GGUFWriter(str(outfile), "mimo_tokenizer", use_temp_file=True)
    writer.add_name("MiMo-Audio-Tokenizer")

    # Metadata
    writer.add_uint32("mimo_tok.d_model", config["d_model"])
    writer.add_uint32("mimo_tok.encoder_layers", config["encoder_layers"])
    writer.add_uint32("mimo_tok.encoder_heads", config["encoder_attention_heads"])
    writer.add_uint32("mimo_tok.encoder_ffn_dim", config["encoder_ffn_dim"])
    writer.add_uint32("mimo_tok.num_quantizers", config["num_quantizers"])
    writer.add_uint32("mimo_tok.sampling_rate", config["sampling_rate"])
    writer.add_uint32("mimo_tok.hop_length", config["hop_length"])
    writer.add_uint32("mimo_tok.stride_size", config.get("stride_size", 2))
    writer.add_uint32("mimo_tok.avg_pooler", config.get("avg_pooler", 2))
    writer.add_uint32("mimo_tok.kernel_size", config.get("kernel_size", 3))

    # Store codebook sizes (first 8 used by ASR)
    for i, cs in enumerate(config["codebook_size"][:20]):
        writer.add_uint32(f"mimo_tok.codebook_size.{i}", cs)

    mapped = 0
    for hf_name in sorted(tensor_names.keys()):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            continue

        data = handles[tensor_names[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        if data.ndim == 0:
            data = np.array([data.item()], dtype=np.float32)
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        elif data.ndim <= 1:
            data = np.ascontiguousarray(data.astype(np.float32))
            writer.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        else:
            data = np.ascontiguousarray(data.astype(out_dtype))
            writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)
        mapped += 1
        if mapped % 50 == 0:
            print(f"  [{mapped}] {gguf_name}")

    print(f"\nWriting {mapped} tensors to {outfile}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = outfile.stat().st_size / 1024**2
    print(f"Done! {outfile} ({size_mb:.0f} MB, {mapped} tensors)")


if __name__ == "__main__":
    main()
