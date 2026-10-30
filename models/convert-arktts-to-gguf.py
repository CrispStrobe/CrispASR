#!/usr/bin/env python3
"""
Convert Edge0/Audio8-TTS-Realistic-0.6b (ArkttsModel) → GGUF for CrispASR.

Architecture (Audio8-TTS-Realistic-0.6b):
  Base Transformer (Slow LM):
    model_type            = "arktts"
    dim                   = 896
    intermediate_size     = 4864
    n_layer               = 24
    n_head                = 14
    n_local_heads         = 2
    head_dim              = 64
    rope_theta            = 1000000.0
    norm_eps              = 1e-06
    vocab_size            = 155776
    semantic_begin_id     = 151678
    semantic_end_id       = 155773

  Fast Subnetwork (Residual Audio Synthesis / RAS):
    n_fast_layer          = 4
    fast_dim              = 896
    fast_n_head           = 14
    fast_n_local_heads    = 2
    fast_head_dim         = 64
    fast_intermediate_size = 4864
    num_codebooks         = 10
    codebook_size         = 4096

  Codec Decoder:
    sample_rate           = 44100
    frame_size            = 2048
    hop_length            = 512
    channels              = 1536
    strides               = [8, 8, 4, 2]

Usage:
    python models/convert-arktts-to-gguf.py \\
        --input Edge0/Audio8-TTS-Realistic-0.6b \\
        --output audio8-tts-realistic-0.6b-f16.gguf \\
        --dtype f16
"""

from __future__ import annotations

import argparse
import json
import os
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


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(
        snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.pth", "*.json", "*.txt"],
        )
    )


def write_tensor(writer: GGUFWriter, name: str, data: np.ndarray, dtype: str):
    """Write tensor to GGUF using target quantization/float type."""
    is_1d_or_norm = data.ndim == 1 or "norm" in name or "bias" in name or "alpha" in name
    if dtype == "f16":
        if is_1d_or_norm:
            writer.add_tensor(name, data.astype(np.float32))
        else:
            writer.add_tensor(name, data.astype(np.float16))
    elif dtype == "bf16":
        if is_1d_or_norm:
            writer.add_tensor(name, data.astype(np.float32))
        else:
            writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.BF16)
    else:
        writer.add_tensor(name, data.astype(np.float32))


def convert_arktts(
    model_dir: Path,
    output_path: Path,
    dtype: str = "f16",
    include_codec: bool = True,
):
    config_path = model_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {model_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    writer = GGUFWriter(str(output_path), "arktts")
    writer.add_name("Audio8-TTS-Realistic-0.6b")

    # Architecture metadata
    dim = cfg.get("dim", 896)
    intermediate_size = cfg.get("intermediate_size", 4864)
    n_layer = cfg.get("n_layer", 24)
    n_head = cfg.get("n_head", 14)
    n_local_heads = cfg.get("n_local_heads", 2)
    head_dim = cfg.get("head_dim", 64)
    norm_eps = cfg.get("norm_eps", 1e-6)
    rope_theta = cfg.get("rope_theta", 1000000.0)
    vocab_size = cfg.get("vocab_size", 155776)
    num_codebooks = cfg.get("num_codebooks", 10)
    codebook_size = cfg.get("codebook_size", 4096)
    semantic_begin_id = cfg.get("semantic_begin_id", 151678)
    semantic_end_id = cfg.get("semantic_end_id", 155773)

    fast_dim = cfg.get("fast_dim", 896)
    fast_intermediate_size = cfg.get("fast_intermediate_size", 4864)
    n_fast_layer = cfg.get("n_fast_layer", 4)
    fast_n_head = cfg.get("fast_n_head", 14)
    fast_n_local_heads = cfg.get("fast_n_local_heads", 2)
    fast_head_dim = cfg.get("fast_head_dim", 64)

    writer.add_uint32("arktts.dim", dim)
    writer.add_uint32("arktts.intermediate_size", intermediate_size)
    writer.add_uint32("arktts.n_layer", n_layer)
    writer.add_uint32("arktts.n_head", n_head)
    writer.add_uint32("arktts.n_local_heads", n_local_heads)
    writer.add_uint32("arktts.head_dim", head_dim)
    writer.add_float32("arktts.norm_eps", norm_eps)
    writer.add_float32("arktts.rope_theta", rope_theta)
    writer.add_uint32("arktts.vocab_size", vocab_size)
    writer.add_uint32("arktts.num_codebooks", num_codebooks)
    writer.add_uint32("arktts.codebook_size", codebook_size)
    writer.add_uint32("arktts.semantic_begin_id", semantic_begin_id)
    writer.add_uint32("arktts.semantic_end_id", semantic_end_id)

    writer.add_uint32("arktts.fast_dim", fast_dim)
    writer.add_uint32("arktts.fast_intermediate_size", fast_intermediate_size)
    writer.add_uint32("arktts.n_fast_layer", n_fast_layer)
    writer.add_uint32("arktts.fast_n_head", fast_n_head)
    writer.add_uint32("arktts.fast_n_local_heads", fast_n_local_heads)
    writer.add_uint32("arktts.fast_head_dim", fast_head_dim)

    codec_sample_rate = cfg.get("codec_sample_rate", 44100)
    codec_frame_size = cfg.get("codec_frame_size", 2048)
    writer.add_uint32("arktts.codec.sample_rate", codec_sample_rate)
    writer.add_uint32("arktts.codec.frame_size", codec_frame_size)
    writer.add_uint32("arktts.codec.hop_length", 512)

    # 1. Convert Transformer & Fast layers from safetensors
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        sys.exit(f"model.safetensors not found in {model_dir}")

    print(f"Reading tensors from {st_path}...")
    with safe_open(str(st_path), framework="pt", device="cpu") as sf:
        for key in sf.keys():
            t = sf.get_tensor(key).float().numpy()
            write_tensor(writer, key, t, dtype)

    # 2. Convert Codec from codec.pth (if present)
    codec_path = model_dir / "codec.pth"
    if include_codec and codec_path.exists():
        print(f"Reading codec tensors from {codec_path}...")
        codec_sd = torch.load(str(codec_path), map_location="cpu")
        if isinstance(codec_sd, dict) and "state_dict" in codec_sd:
            codec_sd = codec_sd["state_dict"]
        for key, tensor in codec_sd.items():
            if isinstance(tensor, torch.Tensor):
                if "parametrizations.weight.original0" in key:
                    g = tensor.float()
                    v = codec_sd[key.replace("original0", "original1")].float()
                    folded_w = torch._weight_norm(v, g, 0).numpy()
                    out_name = "codec." + key.replace(".parametrizations.weight.original0", ".weight")
                    write_tensor(writer, out_name, folded_w, dtype)
                elif "parametrizations.weight.original1" in key:
                    continue
                elif "weight_g" in key:
                    g = tensor.float()
                    v = codec_sd[key.replace("weight_g", "weight_v")].float()
                    folded_w = torch._weight_norm(v, g, 0).numpy()
                    out_name = "codec." + key.replace(".weight_g", ".weight")
                    write_tensor(writer, out_name, folded_w, dtype)
                elif "weight_v" in key:
                    continue
                else:
                    arr = tensor.float().numpy()
                    write_tensor(writer, f"codec.{key}", arr, dtype)

    # 3. Add tokenizer if tokenizer.json present
    tok_path = model_dir / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        if vocab:
            sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
            tokens = [token for token, _ in sorted_tokens]
            scores = [0.0] * len(tokens)
            token_types = [1] * len(tokens)
            writer.add_tokenizer_model("gpt2")
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(token_types)

            raw_merges = tok_data.get("model", {}).get("merges", [])
            merges = []
            for m in raw_merges:
                if isinstance(m, list):
                    merges.append(" ".join(m))
                elif isinstance(m, str) and m and not m.startswith("#"):
                    merges.append(m)
            if merges:
                writer.add_token_merges(merges)

    print(f"Writing GGUF to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("GGUF conversion completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Convert Audio8-TTS (Arktts) to GGUF")
    parser.add_argument("--input", "-i", type=str, required=True, help="HF model ID or directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--dtype", type=str, default="f16", choices=["f16", "bf16", "f32"], help="Weight format")
    parser.add_argument("--no-codec", action="store_true", help="Omit codec vocoder weights")
    args = parser.parse_args()

    model_dir = load_model_dir(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_arktts(model_dir, output_path, dtype=args.dtype, include_codec=not args.no_codec)


if __name__ == "__main__":
    main()
