#!/usr/bin/env python3
"""
Convert Aratako/MioTTS-{0.6B,1.7B} + MioCodec-25Hz-24kHz to GGUF F16.

MioTTS-0.6B is a standard Qwen3ForCausalLM (28 layers, 1024 hidden, GQA 16/8)
trained to generate speech tokens <|s_0|>...<|s_12799|> in its vocabulary.
The codec (MioCodec) converts those token indices → waveform using:
  FSQ dequant → wave_prenet transformer → conv_upsample → ResNet →
  wave_decoder transformer (AdaLN-Zero conditioned on 128-d speaker emb) →
  ResNet → iSTFT head → 24kHz audio.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    TMPDIR=/mnt/volume1/tmp-overflow \\
    python models/convert-miotts-to-gguf.py \\
        --llm /mnt/storage/models/miotts-0.6b \\
        --codec /mnt/storage/models/miocodec-25hz-24khz \\
        --output /mnt/storage/gguf-models/miotts-0.6b-f16.gguf \\
        --dtype f16
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
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")


def resolve_file(directory: Path, name: str) -> Path | None:
    """Find a safetensors or bin file in the model directory."""
    for ext in (".safetensors", ".bin"):
        p = directory / f"{name}{ext}"
        if p.exists():
            return p
    # Single-file models
    p = directory / "model.safetensors"
    if p.exists():
        return p
    return None


def load_tensor(sf, name: str) -> np.ndarray:
    """Load a tensor from safetensors as float32 numpy (handles bf16)."""
    t = sf.get_tensor(name)
    if hasattr(t, 'numpy'):
        # torch tensor
        return t.float().numpy()
    return t.astype(np.float32) if t.dtype != np.float32 else t


def write_tensor(writer: GGUFWriter, name: str, data: np.ndarray, dtype: str):
    """Write a tensor to GGUF with appropriate type."""
    if dtype == "f16":
        # Norms and biases stay F32
        is_norm = "norm" in name or "bias" in name or name.endswith(".weight") and data.ndim == 1
        if is_norm or data.ndim == 1:
            writer.add_tensor(name, data.astype(np.float32))
        else:
            writer.add_tensor(name, data.astype(np.float16))
    else:
        writer.add_tensor(name, data.astype(np.float32))


def convert_llm(writer: GGUFWriter, llm_dir: Path, dtype: str):
    """Convert the Qwen3 LLM weights."""
    model_file = resolve_file(llm_dir, "model")
    if model_file is None:
        sys.exit(f"No model file found in {llm_dir}")

    print(f"  LLM: {model_file}")
    with safe_open(str(model_file), framework="pt") as sf:
        keys = list(sf.keys())
        print(f"  LLM tensors: {len(keys)}")

        for key in sorted(keys):
            data = load_tensor(sf, key)
            # Map HF Qwen3 names to our GGUF names
            gguf_name = map_llm_name(key)
            if gguf_name is None:
                print(f"  [skip] {key} {data.shape}")
                continue
            write_tensor(writer, gguf_name, data, dtype)
            if data.size < 1000:
                print(f"  {key} -> {gguf_name} {data.shape}")


def map_llm_name(hf_name: str) -> str | None:
    """Map HuggingFace Qwen3 weight names to GGUF convention."""
    # Standard Qwen3 mapping
    name = hf_name
    # model.embed_tokens.weight -> token_embd.weight
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    # model.norm.weight -> output_norm.weight
    if name == "model.norm.weight":
        return "output_norm.weight"
    # lm_head.weight -> output.weight (if not tied)
    if name == "lm_head.weight":
        return "output.weight"

    # model.layers.N.* -> blk.N.*
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer_idx = parts[2]
        rest = ".".join(parts[3:])

        prefix = f"blk.{layer_idx}"

        mapping = {
            "input_layernorm.weight": f"{prefix}.attn_norm.weight",
            "post_attention_layernorm.weight": f"{prefix}.ffn_norm.weight",
            "self_attn.q_proj.weight": f"{prefix}.attn_q.weight",
            "self_attn.k_proj.weight": f"{prefix}.attn_k.weight",
            "self_attn.v_proj.weight": f"{prefix}.attn_v.weight",
            "self_attn.o_proj.weight": f"{prefix}.attn_output.weight",
            "self_attn.q_norm.weight": f"{prefix}.attn_q_norm.weight",
            "self_attn.k_norm.weight": f"{prefix}.attn_k_norm.weight",
            "mlp.gate_proj.weight": f"{prefix}.ffn_gate.weight",
            "mlp.up_proj.weight": f"{prefix}.ffn_up.weight",
            "mlp.down_proj.weight": f"{prefix}.ffn_down.weight",
        }
        return mapping.get(rest)

    return None


def convert_codec(writer: GGUFWriter, codec_dir: Path, dtype: str):
    """Convert MioCodec decode-path weights."""
    model_file = resolve_file(codec_dir, "model")
    if model_file is None:
        sys.exit(f"No model file found in {codec_dir}")

    print(f"  Codec: {model_file}")
    with safe_open(str(model_file), framework="pt") as sf:
        keys = list(sf.keys())
        print(f"  Codec tensors: {len(keys)}")

        # Only extract decode-path weights:
        # - local_quantizer (FSQ projection layers)
        # - wave_prenet, wave_decoder, wave_conv_upsample
        # - wave_prior_net, wave_post_net
        # - istft_head
        # - global_encoder (for reference audio → speaker embedding)
        decode_prefixes = (
            "local_quantizer.",
            "wave_prenet.",
            "wave_decoder.",
            "wave_conv_upsample.",
            "wave_prior_net.",
            "wave_post_net.",
            "wave_upsampler.",
            "istft_head.",
            "global_encoder.",
        )

        count = 0
        # Fuse weight_norm parametrizations at convert time:
        # actual_weight = original0 * original1 / norm(original1, dim=1, keepdim=True)
        fused_wn = {}
        for key in sorted(keys):
            if "parametrizations.weight.original" in key:
                # e.g. wave_upsampler.upsample_layers.0.parametrizations.weight.original0
                base = key.rsplit(".parametrizations.weight.", 1)[0]
                suffix = key.rsplit(".", 1)[1]  # original0 or original1
                if base not in fused_wn:
                    fused_wn[base] = {}
                fused_wn[base][suffix] = load_tensor(sf, key)

        for key in sorted(keys):
            if not any(key.startswith(p) for p in decode_prefixes):
                continue
            if "parametrizations.weight.original" in key:
                continue  # handled via fusion below
            data = load_tensor(sf, key)
            gguf_name = f"codec.{key}"
            write_tensor(writer, gguf_name, data, dtype)
            count += 1

        # Write fused weight_norm weights
        for base, parts in fused_wn.items():
            if not any(base.startswith(p) for p in decode_prefixes):
                continue
            if "original0" in parts and "original1" in parts:
                g = parts["original0"]  # (out_ch, 1, 1) scale
                v = parts["original1"]  # (out_ch, in_ch, k) weight
                # weight_norm: w = g * v / ||v||_2
                norm = np.sqrt(np.sum(v * v, axis=(1, 2), keepdims=True) + 1e-12)
                fused = g * v / norm
                gguf_name = f"codec.{base}.weight"
                write_tensor(writer, gguf_name, fused, dtype)
                count += 1
                print(f"  [fused weight_norm] {base} → {fused.shape}")

        print(f"  Codec decode tensors written: {count}")


def main():
    parser = argparse.ArgumentParser(description="Convert MioTTS + MioCodec to GGUF")
    parser.add_argument("--llm", required=True, help="Path to MioTTS-0.6B directory")
    parser.add_argument("--codec", required=True, help="Path to MioCodec-25Hz-24kHz directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args()

    llm_dir = Path(args.llm)
    codec_dir = Path(args.codec)

    # Load LLM config
    config_path = llm_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"config.json not found in {llm_dir}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    print(f"MioTTS converter")
    print(f"  Model type: {config.get('model_type', '?')}")
    print(f"  Hidden size: {config.get('hidden_size', '?')}")
    print(f"  Layers: {config.get('num_hidden_layers', '?')}")
    print(f"  Heads: {config.get('num_attention_heads', '?')}")
    print(f"  KV heads: {config.get('num_key_value_heads', '?')}")
    print(f"  Vocab: {config.get('vocab_size', '?')}")

    # Create GGUF
    writer = GGUFWriter(args.output, "miotts", use_temp_file=True)

    # Metadata
    writer.add_name("MioTTS")
    writer.add_description(f"MioTTS {config.get('hidden_size', '?')}d + MioCodec-25Hz-24kHz")
    writer.add_file_type(GGMLQuantizationType.F16 if args.dtype == "f16" else GGMLQuantizationType.F32)

    # Architecture params
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    hidden = config["hidden_size"]
    intermediate = config.get("intermediate_size", hidden * 3)
    vocab = config["vocab_size"]
    head_dim = config.get("head_dim", hidden // n_heads)
    rope_theta = config.get("rope_theta", 1000000.0)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)

    writer.add_uint32("miotts.block_count", n_layers)
    writer.add_uint32("miotts.attention.head_count", n_heads)
    writer.add_uint32("miotts.attention.head_count_kv", n_kv_heads)
    writer.add_uint32("miotts.embedding_length", hidden)
    writer.add_uint32("miotts.feed_forward_length", intermediate)
    writer.add_uint32("miotts.vocab_size", vocab)
    writer.add_uint32("miotts.head_dim", head_dim)
    writer.add_float32("miotts.rope_theta", rope_theta)
    writer.add_float32("miotts.rms_norm_eps", rms_norm_eps)
    writer.add_uint32("miotts.context_length", config.get("max_position_embeddings", 32768))

    # Codec params — detect from config.yaml if available, else use v2 defaults
    codec_dir = Path(args.codec)
    codec_cfg_path = codec_dir / "config.yaml"
    codec_sr = 44100
    codec_n_fft = 392
    codec_hop = 98
    if codec_cfg_path.exists():
        import re
        cfg_text = codec_cfg_path.read_text()
        m = re.search(r'sample_rate:\s*(\d+)', cfg_text)
        if m: codec_sr = int(m.group(1))
        m = re.search(r'n_fft:\s*(\d+)', cfg_text)
        if m: codec_n_fft = int(m.group(1))
        m = re.search(r'hop_length:\s*(\d+)', cfg_text)
        if m: codec_hop = int(m.group(1))
    writer.add_uint32("miotts.codec.sample_rate", codec_sr)
    writer.add_uint32("miotts.codec.frame_rate", 25)
    writer.add_uint32("miotts.codec.n_fft", codec_n_fft)
    writer.add_uint32("miotts.codec.hop_length", codec_hop)
    writer.add_uint32("miotts.codec.codebook_size", 12800)
    # FSQ levels: [8, 8, 8, 5, 5]
    writer.add_array("miotts.codec.fsq_levels", [8, 8, 8, 5, 5])
    writer.add_uint32("miotts.codec.global_dim", 128)
    writer.add_uint32("miotts.codec.wave_dim", 512)
    writer.add_uint32("miotts.codec.wave_prenet_layers", 6)
    writer.add_uint32("miotts.codec.wave_decoder_layers", 8)

    # Speech token range in the LLM vocabulary
    # From tokenizer analysis: speech tokens are ids 151669..164468
    writer.add_uint32("miotts.speech_token_start", 151669)
    writer.add_uint32("miotts.speech_token_end", 164469)  # exclusive

    # EOS token for generation stopping
    writer.add_uint32("miotts.eos_token_id", config.get("eos_token_id", 151645))

    # NOTE: tokenizer is loaded from tokenizer.json at runtime by the C++ code
    # (miotts_load_tokenizer). We don't embed it in the GGUF because the gguf
    # Python library v0.19 writes string arrays as GGUF type 9 which our
    # vendored ggml C reader doesn't support. Place tokenizer.json next to
    # the GGUF or in the same directory as the source model.

    print("\nConverting LLM weights...")
    convert_llm(writer, llm_dir, args.dtype)

    print("\nConverting codec weights...")
    convert_codec(writer, codec_dir, args.dtype)

    print(f"\nWriting GGUF to {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done! Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
