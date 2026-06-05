#!/usr/bin/env python3
"""Convert Kyutai STT (stt-1b-en_fr or stt-2.6b-en) to GGUF.

Architecture: Mimi audio codec (encoder) + multistream transformer decoder.
Two separate weight files combined into one GGUF:
  - model.safetensors (STT language model)
  - mimi-pytorch-*.safetensors (Mimi codec)
  - tokenizer_*.model (SentencePiece)

Usage:
  python models/convert-kyutai-stt-to-gguf.py \
      --input kyutai/stt-1b-en_fr \
      --output kyutai-stt-1b.gguf
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser(description="Convert Kyutai STT to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    print(f"Loading: {args.input}")

    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    import sentencepiece as spm

    cache_dir = os.environ.get("HF_HOME", None)
    if cache_dir:
        cache_dir += "/hub"

    # Resolve file paths
    if os.path.isdir(args.input):
        base = args.input
        config_path = os.path.join(base, "config.json")
    else:
        config_path = hf_hub_download(args.input, "config.json", cache_dir=cache_dir)
        base = os.path.dirname(config_path)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    mimi_name = config.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors")
    tokenizer_name = config.get("tokenizer_name", "tokenizer_en_fr_audio_8000.model")

    # Download model files
    if not os.path.isdir(args.input):
        for fname in ["model.safetensors", mimi_name, tokenizer_name]:
            hf_hub_download(args.input, fname, cache_dir=cache_dir)

    lm_path = os.path.join(base, "model.safetensors")
    mimi_path = os.path.join(base, mimi_name)
    tok_path = os.path.join(base, tokenizer_name)

    # Load weights
    lm_sd = {}
    with safe_open(lm_path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            lm_sd[key] = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
    print(f"  LM: {len(lm_sd)} tensors")

    mimi_sd = {}
    with safe_open(mimi_path, framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            mimi_sd[key] = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
    print(f"  Mimi: {len(mimi_sd)} tensors")

    # Pre-compute codebook embeddings from embedding_sum / cluster_usage
    codebook_keys = set()
    codebook_embeddings = {}
    for key in list(mimi_sd.keys()):
        if key.endswith("._codebook.embedding_sum"):
            prefix = key[: -len("embedding_sum")]
            usage_key = prefix + "cluster_usage"
            init_key = prefix + "_initialized"

            embedding_sum = mimi_sd[key]   # [num_codes, dim]
            cluster_usage = mimi_sd[usage_key]  # [num_codes]
            # Clamp to avoid division by zero
            cluster_usage = np.maximum(cluster_usage, 1e-5)
            embedding = embedding_sum / cluster_usage[:, None]

            emb_key = prefix + "embedding"
            codebook_embeddings[emb_key] = embedding.astype(np.float32)
            codebook_keys.update([key, usage_key])
            if init_key in mimi_sd:
                codebook_keys.add(init_key)

    print(f"  Pre-computed {len(codebook_embeddings)} codebook embeddings")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    vocab_size = sp.GetPieceSize()
    print(f"  Tokenizer: {vocab_size} tokens")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "kyutai-stt")
    writer.add_name("Kyutai-STT")

    # Write config as GGUF KV
    writer.add_uint32("kyutai.card", config["card"])  # 2048 codebook size
    writer.add_uint32("kyutai.n_q", config["n_q"])  # 32 codebooks
    writer.add_uint32("kyutai.dim", config["dim"])  # 2048
    writer.add_uint32("kyutai.text_card", config["text_card"])  # 8000
    writer.add_uint32("kyutai.num_heads", config["num_heads"])  # 16
    writer.add_uint32("kyutai.num_layers", config["num_layers"])  # 16
    writer.add_uint32("kyutai.context", config["context"])  # 750
    writer.add_float32("kyutai.max_period", config["max_period"])  # 100000
    writer.add_float32("kyutai.hidden_scale", config["hidden_scale"])  # 4.125
    writer.add_uint32("kyutai.existing_text_padding_id", config["existing_text_padding_id"])  # 3
    writer.add_float32("kyutai.stt.audio_delay_seconds", config["stt_config"]["audio_delay_seconds"])

    # Mimi-specific hparams (fixed for the released Mimi codec)
    writer.add_uint32("kyutai.mimi.encoder_dim", 512)
    writer.add_uint32("kyutai.mimi.encoder_num_heads", 8)
    writer.add_uint32("kyutai.mimi.encoder_num_layers", 8)
    writer.add_uint32("kyutai.mimi.encoder_context", 250)
    writer.add_uint32("kyutai.mimi.codebook_dim", 256)
    writer.add_uint32("kyutai.mimi.n_q_semantic", 1)
    writer.add_uint32("kyutai.mimi.n_q_acoustic", config["n_q"] - 1)  # 31
    # SEANet encoder stride = 4*5*6*8 = 960, then downsample stride=2
    # Final rate = 24000 / (960 * 2) = 12.5 Hz
    writer.add_float32("kyutai.mimi.frame_rate", 12.5)
    writer.add_uint32("kyutai.mimi.sample_rate", 24000)

    # Tokenizer vocab
    vocab_list = [sp.IdToPiece(i) for i in range(vocab_size)]
    writer.add_array("tokenizer.ggml.tokens", vocab_list)

    # Write tensors
    print(f"\nWriting: {args.output}")

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

    # Tensor name shortening to stay under GGUF's 64-char limit.
    def shorten_name(name):
        name = name.replace("encoder_transformer.transformer.", "enc_tfm.")
        name = name.replace("decoder_transformer.transformer.", "dec_tfm.")
        name = name.replace("self_attn.in_proj_weight", "attn.qkv_w")
        name = name.replace("self_attn.out_proj.weight", "attn.out_w")
        name = name.replace("layer_scale_1.scale", "ls1")
        name = name.replace("layer_scale_2.scale", "ls2")
        name = name.replace("linear1.weight", "ffn_up_w")
        name = name.replace("linear2.weight", "ffn_down_w")
        name = name.replace("gating.linear_in.weight", "gating_in_w")
        name = name.replace("gating.linear_out.weight", "gating_out_w")
        name = name.replace("conv.conv.conv.", "conv3.")
        name = name.replace("conv.conv.", "conv2.")
        name = name.replace("block.1.conv2.", "blk1.")
        name = name.replace("block.3.conv2.", "blk3.")
        return name

    tensor_count = 0

    # LM tensors (prefix with "lm.", shorten names)
    for name, tensor in sorted(lm_sd.items()):
        gguf_name = shorten_name("lm." + name)
        if "norm" in name or name.endswith(".bias") or len(tensor.shape) <= 1:
            data = f32(tensor)
            dtype_str = "F32"
        else:
            data = f16(tensor)
            dtype_str = "F16"
        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {gguf_name:60s} {str(tensor.shape):20s} {dtype_str}")

    # Skip decoder/upsample tensors (not needed for STT)
    skip_prefixes = ("decoder.", "decoder_transformer.", "upsample.")

    # Mimi tensors (prefix with "mimi.", skip raw codebook training artifacts)
    for name, tensor in sorted(mimi_sd.items()):
        if name in codebook_keys:
            continue  # skip embedding_sum, cluster_usage, _initialized
        # Skip decoder (not needed for STT encode-only path)
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        gguf_name = shorten_name("mimi." + name)
        assert len(gguf_name) < 64, f"Tensor name too long ({len(gguf_name)}): {gguf_name}"
        if "norm" in name or name.endswith(".bias") or len(tensor.shape) <= 1:
            data = f32(tensor)
            dtype_str = "F32"
        else:
            data = f16(tensor)
            dtype_str = "F16"
        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {gguf_name:60s} {str(tensor.shape):20s} {dtype_str}")

    # Pre-computed codebook embeddings
    for name, tensor in sorted(codebook_embeddings.items()):
        gguf_name = shorten_name("mimi." + name)
        assert len(gguf_name) < 64, f"Codebook name too long ({len(gguf_name)}): {gguf_name}"
        data = f32(tensor)  # codebooks always F32 for precision
        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        print(f"  [{tensor_count}] {gguf_name:60s} {str(tensor.shape):20s} F32 (computed)")

    print(f"  ... total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
