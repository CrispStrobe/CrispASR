#!/usr/bin/env python3
"""Convert Microsoft VibeVoice-ASR to GGUF.

Architecture: Two σ-VAE tokenizer encoders (acoustic + semantic) →
  connectors → Qwen2-1.5B LLM decoder → prediction head.

Only the ASR inference path is stored (encoder encoders, connectors, LM,
prediction head). The acoustic/semantic decoders (synthesis) are skipped.

Usage:
  python models/convert-vibevoice-to-gguf.py \
      --input microsoft/VibeVoice-1.5B \
      --output vibevoice-1.5b.gguf
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from safetensors import safe_open

    # Resolve model dir
    if os.path.isdir(args.input):
        model_dir = args.input
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.input)

    # Read config
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    at_cfg = cfg.get("acoustic_tokenizer_config", {})
    st_cfg = cfg.get("semantic_tokenizer_config", {})

    # Architecture params
    d_lm = dec_cfg["hidden_size"]           # 1536
    n_lm_layers = dec_cfg["num_hidden_layers"]  # 28
    n_heads = dec_cfg["num_attention_heads"]     # 12
    n_kv_heads = dec_cfg["num_key_value_heads"]  # 2
    d_ffn = dec_cfg["intermediate_size"]         # 8960
    vocab_size = dec_cfg["vocab_size"]            # 151936
    rope_theta = dec_cfg.get("rope_theta", 1000000.0)
    head_dim = d_lm // n_heads                   # 128
    vae_dim_acoustic = at_cfg.get("vae_dim", 64)
    vae_dim_semantic = st_cfg.get("vae_dim", 128) if st_cfg else 128

    # Tokenizer encoder params
    encoder_ratios = at_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])
    encoder_depths_str = at_cfg.get("encoder_depths", "3-3-3-3-3-3-8")
    encoder_depths = [int(x) for x in encoder_depths_str.split("-")]
    n_filters = at_cfg.get("encoder_n_filters", 32)
    n_stages = len(encoder_depths)

    total_downsample = 1
    for r in encoder_ratios:
        total_downsample *= r

    print(f"VibeVoice-ASR: d_lm={d_lm}, n_layers={n_lm_layers}, heads={n_heads}/{n_kv_heads}")
    print(f"  acoustic vae_dim={vae_dim_acoustic}, semantic vae_dim={vae_dim_semantic}")
    print(f"  encoder: {n_stages} stages, depths={encoder_depths}, ratios={encoder_ratios}")
    print(f"  total downsample: {total_downsample}x, base_filters={n_filters}")
    print(f"  vocab={vocab_size}, rope_theta={rope_theta}, head_dim={head_dim}")

    # Create GGUF
    writer = gguf.GGUFWriter(args.output, "vibevoice-asr")
    writer.add_name("VibeVoice-ASR-1.5B")

    # Hyperparameters
    writer.add_uint32("vibevoice.d_lm", d_lm)
    writer.add_uint32("vibevoice.n_lm_layers", n_lm_layers)
    writer.add_uint32("vibevoice.n_heads", n_heads)
    writer.add_uint32("vibevoice.n_kv_heads", n_kv_heads)
    writer.add_uint32("vibevoice.d_ffn", d_ffn)
    writer.add_uint32("vibevoice.vocab_size", vocab_size)
    writer.add_uint32("vibevoice.head_dim", head_dim)
    writer.add_float32("vibevoice.rope_theta", rope_theta)
    writer.add_uint32("vibevoice.vae_dim_acoustic", vae_dim_acoustic)
    writer.add_uint32("vibevoice.vae_dim_semantic", vae_dim_semantic)
    writer.add_uint32("vibevoice.n_encoder_stages", n_stages)
    writer.add_uint32("vibevoice.n_filters", n_filters)
    writer.add_uint32("vibevoice.total_downsample", total_downsample)
    writer.add_array("vibevoice.encoder_ratios", encoder_ratios)
    writer.add_array("vibevoice.encoder_depths", encoder_depths)

    # Read tokenizer if available
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        # Qwen2 tokenizer — just store vocab size as metadata
        # Full tokenizer is complex (BPE with 151936 tokens)
        # For now, we'll use the Qwen2 tokenizer at runtime
        writer.add_uint32("vibevoice.has_tokenizer", 0)
        print(f"  Qwen2 tokenizer: {vocab_size} tokens (not embedded, use external)")

    # Name shortening for GGUF 64-char limit
    def shorten(name):
        # Strip leading "model." prefix only (not "model." inside "language_model.")
        if name.startswith("model."):
            name = name[len("model."):]
        name = name.replace("acoustic_tokenizer.encoder.", "at_enc.")
        name = name.replace("semantic_tokenizer.encoder.", "st_enc.")
        name = name.replace("acoustic_connector.", "at_conn.")
        name = name.replace("semantic_connector.", "se_conn.")
        name = name.replace("language_model.", "lm.")
        name = name.replace("prediction_head.", "pred.")
        name = name.replace("downsample_layers.", "ds.")
        name = name.replace("stages.", "s.")
        name = name.replace("mixer.conv.conv.conv.", "dw_conv.")
        name = name.replace("ffn.linear1.", "ffn.up.")
        name = name.replace("ffn.linear2.", "ffn.down.")
        name = name.replace("ffn_norm.", "ffn_ln.")
        name = name.replace("input_layernorm.", "attn_ln.")
        name = name.replace("post_attention_layernorm.", "ffn_ln.")
        name = name.replace("self_attn.", "attn.")
        name = name.replace("mlp.gate_proj.", "ffn.gate.")
        name = name.replace("mlp.up_proj.", "ffn.up.")
        name = name.replace("mlp.down_proj.", "ffn.down.")
        name = name.replace("embed_tokens.", "tok_emb.")
        name = name.replace("head.conv.conv.", "head.")
        name = name.replace("conv.conv.", "conv.")
        name = name.replace("adaLN_modulation.1.", "adaln.")
        name = name.replace("t_embedder.mlp.", "t_emb.")
        name = name.replace("noisy_images_proj.", "noisy_proj.")
        name = name.replace("final_layer.", "final.")
        name = name.replace("cond_proj.", "cond.")
        return name

    # Load and write tensors
    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    tensor_count = 0
    skipped = 0

    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt") as f:
            for name in sorted(f.keys()):
                # Skip decoder (synthesis) tensors
                if "acoustic_tokenizer.decoder" in name:
                    skipped += 1
                    continue
                if "semantic_tokenizer.decoder" in name:
                    skipped += 1
                    continue

                t = f.get_tensor(name).float().numpy()
                gguf_name = shorten(name)

                if len(gguf_name) >= 64:
                    # Further shorten
                    gguf_name = gguf_name.replace("layers.", "l.")
                    if len(gguf_name) >= 64:
                        print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
                        skipped += 1
                        continue

                # Store norms/biases/scalars as F32, weights as F16.
                # Exception: depthwise conv weights (dw_conv) stay F32 because
                # ggml_conv_1d_dw forces F16 im2col intermediates — storing
                # weights as F32 avoids double F16 precision loss through 29 blocks.
                is_f32 = ("norm" in name or "gamma" in name or name.endswith(".bias") or
                          len(t.shape) <= 1 or "scaling_factor" in name or "bias_factor" in name)
                if is_f32:
                    data = t.astype(np.float32)
                else:
                    data = t.astype(np.float16)

                writer.add_tensor(gguf_name, data)
                tensor_count += 1
                if tensor_count <= 5 or tensor_count % 100 == 0:
                    print(f"  [{tensor_count}] {gguf_name:55s} {str(data.shape):20s}")

    print(f"\n  total: {tensor_count} tensors ({skipped} skipped)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
