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
import torch

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-decoder", action="store_true",
                        help="Include audio decoder tensors for TTS support")
    args = parser.parse_args()

    from safetensors import safe_open

    # Resolve model dir
    if os.path.isdir(args.input):
        model_dir = args.input
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.input)

    # Read config
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
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
    tts_n_layers = cfg.get("tts_backbone_num_hidden_layers", 0)

    print(f"  vocab={vocab_size}, rope_theta={rope_theta}, head_dim={head_dim}")
    if tts_n_layers > 0:
        print(f"  TTS LM: {tts_n_layers} layers (streaming model)")

    # Pre-scan tensors to detect ACTUAL base LM layer count before writing metadata.
    # Realtime-0.5B's config says 24 base LM layers but ships only 4 — every metadata
    # write happens once with the corrected value.
    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    detected_base_layers = set()
    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                if "language_model.layers." in name and "tts_" not in name:
                    try:
                        layer = int(name.split("language_model.layers.")[1].split(".")[0])
                        detected_base_layers.add(layer)
                    except (ValueError, IndexError):
                        pass
    if detected_base_layers:
        actual_base_layers = max(detected_base_layers) + 1
        if actual_base_layers != n_lm_layers:
            print(f"  NOTE: actual base LM layers = {actual_base_layers} (config says {n_lm_layers})")
            n_lm_layers = actual_base_layers

    # Create GGUF
    model_name = os.path.basename(args.input.rstrip("/"))
    arch = "vibevoice-tts" if tts_n_layers > 0 else "vibevoice-asr"
    writer = gguf.GGUFWriter(args.output, arch)
    writer.add_name(model_name)

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
    writer.add_uint32("vibevoice.has_decoder", 1 if args.include_decoder else 0)
    writer.add_array("vibevoice.encoder_depths", encoder_depths)

    # TTS-specific metadata (VibeVoice-Realtime streaming model)
    tts_n_layers = cfg.get("tts_backbone_num_hidden_layers", 0)
    writer.add_uint32("vibevoice.tts_n_layers", tts_n_layers)
    diff_cfg = cfg.get("diffusion_head_config", {})
    if diff_cfg:
        writer.add_string("vibevoice.prediction_type", diff_cfg.get("prediction_type", "v_prediction"))
        writer.add_string("vibevoice.beta_schedule", diff_cfg.get("ddpm_beta_schedule", "cosine"))
        writer.add_uint32("vibevoice.ddpm_num_steps", diff_cfg.get("ddpm_num_steps", 1000))
        writer.add_uint32("vibevoice.ddpm_inference_steps", diff_cfg.get("ddpm_num_inference_steps", 20))

    # Embed Qwen2 vocabulary for token ID → string decoding.
    # The VibeVoice snapshot itself does not ship tokenizer files; the
    # official HF converter falls back to Qwen/Qwen2.5-7B.
    try:
        from transformers import AutoTokenizer

        tok = None
        tok_errs = []
        for tok_src in (model_dir, "Qwen/Qwen2.5-7B"):
            try:
                tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
                print(f"  loaded tokenizer from: {tok_src}")
                break
            except Exception as e:
                tok_errs.append(f"{tok_src}: {e}")

        if tok is None:
            raise RuntimeError(" ; ".join(tok_errs))

        vocab_map = tok.get_vocab()
        inv = {v: k for k, v in vocab_map.items()}
        max_id = max(inv.keys())
        vocab_list = [inv.get(i, f"<unk_{i}>") for i in range(max_id + 1)]
        writer.add_array("tokenizer.ggml.tokens", vocab_list)
        writer.add_uint32("vibevoice.has_tokenizer", 1)
        print(f"  Qwen2 tokenizer: {len(vocab_list)} tokens embedded")
    except Exception as e:
        writer.add_uint32("vibevoice.has_tokenizer", 0)
        print(f"  Tokenizer not embedded: {e}")

    # Name shortening for GGUF 64-char limit
    def shorten(name):
        # Strip leading "model." prefix only (not "model." inside "language_model.")
        if name.startswith("model."):
            name = name[len("model."):]
        # tts_eos_classifier is at top level (no model. prefix)
        name = name.replace("tts_eos_classifier.", "tts_eos.")
        name = name.replace("acoustic_tokenizer.encoder.", "at_enc.")
        name = name.replace("acoustic_tokenizer.decoder.", "at_dec.")
        name = name.replace("semantic_tokenizer.encoder.", "st_enc.")
        name = name.replace("semantic_tokenizer.decoder.", "st_dec.")
        name = name.replace("upsample_layers.", "us.")
        name = name.replace("convtr.convtr.", "convtr.")
        name = name.replace("head.conv.conv.", "head.")
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
        name = name.replace("tts_language_model.", "tts_lm.")
        name = name.replace("tts_input_types.", "tts_types.")
        return name

    # Load and write tensors
    tensor_count = 0
    skipped = 0

    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt") as f:
            for name in sorted(f.keys()):
                # Skip decoder (synthesis) tensors unless --include-decoder
                if not args.include_decoder:
                    if "acoustic_tokenizer.decoder" in name:
                        skipped += 1
                        continue
                    if "semantic_tokenizer.decoder" in name:
                        skipped += 1
                        continue

                gguf_name = shorten(name)

                if len(gguf_name) >= 64:
                    gguf_name = gguf_name.replace("layers.", "l.")
                    if len(gguf_name) >= 64:
                        print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
                        skipped += 1
                        continue

                # Load in native dtype to avoid doubling peak RAM.
                # (7B token embedding is ~600 MB BF16; .float() first would be 1.2 GB)
                raw = f.get_tensor(name)

                # Store norms/biases/scalars as F32, weights as F16.
                # Exception: depthwise conv weights (dw_conv) stay F32 because
                # ggml_conv_1d_dw forces F16 im2col intermediates — storing
                # weights as F32 avoids double F16 precision loss through 29 blocks.
                is_f32 = ("norm" in name or "gamma" in name or name.endswith(".bias") or
                          raw.ndim <= 1 or "scaling_factor" in name or "bias_factor" in name)
                if is_f32:
                    data = raw.to(torch.float32).numpy()
                else:
                    data = raw.to(torch.float16).numpy()

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
