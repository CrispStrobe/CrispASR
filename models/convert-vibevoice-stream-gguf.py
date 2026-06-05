#!/usr/bin/env python3
"""Streaming GGUF converter for large VibeVoice models (7B+).

Two-pass approach:
  Pass 1: scan safetensors, collect tensor metadata (no data loaded)
  Pass 2: load each tensor one at a time, convert dtype, write to GGUF file

Peak RAM: ~2 GB regardless of model size (only one tensor in memory at a time).

Usage:
  python models/convert-vibevoice-stream-gguf.py \
      --input vibevoice/VibeVoice-7B \
      --output vibevoice-7b-f16.gguf \
      --include-decoder
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf

from safetensors import safe_open


def shorten(name):
    if name.startswith("model."):
        name = name[len("model."):]
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
    name = name.replace("conv.conv.", "conv.")
    name = name.replace("adaLN_modulation.1.", "adaln.")
    name = name.replace("t_embedder.mlp.", "t_emb.")
    name = name.replace("noisy_images_proj.", "noisy_proj.")
    name = name.replace("final_layer.", "final.")
    name = name.replace("cond_proj.", "cond.")
    name = name.replace("tts_language_model.", "tts_lm.")
    name = name.replace("tts_input_types.", "tts_types.")
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-decoder", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        model_dir = args.input
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.input)

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    dec_cfg = cfg.get("decoder_config", {})
    at_cfg = cfg.get("acoustic_tokenizer_config", {})
    st_cfg = cfg.get("semantic_tokenizer_config", {})

    d_lm = dec_cfg["hidden_size"]
    n_lm_layers = dec_cfg["num_hidden_layers"]
    n_heads = dec_cfg["num_attention_heads"]
    n_kv_heads = dec_cfg["num_key_value_heads"]
    d_ffn = dec_cfg["intermediate_size"]
    vocab_size = dec_cfg["vocab_size"]
    rope_theta = dec_cfg.get("rope_theta", 1000000.0)
    head_dim = d_lm // n_heads
    vae_dim_acoustic = at_cfg.get("vae_dim", 64)
    vae_dim_semantic = st_cfg.get("vae_dim", 128) if st_cfg else 128
    encoder_ratios = at_cfg.get("encoder_ratios", [8, 5, 5, 4, 2, 2])
    encoder_depths_str = at_cfg.get("encoder_depths", "3-3-3-3-3-3-8")
    encoder_depths = [int(x) for x in encoder_depths_str.split("-")]
    n_filters = at_cfg.get("encoder_n_filters", 32)
    n_stages = len(encoder_depths)
    total_downsample = 1
    for r in encoder_ratios:
        total_downsample *= r
    tts_n_layers = cfg.get("tts_backbone_num_hidden_layers", 0)

    print(f"VibeVoice: d_lm={d_lm}, n_layers={n_lm_layers}, heads={n_heads}/{n_kv_heads}, vocab={vocab_size}")

    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])

    # Detect actual base LM layers
    detected_layers = set()
    for shard in shard_files:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for name in f.keys():
                if "language_model.layers." in name and "tts_" not in name:
                    try:
                        detected_layers.add(int(name.split("language_model.layers.")[1].split(".")[0]))
                    except (ValueError, IndexError):
                        pass
    if detected_layers:
        actual = max(detected_layers) + 1
        if actual != n_lm_layers:
            print(f"  actual base LM layers: {actual} (config: {n_lm_layers})")
            n_lm_layers = actual

    # ── Pass 1: scan all tensors, collect metadata ──
    print("Pass 1: scanning tensors...")
    tensor_info = []  # (gguf_name, shard, orig_name, shape, is_f32)
    for shard in shard_files:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for name in sorted(f.keys()):
                if not args.include_decoder:
                    if "acoustic_tokenizer.decoder" in name or "semantic_tokenizer.decoder" in name:
                        continue
                gguf_name = shorten(name)
                if len(gguf_name) >= 64:
                    gguf_name = gguf_name.replace("layers.", "l.")
                    if len(gguf_name) >= 64:
                        continue
                # Get shape without loading data
                meta = f.get_slice(name)
                shape = list(meta.get_shape())
                ndim = len(shape)
                is_f32 = ("norm" in name or "gamma" in name or name.endswith(".bias") or
                          ndim <= 1 or "scaling_factor" in name or "bias_factor" in name)
                tensor_info.append((gguf_name, shard, name, shape, is_f32))

    print(f"  found {len(tensor_info)} tensors")

    # ── Create GGUF writer and add metadata ──
    arch = "vibevoice-tts" if tts_n_layers > 0 else "vibevoice-asr"
    writer = gguf.GGUFWriter(args.output, arch)
    writer.add_name(os.path.basename(args.input.rstrip("/")))
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
    writer.add_uint32("vibevoice.tts_n_layers", tts_n_layers)
    diff_cfg = cfg.get("diffusion_head_config", {})
    if diff_cfg:
        writer.add_string("vibevoice.prediction_type", diff_cfg.get("prediction_type", "v_prediction"))
        writer.add_string("vibevoice.beta_schedule", diff_cfg.get("ddpm_beta_schedule", "cosine"))
    writer.add_uint32("vibevoice.has_tokenizer", 0)  # skip tokenizer to save memory

    # ── Pass 2: load tensors one at a time, store as memory-mapped files ──
    # This keeps peak RAM under ~2 GB even for 9B+ parameter models.
    print("Pass 2: converting tensors (memory-mapped)...")
    import gc
    import tempfile

    # Use output directory for temp files (ensure enough disk space)
    tmpdir = tempfile.mkdtemp(prefix="gguf_convert_", dir=os.path.dirname(os.path.abspath(args.output)))
    mmap_files = []
    current_shard = None
    current_f = None
    count = 0

    for gguf_name, shard, orig_name, shape, is_f32 in tensor_info:
        if shard != current_shard:
            if current_f is not None:
                del current_f
                gc.collect()
            current_f = safe_open(os.path.join(model_dir, shard), framework="pt")
            current_shard = shard

        raw = current_f.get_tensor(orig_name)
        if is_f32:
            data = raw.to(torch.float32).numpy()
            dtype = np.float32
        else:
            data = raw.to(torch.float16).numpy()
            dtype = np.float16
        del raw

        # Write to memmap file on disk (frees RAM)
        mmap_path = os.path.join(tmpdir, f"t{count:05d}.bin")
        mmap_files.append(mmap_path)
        flat = data.ravel()
        flat.tofile(mmap_path)
        orig_shape = data.shape
        del data, flat
        gc.collect()

        # Read back as memmap (data on disk, not in RAM)
        total_elements = 1
        for s in orig_shape:
            total_elements *= s
        if total_elements == 0:
            total_elements = 1
        mm_ref = np.memmap(mmap_path, dtype=dtype, mode="r", shape=(total_elements,))
        if len(orig_shape) > 1:
            mm_ref = mm_ref.reshape(orig_shape)

        writer.add_tensor(gguf_name, mm_ref)
        count += 1

        if count <= 5 or count % 200 == 0:
            print(f"  [{count}/{len(tensor_info)}] {gguf_name} {'F32' if is_f32 else 'F16'}")

    if current_f is not None:
        del current_f
    gc.collect()

    print(f"  total: {count} tensors")
    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e9:.2f} GB, {count} tensors)")

    # Clean up temp memmap files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
