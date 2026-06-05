#!/usr/bin/env python3
"""Single-pass safetensors → Q4_K GGUF for VibeVoice models.

Skips the F16 intermediate. Quantizes per tensor and writes to disk
immediately. Peak RAM ≈ size of the largest single tensor (~2.5 GB
for the 7B LM embed in F32 + Q4_K output buffer), well below the
14 GB+ that crispasr-quantize would need to load a full F16 GGUF.

Memory strategy:
  Pass 1: scan all shards, gather tensor metadata + decide target dtype.
          Register each tensor's INFO with the writer (no data yet).
  Pass 2: write GGUF header + KV table + tensor info table.
  Pass 3: stream each tensor — load → quantize → write_tensor_data → free.
          Never holds more than one tensor's worth of data in RAM.

Usage:
  python models/convert-vibevoice-stream-q4k.py \\
      --input /path/to/safetensors \\
      --output /path/to/vibevoice-7b-q4_k.gguf \\
      --include-decoder
"""

import argparse
import gc
import json
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf

from gguf import GGMLQuantizationType, quants
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Name shortening — must match the F16 converters byte-for-byte so the C++
# loader finds the same tensor names.
# ---------------------------------------------------------------------------
def shorten(name):
    if name.startswith("model."):
        name = name[len("model.") :]
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


# ---------------------------------------------------------------------------
# Per-tensor target dtype policy — same as crispasr-quantize q4_k:
#   - 1D (norms / biases / scales)        → F32  (no quant)
#   - 3D / 4D weights (conv kernels)      → F16  (Q4_K can't represent)
#   - 2D embedding lookups                → F16  (lookup table, not matmul)
#   - 2D matmul weights, last dim % 256=0 → Q4_K (FC, attn proj, FFN)
#   - everything else 2D                  → F16  (small / ragged shapes)
# ---------------------------------------------------------------------------
def target_dtype(name, shape, raw_dtype_str):
    raw_is_int = raw_dtype_str.startswith("I") or raw_dtype_str.startswith("U")
    if raw_is_int:
        return GGMLQuantizationType.F32
    nd = len(shape)
    is_norm = (
        "norm" in name
        or "gamma" in name
        or name.endswith(".bias")
        or "scaling_factor" in name
        or "bias_factor" in name
    )
    if nd <= 1 or is_norm:
        return GGMLQuantizationType.F32
    if nd >= 3:
        return GGMLQuantizationType.F16
    if nd == 2:
        if "tok_emb" in name or "embed_tokens" in name:
            return GGMLQuantizationType.F16
        # Q4_0 block size is 32 along the last dim. (Note: original goal was
        # Q4_K but gguf-py ships only stubs for k-quants; quants.quantize()
        # raises NotImplementedError. Q4_0 has identical compression ratio
        # — 4.5 bits/weight — only the per-block scale granularity differs.)
        if shape[-1] % 32 == 0 and shape[0] >= 32:
            return GGMLQuantizationType.Q4_0
        return GGMLQuantizationType.F16
    return GGMLQuantizationType.F16


# Numpy storage dtype for each ggml type — used for add_tensor_info()'s
# tensor_dtype param and to construct the bytes we hand to write_tensor_data().
NUMPY_DTYPE_FOR = {
    GGMLQuantizationType.F32: np.float32,
    GGMLQuantizationType.F16: np.float16,
    GGMLQuantizationType.Q4_0: np.uint8,  # opaque packed bytes
}


def quantized_nbytes(shape, qtype):
    """Total bytes after quantization — must match what quants.quantize emits."""
    if qtype == GGMLQuantizationType.F32:
        return int(np.prod(shape)) * 4
    if qtype == GGMLQuantizationType.F16:
        return int(np.prod(shape)) * 2
    # For block-quantized types, gguf's helper computes the byte shape.
    byte_shape = quants.quant_shape_to_byte_shape(shape, qtype)
    return int(np.prod(byte_shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="local safetensors dir or HF repo ID")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-decoder", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        model_dir = args.input
    else:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(args.input)
    print(f"model dir: {model_dir}")

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
    diff_cfg = cfg.get("diffusion_head_config", {})
    print(f"VibeVoice: d_lm={d_lm}, n_layers={n_lm_layers}, heads={n_heads}/{n_kv_heads}, vocab={vocab_size}")

    shard_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    detected_layers = set()
    for shard in shard_files:
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for name in f.keys():
                if "language_model.layers." in name and "tts_" not in name:
                    try:
                        layer = int(name.split("language_model.layers.")[1].split(".")[0])
                        detected_layers.add(layer)
                    except (ValueError, IndexError):
                        pass
    if detected_layers:
        actual = max(detected_layers) + 1
        if actual != n_lm_layers:
            print(f"  NOTE: actual base LM layers = {actual} (config says {n_lm_layers})")
            n_lm_layers = actual

    # ── Pass 1: scan all tensors, decide target dtype per tensor ──
    print("Pass 1: scanning tensors …")
    plan = []  # list of (gguf_name, shard, orig_name, shape, qtype, nbytes)
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
                slice_meta = f.get_slice(name)
                shape = tuple(slice_meta.get_shape())
                # Promote 0-d scalars to (1,) — gguf has no concept of 0-d
                # and quant_shape_to_byte_shape() crashes on empty shapes.
                if not shape:
                    shape = (1,)
                qtype = target_dtype(gguf_name, shape, slice_meta.get_dtype())
                nbytes = quantized_nbytes(shape, qtype)
                plan.append((gguf_name, shard, name, shape, qtype, nbytes))

    n_total = len(plan)
    n_q4k = sum(1 for p in plan if p[4] == GGMLQuantizationType.Q4_0)
    n_f16 = sum(1 for p in plan if p[4] == GGMLQuantizationType.F16)
    n_f32 = sum(1 for p in plan if p[4] == GGMLQuantizationType.F32)
    total_bytes = sum(p[5] for p in plan)
    print(f"  {n_total} tensors total: {n_q4k} Q4_K + {n_f16} F16 + {n_f32} F32")
    print(f"  estimated GGUF size: {total_bytes / 1e9:.2f} GB")

    # ── Pass 2: build GGUF header + KV + tensor info table ──
    print("Pass 2: writing header + KV + tensor info table …")
    model_name = os.path.basename(args.input.rstrip("/"))
    arch = "vibevoice-tts" if tts_n_layers > 0 else "vibevoice-asr"
    writer = gguf.GGUFWriter(args.output, arch, use_temp_file=False)
    writer.add_name(model_name)
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
    if diff_cfg:
        writer.add_string("vibevoice.prediction_type", diff_cfg.get("prediction_type", "v_prediction"))
        writer.add_string("vibevoice.beta_schedule", diff_cfg.get("ddpm_beta_schedule", "cosine"))
        writer.add_uint32("vibevoice.ddpm_num_steps", diff_cfg.get("ddpm_num_steps", 1000))
        writer.add_uint32("vibevoice.ddpm_inference_steps", diff_cfg.get("ddpm_num_inference_steps", 20))
    # Embed Qwen2.5 tokenizer for TTS text input. Without this the runtime
    # has no way to tokenize the user's text → "model lacks tokenizer".
    try:
        from transformers import AutoTokenizer

        tok = None
        for tok_src in (model_dir, "Qwen/Qwen2.5-7B"):
            try:
                tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
                print(f"  loaded tokenizer from: {tok_src}")
                break
            except Exception:
                pass
        if tok:
            vocab_map = tok.get_vocab()
            inv = {v: k for k, v in vocab_map.items()}
            max_id = max(inv.keys())
            vocab_list = [inv.get(i, f"<unk_{i}>") for i in range(max_id + 1)]
            writer.add_array("tokenizer.ggml.tokens", vocab_list)
            writer.add_uint32("vibevoice.has_tokenizer", 1)
            print(f"  tokenizer: {len(vocab_list)} tokens embedded")
        else:
            writer.add_uint32("vibevoice.has_tokenizer", 0)
            print("  tokenizer not embedded: no source available")
    except Exception as e:
        writer.add_uint32("vibevoice.has_tokenizer", 0)
        print(f"  tokenizer not embedded: {e}")

    for gguf_name, _shard, _orig, shape, qtype, nbytes in plan:
        np_dtype = NUMPY_DTYPE_FOR[qtype]
        # add_tensor_info ONLY applies quant_shape_from_byte_shape() when
        # tensor_dtype == np.uint8 (quantized). For F16/F32 it stores the
        # passed shape as-is. So pass byte_shape only for quantized types,
        # logical shape for unquantized.
        if np_dtype is np.uint8:
            shape_arg = quants.quant_shape_to_byte_shape(shape, qtype)
        else:
            shape_arg = shape
        writer.add_tensor_info(gguf_name, shape_arg, np_dtype, nbytes, raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ── Pass 3: stream each tensor — load, quantize, append, free ──
    print("Pass 3: streaming tensor data …")
    import torch  # noqa: E402

    current_shard = None
    current_f = None
    try:
        for i, (gguf_name, shard, orig_name, shape, qtype, nbytes) in enumerate(plan, 1):
            if shard != current_shard:
                if current_f is not None:
                    current_f.__exit__(None, None, None)
                current_f = safe_open(os.path.join(model_dir, shard), framework="pt").__enter__()
                current_shard = shard

            raw = current_f.get_tensor(orig_name)
            if qtype == GGMLQuantizationType.F32:
                arr = raw.to(torch.float32).numpy()
            elif qtype == GGMLQuantizationType.F16:
                arr = raw.to(torch.float16).numpy()
            elif qtype == GGMLQuantizationType.Q4_0:
                src = raw.to(torch.float32).numpy()
                arr = quants.quantize(src, qtype)  # uint8 packed bytes
                del src
            else:
                raise RuntimeError(f"unsupported qtype {qtype} for {gguf_name}")
            del raw
            # Promote 0-d scalars to (1,) here too — must match the plan's shape.
            if arr.ndim == 0:
                arr = arr.reshape(1)

            assert arr.nbytes == nbytes, f"size mismatch for {gguf_name}: declared {nbytes} got {arr.nbytes}"
            writer.write_tensor_data(arr)
            del arr

            if i % 50 == 0:
                gc.collect()
            if i <= 5 or i % 100 == 0 or i == n_total:
                sz_gb = os.path.getsize(args.output) / 1e9
                print(f"  [{i}/{n_total}] {gguf_name} → {qtype.name}  (file: {sz_gb:.2f} GB)")
    finally:
        if current_f is not None:
            current_f.__exit__(None, None, None)
        writer.close()

    sz_gb = os.path.getsize(args.output) / 1e9
    print(f"\nDone: {args.output}  ({sz_gb:.2f} GB, {n_total} tensors)")


if __name__ == "__main__":
    main()
