#!/usr/bin/env python3
"""
Convert Qwen/Qwen3-TTS-Tokenizer-12Hz (RVQ codec) → GGUF.

Pairs with `convert-qwen3-tts-to-gguf.py`. The talker LM emits
multi-codebook RVQ codes; this codec turns them back into 24 kHz
waveform.

Architecture (from public config.json, April 2026):

  Encoder (used during voice cloning to extract speech tokens):
    hidden_size      = 512
    num_hidden_layers = 8
    num_attention_heads = 8
    intermediate_size = 2048
    sliding_window    = 250
    num_quantizers    = 32       (encoder side has more codebooks than decoder)
    codebook_dim      = 256
    codebook_size     = 2048
    upsampling_ratios = [8, 6, 5, 4]   (matched to encode_downsample_rate=1920)

  Decoder (the synthesis path — what we need to run on talker output):
    hidden_size       = 512
    num_hidden_layers = 8
    num_attention_heads = 16
    intermediate_size = 1024
    sliding_window    = 72
    num_quantizers    = 16            (talker emits 16 codes per frame)
    codebook_dim      = 512
    codebook_size     = 2048
    latent_dim        = 1024
    decoder_dim       = 1536
    upsample_rates    = [8, 5, 4, 3]  (decode_upsample_rate=1920 = 8×5×4×3 / 8)

  Audio:
    input_sample_rate  = 24000
    output_sample_rate = 24000
    frame_rate         = 12.5 Hz
    semantic_quantizers = 1   (decoder gets one extra "semantic" codebook)
    semantic_codebook_size = 4096

Usage:

    python models/convert-qwen3-tts-tokenizer-to-gguf.py \\
        --input Qwen/Qwen3-TTS-Tokenizer-12Hz \\
        --output qwen3-tts-tokenizer-12hz.gguf
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

try:
    import torch
except ImportError:
    sys.exit("pip install torch")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("pip install huggingface_hub")


def load_model_dir(model_id: str) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping
#
# We don't yet know the exact safetensor key prefixes — Qwen ships a custom
# `Qwen3TTSTokenizerV2Model` class. The strategy below covers the patterns
# we've seen in similar RVQ codecs (Mimi, Encodec, MiMo-Tokenizer) and
# emits a clear `[WARN unmapped]` for anything that slips through so we
# can extend this on first run.
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    if hf_name.endswith("num_batches_tracked"):
        return None

    n = hf_name

    # Encoder/decoder top-level
    n = n.replace("model.encoder.", "encoder.")
    n = n.replace("model.decoder.", "decoder.")

    # Vector-quantizer codebooks
    n = n.replace(".quantizer.layers.", ".rvq.")
    n = n.replace(".codebook.weight", ".codebook")
    n = n.replace(".embed.weight", ".codebook")  # alternative name some codecs use

    # Convolutions in the up/down-sampling stacks
    n = n.replace(".conv.weight", ".conv_w")
    n = n.replace(".conv.bias", ".conv_b")
    n = n.replace(".transposed_conv.weight", ".tconv_w")
    n = n.replace(".transposed_conv.bias", ".tconv_b")
    n = n.replace(".weight_g", ".wn_g")
    n = n.replace(".weight_v", ".wn_v")  # weight-norm parametrisation

    # Transformer blocks inside the encoder/decoder
    n = n.replace(".layers.", ".blk.")
    n = n.replace(".self_attn.q_proj.", ".attn_q.")
    n = n.replace(".self_attn.k_proj.", ".attn_k.")
    n = n.replace(".self_attn.v_proj.", ".attn_v.")
    n = n.replace(".self_attn.o_proj.", ".attn_output.")
    n = n.replace(".input_layernorm.", ".attn_norm.")
    n = n.replace(".post_attention_layernorm.", ".ffn_norm.")
    n = n.replace(".mlp.gate_proj.", ".ffn_gate.")
    n = n.replace(".mlp.up_proj.", ".ffn_up.")
    n = n.replace(".mlp.down_proj.", ".ffn_down.")

    return n


def main():
    ap = argparse.ArgumentParser(description="Convert Qwen3-TTS-Tokenizer-12Hz to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. Qwen/Qwen3-TTS-Tokenizer-12Hz) or local dir")
    ap.add_argument("--output", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    enc = cfg.get("encoder_config", cfg.get("encoder", {}))
    dec = cfg.get("decoder_config", cfg.get("decoder", {}))

    print(f"\nQwen3-TTS-Tokenizer-12Hz")
    print(f"  Encoder:  {enc.get('num_hidden_layers')}L  hidden={enc.get('hidden_size')}  "
          f"heads={enc.get('num_attention_heads')}  ff={enc.get('intermediate_size')}  "
          f"rvq={enc.get('num_quantizers')}×{enc.get('codebook_size')}")
    print(f"  Decoder:  {dec.get('num_hidden_layers')}L  hidden={dec.get('hidden_size')}  "
          f"heads={dec.get('num_attention_heads')}  ff={dec.get('intermediate_size')}  "
          f"rvq={dec.get('num_quantizers')}×{dec.get('codebook_size')}")
    print(f"  SR:       {cfg.get('sampling_rate', 24000)} Hz   "
          f"frame={cfg.get('frame_rate', 12.5)} Hz")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors: {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="qwen3tts_tokenizer", use_temp_file=True)
    w.add_name("qwen3-tts-tokenizer-12hz")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    # Top-level audio params — Qwen3-TTS-Tokenizer-12Hz uses
    # input_sample_rate / output_sample_rate (not the generic
    # `sampling_rate`), and the encoder frame rate lives inside
    # encoder_config as `_frame_rate`. Persist all of them so the
    # runtime can drive the codec at the right rate without re-reading
    # the YAML.
    u32("qwen3tts_codec.input_sample_rate",
        cfg.get("input_sample_rate", cfg.get("sampling_rate", 24000)))
    u32("qwen3tts_codec.output_sample_rate",
        cfg.get("output_sample_rate", cfg.get("sampling_rate", 24000)))
    f32("qwen3tts_codec.frame_rate",
        enc.get("_frame_rate", cfg.get("frame_rate", 12.5)))
    u32("qwen3tts_codec.encode_downsample", cfg.get("encode_downsample_rate", 1920))
    u32("qwen3tts_codec.decode_upsample",   cfg.get("decode_upsample_rate", 1920))
    u32("qwen3tts_codec.encoder_valid_num_quantizers",
        cfg.get("encoder_valid_num_quantizers", 16))

    # encoder
    u32("qwen3tts_codec.enc.n_layers",   enc.get("num_hidden_layers", 8))
    u32("qwen3tts_codec.enc.d_model",    enc.get("hidden_size", 512))
    u32("qwen3tts_codec.enc.n_heads",    enc.get("num_attention_heads", 8))
    u32("qwen3tts_codec.enc.ff_dim",     enc.get("intermediate_size", 2048))
    u32("qwen3tts_codec.enc.n_quantizers", enc.get("num_quantizers", 32))
    u32("qwen3tts_codec.enc.codebook_dim", enc.get("codebook_dim", 256))
    u32("qwen3tts_codec.enc.codebook_size", enc.get("codebook_size", 2048))
    u32("qwen3tts_codec.enc.sliding_window", enc.get("sliding_window", 250))
    if "upsampling_ratios" in enc:
        w.add_array("qwen3tts_codec.enc.upsample_ratios", list(enc["upsampling_ratios"]))

    # decoder
    u32("qwen3tts_codec.dec.n_layers",   dec.get("num_hidden_layers", 8))
    u32("qwen3tts_codec.dec.d_model",    dec.get("hidden_size", 512))
    u32("qwen3tts_codec.dec.n_heads",    dec.get("num_attention_heads", 16))
    u32("qwen3tts_codec.dec.ff_dim",     dec.get("intermediate_size", 1024))
    u32("qwen3tts_codec.dec.n_quantizers", dec.get("num_quantizers", 16))
    u32("qwen3tts_codec.dec.codebook_dim", dec.get("codebook_dim", 512))
    u32("qwen3tts_codec.dec.codebook_size", dec.get("codebook_size", 2048))
    u32("qwen3tts_codec.dec.latent_dim", dec.get("latent_dim", 1024))
    u32("qwen3tts_codec.dec.decoder_dim", dec.get("decoder_dim", 1536))
    u32("qwen3tts_codec.dec.sliding_window", dec.get("sliding_window", 72))
    if "upsample_rates" in dec:
        w.add_array("qwen3tts_codec.dec.upsample_rates", list(dec["upsample_rates"]))

    # Semantic quantisers (Qwen3 codec has a small "semantic" head
    # separate from the main RVQ stack — encoded redundantly into both
    # the encoder_config and decoder_config sections in the HF YAML).
    u32("qwen3tts_codec.semantic_quantizers",
        dec.get("num_semantic_quantizers",
            enc.get("num_semantic_quantizers",
                cfg.get("semantic_quantizers", 1))))
    u32("qwen3tts_codec.semantic_codebook_size",
        dec.get("semantic_codebook_size",
            cfg.get("semantic_codebook_size", 4096)))

    # Decoder-side numerical params we'll need at runtime.
    f32("qwen3tts_codec.dec.rope_theta", dec.get("rope_theta", 10000.0))
    f32("qwen3tts_codec.dec.rms_norm_eps", dec.get("rms_norm_eps", 1e-5))
    u32("qwen3tts_codec.dec.head_dim", dec.get("head_dim", 64))
    u32("qwen3tts_codec.dec.max_pos", dec.get("max_position_embeddings", 8000))
    u32("qwen3tts_codec.dec.vq_hidden_dim",
        dec.get("vector_quantization_hidden_dimension", 512))
    f32("qwen3tts_codec.dec.layer_scale_initial",
        dec.get("layer_scale_initial_scale", 0.01))
    if "upsampling_ratios" in dec:
        w.add_array("qwen3tts_codec.dec.upsample_ratios_extra",
                    list(dec["upsampling_ratios"]))

    # Encoder-side numerical params (used during voice clone from a
    # reference audio signal).
    f32("qwen3tts_codec.enc.rope_theta", enc.get("rope_theta", 10000.0))
    f32("qwen3tts_codec.enc.norm_eps",   enc.get("norm_eps", 1e-5))
    u32("qwen3tts_codec.enc.head_dim",   enc.get("head_dim", 64))
    u32("qwen3tts_codec.enc.max_pos",    enc.get("max_position_embeddings", 8000))
    u32("qwen3tts_codec.enc.kernel_size", enc.get("kernel_size", 7))
    u32("qwen3tts_codec.enc.last_kernel_size", enc.get("last_kernel_size", 3))
    u32("qwen3tts_codec.enc.compress",   enc.get("compress", 2))
    u32("qwen3tts_codec.enc.dilation_growth_rate",
        enc.get("dilation_growth_rate", 2))
    u32("qwen3tts_codec.enc.num_filters", enc.get("num_filters", 64))
    u32("qwen3tts_codec.enc.num_residual_layers",
        enc.get("num_residual_layers", 1))
    u32("qwen3tts_codec.enc.upsample_groups",
        enc.get("upsample_groups", 512))
    u32("qwen3tts_codec.enc.residual_kernel_size",
        enc.get("residual_kernel_size", 3))
    u32("qwen3tts_codec.enc.audio_channels",
        enc.get("audio_channels", 1))
    f32("qwen3tts_codec.enc.layer_scale_initial",
        enc.get("layer_scale_initial_scale", 0.01))
    f32("qwen3tts_codec.enc.trim_right_ratio",
        enc.get("trim_right_ratio", 1.0))

    n_mapped = 0
    n_skipped = 0
    skipped_examples = []
    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue
        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped += 1
        if n_mapped <= 30 or n_mapped % 100 == 0:
            print(f"  [{n_mapped}] {gn:55s} {t.shape}  {t.dtype}")

    print(f"\nMapped: {n_mapped}, skipped: {n_skipped}")
    if skipped_examples:
        print("Examples:", "\n".join(skipped_examples), file=sys.stderr)

    print(f"Writing {out_path}…")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {n_mapped} tensors)")


if __name__ == "__main__":
    main()
