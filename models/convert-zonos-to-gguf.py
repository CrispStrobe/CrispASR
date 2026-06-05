#!/usr/bin/env python3
"""
Convert Zyphra/Zonos-v0.1-transformer HuggingFace safetensors -> GGUF F16
for the CrispASR `zonos_tts` backend.

Zonos is a ~500M-param text-to-speech model (Apache 2.0) with:
  - Prefix conditioning: phoneme embeddings + speaker + emotion + pitch/rate/etc
  - AR backbone: 26-layer GPT-style transformer (d=2048, GQA 16/4) with
    fused QKV in_proj, SwiGLU FFN, LayerNorm (not RMSNorm)
  - 9 codebook embeddings (1026 entries each) + 9 prediction heads (1025 outputs)
  - Output: DAC 44.1 kHz codec codes (decoded by separate dac_44khz GGUF)

Architecture (from Zyphra/Zonos-v0.1-transformer/config.json):

    backbone.d_model              = 2048
    backbone.attn_mlp_d_intermediate = 8192
    backbone.n_layer              = 26
    backbone.attn_cfg.num_heads   = 16
    backbone.attn_cfg.num_heads_kv = 4
    backbone.norm_epsilon         = 1e-5
    backbone.rms_norm             = false (uses LayerNorm with bias)
    n_codebooks                   = 9
    codebook_size                 = 1024
    eos_token_id                  = 1024
    masked_token_id               = 1025

Weight layout:
  - backbone.layers.N.mixer.in_proj.weight: [3072, 2048] - fused Q/K/V
    Split as Q=[n_heads*head_dim], K=[n_kv_heads*head_dim], V=[n_kv_heads*head_dim]
    where n_heads=16, n_kv_heads=4, head_dim=128
    Q: rows [0:2048], K: rows [2048:2560], V: rows [2560:3072]
  - backbone.layers.N.mixer.out_proj.weight: [2048, 2048]
  - backbone.layers.N.mlp.fc1.weight: [16384, 2048] - fused gate+up
    gate: rows [0:8192], up: rows [8192:16384]
  - backbone.layers.N.mlp.fc2.weight: [2048, 8192] - down projection
  - backbone.layers.N.norm.{weight,bias}: [2048] - pre-attn LayerNorm
  - backbone.layers.N.norm2.{weight,bias}: [2048] - pre-FFN LayerNorm
  - backbone.norm_f.{weight,bias}: [2048] - final LayerNorm
  - embeddings.K.weight: [1026, 2048] - per-codebook token embeddings
  - heads.K.weight: [1025, 2048] - per-codebook prediction heads
  - prefix_conditioner.* - conditioning modules

Usage:
    python models/convert-zonos-to-gguf.py \\
        --input Zyphra/Zonos-v0.1-transformer \\
        --output /mnt/storage/zonos-tts/zonos-v0.1-transformer-f16.gguf

    # With quantization:
    python models/convert-zonos-to-gguf.py \\
        --input Zyphra/Zonos-v0.1-transformer \\
        --output /mnt/storage/zonos-tts/zonos-v0.1-transformer-q8_0.gguf \\
        --quant q8_0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
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


# ---------------------------------------------------------------------------
# eSpeak phoneme symbol table (from zonos/conditioning.py)
# Must match exactly for the C++ runtime to interpret phoneme embeddings.
# ---------------------------------------------------------------------------

_punctuation = ';:,.!?¡¿—…"«»""() *~-/\\&'
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)
PHONEME_SYMBOLS = [*_punctuation, *_letters, *_letters_ipa]

# Language codes supported by Zonos (from conditioning.py)
LANGUAGE_CODES = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue',
]


def load_model_dir(model_id: str) -> Path:
    """Load model from local path or download from HF hub."""
    p = Path(model_id)
    if p.is_dir():
        return p
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json",
    ]))


# ---------------------------------------------------------------------------
# Tensor name mapping: HF Zonos -> GGUF
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map HuggingFace weight name to GGUF tensor name."""
    n = hf_name

    # Final norm (must be done before the .norm. replacement below)
    n = n.replace("backbone.norm_f.", "backbone.output_norm.")

    # Backbone transformer layers
    n = n.replace("backbone.layers.", "backbone.blk.")
    n = n.replace(".mixer.in_proj.", ".attn_qkv.")  # fused Q/K/V
    n = n.replace(".mixer.out_proj.", ".attn_output.")
    n = n.replace(".mlp.fc1.", ".ffn_gate_up.")  # fused gate+up (SwiGLU)
    n = n.replace(".mlp.fc2.", ".ffn_down.")

    # Pre-FFN LayerNorm (.norm2. must be replaced BEFORE .norm.)
    n = n.replace(".norm2.", ".ffn_norm.")

    # Pre-attn LayerNorm -- only inside backbone.blk.N, not prefix_conditioner
    if n.startswith("backbone.blk.") and ".norm." in n:
        n = n.replace(".norm.", ".attn_norm.")

    # Codebook embeddings and heads (keep as-is)
    # embeddings.K.weight -> embeddings.K.weight
    # heads.K.weight -> heads.K.weight

    # Prefix conditioner (keep structure)
    # prefix_conditioner.* -> prefix_conditioner.*

    return n


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

QUANT_MAP = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "bf16": GGMLQuantizationType.BF16,
    "q8_0": GGMLQuantizationType.Q8_0,
    "q4_0": GGMLQuantizationType.Q4_0,
    "q4_1": GGMLQuantizationType.Q4_1,
    "q5_0": GGMLQuantizationType.Q5_0,
    "q5_1": GGMLQuantizationType.Q5_1,
    "q4_k": GGMLQuantizationType.Q4_K_M if hasattr(GGMLQuantizationType, 'Q4_K_M') else GGMLQuantizationType.Q4_0,
}


def choose_quant(name: str, shape: list, quant_type: str) -> GGMLQuantizationType:
    """Choose quantization type for a given tensor."""
    if quant_type in ("f32", "f16", "bf16"):
        return QUANT_MAP[quant_type]

    ndim = len(shape)

    # 1D tensors (norms, biases, uncond_vectors): always F32
    if ndim == 1:
        return GGMLQuantizationType.F32

    # Small tensors (embeddings < 256 rows, Fourier weights): keep F16
    if ndim == 2 and min(shape) < 256:
        return GGMLQuantizationType.F16

    # Large 2D tensors: apply requested quantization
    return QUANT_MAP.get(quant_type, GGMLQuantizationType.F16)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Zonos safetensors to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID or local directory path")
    ap.add_argument("--output", required=True,
                    help="Output GGUF file path")
    ap.add_argument("--quant", default="f16",
                    choices=list(QUANT_MAP.keys()),
                    help="Quantization type (default: f16)")
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        sys.exit(f"No config.json found in {model_dir}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    backbone_cfg = config["backbone"]
    prefix_cfg = config["prefix_conditioner"]

    d_model = backbone_cfg["d_model"]
    n_layer = backbone_cfg["n_layer"]
    n_heads = backbone_cfg["attn_cfg"]["num_heads"]
    n_kv_heads = backbone_cfg["attn_cfg"]["num_heads_kv"]
    head_dim = d_model // n_heads
    ff_dim = backbone_cfg["attn_mlp_d_intermediate"]
    norm_eps = backbone_cfg["norm_epsilon"]
    eos_token_id = config.get("eos_token_id", 1024)
    masked_token_id = config.get("masked_token_id", 1025)

    print(f"Zonos config:", file=sys.stderr)
    print(f"  d_model={d_model}, n_layer={n_layer}", file=sys.stderr)
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}",
          file=sys.stderr)
    print(f"  ff_dim={ff_dim}, norm_eps={norm_eps}", file=sys.stderr)
    print(f"  eos_token_id={eos_token_id}, masked_token_id={masked_token_id}",
          file=sys.stderr)

    # Find safetensors file
    st_files = list(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"No .safetensors files found in {model_dir}")
    st_path = st_files[0]
    print(f"Loading weights from {st_path}...", file=sys.stderr)

    # Create GGUF writer
    writer = GGUFWriter(args.output, "zonos-tts")

    # Write architecture metadata
    writer.add_uint32("zonos.d_model", d_model)
    writer.add_uint32("zonos.n_layer", n_layer)
    writer.add_uint32("zonos.n_heads", n_heads)
    writer.add_uint32("zonos.n_kv_heads", n_kv_heads)
    writer.add_uint32("zonos.head_dim", head_dim)
    writer.add_uint32("zonos.ff_dim", ff_dim)
    writer.add_float32("zonos.norm_eps", norm_eps)
    writer.add_uint32("zonos.eos_token_id", eos_token_id)
    writer.add_uint32("zonos.masked_token_id", masked_token_id)

    # DAC codec metadata (the decoder is a separate GGUF; these are for
    # the AR codebook structure)
    n_codebooks = 9
    codebook_size = 1024  # DAC uses 1024 per codebook
    writer.add_uint32("zonos.n_codebooks", n_codebooks)
    writer.add_uint32("zonos.codebook_size", codebook_size)
    writer.add_uint32("zonos.sample_rate", 44100)

    # RoPE metadata
    writer.add_float32("zonos.rope_theta", 10000.0)
    writer.add_uint32("zonos.rope_dim", head_dim)  # full head_dim for RoPE

    # Conditioner metadata
    writer.add_uint32("zonos.n_conditioners",
                      len(prefix_cfg["conditioners"]))
    for i, cond in enumerate(prefix_cfg["conditioners"]):
        ctype = cond["type"]
        cname = cond["name"]
        writer.add_string(f"zonos.conditioner.{i}.type", ctype)
        writer.add_string(f"zonos.conditioner.{i}.name", cname)
        if "cond_dim" in cond:
            writer.add_uint32(f"zonos.conditioner.{i}.cond_dim", cond["cond_dim"])
        if "input_dim" in cond:
            writer.add_uint32(f"zonos.conditioner.{i}.input_dim", cond["input_dim"])
        if "min_val" in cond:
            writer.add_int32(f"zonos.conditioner.{i}.min_val", cond["min_val"])
        if "max_val" in cond:
            writer.add_int32(f"zonos.conditioner.{i}.max_val", cond["max_val"])
        if "std" in cond:
            writer.add_float32(f"zonos.conditioner.{i}.std", cond["std"])
        has_uncond = cond.get("uncond_type", "none") == "learned"
        writer.add_bool(f"zonos.conditioner.{i}.has_uncond", has_uncond)
        has_proj = cond.get("projection", "none") != "none"
        writer.add_string(f"zonos.conditioner.{i}.projection",
                          cond.get("projection", "none"))

    # Phoneme vocabulary (for the C++ runtime to map characters to IDs)
    writer.add_uint32("zonos.phoneme_vocab_size", len(PHONEME_SYMBOLS) + 4)
    phoneme_vocab_str = "\n".join(PHONEME_SYMBOLS)
    writer.add_string("zonos.phoneme_symbols", phoneme_vocab_str)
    # Special token IDs
    writer.add_uint32("zonos.phoneme_pad_id", 0)
    writer.add_uint32("zonos.phoneme_unk_id", 1)
    writer.add_uint32("zonos.phoneme_bos_id", 2)
    writer.add_uint32("zonos.phoneme_eos_id", 3)

    # Language codes
    writer.add_uint32("zonos.n_languages", len(LANGUAGE_CODES))
    writer.add_string("zonos.language_codes", "\n".join(LANGUAGE_CODES))

    # Prefix conditioner projection type
    writer.add_string("zonos.prefix_projection", prefix_cfg.get("projection", "none"))

    # Write tensors
    n_written = 0
    with safe_open(st_path, framework="pt") as f:
        for hf_name in sorted(f.keys()):
            gguf_name = map_tensor_name(hf_name)
            if gguf_name is None:
                print(f"  skip: {hf_name}", file=sys.stderr)
                continue

            tensor = f.get_tensor(hf_name)
            shape = list(tensor.shape)

            # Choose quantization
            qtype = choose_quant(gguf_name, shape, args.quant)

            # Convert to the target numpy dtype.
            # BF16 must go through F32 first (numpy has no bf16).
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()

            if qtype == GGMLQuantizationType.F32:
                data = np.ascontiguousarray(tensor.float().numpy())
            elif qtype == GGMLQuantizationType.F16:
                data = np.ascontiguousarray(tensor.float().numpy().astype(np.float16))
            elif qtype in (GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0,
                           GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q5_0,
                           GGMLQuantizationType.Q5_1):
                data_f32 = np.ascontiguousarray(tensor.float().numpy())
                try:
                    data = gguf.quantize(data_f32, qtype)
                except Exception as e:
                    print(f"  [WARN] {gguf_name}: quantize failed ({e}), keeping F16",
                          file=sys.stderr)
                    data = np.ascontiguousarray(tensor.float().numpy().astype(np.float16))
                    qtype = GGMLQuantizationType.F16
            else:
                data = np.ascontiguousarray(tensor.float().numpy().astype(np.float16))
                qtype = GGMLQuantizationType.F16

            writer.add_tensor(gguf_name, data, raw_dtype=qtype)
            n_written += 1

            if n_written <= 5 or n_written % 50 == 0:
                print(f"  [{n_written}] {hf_name} -> {gguf_name} "
                      f"{shape} {qtype.name}", file=sys.stderr)

    print(f"\nTotal tensors written: {n_written}", file=sys.stderr)
    print(f"Writing GGUF to {args.output}...", file=sys.stderr)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Print file size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. Output: {args.output} ({size_mb:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
