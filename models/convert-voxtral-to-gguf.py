#!/usr/bin/env python3
"""
Convert mistralai/Voxtral-Mini-3B-2507 (HuggingFace safetensors) → GGUF F16.

Architecture (verified against config.json + safetensors index):

  Audio encoder (audio_tower.*) — Whisper-large-v3 encoder verbatim:
    conv1                                 (1280, 128, 3)   stride 1
    conv2                                 (1280, 1280, 3)  stride 2
    embed_positions                       (1500, 1280)     learned absolute pos
    32 × encoder block (Whisper-style pre-LN, biased, GELU FFN):
      self_attn_layer_norm.{weight,bias}  (1280,)
      self_attn.q_proj.{weight,bias}      (1280, 1280)
      self_attn.k_proj.weight             (1280, 1280)   (NO bias — Whisper quirk)
      self_attn.v_proj.{weight,bias}      (1280, 1280)
      self_attn.out_proj.{weight,bias}    (1280, 1280)
      final_layer_norm.{weight,bias}      (1280,)
      fc1.{weight,bias}                   (5120, 1280)
      fc2.{weight,bias}                   (1280, 5120)
    layer_norm.{weight,bias}              (1280,)        post-encoder norm

  Multi-modal projector (multi_modal_projector.*) — stack-4-frames + 2× Linear:
    linear_1.weight                       (3072, 5120)   = Linear(5120 → 3072)
    linear_2.weight                       (3072, 3072)   = Linear(3072 → 3072)

  Llama-style LLM (language_model.*, 30 layers, GQA 32/8):
    embed_tokens.weight                   (131072, 3072)
    per layer:
      input_layernorm.weight              (3072,)
      self_attn.q_proj.weight             (4096, 3072)   32 heads × 128
      self_attn.k_proj.weight             (1024, 3072)   8 KV heads × 128
      self_attn.v_proj.weight             (1024, 3072)
      self_attn.o_proj.weight             (3072, 4096)
      post_attention_layernorm.weight     (3072,)
      mlp.gate_proj.weight                (8192, 3072)
      mlp.up_proj.weight                  (8192, 3072)
      mlp.down_proj.weight                (3072, 8192)
    norm.weight                           (3072,)
    lm_head.weight                        (131072, 3072)

GGUF tensor naming (mirrors what the C++ loader expects, see voxtral.cpp):

  audio.conv.{1,2}.{weight,bias}                F16/F32
  audio.embed_positions                         F32
  audio.blk.{i}.attn_norm.{weight,bias}         F32
  audio.blk.{i}.attn_{q,k,v,out}.{weight,bias?} F16/F32
  audio.blk.{i}.ffn_norm.{weight,bias}          F32
  audio.blk.{i}.ffn_{up,down}.{weight,bias}     F16/F32
  audio.ln_post.{weight,bias}                   F32

  proj1.weight                                  F16
  proj2.weight                                  F16

  token_embd.weight                             F16
  blk.{i}.attn_norm.weight                      F32
  blk.{i}.attn_{q,k,v,output}.weight            F16
  blk.{i}.ffn_norm.weight                       F32
  blk.{i}.ffn_{gate,up,down}.weight             F16
  output_norm.weight                            F32
  output.weight                                 F16

GGUF metadata keys (under `voxtral.*`):
  voxtral.sample_rate, n_mels, n_fft, hop_length, win_length
  voxtral.audio.{n_layers, d_model, n_heads, head_dim, ff_dim, max_pos}
  voxtral.proj.in_dim       = 5120  (= audio.d_model × 4 frames stacked)
  voxtral.proj.out_dim      = 3072
  voxtral.proj.frame_stack  = 4
  voxtral.llm.{n_layers, d_model, n_heads, n_kv_heads, head_dim, ff_dim,
               rope_theta, rms_norm_eps, vocab_size, max_pos}
  voxtral.audio_token_id    = 24

Tokenizer is stored separately:
  tokenizer.tekken.vocab    raw bytes for the 150k vocab entries (concatenated
                            with length prefixes)
  tokenizer.tekken.specials list of special-token strings
  tokenizer.tekken.pre_pattern  the tiktoken-style pre-split regex
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    from safetensors import safe_open
except ImportError:
    sys.exit("pip install safetensors")


# ---------------------------------------------------------------------------
# Tensor name remapping
# ---------------------------------------------------------------------------

DIRECT = {
    # Audio encoder front-end
    "audio_tower.conv1.weight":           "audio.conv.1.weight",
    "audio_tower.conv1.bias":             "audio.conv.1.bias",
    "audio_tower.conv2.weight":           "audio.conv.2.weight",
    "audio_tower.conv2.bias":             "audio.conv.2.bias",
    "audio_tower.embed_positions.weight": "audio.embed_positions",
    "audio_tower.layer_norm.weight":      "audio.ln_post.weight",
    "audio_tower.layer_norm.bias":        "audio.ln_post.bias",
    # Multi-modal projector
    "multi_modal_projector.linear_1.weight": "proj1.weight",
    "multi_modal_projector.linear_2.weight": "proj2.weight",
    # LLM top level
    "language_model.model.embed_tokens.weight": "token_embd.weight",
    "language_model.model.norm.weight":         "output_norm.weight",
    "language_model.lm_head.weight":            "output.weight",
}

# Per-layer regex patterns
AUDIO_LAYER_PATTERNS = [
    (r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight", "audio.blk.{}.attn_norm.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.bias",   "audio.blk.{}.attn_norm.bias"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight",    "audio.blk.{}.attn_q.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias",      "audio.blk.{}.attn_q.bias"),
    # Note: K has weight only (Whisper quirk)
    (r"audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight",    "audio.blk.{}.attn_k.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight",    "audio.blk.{}.attn_v.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias",      "audio.blk.{}.attn_v.bias"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.weight",  "audio.blk.{}.attn_out.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.bias",    "audio.blk.{}.attn_out.bias"),
    (r"audio_tower\.layers\.(\d+)\.final_layer_norm\.weight",     "audio.blk.{}.ffn_norm.weight"),
    (r"audio_tower\.layers\.(\d+)\.final_layer_norm\.bias",       "audio.blk.{}.ffn_norm.bias"),
    (r"audio_tower\.layers\.(\d+)\.fc1\.weight",                  "audio.blk.{}.ffn_up.weight"),
    (r"audio_tower\.layers\.(\d+)\.fc1\.bias",                    "audio.blk.{}.ffn_up.bias"),
    (r"audio_tower\.layers\.(\d+)\.fc2\.weight",                  "audio.blk.{}.ffn_down.weight"),
    (r"audio_tower\.layers\.(\d+)\.fc2\.bias",                    "audio.blk.{}.ffn_down.bias"),
]

LLM_LAYER_PATTERNS = [
    (r"language_model\.model\.layers\.(\d+)\.input_layernorm\.weight",          "blk.{}.attn_norm.weight"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",        "blk.{}.attn_q.weight"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",        "blk.{}.attn_k.weight"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",        "blk.{}.attn_v.weight"),
    (r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",        "blk.{}.attn_output.weight"),
    (r"language_model\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "blk.{}.ffn_norm.weight"),
    (r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",           "blk.{}.ffn_gate.weight"),
    (r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",             "blk.{}.ffn_up.weight"),
    (r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",           "blk.{}.ffn_down.weight"),
]


def remap_name(hf_name: str) -> str | None:
    if hf_name in DIRECT:
        return DIRECT[hf_name]
    for pat, tmpl in AUDIO_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    for pat, tmpl in LLM_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Norms, biases, 1-D tensors, embed_positions stay F32 for accuracy."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name or "ln_post" in gguf_name:
        return True
    if "embed_positions" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Tekken tokenizer serialization
#
# We embed the tokenizer in the GGUF as three blobs so the C++ runtime
# doesn't need a separate file:
#
#   tokenizer.tekken.vocab     length-prefixed concatenation of token_bytes
#                              entries (one per rank, in rank order). Each
#                              entry is u16 length + raw bytes.
#   tokenizer.tekken.specials  GGUF string array, one entry per special token
#                              (in rank order; rank IS the token id)
#   tokenizer.tekken.pattern   the pre-tokenizer regex (uses Unicode property
#                              classes — the C++ side has to handle this with
#                              either a Unicode regex lib or a hand-rolled
#                              approximation)
# ---------------------------------------------------------------------------

def serialize_tekken_vocab(tekken: dict) -> bytes:
    """Encode the 150k vocab entries as length-prefixed (u16 len + bytes)."""
    vocab = tekken["vocab"]
    out = bytearray()
    for entry in vocab:
        b = base64.b64decode(entry["token_bytes"])
        if len(b) > 65535:
            raise ValueError(f"vocab entry too long: {len(b)} bytes")
        out += struct.pack("<H", len(b))
        out += b
    return bytes(out)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")
    with open(input_dir / "config.json", "r") as f:
        cfg = json.load(f)
    audio = cfg["audio_config"]
    text  = cfg["text_config"]

    safetensor_files = sorted(input_dir.glob("model-*.safetensors"))
    if not safetensor_files:
        # try a single-file layout too
        safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        sys.exit(f"no safetensors files found in {input_dir}")
    print(f"  shards: {[p.name for p in safetensor_files]}")

    # Tekken tokenizer
    tekken_path = input_dir / "tekken.json"
    if not tekken_path.exists():
        sys.exit(f"missing tekken.json in {input_dir}")
    with open(tekken_path, "r", encoding="utf-8") as f:
        tekken = json.load(f)
    tekken_cfg = tekken.get("config", {})
    n_specials = len(tekken.get("special_tokens", []))
    n_vocab    = len(tekken.get("vocab", []))
    print(f"  tekken: {n_specials} specials + {n_vocab} BPE = {n_specials + n_vocab} total")

    # ----- Write GGUF -----
    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="voxtral")

    # Audio params
    writer.add_uint32("voxtral.sample_rate",      16000)
    writer.add_uint32("voxtral.n_mels",           audio.get("num_mel_bins", 128))
    writer.add_uint32("voxtral.n_fft",            400)
    writer.add_uint32("voxtral.win_length",       400)
    writer.add_uint32("voxtral.hop_length",       160)
    writer.add_uint32("voxtral.audio.n_layers",   audio.get("num_hidden_layers", 32))
    writer.add_uint32("voxtral.audio.d_model",    audio.get("hidden_size", 1280))
    writer.add_uint32("voxtral.audio.n_heads",    audio.get("num_attention_heads", 20))
    writer.add_uint32("voxtral.audio.head_dim",   audio.get("head_dim", 64))
    writer.add_uint32("voxtral.audio.ff_dim",     audio.get("intermediate_size", 5120))
    writer.add_uint32("voxtral.audio.max_pos",    audio.get("max_source_positions", 1500))

    # Projector params
    writer.add_uint32("voxtral.proj.in_dim",      5120)  # audio.d_model × 4
    writer.add_uint32("voxtral.proj.out_dim",     text.get("hidden_size", 3072))
    writer.add_uint32("voxtral.proj.frame_stack", 4)

    # LLM params
    writer.add_uint32("voxtral.llm.n_layers",     text.get("num_hidden_layers", 30))
    writer.add_uint32("voxtral.llm.d_model",      text.get("hidden_size", 3072))
    writer.add_uint32("voxtral.llm.n_heads",      text.get("num_attention_heads", 32))
    writer.add_uint32("voxtral.llm.n_kv_heads",   text.get("num_key_value_heads", 8))
    writer.add_uint32("voxtral.llm.head_dim",     text.get("head_dim", 128))
    writer.add_uint32("voxtral.llm.ff_dim",       text.get("intermediate_size", 8192))
    writer.add_float32("voxtral.llm.rope_theta",  float(text.get("rope_theta", 1e8)))
    writer.add_float32("voxtral.llm.rms_norm_eps", float(text.get("rms_norm_eps", 1e-5)))
    writer.add_uint32("voxtral.llm.vocab_size",   text.get("vocab_size", 131072))
    writer.add_uint32("voxtral.llm.max_pos",      text.get("max_position_embeddings", 131072))
    writer.add_uint32("voxtral.audio_token_id",   cfg.get("audio_token_id", 24))

    # Tekken tokenizer (custom blobs, NOT the standard tokenizer.ggml.* keys
    # because Tekken isn't GPT-2 BPE — it's a tiktoken-style rank BPE)
    writer.add_string("tokenizer.tekken.pattern", tekken_cfg.get("pattern", ""))
    specials = [s["token_str"] for s in tekken.get("special_tokens", [])]
    writer.add_array("tokenizer.tekken.specials", specials)
    # Vocab as a single bytes blob (length-prefixed entries)
    vocab_blob = serialize_tekken_vocab(tekken)
    print(f"  vocab blob: {len(vocab_blob)/1024:.1f} KB")
    # gguf-py doesn't have a direct "raw bytes" type, so use uint8 array
    writer.add_array("tokenizer.tekken.vocab",
                     np.frombuffer(vocab_blob, dtype=np.uint8).tolist())
    writer.add_uint32("tokenizer.tekken.n_specials", len(specials))
    writer.add_uint32("tokenizer.tekken.n_vocab",    n_vocab)

    # ----- Tensors -----
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_skipped = 0
    skipped_names: list[str] = []

    for sf_path in safetensor_files:
        print(f"  reading {sf_path.name}")
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            for hf_name in sorted(f.keys()):
                gguf_name = remap_name(hf_name)
                if gguf_name is None:
                    n_skipped += 1
                    skipped_names.append(hf_name)
                    continue
                t = f.get_tensor(hf_name)
                if "bfloat" in str(t.dtype):
                    t = t.float()
                arr = t.numpy()
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                if is_f32_tensor(gguf_name, arr.shape):
                    arr = arr.astype(np.float32)
                    n_f32 += 1
                else:
                    arr = arr.astype(np.float16)
                    n_f16 += 1
                writer.add_tensor(gguf_name, arr)
                n_written += 1
                if n_written <= 25 or n_written % 100 == 0:
                    print(f"    {gguf_name:50s} {str(arr.shape):26s} {arr.dtype}")

    print(f"\n  total: {n_written} tensors  (F16: {n_f16}, F32: {n_f32})  "
          f"skipped: {n_skipped}")
    if skipped_names:
        print("  skipped tensors (first 10):")
        for n in skipped_names[:10]:
            print(f"    {n}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Voxtral-Mini-3B-2507 HF safetensors → GGUF F16")
    p.add_argument("--input",  required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
