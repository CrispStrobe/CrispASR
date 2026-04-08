#!/usr/bin/env python3
"""
Convert Qwen/Qwen3-ASR-0.6B (HuggingFace safetensors) → GGUF F16.

Architecture (verified against actual safetensors shapes):

  Audio tower (qwen3-asr-specific):
    conv2d1                       (480, 1,   3, 3)   stride-2 freq subsampling
    conv2d2                       (480, 480, 3, 3)
    conv2d3                       (480, 480, 3, 3)
    conv_out                      (896, 7680)         linear: 480ch × 16freq → 896
    18 × encoder block (Whisper-block style, pre-LN, biased):
      self_attn_layer_norm.{weight,bias}   (896,)
      self_attn.{q,k,v,out}_proj.{weight,bias}   (896, 896)
      final_layer_norm.{weight,bias}       (896,)
      fc1.{weight,bias}                    (3584, 896)
      fc2.{weight,bias}                    (896, 3584)
    ln_post.{weight,bias}         (896,)
    proj1.{weight,bias}           (896, 896)         projector head
    proj2.{weight,bias}           (1024, 896)        → matches LLM hidden

  Text decoder (stock Qwen3 0.6B, 28 layers):
    embed_tokens                  (151936, 1024)
    per layer:
      input_layernorm                       (1024,)   RMSNorm, weight-only
      self_attn.q_proj              (2048, 1024)      16 heads × 128
      self_attn.k_proj              (1024, 1024)      8  KV heads × 128 (GQA)
      self_attn.v_proj              (1024, 1024)
      self_attn.o_proj              (1024, 2048)
      self_attn.q_norm              (128,)            Qwen3 per-head Q RMSNorm
      self_attn.k_norm              (128,)
      post_attention_layernorm      (1024,)
      mlp.gate_proj                 (3072, 1024)      SwiGLU
      mlp.up_proj                   (3072, 1024)
      mlp.down_proj                 (1024, 3072)
    model.norm                    (1024,)
    lm_head                       (151936, 1024)

GGUF tensor naming (mirrors what the C++ loader will expect):

  audio.conv.{1,2,3}.{weight,bias}            F16/F32
  audio.conv_out.{weight,bias}                F16/F32
  audio.blk.{i}.attn_norm.{weight,bias}       F32
  audio.blk.{i}.attn_{q,k,v,out}.{weight,bias}  F16/F32
  audio.blk.{i}.ffn_norm.{weight,bias}        F32
  audio.blk.{i}.ffn_up.{weight,bias}          F16/F32  (fc1)
  audio.blk.{i}.ffn_down.{weight,bias}        F16/F32  (fc2)
  audio.ln_post.{weight,bias}                 F32
  audio.proj1.{weight,bias}                   F16/F32
  audio.proj2.{weight,bias}                   F16/F32

  token_embd.weight                           F16
  blk.{i}.attn_norm.weight                    F32
  blk.{i}.attn_{q,k,v,output}.weight          F16
  blk.{i}.attn_{q,k}_norm.weight              F32
  blk.{i}.ffn_norm.weight                     F32
  blk.{i}.ffn_{gate,up,down}.weight           F16
  output_norm.weight                          F32
  output.weight                               F16

GGUF metadata keys (under `qwen3asr.*`):
  qwen3asr.sample_rate            = 16000
  qwen3asr.n_mels                 = 128
  qwen3asr.n_fft                  = 400
  qwen3asr.win_length             = 400
  qwen3asr.hop_length             = 160
  qwen3asr.audio.n_layers         = 18
  qwen3asr.audio.d_model          = 896
  qwen3asr.audio.n_heads          = 14
  qwen3asr.audio.head_dim         = 64
  qwen3asr.audio.ff_dim           = 3584
  qwen3asr.audio.conv_channels    = 480
  qwen3asr.audio.proj_dim         = 1024
  qwen3asr.audio.max_source_pos   = 1500
  qwen3asr.llm.n_layers           = 28
  qwen3asr.llm.d_model            = 1024
  qwen3asr.llm.n_heads            = 16
  qwen3asr.llm.n_kv_heads         = 8
  qwen3asr.llm.head_dim           = 128
  qwen3asr.llm.ff_dim             = 3072
  qwen3asr.llm.rope_theta         = 1000000
  qwen3asr.llm.rms_norm_eps       = 1e-6
  qwen3asr.llm.vocab_size         = 151936
  qwen3asr.llm.max_pos            = 65536
  qwen3asr.audio_start_token_id   = 151669
  qwen3asr.audio_end_token_id     = 151670
  qwen3asr.audio_pad_token_id     = 151676
  qwen3asr.eos_token_id           = 151645
  qwen3asr.pad_token_id           = 151643

  tokenizer.ggml.model            = "gpt2"
  tokenizer.ggml.tokens           = [151936 strings from vocab.json]
  tokenizer.ggml.merges           = [BPE merges from merges.txt]
"""

from __future__ import annotations

import argparse
import json
import re
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

# Direct mappings (everything that isn't per-layer)
DIRECT = {
    # Audio tower — conv front-end
    "thinker.audio_tower.conv2d1.weight":  "audio.conv.1.weight",
    "thinker.audio_tower.conv2d1.bias":    "audio.conv.1.bias",
    "thinker.audio_tower.conv2d2.weight":  "audio.conv.2.weight",
    "thinker.audio_tower.conv2d2.bias":    "audio.conv.2.bias",
    "thinker.audio_tower.conv2d3.weight":  "audio.conv.3.weight",
    "thinker.audio_tower.conv2d3.bias":    "audio.conv.3.bias",
    "thinker.audio_tower.conv_out.weight": "audio.conv_out.weight",
    "thinker.audio_tower.conv_out.bias":   "audio.conv_out.bias",
    # Audio tower — final norm + projector head
    "thinker.audio_tower.ln_post.weight":  "audio.ln_post.weight",
    "thinker.audio_tower.ln_post.bias":    "audio.ln_post.bias",
    "thinker.audio_tower.proj1.weight":    "audio.proj1.weight",
    "thinker.audio_tower.proj1.bias":      "audio.proj1.bias",
    "thinker.audio_tower.proj2.weight":    "audio.proj2.weight",
    "thinker.audio_tower.proj2.bias":      "audio.proj2.bias",
    # Text decoder — top level
    "thinker.model.embed_tokens.weight":   "token_embd.weight",
    "thinker.model.norm.weight":           "output_norm.weight",
    "thinker.lm_head.weight":              "output.weight",
}

# Per-layer regex patterns (audio encoder body — Whisper-block style)
AUDIO_LAYER_PATTERNS = [
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight", "audio.blk.{}.attn_norm.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.bias",   "audio.blk.{}.attn_norm.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight",    "audio.blk.{}.attn_q.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias",      "audio.blk.{}.attn_q.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight",    "audio.blk.{}.attn_k.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.bias",      "audio.blk.{}.attn_k.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight",    "audio.blk.{}.attn_v.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias",      "audio.blk.{}.attn_v.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.weight",  "audio.blk.{}.attn_out.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.bias",    "audio.blk.{}.attn_out.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.final_layer_norm\.weight",     "audio.blk.{}.ffn_norm.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.final_layer_norm\.bias",       "audio.blk.{}.ffn_norm.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.fc1\.weight",                  "audio.blk.{}.ffn_up.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.fc1\.bias",                    "audio.blk.{}.ffn_up.bias"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.fc2\.weight",                  "audio.blk.{}.ffn_down.weight"),
    (r"thinker\.audio_tower\.layers\.(\d+)\.fc2\.bias",                    "audio.blk.{}.ffn_down.bias"),
]

# Per-layer regex patterns (text decoder — Qwen3 0.6B)
TEXT_LAYER_PATTERNS = [
    (r"thinker\.model\.layers\.(\d+)\.input_layernorm\.weight",            "blk.{}.attn_norm.weight"),
    (r"thinker\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",          "blk.{}.attn_q.weight"),
    (r"thinker\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",          "blk.{}.attn_k.weight"),
    (r"thinker\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",          "blk.{}.attn_v.weight"),
    (r"thinker\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",          "blk.{}.attn_output.weight"),
    (r"thinker\.model\.layers\.(\d+)\.self_attn\.q_norm\.weight",          "blk.{}.attn_q_norm.weight"),
    (r"thinker\.model\.layers\.(\d+)\.self_attn\.k_norm\.weight",          "blk.{}.attn_k_norm.weight"),
    (r"thinker\.model\.layers\.(\d+)\.post_attention_layernorm\.weight",   "blk.{}.ffn_norm.weight"),
    (r"thinker\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",             "blk.{}.ffn_gate.weight"),
    (r"thinker\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",               "blk.{}.ffn_up.weight"),
    (r"thinker\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",             "blk.{}.ffn_down.weight"),
]


def remap_name(hf_name: str) -> str | None:
    if hf_name in DIRECT:
        return DIRECT[hf_name]
    for pat, tmpl in AUDIO_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    for pat, tmpl in TEXT_LAYER_PATTERNS:
        m = re.match(pat, hf_name)
        if m:
            return tmpl.format(m.group(1))
    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Norms, biases, 1-D tensors stay F32 for accuracy."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name or "ln_post" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")
    with open(input_dir / "config.json", "r") as f:
        cfg = json.load(f)
    thinker = cfg["thinker_config"]
    audio   = thinker["audio_config"]
    text    = thinker["text_config"]

    safetensor_files = sorted(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        sys.exit(f"no safetensors files found in {input_dir}")

    # Tokenizer
    with open(input_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    sorted_vocab = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    vocab_size = text.get("vocab_size", 151936)
    tokens = [tok for tok, _ in sorted_vocab]
    while len(tokens) < vocab_size:
        tokens.append(f"[PAD{len(tokens)}]")

    # Pull the special tokens (e.g. <|im_start|>, <|audio_pad|>) out of
    # tokenizer_config.json's added_tokens_decoder and patch them into the
    # token list at their proper IDs. vocab.json itself only contains the
    # 151643 regular BPE tokens; the special tokens live in the added_tokens
    # block, and without this patch they end up as "[PAD<id>]" placeholders
    # in the GGUF, which breaks the C++ BPE encoder's special-token lookup.
    tcfg_path = input_dir / "tokenizer_config.json"
    if tcfg_path.exists():
        with open(tcfg_path, "r", encoding="utf-8") as f:
            tcfg = json.load(f)
        added = tcfg.get("added_tokens_decoder", {})
        for tid_str, info in added.items():
            tid = int(tid_str)
            content = info.get("content")
            if content and 0 <= tid < len(tokens):
                tokens[tid] = content
        print(f"  patched {len(added)} added/special tokens from tokenizer_config.json")
    print(f"  vocab: {len(tokens)} tokens")

    merges_path = input_dir / "merges.txt"
    merges = []
    if merges_path.exists():
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
        print(f"  merges: {len(merges)}")

    # ----- write GGUF -----
    print(f"Writing: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="qwen3asr")

    # Audio params
    writer.add_uint32("qwen3asr.sample_rate",          16000)
    writer.add_uint32("qwen3asr.n_mels",               audio.get("num_mel_bins", 128))
    writer.add_uint32("qwen3asr.n_fft",                400)
    writer.add_uint32("qwen3asr.win_length",           400)
    writer.add_uint32("qwen3asr.hop_length",           160)
    writer.add_uint32("qwen3asr.audio.n_layers",       audio.get("encoder_layers", 18))
    writer.add_uint32("qwen3asr.audio.d_model",        audio.get("d_model", 896))
    writer.add_uint32("qwen3asr.audio.n_heads",        audio.get("encoder_attention_heads", 14))
    writer.add_uint32("qwen3asr.audio.head_dim",       audio.get("d_model", 896) // audio.get("encoder_attention_heads", 14))
    writer.add_uint32("qwen3asr.audio.ff_dim",         audio.get("encoder_ffn_dim", 3584))
    writer.add_uint32("qwen3asr.audio.conv_channels",  audio.get("downsample_hidden_size", 480))
    writer.add_uint32("qwen3asr.audio.proj_dim",       audio.get("output_dim", 1024))
    writer.add_uint32("qwen3asr.audio.max_source_pos", audio.get("max_source_positions", 1500))

    # LLM params
    writer.add_uint32("qwen3asr.llm.n_layers",     text.get("num_hidden_layers", 28))
    writer.add_uint32("qwen3asr.llm.d_model",      text.get("hidden_size", 1024))
    writer.add_uint32("qwen3asr.llm.n_heads",      text.get("num_attention_heads", 16))
    writer.add_uint32("qwen3asr.llm.n_kv_heads",   text.get("num_key_value_heads", 8))
    writer.add_uint32("qwen3asr.llm.head_dim",     text.get("head_dim", 128))
    writer.add_uint32("qwen3asr.llm.ff_dim",       text.get("intermediate_size", 3072))
    writer.add_float32("qwen3asr.llm.rope_theta",  float(text.get("rope_theta", 1000000)))
    writer.add_float32("qwen3asr.llm.rms_norm_eps", float(text.get("rms_norm_eps", 1e-6)))
    writer.add_uint32("qwen3asr.llm.vocab_size",   vocab_size)
    writer.add_uint32("qwen3asr.llm.max_pos",      text.get("max_position_embeddings", 65536))

    # Special tokens
    writer.add_uint32("qwen3asr.audio_start_token_id", thinker.get("audio_start_token_id", 151669))
    writer.add_uint32("qwen3asr.audio_end_token_id",   thinker.get("audio_end_token_id", 151670))
    writer.add_uint32("qwen3asr.audio_pad_token_id",   thinker.get("audio_token_id", 151676))
    writer.add_uint32("qwen3asr.eos_token_id",         151645)
    writer.add_uint32("qwen3asr.pad_token_id",         151643)

    # Tokenizer
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    if merges:
        writer.add_token_merges(merges)

    # ----- Mel filterbank + Hann window (baked from WhisperFeatureExtractor) -----
    # Pull these from the HF processor to avoid re-implementing the Slaney mel
    # filterbank in C++. Same pattern as parakeet's preprocessor.fb / window.
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(str(input_dir))
    except Exception:
        # Fall back to qwen-asr's own processor (which uses WhisperFeatureExtractor)
        from qwen_asr import Qwen3ASRModel
        wrapper = Qwen3ASRModel.from_pretrained(str(input_dir), dtype="float32",
                                                 device_map="cpu")
        fe = wrapper.processor.feature_extractor
    mel_filters = np.asarray(fe.mel_filters, dtype=np.float32)  # (n_freqs, n_mels)
    print(f"  mel_filters shape: {mel_filters.shape}")
    writer.add_tensor("audio.mel_filters", mel_filters)
    # WhisperFeatureExtractor uses scipy/librosa periodic hann of length n_fft=400.
    # Equivalent to torch.hann_window(400, periodic=True) = 0.5 - 0.5*cos(2*pi*n/N).
    n_fft_w = 400
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft_w) / n_fft_w)).astype(np.float32)
    writer.add_tensor("audio.mel_window", win)
    print(f"  mel_window shape: {win.shape}")

    # ----- tensors -----
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
                # safetensors gives torch tensors; bf16 needs float32 detour
                if t.dtype.is_floating_point and t.dtype.itemsize == 2 and "bfloat" in str(t.dtype):
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
                    print(f"    {gguf_name:50s} {str(arr.shape):24s} {arr.dtype}")

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
    p = argparse.ArgumentParser(description="Convert Qwen3-ASR-0.6B HF safetensors → GGUF F16")
    p.add_argument("--input",  required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
