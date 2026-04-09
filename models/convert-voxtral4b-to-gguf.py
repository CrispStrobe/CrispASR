#!/usr/bin/env python3
"""
Convert mistralai/Voxtral-Mini-4B-Realtime-2602 (HF safetensors) → GGUF F16.

Architecture differences from Voxtral-Mini-3B-2507:

  Audio encoder (32 layers):
    - RoPE (no learned absolute pos embed)
    - SwiGLU FFN (gate_proj + up_proj + down_proj) instead of GELU fc1/fc2
    - RMSNorm (no bias) instead of LayerNorm (with bias)
    - Sliding window attention (750)
    - Biases on Q, V, O, down_proj (K has no bias — preserved Whisper quirk)
    - Tensor name changes: out_proj → o_proj, fc1/fc2 → gate/up/down_proj

  LLM (26 layers):
    - FFN dim 9216 (vs 8192)
    - RoPE θ=1e6 (vs 1e8)
    - Sliding window attention (8192)
    - Tied embeddings (no lm_head)
    - NEW: adaptive RMSNorm per layer (ada_rms_norm.linear1/2)

  Projector: identical topology to 3B.
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
# Tensor name remapping (HF → GGUF)
# ---------------------------------------------------------------------------

DIRECT = {
    # Audio encoder front-end (embedder submodule in 4B)
    "audio_tower.embedder.conv1.weight": "audio.conv.1.weight",
    "audio_tower.embedder.conv1.bias":   "audio.conv.1.bias",
    "audio_tower.embedder.conv2.weight": "audio.conv.2.weight",
    "audio_tower.embedder.conv2.bias":   "audio.conv.2.bias",
    # Audio encoder post-norm (RMSNorm, no bias)
    "audio_tower.norm.weight":           "audio.ln_post.weight",
    # Multi-modal projector
    "multi_modal_projector.linear_1.weight": "proj1.weight",
    "multi_modal_projector.linear_2.weight": "proj2.weight",
    # LLM top level
    "language_model.model.embed_tokens.weight": "token_embd.weight",
    "language_model.model.norm.weight":         "output_norm.weight",
    # No lm_head — tied to token_embd. The C++ runtime handles this.
}

# Audio encoder per-layer (4B naming: o_proj, gate/up/down_proj, RMSNorm)
AUDIO_LAYER_PATTERNS = [
    # Attention norms (RMSNorm, weight only)
    (r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.weight", "audio.blk.{}.attn_norm.weight"),
    (r"audio_tower\.layers\.(\d+)\.final_layer_norm\.weight",     "audio.blk.{}.ffn_norm.weight"),
    # Attention Q/K/V/O
    (r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.weight",    "audio.blk.{}.attn_q.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.bias",      "audio.blk.{}.attn_q.bias"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight",    "audio.blk.{}.attn_k.weight"),
    # K has no bias (Whisper quirk preserved)
    (r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.weight",    "audio.blk.{}.attn_v.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.bias",      "audio.blk.{}.attn_v.bias"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.o_proj\.weight",    "audio.blk.{}.attn_out.weight"),
    (r"audio_tower\.layers\.(\d+)\.self_attn\.o_proj\.bias",      "audio.blk.{}.attn_out.bias"),
    # SwiGLU FFN: gate + up + down (down has bias)
    (r"audio_tower\.layers\.(\d+)\.mlp\.gate_proj\.weight",       "audio.blk.{}.ffn_gate.weight"),
    (r"audio_tower\.layers\.(\d+)\.mlp\.up_proj\.weight",         "audio.blk.{}.ffn_up.weight"),
    (r"audio_tower\.layers\.(\d+)\.mlp\.down_proj\.weight",       "audio.blk.{}.ffn_down.weight"),
    (r"audio_tower\.layers\.(\d+)\.mlp\.down_proj\.bias",         "audio.blk.{}.ffn_down.bias"),
]

# LLM per-layer (26 layers, includes ada_rms_norm)
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
    # Adaptive RMSNorm (time conditioning)
    (r"language_model\.model\.layers\.(\d+)\.ada_rms_norm\.linear1\.weight",   "blk.{}.ada_norm_down.weight"),
    (r"language_model\.model\.layers\.(\d+)\.ada_rms_norm\.linear2\.weight",   "blk.{}.ada_norm_up.weight"),
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
    """Norms, biases, 1-D tensors stay F32 for accuracy."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name or "ln_post" in gguf_name:
        return True
    if "ada_norm" in gguf_name:
        return True  # small tensors (32×3072), keep F32
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Tekken tokenizer serialization (same as 3B)
# ---------------------------------------------------------------------------

def serialize_tekken_vocab(tekken: dict) -> bytes:
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

    safetensor_files = sorted(input_dir.glob("model*.safetensors"))
    # Exclude consolidated.safetensors (different naming convention)
    safetensor_files = [p for p in safetensor_files if "consolidated" not in p.name]
    if not safetensor_files:
        sys.exit(f"no model safetensors files found in {input_dir}")
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
    writer = gguf.GGUFWriter(str(out_path), arch="voxtral4b")

    # Audio params
    writer.add_uint32("voxtral4b.sample_rate",      16000)
    writer.add_uint32("voxtral4b.n_mels",           audio.get("num_mel_bins", 128))
    writer.add_uint32("voxtral4b.n_fft",            400)
    writer.add_uint32("voxtral4b.win_length",       400)
    writer.add_uint32("voxtral4b.hop_length",       160)
    writer.add_uint32("voxtral4b.audio.n_layers",   audio.get("num_hidden_layers", 32))
    writer.add_uint32("voxtral4b.audio.d_model",    audio.get("hidden_size", 1280))
    writer.add_uint32("voxtral4b.audio.n_heads",    audio.get("num_attention_heads", 32))
    writer.add_uint32("voxtral4b.audio.head_dim",   audio.get("head_dim", 64))
    writer.add_uint32("voxtral4b.audio.ff_dim",     audio.get("intermediate_size", 5120))
    writer.add_uint32("voxtral4b.audio.max_pos",    audio.get("max_position_embeddings", 1500))
    writer.add_float32("voxtral4b.audio.rope_theta", float(audio.get("rope_theta", 1e6)))
    writer.add_uint32("voxtral4b.audio.sliding_window", audio.get("sliding_window", 750))

    # Projector params
    writer.add_uint32("voxtral4b.proj.in_dim",      5120)
    writer.add_uint32("voxtral4b.proj.out_dim",     text.get("hidden_size", 3072))
    writer.add_uint32("voxtral4b.proj.frame_stack", cfg.get("downsample_factor", 4))

    # LLM params
    writer.add_uint32("voxtral4b.llm.n_layers",     text.get("num_hidden_layers", 26))
    writer.add_uint32("voxtral4b.llm.d_model",      text.get("hidden_size", 3072))
    writer.add_uint32("voxtral4b.llm.n_heads",      text.get("num_attention_heads", 32))
    writer.add_uint32("voxtral4b.llm.n_kv_heads",   text.get("num_key_value_heads", 8))
    writer.add_uint32("voxtral4b.llm.head_dim",     text.get("head_dim", 128))
    writer.add_uint32("voxtral4b.llm.ff_dim",       text.get("intermediate_size", 9216))
    writer.add_float32("voxtral4b.llm.rope_theta",  float(text.get("rope_theta", 1e6)))
    writer.add_float32("voxtral4b.llm.rms_norm_eps", float(text.get("rms_norm_eps", 1e-5)))
    writer.add_uint32("voxtral4b.llm.vocab_size",   text.get("vocab_size", 131072))
    writer.add_uint32("voxtral4b.llm.max_pos",      text.get("max_position_embeddings", 131072))
    writer.add_uint32("voxtral4b.llm.sliding_window", text.get("sliding_window", 8192))
    writer.add_uint32("voxtral4b.llm.ada_norm_dim", text.get("ada_rms_norm_t_cond_dim", 32))
    writer.add_bool  ("voxtral4b.llm.tied_embeddings", True)
    writer.add_uint32("voxtral4b.audio_token_id",   cfg.get("audio_token_id", 24))

    # Tekken tokenizer
    writer.add_string("tokenizer.tekken.pattern", tekken_cfg.get("pattern", ""))
    specials = [s["token_str"] for s in tekken.get("special_tokens", [])]
    writer.add_array("tokenizer.tekken.specials", specials)
    vocab_blob = serialize_tekken_vocab(tekken)
    print(f"  vocab blob: {len(vocab_blob)/1024:.1f} KB")
    vocab_f32 = np.frombuffer(vocab_blob, dtype=np.uint8).astype(np.float32)
    writer.add_tensor("tokenizer.tekken.vocab_tensor", vocab_f32)
    writer.add_uint32("tokenizer.tekken.n_specials", len(specials))
    writer.add_uint32("tokenizer.tekken.n_vocab",    n_vocab)

    # Mel filterbank + Hann window
    # IMPORTANT: Voxtral Realtime uses SLANEY mel filters (not HTK/Whisper).
    # All reference implementations (voxtral.c, voxmlx, voxtral-rs) use Slaney.
    def build_slaney_mel_filters(sr=16000, n_fft=400, n_mels=128, f_min=0.0, f_max=8000.0):
        """Slaney-style mel filter bank matching mistral_common/audio.py."""
        n_freqs = n_fft // 2 + 1
        def hz_to_mel(f):
            min_log_hz, min_log_mel = 1000.0, 15.0
            logstep = 27.0 / np.log(6.4)
            mels = 3.0 * np.asarray(f, dtype=np.float64) / 200.0
            mask = np.asarray(f) >= min_log_hz
            mels[mask] = min_log_mel + np.log(np.asarray(f)[mask] / min_log_hz) * logstep
            return mels
        def mel_to_hz(m):
            min_log_hz, min_log_mel = 1000.0, 15.0
            logstep = np.log(6.4) / 27.0
            freq = 200.0 * np.asarray(m, dtype=np.float64) / 3.0
            mask = np.asarray(m) >= min_log_mel
            freq[mask] = min_log_hz * np.exp(logstep * (np.asarray(m)[mask] - min_log_mel))
            return freq
        fft_freqs = np.linspace(0, sr / 2, n_freqs)
        mel_min = hz_to_mel(np.array([f_min]))[0]
        mel_max = hz_to_mel(np.array([f_max]))[0]
        mel_freqs = np.linspace(mel_min, mel_max, n_mels + 2)
        filter_freqs = mel_to_hz(mel_freqs)
        filter_diff = np.diff(filter_freqs)
        slopes = filter_freqs[np.newaxis, :] - fft_freqs[:, np.newaxis]
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
        enorm = 2.0 / (filter_freqs[2:n_mels+2] - filter_freqs[:n_mels])
        fb *= enorm[np.newaxis, :]
        return fb.astype(np.float32)  # (n_freqs, n_mels)

    mel_filters = build_slaney_mel_filters()
    print(f"  mel_filters shape: {mel_filters.shape}")
    writer.add_tensor("audio.mel_filters", mel_filters)
    n_fft_w = 400
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft_w) / n_fft_w)).astype(np.float32)
    writer.add_tensor("audio.mel_window", win)

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
        print("  skipped tensors:")
        for n in skipped_names:
            print(f"    {n}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Voxtral-Mini-4B-Realtime-2602 HF safetensors → GGUF F16")
    p.add_argument("--input",  required=True, type=Path, help="HF model directory")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
