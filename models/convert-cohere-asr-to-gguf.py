#!/usr/bin/env python3
"""Convert CohereForAI/cohere-transcribe-03-2026 safetensors → GGUF F16.

Architecture:
  Conformer encoder (48 layers, d_model=1280, 8 heads, 160 hd, 5120 FFN,
                     conv kernel=9, 3×stride-2 Conv2d pre-encode → ×8 downsample)
  Linear enc→dec projection (1280 → 1024)
  Transformer decoder (8 layers, d=1024, 8 heads, 128 hd, 4096 FFN,
                       max 1024 ctx, 16384-token vocab)
  Audio: 16 kHz, 128 mels, n_fft=512, hop=160, win=400

The spec for this model (defaults from the C++ loader / cohere-arch.h):
  encoder: 48 layers, d=1280, heads=8, head_dim=160, ffn=5120, conv_k=9
  decoder: 8 layers, d=1024, heads=8, head_dim=128, ffn=4096, max_ctx=1024
  audio: sr=16000, n_mels=128, n_fft=512, hop=160, win=400

However the model spec passed in the task description uses:
  encoder: 32 layers, d=1280, heads=20, head_dim=64, ffn=5120, conv_k=31
  decoder: 4 layers, d=1280, heads=20, head_dim=64, ffn=5120, max_ctx=448
  audio: sr=16000, n_mels=128, n_fft=400, hop=160, win=400
The converter reads these from the actual model config.json, so either variant
is handled automatically.

Usage:
  python models/convert-cohere-asr-to-gguf.py \\
      --input CohereForAI/cohere-transcribe-03-2026 \\
      --output /mnt/storage/models/cohere-transcribe.gguf

  # or from a local directory:
  python models/convert-cohere-asr-to-gguf.py \\
      --input /mnt/storage/cohere-transcribe-03-2026 \\
      --output /mnt/storage/models/cohere-transcribe.gguf

Requires: pip install gguf safetensors torch transformers huggingface_hub
The model is gated — run `huggingface-cli login` and accept the license at
https://huggingface.co/CohereForAI/cohere-transcribe-03-2026 first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    try:
        import gguf
    except ImportError:
        sys.exit("gguf not found — run: pip install gguf")

try:
    from safetensors import safe_open
except ImportError:
    sys.exit("safetensors not found — run: pip install safetensors")

try:
    import torch
except ImportError:
    sys.exit("torch not found — run: pip install torch")


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def resolve_model_dir(input_path: str) -> Path:
    """Return a local directory with the model files.

    If input_path is a local directory that exists, return it directly.
    Otherwise treat it as a HuggingFace repo ID and download via snapshot.
    """
    p = Path(input_path)
    if p.is_dir():
        return p

    # Try HF download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub not found — run: pip install huggingface_hub\n"
            "Or pass a local directory with --input."
        )

    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        cache_dir = os.path.join(cache_dir, "hub")

    print(f"  Downloading {input_path} from HuggingFace...")
    local = snapshot_download(
        input_path,
        cache_dir=cache_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    return Path(local)


def load_safetensors(model_dir: Path) -> dict[str, np.ndarray]:
    """Load all safetensors shards from a model directory into a single dict."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        # Get unique shard filenames
        shards = sorted(set(index["weight_map"].values()))
    else:
        # Single-file model
        single = model_dir / "model.safetensors"
        if not single.exists():
            sys.exit(f"No safetensors files found in {model_dir}")
        shards = ["model.safetensors"]

    tensors: dict[str, np.ndarray] = {}
    for shard in shards:
        path = model_dir / shard
        print(f"  Loading shard: {shard}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                # Convert bfloat16 to float32 (bfloat16 has no direct numpy type)
                if t.dtype == torch.bfloat16:
                    t = t.to(torch.float32)
                tensors[key] = t.numpy()

    print(f"  Loaded {len(tensors)} tensors from {len(shards)} shard(s)")
    return tensors


def load_config(model_dir: Path) -> dict:
    """Load config.json from the model directory."""
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print("  WARNING: config.json not found, using spec defaults", file=sys.stderr)
        return {}
    with open(config_file) as f:
        return json.load(f)


def load_tokenizer(model_dir: Path) -> list[str]:
    """Extract vocabulary from the HF tokenizer files.

    Tries tokenizer.json (byte-level BPE / tiktoken) first, then
    tokenizer_config.json for a SentencePiece fallback.
    """
    tok_file = model_dir / "tokenizer.json"
    if tok_file.exists():
        with open(tok_file, encoding="utf-8") as f:
            tok = json.load(f)

        # tiktoken / BPE — vocab is in model.vocab as {token: id}
        model_section = tok.get("model", {})
        raw_vocab = model_section.get("vocab", {})

        if raw_vocab:
            vocab = [""] * len(raw_vocab)
            for token, idx in raw_vocab.items():
                if idx < len(vocab):
                    vocab[idx] = token
            # Fill any added_tokens that may extend beyond the base vocab
            for entry in tok.get("added_tokens", []):
                idx = entry["id"]
                if idx >= len(vocab):
                    vocab.extend([""] * (idx - len(vocab) + 1))
                vocab[idx] = entry["content"]
            return vocab

        # merges-only format (some tiktoken variants)
        # fall through to added_tokens only
        added = {e["id"]: e["content"] for e in tok.get("added_tokens", [])}
        if added:
            max_id = max(added.keys())
            vocab = [""] * (max_id + 1)
            for idx, tok_str in added.items():
                vocab[idx] = tok_str
            return vocab

    # SentencePiece fallback
    spm_files = list(model_dir.glob("*.model"))
    if spm_files:
        try:
            import sentencepiece as spm
        except ImportError:
            sys.exit("sentencepiece not found — run: pip install sentencepiece")
        sp = spm.SentencePieceProcessor(model_file=str(spm_files[0]))
        vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  Vocabulary: {len(vocab)} SentencePiece tokens from {spm_files[0].name}")
        return vocab

    # Last resort: try loading via transformers AutoTokenizer
    try:
        from transformers import AutoTokenizer
        print("  Loading tokenizer via transformers AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True)
        vocab_dict = tokenizer.get_vocab()
        vocab = [""] * len(vocab_dict)
        for tok_str, idx in vocab_dict.items():
            if idx < len(vocab):
                vocab[idx] = tok_str
        print(f"  Vocabulary: {len(vocab)} tokens via AutoTokenizer")
        return vocab
    except Exception as e:
        print(f"  WARNING: Could not load tokenizer ({e}), embedding vocab_size=0", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Architecture hyperparams from config.json
# ---------------------------------------------------------------------------

def extract_hparams(config: dict) -> dict:
    """Extract architecture hyperparameters from HF config.json.

    The Cohere Transcribe model ships its own remote code so the exact
    field names depend on that code's version. We probe several plausible
    layouts and fall back to the task-spec defaults.
    """
    hp = {
        # defaults from the task spec (cohere-transcribe-03-2026)
        "vocab_size": 256000,    # multilingual BPE, overridden below
        # encoder
        "enc_n_layers": 32,
        "enc_d_model": 1280,
        "enc_n_heads": 20,
        "enc_head_dim": 64,
        "enc_ffn_dim": 5120,
        "enc_conv_k": 31,
        # decoder
        "dec_n_layers": 4,
        "dec_d_model": 1280,
        "dec_n_heads": 20,
        "dec_head_dim": 64,
        "dec_ffn_dim": 5120,
        "dec_max_ctx": 448,
        # audio
        "sample_rate": 16000,
        "n_mels": 128,
        "n_fft": 400,
        "hop_length": 160,
        "win_length": 400,
    }

    if not config:
        return hp

    # Top-level vocab_size
    if "vocab_size" in config:
        hp["vocab_size"] = config["vocab_size"]

    # Encoder — look for a nested "encoder_config" or "audio_encoder_config" etc.
    enc_cfg = (
        config.get("encoder_config") or
        config.get("audio_encoder_config") or
        config.get("encoder") or
        config.get("conformer_config") or
        {}
    )
    if enc_cfg:
        _pick(hp, enc_cfg, "enc_n_layers",  ["num_hidden_layers", "n_layers", "num_layers", "n_conformer_layers"])
        _pick(hp, enc_cfg, "enc_d_model",   ["hidden_size", "d_model", "encoder_dim"])
        _pick(hp, enc_cfg, "enc_n_heads",   ["num_attention_heads", "n_heads", "attention_heads"])
        _pick(hp, enc_cfg, "enc_head_dim",  ["head_dim", "attention_head_size"])
        _pick(hp, enc_cfg, "enc_ffn_dim",   ["intermediate_size", "ffn_dim", "ff_dim", "feed_forward_expansion_factor"])
        _pick(hp, enc_cfg, "enc_conv_k",    ["conv_kernel_size", "kernel_size", "depthwise_conv_kernel_size"])

    # Decoder
    dec_cfg = (
        config.get("decoder_config") or
        config.get("text_config") or
        config.get("decoder") or
        {}
    )
    if dec_cfg:
        _pick(hp, dec_cfg, "dec_n_layers",  ["num_hidden_layers", "n_layers", "num_layers"])
        _pick(hp, dec_cfg, "dec_d_model",   ["hidden_size", "d_model"])
        _pick(hp, dec_cfg, "dec_n_heads",   ["num_attention_heads", "n_heads"])
        _pick(hp, dec_cfg, "dec_head_dim",  ["head_dim", "attention_head_size"])
        _pick(hp, dec_cfg, "dec_ffn_dim",   ["intermediate_size", "ffn_dim"])
        _pick(hp, dec_cfg, "dec_max_ctx",   ["max_position_embeddings", "max_ctx", "max_length"])

    # Audio / feature extractor config
    audio_cfg = (
        config.get("feature_extractor") or
        config.get("audio_config") or
        config.get("preprocessor_config") or
        {}
    )
    # Also check top-level for flat configs
    flat = config
    for src in (audio_cfg, flat):
        _pick(hp, src, "sample_rate",  ["sampling_rate", "sample_rate"])
        _pick(hp, src, "n_mels",       ["num_mel_bins", "n_mels", "n_mel_channels", "num_mel_channels"])
        _pick(hp, src, "n_fft",        ["n_fft", "fft_size"])
        _pick(hp, src, "hop_length",   ["hop_length", "hop_size", "stride"])
        _pick(hp, src, "win_length",   ["win_length", "window_size"])

    # Derive head_dim if not set
    if hp["enc_head_dim"] == 64 and hp["enc_d_model"] > 0 and hp["enc_n_heads"] > 0:
        derived = hp["enc_d_model"] // hp["enc_n_heads"]
        if derived != hp["enc_head_dim"]:
            hp["enc_head_dim"] = derived
    if hp["dec_head_dim"] == 64 and hp["dec_d_model"] > 0 and hp["dec_n_heads"] > 0:
        derived = hp["dec_d_model"] // hp["dec_n_heads"]
        if derived != hp["dec_head_dim"]:
            hp["dec_head_dim"] = derived

    return hp


def _pick(dst: dict, src: dict, dst_key: str, src_keys: list[str]) -> None:
    """Update dst[dst_key] from the first matching key in src."""
    for k in src_keys:
        if k in src and src[k] is not None:
            dst[dst_key] = int(src[k])
            return


# ---------------------------------------------------------------------------
# Mel filterbank and Hann window generation
# ---------------------------------------------------------------------------

def make_mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Compute a mel filterbank matrix [1, n_mels, n_freqs] in F32.

    Uses the HTK formula (matching librosa's mel_filters default).
    """
    n_freqs = n_fft // 2 + 1
    fmin = 0.0
    fmax = sample_rate / 2.0

    # Mel scale conversion
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # FFT bin frequencies
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    # Build filterbank [n_mels, n_freqs]
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]
        for k in range(n_freqs):
            f = fft_freqs[k]
            if f_left <= f <= f_center:
                fb[m, k] = (f - f_left) / (f_center - f_left + 1e-10)
            elif f_center < f <= f_right:
                fb[m, k] = (f_right - f) / (f_right - f_center + 1e-10)

    # Add batch dimension → [1, n_mels, n_freqs]
    return fb[np.newaxis, :, :].astype(np.float32)


def make_hann_window(win_length: int) -> np.ndarray:
    """Generate a Hann window of length win_length in F32."""
    n = np.arange(win_length, dtype=np.float32)
    window = 0.5 * (1.0 - np.cos(2.0 * math.pi * n / (win_length - 1)))
    return window.astype(np.float32)


# ---------------------------------------------------------------------------
# Tensor name remapping: PyTorch → GGUF
# ---------------------------------------------------------------------------

def remap_name(pt_name: str, hp: dict) -> Optional[str]:
    """Map a PyTorch state-dict key to the GGUF tensor name.

    Returns None to drop the tensor entirely.

    The Cohere Transcribe model ships its own remote modeling code.
    Observed naming conventions (inferred from the cohere.py reference
    backend, the task spec, and common Conformer/Whisper patterns):

      encoder.conv_subsampler.conv_layers.0.weight
        → enc.pre.conv.0.weight
      encoder.conv_subsampler.conv_layers.0.bias
        → enc.pre.conv.0.bias
      ... (indices 0, 2, 3, 5, 6) ...
      encoder.conv_subsampler.out_proj.weight
        → enc.pre.out.weight

      encoder.conformer_layers.N.feed_forward1.layer_norm.weight
        → enc.blk.N.ff1.norm.weight
      encoder.conformer_layers.N.feed_forward1.linear_in.weight
        → enc.blk.N.ff1.up.weight
      encoder.conformer_layers.N.feed_forward1.linear_out.weight
        → enc.blk.N.ff1.down.weight
      encoder.conformer_layers.N.self_attention.pos_bias_u
        → enc.blk.N.attn.pos_bias_u
      encoder.conformer_layers.N.self_attention.linear_pos.weight
        → enc.blk.N.attn.pos.weight
      encoder.conformer_layers.N.self_attention.linear_q.weight
        → enc.blk.N.attn.q.weight
      encoder.conformer_layers.N.conv_module.batch_norm.{weight,bias,running_mean,running_var}
        → enc.blk.N.conv.bn.{weight,bias,mean,var}
      encoder.conformer_layers.N.final_layer_norm.weight
        → enc.blk.N.out_norm.weight

      enc_to_dec_proj.weight  / encoder_proj.weight
        → enc.proj.weight

      decoder.embed_tokens.weight     → dec.emb.weight
      decoder.embed_positions.weight  → dec.pos.weight
      decoder.layer_norm.weight       → dec.emb_ln.weight
      decoder.layers.N.self_attn_layer_norm.weight  → dec.blk.N.attn_ln.weight
      decoder.layers.N.self_attn.q_proj.weight      → dec.blk.N.attn_q.weight
      decoder.layers.N.encoder_attn_layer_norm.weight → dec.blk.N.cross_ln.weight
      decoder.layers.N.encoder_attn.q_proj.weight     → dec.blk.N.cross_q.weight
      decoder.layers.N.final_layer_norm.weight        → dec.blk.N.ffn_ln.weight
      decoder.layers.N.fc1.weight                     → dec.blk.N.ffn_up.weight
      decoder.layers.N.fc2.weight                     → dec.blk.N.ffn_down.weight
      decoder.final_layer_norm.weight  → dec.out_ln.weight
      lm_head.weight                   → dec.head.weight
    """
    n = pt_name

    # --- drop num_batches_tracked (BatchNorm bookkeeping) ---
    if n.endswith(".num_batches_tracked"):
        return None

    # =====================================================================
    # Feature extraction (mel_fb / window are synthesized, not in the SD)
    # =====================================================================

    # =====================================================================
    # Pre-encode Conv2D subsampling
    # conv_subsampler / pre_encode / conv_subsample (remote code variant)
    # =====================================================================
    for prefix in ("encoder.conv_subsampler.", "encoder.pre_encode.", "encoder.conv_subsample."):
        if n.startswith(prefix):
            rest = n[len(prefix):]
            # conv_layers.{idx}.{weight,bias}
            if rest.startswith("conv_layers."):
                parts = rest[len("conv_layers."):].split(".", 1)
                if len(parts) == 2:
                    idx, suffix = parts
                    return f"enc.pre.conv.{idx}.{suffix}"
            # out_proj / linear (projection after flatten)
            if rest in ("out_proj.weight", "linear.weight", "out.weight", "proj.weight"):
                return "enc.pre.out.weight"
            if rest in ("out_proj.bias", "linear.bias", "out.bias", "proj.bias"):
                return "enc.pre.out.bias"
            # Sequential numeric: some implementations name layers 0,1,... inside a Sequential
            # e.g. encoder.pre_encode.0.weight -> enc.pre.conv.0.weight
            parts2 = rest.split(".", 1)
            if len(parts2) == 2 and parts2[0].isdigit():
                return f"enc.pre.conv.{parts2[0]}.{parts2[1]}"
            print(f"  [warn] unmapped pre-encode tensor: {n}", file=sys.stderr)
            return None

    # =====================================================================
    # Encoder conformer layers
    # Handles both encoder.conformer_layers.N.* and encoder.layers.N.*
    # =====================================================================
    for enc_prefix in ("encoder.conformer_layers.", "encoder.layers."):
        if n.startswith(enc_prefix):
            rest = n[len(enc_prefix):]
            dot = rest.find(".")
            if dot < 0:
                return None
            layer_id = rest[:dot]
            sub = rest[dot + 1:]

            g = _remap_enc_sub(layer_id, sub)
            if g is None:
                print(f"  [warn] unmapped encoder tensor: {n}", file=sys.stderr)
            return g

    # =====================================================================
    # Encoder → decoder projection
    # =====================================================================
    for proj_name in ("enc_to_dec_proj", "encoder_proj", "enc_proj", "audio_proj",
                      "encoder.linear", "model.enc_to_dec_proj"):
        if n == f"{proj_name}.weight":
            return "enc.proj.weight"
        if n == f"{proj_name}.bias":
            return "enc.proj.bias"

    # =====================================================================
    # Decoder
    # =====================================================================
    # embeddings
    for emb_name in ("model.embed_tokens", "decoder.embed_tokens", "embed_tokens"):
        if n == f"{emb_name}.weight":
            return "dec.emb.weight"
    for pos_name in ("model.embed_positions", "decoder.embed_positions", "embed_positions"):
        if n == f"{pos_name}.weight":
            return "dec.pos.weight"

    # embedding layer norm
    for emb_ln in ("decoder.layer_norm", "model.decoder_layer_norm",
                   "decoder.embed_layer_norm", "model.embed_layer_norm"):
        if n == f"{emb_ln}.weight":
            return "dec.emb_ln.weight"
        if n == f"{emb_ln}.bias":
            return "dec.emb_ln.bias"

    # decoder layers
    for dec_prefix in ("decoder.layers.", "model.decoder.layers.",
                       "text_decoder.layers.", "model.layers."):
        if n.startswith(dec_prefix):
            rest = n[len(dec_prefix):]
            dot = rest.find(".")
            if dot < 0:
                return None
            layer_id = rest[:dot]
            sub = rest[dot + 1:]
            g = _remap_dec_sub(layer_id, sub)
            if g is None:
                print(f"  [warn] unmapped decoder tensor: {n}", file=sys.stderr)
            return g

    # decoder final layer norm
    for out_ln in ("decoder.final_layer_norm", "model.final_layer_norm",
                   "text_decoder.layer_norm", "model.decoder_norm"):
        if n == f"{out_ln}.weight":
            return "dec.out_ln.weight"
        if n == f"{out_ln}.bias":
            return "dec.out_ln.bias"

    # output head (lm_head / dec_head / projection)
    for head_name in ("lm_head", "decoder.lm_head", "model.lm_head",
                      "output_projection", "proj_out"):
        if n == f"{head_name}.weight":
            return "dec.head.weight"
        if n == f"{head_name}.bias":
            return "dec.head.bias"

    # Encoder final layer norm (used before projection in some versions)
    for enc_ln in ("encoder.layer_norm", "encoder.final_layer_norm"):
        if n == f"{enc_ln}.weight":
            # Store as enc.out_norm.weight (used as pre-proj norm)
            return "enc.out_norm.weight"
        if n == f"{enc_ln}.bias":
            return "enc.out_norm.bias"

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


def _remap_enc_sub(layer_id: str, sub: str) -> Optional[str]:
    """Map an encoder conformer-layer sub-path to a GGUF name."""
    pfx = f"enc.blk.{layer_id}"

    # --- Feed-forward 1 ---
    for ff1_prefix in ("feed_forward1.", "feed_forward_macaron.", "ff1.",
                       "ffn1.", "macaron_ff.", "macaron1."):
        if sub.startswith(ff1_prefix):
            s = sub[len(ff1_prefix):]
            s = (s.replace("layer_norm.", "norm.")
                  .replace("layer_norm_", "norm.")
                  .replace("linear_in.", "up.")
                  .replace("linear1.", "up.")
                  .replace("w_1.", "up.")
                  .replace("linear_out.", "down.")
                  .replace("linear2.", "down.")
                  .replace("w_2.", "down."))
            return f"{pfx}.ff1.{s}"

    # --- Self-attention ---
    for attn_prefix in ("self_attention.", "self_attn.", "mhsa.", "multihead_attn."):
        if sub.startswith(attn_prefix):
            s = sub[len(attn_prefix):]
            # pos_bias_u / pos_bias_v (scalar biases, no weight suffix)
            if s in ("pos_bias_u", "pos_bias_v"):
                return f"{pfx}.attn.{s}"
            # layer norm inside attention
            s = (s.replace("layer_norm.", "norm.")
                  .replace("layer_norm_", "norm.")
                  .replace("norm.", "norm.")
                  # position projection
                  .replace("linear_pos.", "pos.")
                  .replace("pos_proj.", "pos.")
                  # Q/K/V/out
                  .replace("linear_q.", "q.")
                  .replace("q_proj.", "q.")
                  .replace("linear_k.", "k.")
                  .replace("k_proj.", "k.")
                  .replace("linear_v.", "v.")
                  .replace("v_proj.", "v.")
                  .replace("linear_out.", "out.")
                  .replace("out_proj.", "out.")
                  .replace("linear_out.", "out."))
            return f"{pfx}.attn.{s}"

    # attention layer norm (standalone, outside the sub-module in some configs)
    if sub in ("norm_self_att.weight", "norm_self_att.bias",
               "self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
               "attn_layer_norm.weight", "attn_layer_norm.bias"):
        suffix = sub.split(".", 1)[1]
        return f"{pfx}.attn.norm.{suffix}"

    # --- Convolution module ---
    for conv_prefix in ("conv_module.", "convolution_module.", "conv.", "conv_block."):
        if sub.startswith(conv_prefix):
            s = sub[len(conv_prefix):]
            s = (s.replace("layer_norm.", "norm.")
                  .replace("layer_norm_", "norm.")
                  .replace("norm.", "norm.")
                  .replace("pointwise_conv1.", "pw1.")
                  .replace("pw1_conv.", "pw1.")
                  .replace("depthwise_conv.", "dw.")
                  .replace("dw_conv.", "dw.")
                  .replace("batch_norm.", "bn.")
                  .replace("bn.", "bn.")
                  .replace("pointwise_conv2.", "pw2.")
                  .replace("pw2_conv.", "pw2."))
            # Remap BatchNorm running stats
            s = s.replace("bn.running_mean", "bn.mean").replace("bn.running_var", "bn.var")
            return f"{pfx}.conv.{s}"

    # conv layer norm (standalone)
    if sub in ("norm_conv.weight", "norm_conv.bias",
               "conv_layer_norm.weight", "conv_layer_norm.bias"):
        suffix = sub.split(".", 1)[1]
        return f"{pfx}.conv.norm.{suffix}"

    # --- Feed-forward 2 ---
    for ff2_prefix in ("feed_forward2.", "feed_forward.", "ff2.",
                       "ffn2.", "ffn.", "macaron2.", "macaron_ff2."):
        if sub.startswith(ff2_prefix):
            s = sub[len(ff2_prefix):]
            s = (s.replace("layer_norm.", "norm.")
                  .replace("layer_norm_", "norm.")
                  .replace("linear_in.", "up.")
                  .replace("linear1.", "up.")
                  .replace("w_1.", "up.")
                  .replace("linear_out.", "down.")
                  .replace("linear2.", "down.")
                  .replace("w_2.", "down."))
            return f"{pfx}.ff2.{s}"

    # --- Output norm ---
    for out_norm in ("final_layer_norm.", "norm.", "layer_norm.", "out_norm."):
        if sub.startswith(out_norm):
            suffix = sub[len(out_norm):]
            return f"{pfx}.out_norm.{suffix}"

    # bare suffixes
    if sub in ("final_layer_norm.weight", "norm.weight",
               "layer_norm.weight", "out_norm.weight"):
        return f"{pfx}.out_norm.weight"
    if sub in ("final_layer_norm.bias", "norm.bias",
               "layer_norm.bias", "out_norm.bias"):
        return f"{pfx}.out_norm.bias"

    return None


def _remap_dec_sub(layer_id: str, sub: str) -> Optional[str]:
    """Map a decoder layer sub-path to a GGUF name."""
    pfx = f"dec.blk.{layer_id}"

    # --- Self-attention layer norm ---
    if sub in ("self_attn_layer_norm.weight", "self_attn_norm.weight",
               "norm1.weight", "ln1.weight", "attn_layer_norm.weight"):
        return f"{pfx}.attn_ln.weight"
    if sub in ("self_attn_layer_norm.bias", "self_attn_norm.bias",
               "norm1.bias", "ln1.bias", "attn_layer_norm.bias"):
        return f"{pfx}.attn_ln.bias"

    # --- Self-attention projections ---
    for sa_prefix in ("self_attn.", "self_attention."):
        if sub.startswith(sa_prefix):
            s = sub[len(sa_prefix):]
            s = (s.replace("q_proj.", "attn_q.")
                  .replace("k_proj.", "attn_k.")
                  .replace("v_proj.", "attn_v.")
                  .replace("out_proj.", "attn_o.")
                  .replace("linear_q.", "attn_q.")
                  .replace("linear_k.", "attn_k.")
                  .replace("linear_v.", "attn_v.")
                  .replace("linear_out.", "attn_o."))
            return f"{pfx}.{s}"

    # bare self-attn names
    for pat, rep in (("q_proj.", "attn_q."), ("k_proj.", "attn_k."),
                     ("v_proj.", "attn_v."), ("out_proj.", "attn_o.")):
        if sub.startswith(pat):
            return f"{pfx}.{sub.replace(pat, rep, 1)}"

    # --- Cross-attention layer norm ---
    if sub in ("encoder_attn_layer_norm.weight", "cross_attn_layer_norm.weight",
               "norm2.weight", "ln2.weight"):
        return f"{pfx}.cross_ln.weight"
    if sub in ("encoder_attn_layer_norm.bias", "cross_attn_layer_norm.bias",
               "norm2.bias", "ln2.bias"):
        return f"{pfx}.cross_ln.bias"

    # --- Cross-attention projections ---
    for ca_prefix in ("encoder_attn.", "cross_attn.", "cross_attention."):
        if sub.startswith(ca_prefix):
            s = sub[len(ca_prefix):]
            s = (s.replace("q_proj.", "cross_q.")
                  .replace("k_proj.", "cross_k.")
                  .replace("v_proj.", "cross_v.")
                  .replace("out_proj.", "cross_o.")
                  .replace("linear_q.", "cross_q.")
                  .replace("linear_k.", "cross_k.")
                  .replace("linear_v.", "cross_v.")
                  .replace("linear_out.", "cross_o."))
            return f"{pfx}.{s}"

    # --- FFN layer norm ---
    if sub in ("final_layer_norm.weight", "ffn_layer_norm.weight",
               "norm3.weight", "ln3.weight"):
        return f"{pfx}.ffn_ln.weight"
    if sub in ("final_layer_norm.bias", "ffn_layer_norm.bias",
               "norm3.bias", "ln3.bias"):
        return f"{pfx}.ffn_ln.bias"

    # --- FFN projections ---
    if sub in ("fc1.weight", "linear1.weight", "ffn.linear_in.weight",
               "ffn.w_1.weight", "ffn_up.weight"):
        return f"{pfx}.ffn_up.weight"
    if sub in ("fc1.bias", "linear1.bias", "ffn.linear_in.bias",
               "ffn.w_1.bias", "ffn_up.bias"):
        return f"{pfx}.ffn_up.bias"
    if sub in ("fc2.weight", "linear2.weight", "ffn.linear_out.weight",
               "ffn.w_2.weight", "ffn_down.weight"):
        return f"{pfx}.ffn_down.weight"
    if sub in ("fc2.bias", "linear2.bias", "ffn.linear_out.bias",
               "ffn.w_2.bias", "ffn_down.bias"):
        return f"{pfx}.ffn_down.bias"

    return None


# ---------------------------------------------------------------------------
# F16 / F32 decision
# ---------------------------------------------------------------------------

def is_f32(gguf_name: str, shape: tuple) -> bool:
    """Return True if this tensor should be stored as F32.

    F32 tensors: biases, layer norms, batch norm stats, pos biases,
    1-D / small tensors, and the synthesised feature-extraction tensors.
    Large weight matrices are stored as F16.
    """
    if gguf_name.startswith("fe."):
        return True
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if gguf_name.endswith(".bn.mean") or gguf_name.endswith(".bn.var"):
        return True
    if "pos_bias_u" in gguf_name or "pos_bias_v" in gguf_name:
        return True
    if len(shape) <= 1:
        return True
    if gguf_name == "dec.pos.weight":
        return True
    return False


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_path: str, output_path: str) -> None:
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # 1. Resolve local directory
    model_dir = resolve_model_dir(input_path)
    print(f"  Model dir: {model_dir}")

    # 2. Load config and extract hparams
    config = load_config(model_dir)
    hp = extract_hparams(config)
    print(f"\n  Architecture:")
    print(f"    encoder: {hp['enc_n_layers']}L  d={hp['enc_d_model']}  "
          f"h={hp['enc_n_heads']}  hd={hp['enc_head_dim']}  "
          f"ffn={hp['enc_ffn_dim']}  conv_k={hp['enc_conv_k']}")
    print(f"    decoder: {hp['dec_n_layers']}L  d={hp['dec_d_model']}  "
          f"h={hp['dec_n_heads']}  hd={hp['dec_head_dim']}  "
          f"ffn={hp['dec_ffn_dim']}  max_ctx={hp['dec_max_ctx']}")
    print(f"    audio:   sr={hp['sample_rate']}  mels={hp['n_mels']}  "
          f"fft={hp['n_fft']}  hop={hp['hop_length']}  win={hp['win_length']}")

    # 3. Load tokenizer
    vocab = load_tokenizer(model_dir)
    if vocab:
        hp["vocab_size"] = len(vocab)
        print(f"  Vocabulary: {len(vocab)} tokens")
    else:
        print(f"  Vocabulary: empty (will use vocab_size={hp['vocab_size']} from config)")

    # 4. Load safetensors weights
    print(f"\n  Loading weights...")
    sd = load_safetensors(model_dir)

    # 5. Synthesise feature extraction tensors
    n_freqs = hp["n_fft"] // 2 + 1
    mel_fb = make_mel_filterbank(hp["n_mels"], hp["n_fft"], hp["sample_rate"])
    hann_win = make_hann_window(hp["win_length"])
    print(f"  Synthesised mel_fb {mel_fb.shape}, window {hann_win.shape}")

    # 6. Write GGUF
    print(f"\n  Writing GGUF: {output_path}")
    writer = gguf.GGUFWriter(output_path, arch="cohere-transcribe")
    writer.add_name("cohere-transcribe-03-2026")

    # --- Metadata ---
    writer.add_uint32("cohere_transcribe.vocab_size",         hp["vocab_size"])
    writer.add_uint32("cohere_transcribe.encoder.n_layers",   hp["enc_n_layers"])
    writer.add_uint32("cohere_transcribe.encoder.d_model",    hp["enc_d_model"])
    writer.add_uint32("cohere_transcribe.encoder.n_heads",    hp["enc_n_heads"])
    writer.add_uint32("cohere_transcribe.encoder.head_dim",   hp["enc_head_dim"])
    writer.add_uint32("cohere_transcribe.encoder.ffn_dim",    hp["enc_ffn_dim"])
    writer.add_uint32("cohere_transcribe.encoder.conv_kernel",hp["enc_conv_k"])
    writer.add_uint32("cohere_transcribe.decoder.n_layers",   hp["dec_n_layers"])
    writer.add_uint32("cohere_transcribe.decoder.d_model",    hp["dec_d_model"])
    writer.add_uint32("cohere_transcribe.decoder.n_heads",    hp["dec_n_heads"])
    writer.add_uint32("cohere_transcribe.decoder.head_dim",   hp["dec_head_dim"])
    writer.add_uint32("cohere_transcribe.decoder.ffn_dim",    hp["dec_ffn_dim"])
    writer.add_uint32("cohere_transcribe.decoder.max_ctx",    hp["dec_max_ctx"])
    writer.add_uint32("cohere_transcribe.audio.sample_rate",  hp["sample_rate"])
    writer.add_uint32("cohere_transcribe.audio.n_mels",       hp["n_mels"])
    writer.add_uint32("cohere_transcribe.audio.n_fft",        hp["n_fft"])
    writer.add_uint32("cohere_transcribe.audio.hop_length",   hp["hop_length"])
    writer.add_uint32("cohere_transcribe.audio.win_length",   hp["win_length"])

    # --- Tokenizer ---
    if vocab:
        writer.add_array("tokenizer.ggml.tokens", vocab)

    # --- Synthesised feature-extraction tensors ---
    writer.add_tensor("fe.mel_fb", mel_fb)   # F32  [1, n_mels, n_freqs]
    writer.add_tensor("fe.window", hann_win) # F32  [win_length]

    # --- Model weights ---
    n_written = 0
    n_f16 = 0
    n_f32 = 0
    n_dropped = 0
    seen_gguf: set[str] = set()

    for pt_name in sorted(sd.keys()):
        gguf_name = remap_name(pt_name, hp)
        if gguf_name is None:
            n_dropped += 1
            continue

        # Skip duplicates (shouldn't happen with sorted keys, but be safe)
        if gguf_name in seen_gguf:
            print(f"  [warn] duplicate GGUF name {gguf_name!r} from {pt_name!r}, skipping",
                  file=sys.stderr)
            continue
        seen_gguf.add(gguf_name)

        t = sd[pt_name]
        if t.dtype == np.float64:
            t = t.astype(np.float32)

        shape = t.shape
        if is_f32(gguf_name, shape):
            t = t.astype(np.float32)
            n_f32 += 1
        else:
            t = t.astype(np.float16)
            n_f16 += 1

        writer.add_tensor(gguf_name, t)
        n_written += 1

        if n_written <= 30 or n_written % 100 == 0:
            print(f"  [{n_written:4d}] {gguf_name:60s}  {str(shape):25s}  {t.dtype}")

    print(f"\n  Tensors: {n_written} written  (F16={n_f16}, F32={n_f32}), "
          f"{n_dropped} dropped, 2 synthesised (mel_fb + window)")

    # Sanity-check: warn if key enc.proj.weight is missing
    if "enc.proj.weight" not in seen_gguf:
        print("  [warn] enc.proj.weight not found — check tensor name mapping", file=sys.stderr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(output_path)
    print(f"\nDone: {output_path}  ({out_size / 1e9:.2f} GB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CohereForAI/cohere-transcribe-03-2026 safetensors → GGUF F16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From HuggingFace (requires login + license acceptance):
  python models/convert-cohere-asr-to-gguf.py \\
      --input CohereForAI/cohere-transcribe-03-2026 \\
      --output /mnt/storage/models/cohere-transcribe.gguf

  # From a local directory:
  python models/convert-cohere-asr-to-gguf.py \\
      --input /mnt/storage/cohere-transcribe-03-2026 \\
      --output /mnt/storage/models/cohere-transcribe.gguf
""",
    )
    p.add_argument(
        "--input", required=True,
        help="HuggingFace repo ID (e.g. CohereForAI/cohere-transcribe-03-2026) "
             "or local path to extracted model directory",
    )
    p.add_argument(
        "--output", required=True,
        help="Output GGUF file path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
