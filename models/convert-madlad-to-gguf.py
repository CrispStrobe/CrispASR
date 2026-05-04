#!/usr/bin/env python3
"""
Convert google/madlad400-3b-mt (T5ForConditionalGeneration) from safetensors
to GGUF F16.

Architecture: T5ForConditionalGeneration
  - Encoder-decoder transformer, pre-norm (RMSNorm), gated-GELU FFN
  - Relative attention bias (learnable, only stored in layer 0)
  - Shared embedding table for encoder; separate lm_head (tie_word_embeddings=false)
  - SentencePiece Unigram tokenizer (spiece.model, 256000 tokens)

Usage:
  python convert-madlad-to-gguf.py \\
      --input /path/to/madlad_3b \\
      --output /path/to/madlad400-3b-mt-f16.gguf

GGUF tensor naming convention:

  shared.embed.weight                                F16  (vocab_size, d_model)
  lm_head.weight                                     F16  (vocab_size, d_model)

  enc.rel_bias.weight                                F32  (rel_buckets, n_heads)
  enc.final_rms.weight                               F32  (d_model,)

  enc.blk.N.attn_{q,k,v,o}.weight                   F16
  enc.blk.N.attn_rms.weight                          F32
  enc.blk.N.ffn_{gate,up,down}.weight                F16
  enc.blk.N.ffn_rms.weight                           F32

  dec.rel_bias.weight                                F32  (rel_buckets, n_heads)
  dec.final_rms.weight                               F32  (d_model,)

  dec.blk.N.attn_{q,k,v,o}.weight                   F16
  dec.blk.N.attn_rms.weight                          F32
  dec.blk.N.cross_{q,k,v,o}.weight                  F16
  dec.blk.N.cross_rms.weight                         F32
  dec.blk.N.ffn_{gate,up,down}.weight                F16
  dec.blk.N.ffn_rms.weight                           F32

GGUF metadata:
  general.architecture                   = "t5"
  t5.vocab_size                          = 256000
  t5.d_model                             = 1024
  t5.d_kv                                = 128
  t5.d_ff                                = 8192
  t5.n_heads                             = 16
  t5.encoder.n_layers                    = 32
  t5.decoder.n_layers                    = 32
  t5.relative_attention_num_buckets      = 32
  t5.relative_attention_max_distance     = 128
  t5.layer_norm_epsilon                  = 1e-6  (f32)
  t5.tie_word_embeddings                 = 0
  t5.feed_forward_proj                   = "gated-gelu"
  t5.eos_token_id                        = 2
  t5.pad_token_id                        = 1
  t5.decoder_start_token_id              = 0
  tokenizer.ggml.model                   = "t5"
  tokenizer.ggml.tokens                  = [<256000 strings>]
  tokenizer.ggml.scores                  = [<256000 floats>]
"""

from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")


# ---------------------------------------------------------------------------
# Safetensors reader (no torch dependency)
# ---------------------------------------------------------------------------

# Map safetensors dtype strings to numpy dtypes
_ST_DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float32,   # will re-read and convert
    "I32": np.int32,
    "I64": np.int64,
    "I8":  np.int8,
    "U8":  np.uint8,
}

# bfloat16 → float32 conversion helper
def _bf16_bytes_to_f32(raw: bytes) -> np.ndarray:
    """Convert raw bfloat16 bytes to a float32 array."""
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


class SafetensorsReader:
    """
    Memory-mapped safetensors reader.  Reads the JSON header once; individual
    tensors are decoded on demand from the mmap without loading the whole file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        header_len = struct.unpack_from("<Q", self._mm, 0)[0]
        header_raw = bytes(self._mm[8 : 8 + header_len])
        self._header: dict = json.loads(header_raw)
        self._data_offset = 8 + header_len

        self._tensors = {
            k: v for k, v in self._header.items() if k != "__metadata__"
        }
        print(f"  safetensors: {len(self._tensors)} tensors  "
              f"header_len={header_len}  data_offset={self._data_offset}")

    def keys(self) -> list[str]:
        return list(self._tensors.keys())

    def get(self, name: str) -> np.ndarray:
        """Return the named tensor as a float32 numpy array."""
        info = self._tensors[name]
        dtype_str: str = info["dtype"]
        shape: list[int] = info["shape"]
        start, end = info["data_offsets"]

        raw = bytes(self._mm[self._data_offset + start : self._data_offset + end])

        if dtype_str == "BF16":
            arr = _bf16_bytes_to_f32(raw)
        else:
            np_dtype = _ST_DTYPE_MAP.get(dtype_str)
            if np_dtype is None:
                raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")
            arr = np.frombuffer(raw, dtype=np_dtype).copy()

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if shape:
            arr = arr.reshape(shape)
        return arr

    def close(self) -> None:
        self._mm.close()
        self._f.close()


# ---------------------------------------------------------------------------
# Tensor name remapping: safetensors (HuggingFace T5) → GGUF
# ---------------------------------------------------------------------------


def remap_name(pt_name: str) -> str | None:
    """
    Map HuggingFace T5 state-dict keys to GGUF tensor names.
    Returns None for tensors we deliberately skip.
    """
    n = pt_name

    # Shared / global embeddings.
    # MADLAD-400 (tie_word_embeddings=False) ships the shared encoder/
    # decoder embedding under `decoder.embed_tokens.weight` and a
    # separate `lm_head.weight`. flan-t5-small / -base / -large etc.
    # (tie_word_embeddings=True) ship it under `shared.weight` instead;
    # in both cases the runtime expects `shared.embed.weight`.
    if n in ("decoder.embed_tokens.weight", "shared.weight", "encoder.embed_tokens.weight"):
        return "shared.embed.weight"
    if n == "lm_head.weight":
        return "lm_head.weight"

    # Encoder global
    if n == "encoder.final_layer_norm.weight":
        return "enc.final_rms.weight"

    # Decoder global
    if n == "decoder.final_layer_norm.weight":
        return "dec.final_rms.weight"

    # Encoder layers: encoder.block.N.layer.X.*
    if n.startswith("encoder.block."):
        rest = n[len("encoder.block."):]               # "N.layer.X.*"
        layer_id, tail = rest.split(".", 1)             # ("N", "layer.X.*")
        layer_n = int(layer_id)
        result = _remap_enc_block(layer_n, tail)
        if result is None:
            print(f"  [warn] unmapped encoder tensor: {n}", file=sys.stderr)
        return result

    # Decoder layers: decoder.block.N.layer.X.*
    if n.startswith("decoder.block."):
        rest = n[len("decoder.block."):]
        layer_id, tail = rest.split(".", 1)
        layer_n = int(layer_id)
        result = _remap_dec_block(layer_n, tail)
        if result is None:
            print(f"  [warn] unmapped decoder tensor: {n}", file=sys.stderr)
        return result

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


def _remap_enc_block(layer_n: int, tail: str) -> str | None:
    """Map encoder block sub-key to GGUF name."""
    prefix = f"enc.blk.{layer_n}"

    # layer.0 = self-attention
    if tail.startswith("layer.0."):
        sub = tail[len("layer.0."):]
        # Relative attention bias: only stored in layer 0
        if sub == "SelfAttention.relative_attention_bias.weight":
            return "enc.rel_bias.weight"
        for proj, out in [("q", "attn_q"), ("k", "attn_k"),
                          ("v", "attn_v"), ("o", "attn_o")]:
            if sub == f"SelfAttention.{proj}.weight":
                return f"{prefix}.{out}.weight"
        if sub == "layer_norm.weight":
            return f"{prefix}.attn_rms.weight"
        return None

    # layer.1 = FFN (encoder has no cross-attention)
    if tail.startswith("layer.1."):
        sub = tail[len("layer.1."):]
        if sub == "DenseReluDense.wi_0.weight":
            return f"{prefix}.ffn_gate.weight"
        if sub == "DenseReluDense.wi_1.weight":
            return f"{prefix}.ffn_up.weight"
        if sub == "DenseReluDense.wo.weight":
            return f"{prefix}.ffn_down.weight"
        if sub == "layer_norm.weight":
            return f"{prefix}.ffn_rms.weight"
        return None

    return None


def _remap_dec_block(layer_n: int, tail: str) -> str | None:
    """Map decoder block sub-key to GGUF name."""
    prefix = f"dec.blk.{layer_n}"

    # layer.0 = self-attention
    if tail.startswith("layer.0."):
        sub = tail[len("layer.0."):]
        if sub == "SelfAttention.relative_attention_bias.weight":
            return "dec.rel_bias.weight"
        for proj, out in [("q", "attn_q"), ("k", "attn_k"),
                          ("v", "attn_v"), ("o", "attn_o")]:
            if sub == f"SelfAttention.{proj}.weight":
                return f"{prefix}.{out}.weight"
        if sub == "layer_norm.weight":
            return f"{prefix}.attn_rms.weight"
        return None

    # layer.1 = cross-attention
    if tail.startswith("layer.1."):
        sub = tail[len("layer.1."):]
        for proj, out in [("q", "cross_q"), ("k", "cross_k"),
                          ("v", "cross_v"), ("o", "cross_o")]:
            if sub == f"EncDecAttention.{proj}.weight":
                return f"{prefix}.{out}.weight"
        if sub == "layer_norm.weight":
            return f"{prefix}.cross_rms.weight"
        return None

    # layer.2 = FFN
    if tail.startswith("layer.2."):
        sub = tail[len("layer.2."):]
        if sub == "DenseReluDense.wi_0.weight":
            return f"{prefix}.ffn_gate.weight"
        if sub == "DenseReluDense.wi_1.weight":
            return f"{prefix}.ffn_up.weight"
        if sub == "DenseReluDense.wo.weight":
            return f"{prefix}.ffn_down.weight"
        if sub == "layer_norm.weight":
            return f"{prefix}.ffn_rms.weight"
        return None

    return None


# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Return True if this tensor should be stored as F32."""
    # RMS norms (1-D)
    if gguf_name.endswith("_rms.weight"):
        return True
    # Relative attention bias (2-D but small)
    if gguf_name.endswith("rel_bias.weight"):
        return True
    # Safety net: any remaining 1-D tensor
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Tokenizer vocabulary construction
# ---------------------------------------------------------------------------


def build_vocab(model_dir: Path) -> tuple[list[str], list[float]]:
    """
    Build the MADLAD-400 / T5 vocabulary from spiece.model (SentencePiece).

    The SPM model contains exactly vocab_size=256000 pieces with their
    log-probability scores.  Token IDs are in SPM order (0-based).

    Returns:
      tokens: list of 256000 token strings
      scores: list of 256000 floats (SPM log-prob scores)
    """
    spm_path = model_dir / "spiece.model"
    if not spm_path.exists():
        sys.exit(f"Missing spiece.model at {spm_path}")

    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    n = sp.get_piece_size()
    print(f"  spm:     {n} pieces")

    tokens: list[str] = []
    scores: list[float] = []
    for i in range(n):
        tokens.append(sp.id_to_piece(i))
        scores.append(sp.get_score(i))

    return tokens, scores


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")

    # ---- config ----
    cfg_path = input_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"Missing config.json at {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    vocab_size         = cfg.get("vocab_size", 256000)
    d_model            = cfg.get("d_model", 1024)
    d_kv               = cfg.get("d_kv", 128)
    d_ff               = cfg.get("d_ff", 8192)
    n_heads            = cfg.get("num_heads", 16)
    enc_n_layers       = cfg.get("num_layers", 32)
    dec_n_layers       = cfg.get("num_decoder_layers", enc_n_layers)
    rel_buckets        = cfg.get("relative_attention_num_buckets", 32)
    rel_max_dist       = cfg.get("relative_attention_max_distance", 128)
    layer_norm_eps     = cfg.get("layer_norm_epsilon", 1e-6)
    tie_word_emb       = int(cfg.get("tie_word_embeddings", False))
    ff_proj            = cfg.get("feed_forward_proj", "gated-gelu")
    eos_id             = cfg.get("eos_token_id", 2)
    pad_id             = cfg.get("pad_token_id", 1)
    dec_start_id       = cfg.get("decoder_start_token_id", 0)

    print(f"  arch:    T5  d_model={d_model}  d_kv={d_kv}  d_ff={d_ff}  "
          f"n_heads={n_heads}  enc={enc_n_layers}L  dec={dec_n_layers}L  "
          f"vocab={vocab_size}  ff={ff_proj}")

    # ---- safetensors weights ----
    st_path = input_dir / "model.safetensors"
    if not st_path.exists():
        sys.exit(f"Missing model.safetensors at {st_path}")
    print(f"  weights: {st_path}  ({st_path.stat().st_size / 1e9:.2f} GB)")
    reader = SafetensorsReader(st_path)

    # ---- vocabulary ----
    tokens, scores = build_vocab(input_dir)
    if len(tokens) != vocab_size:
        print(f"  [warn] SPM size {len(tokens)} != config vocab_size {vocab_size}",
              file=sys.stderr)

    # ---- write GGUF ----
    print(f"\nWriting: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="t5")

    # Architecture metadata
    writer.add_uint32("t5.vocab_size",                        vocab_size)
    writer.add_uint32("t5.d_model",                           d_model)
    writer.add_uint32("t5.d_kv",                              d_kv)
    writer.add_uint32("t5.d_ff",                              d_ff)
    writer.add_uint32("t5.n_heads",                           n_heads)
    writer.add_uint32("t5.encoder.n_layers",                  enc_n_layers)
    writer.add_uint32("t5.decoder.n_layers",                  dec_n_layers)
    writer.add_uint32("t5.relative_attention_num_buckets",    rel_buckets)
    writer.add_uint32("t5.relative_attention_max_distance",   rel_max_dist)
    writer.add_float32("t5.layer_norm_epsilon",               float(layer_norm_eps))
    writer.add_uint32("t5.tie_word_embeddings",               tie_word_emb)
    writer.add_string("t5.feed_forward_proj",                 ff_proj)
    writer.add_uint32("t5.eos_token_id",                      eos_id)
    writer.add_uint32("t5.pad_token_id",                      pad_id)
    writer.add_uint32("t5.decoder_start_token_id",            dec_start_id)

    # Tokenizer
    writer.add_string("tokenizer.ggml.model", "t5")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", scores)

    n_written = 0
    n_f16 = 0
    n_f32 = 0

    def write_tensor(gguf_name: str, arr: np.ndarray) -> None:
        nonlocal n_written, n_f16, n_f32
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
        if n_written <= 30 or n_written % 50 == 0:
            print(f"  {gguf_name:60s}  {str(arr.shape):25s}  {arr.dtype}")

    # ---- model weights ----
    for pt_name in sorted(reader.keys()):
        gguf_name = remap_name(pt_name)
        if gguf_name is None:
            continue
        arr = reader.get(pt_name)
        write_tensor(gguf_name, arr)

    print(f"\n  total tensors written: {n_written}  (F16: {n_f16}, F32: {n_f32})")

    # ---- sanity check ----
    # Encoder per layer: 4 attn weights + 1 attn_rms + 3 ffn weights + 1 ffn_rms = 9
    # Plus enc.rel_bias (1) + enc.final_rms (1) = enc_n_layers * 9 + 2
    # Decoder per layer: 4 attn + 1 attn_rms + 4 cross + 1 cross_rms + 3 ffn + 1 ffn_rms = 14
    # Plus dec.rel_bias (1) + dec.final_rms (1) = dec_n_layers * 14 + 2
    # Global: shared.embed (1) + lm_head (1) = 2
    expected = (enc_n_layers * 9 + 2) + (dec_n_layers * 14 + 2) + 2
    if n_written == expected:
        print(f"  tensor count check: OK ({n_written} == {expected})")
    else:
        print(f"  [warn] tensor count: got {n_written}, expected {expected}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert google/madlad400-3b-mt (HuggingFace safetensors) → GGUF F16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="path to HuggingFace model directory (containing model.safetensors, "
             "config.json, spiece.model)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output GGUF file path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
