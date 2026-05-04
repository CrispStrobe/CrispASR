#!/usr/bin/env python3
"""
Convert facebook/m2m100_418M (and compatible variants like m2m100_1.2B,
wmt21-dense-24-wide) from HuggingFace PyTorch format to GGUF F16.

Architecture: M2M100ForConditionalGeneration
  - Encoder-decoder transformer, pre-norm, ReLU FFN
  - Shared embedding (encoder + decoder + lm_head share one table)
  - Sinusoidal positional embeddings (generated, not stored in weights)
  - SentencePiece BPE tokenizer with 100 language codes

Usage:
  python convert-m2m100-to-gguf.py \\
      --input /path/to/m2m100_418m \\
      --output /path/to/m2m100-418m-f16.gguf

GGUF tensor naming convention:

  shared.embed.weight                            F16  (vocab_size, d_model)
  enc.pos_emb                                    F32  (max_pos+2, d_model)
  dec.pos_emb                                    F32  (max_pos+2, d_model)

  enc.blk.N.attn_{q,k,v,o}.{weight,bias}        F16 / F32
  enc.blk.N.attn_ln.{weight,bias}               F32
  enc.blk.N.ffn_up.{weight,bias}                F16 / F32
  enc.blk.N.ffn_down.{weight,bias}              F16 / F32
  enc.blk.N.ffn_ln.{weight,bias}                F32
  enc.out_ln.{weight,bias}                       F32

  dec.blk.N.attn_{q,k,v,o}.{weight,bias}        F16 / F32
  dec.blk.N.attn_ln.{weight,bias}               F32
  dec.blk.N.cross_{q,k,v,o}.{weight,bias}       F16 / F32
  dec.blk.N.cross_ln.{weight,bias}              F32
  dec.blk.N.ffn_up.{weight,bias}                F16 / F32
  dec.blk.N.ffn_down.{weight,bias}              F16 / F32
  dec.blk.N.ffn_ln.{weight,bias}                F32
  dec.out_ln.{weight,bias}                       F32

GGUF metadata:
  general.architecture            = "m2m100"
  m2m100.vocab_size               = 128112
  m2m100.d_model                  = 1024
  m2m100.encoder.n_layers         = 12
  m2m100.encoder.n_heads          = 16
  m2m100.encoder.ffn_dim          = 4096
  m2m100.decoder.n_layers         = 12
  m2m100.decoder.n_heads          = 16
  m2m100.decoder.ffn_dim          = 4096
  m2m100.max_position_embeddings  = 1024
  m2m100.scale_embedding          = 1
  m2m100.bos_token_id             = 0
  m2m100.eos_token_id             = 2
  m2m100.pad_token_id             = 1
  m2m100.decoder_start_token_id   = 2
  tokenizer.ggml.model            = "m2m100"
  tokenizer.ggml.tokens           = [<128112 strings>]
  tokenizer.ggml.scores           = [<128112 floats>]
  m2m100.lang_codes               = ["af", "am", ...]
  m2m100.lang_token_ids           = [128004, 128005, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")
try:
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    import sentencepiece as spm
except ImportError:
    sys.exit("pip install sentencepiece")


# ---------------------------------------------------------------------------
# Language codes — the 100 languages supported by M2M-100
# These are the ISO 639-1/3 codes that map to tokens __XX__ at IDs 128004+
# ---------------------------------------------------------------------------

LANG_CODES = [
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br",
    "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu",
    "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo",
    "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb",
    "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr",
    "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl",
    "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq",
    "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr",
    "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu",
]

# Lang tokens start at ID 128004 (after 128000 SPM pieces + 4 special tokens)
LANG_TOKEN_ID_START = 128004


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings (M2M-100 / BART style)
# ---------------------------------------------------------------------------


def make_sinusoidal_embeddings(
    num_positions: int, d_model: int, padding_idx: int = 1
) -> np.ndarray:
    """
    Generate sinusoidal position embeddings as used in M2M-100.

    Row at padding_idx is left as zeros.
    Positions 0..num_positions-1 are filled with sin/cos, skipping padding_idx.

    Returns float32 array of shape (num_positions, d_model).
    """
    half = d_model // 2
    emb = np.zeros((num_positions, d_model), dtype=np.float32)
    positions = np.arange(num_positions)
    dim_idx = np.arange(half)
    freq = np.exp(dim_idx * -(np.log(10000.0) / half))
    for p in positions:
        if p == padding_idx:
            continue
        emb[p, :half] = np.sin(p * freq)
        emb[p, half:] = np.cos(p * freq)
    return emb


# ---------------------------------------------------------------------------
# Tensor name remapping: PyTorch → GGUF
# ---------------------------------------------------------------------------


def remap_name(pt_name: str) -> str | None:
    """
    Map HuggingFace M2M-100 state-dict keys to GGUF tensor names.
    Returns None for tensors we deliberately skip.
    """
    n = pt_name

    # Shared embedding — keep only the canonical copy
    if n == "model.shared.weight":
        return "shared.embed.weight"

    # lm_head is tied to shared embedding — skip the duplicate
    if n == "lm_head.weight":
        return None

    # Encoder embed_tokens is also tied — skip
    if n == "model.encoder.embed_tokens.weight":
        return None

    # Decoder embed_tokens is also tied — skip
    if n == "model.decoder.embed_tokens.weight":
        return None

    # Encoder global layer norm
    if n == "model.encoder.layer_norm.weight":
        return "enc.out_ln.weight"
    if n == "model.encoder.layer_norm.bias":
        return "enc.out_ln.bias"

    # Decoder global layer norm
    if n == "model.decoder.layer_norm.weight":
        return "dec.out_ln.weight"
    if n == "model.decoder.layer_norm.bias":
        return "dec.out_ln.bias"

    # Encoder layers
    if n.startswith("model.encoder.layers."):
        rest = n[len("model.encoder.layers."):]
        layer_id, sub = rest.split(".", 1)
        sub = _remap_enc_sub(sub)
        if sub is None:
            print(f"  [warn] unmapped encoder sub-key: {n}", file=sys.stderr)
            return None
        return f"enc.blk.{layer_id}.{sub}"

    # Decoder layers
    if n.startswith("model.decoder.layers."):
        rest = n[len("model.decoder.layers."):]
        layer_id, sub = rest.split(".", 1)
        sub = _remap_dec_sub(sub)
        if sub is None:
            print(f"  [warn] unmapped decoder sub-key: {n}", file=sys.stderr)
            return None
        return f"dec.blk.{layer_id}.{sub}"

    print(f"  [warn] unmapped tensor: {n}", file=sys.stderr)
    return None


def _remap_enc_sub(sub: str) -> str | None:
    """Map encoder layer sub-key to GGUF suffix."""
    # Self-attention projections
    for proj, out in [("q_proj", "attn_q"), ("k_proj", "attn_k"),
                      ("v_proj", "attn_v"), ("out_proj", "attn_o")]:
        if sub == f"self_attn.{proj}.weight":
            return f"{out}.weight"
        if sub == f"self_attn.{proj}.bias":
            return f"{out}.bias"
    # Self-attention layer norm
    if sub == "self_attn_layer_norm.weight":
        return "attn_ln.weight"
    if sub == "self_attn_layer_norm.bias":
        return "attn_ln.bias"
    # FFN
    if sub == "fc1.weight":
        return "ffn_up.weight"
    if sub == "fc1.bias":
        return "ffn_up.bias"
    if sub == "fc2.weight":
        return "ffn_down.weight"
    if sub == "fc2.bias":
        return "ffn_down.bias"
    # FFN layer norm
    if sub == "final_layer_norm.weight":
        return "ffn_ln.weight"
    if sub == "final_layer_norm.bias":
        return "ffn_ln.bias"
    return None


def _remap_dec_sub(sub: str) -> str | None:
    """Map decoder layer sub-key to GGUF suffix."""
    # Self-attention projections
    for proj, out in [("q_proj", "attn_q"), ("k_proj", "attn_k"),
                      ("v_proj", "attn_v"), ("out_proj", "attn_o")]:
        if sub == f"self_attn.{proj}.weight":
            return f"{out}.weight"
        if sub == f"self_attn.{proj}.bias":
            return f"{out}.bias"
    # Self-attention layer norm
    if sub == "self_attn_layer_norm.weight":
        return "attn_ln.weight"
    if sub == "self_attn_layer_norm.bias":
        return "attn_ln.bias"
    # Cross-attention projections
    for proj, out in [("q_proj", "cross_q"), ("k_proj", "cross_k"),
                      ("v_proj", "cross_v"), ("out_proj", "cross_o")]:
        if sub == f"encoder_attn.{proj}.weight":
            return f"{out}.weight"
        if sub == f"encoder_attn.{proj}.bias":
            return f"{out}.bias"
    # Cross-attention layer norm
    if sub == "encoder_attn_layer_norm.weight":
        return "cross_ln.weight"
    if sub == "encoder_attn_layer_norm.bias":
        return "cross_ln.bias"
    # FFN
    if sub == "fc1.weight":
        return "ffn_up.weight"
    if sub == "fc1.bias":
        return "ffn_up.bias"
    if sub == "fc2.weight":
        return "ffn_down.weight"
    if sub == "fc2.bias":
        return "ffn_down.bias"
    # FFN layer norm
    if sub == "final_layer_norm.weight":
        return "ffn_ln.weight"
    if sub == "final_layer_norm.bias":
        return "ffn_ln.bias"
    return None


# ---------------------------------------------------------------------------
# Dtype policy
# ---------------------------------------------------------------------------


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Return True if this tensor should be stored as F32."""
    # Positional embeddings
    if "pos_emb" in gguf_name:
        return True
    # Biases
    if gguf_name.endswith(".bias"):
        return True
    # Layer norms (weight/bias)
    if "_ln." in gguf_name or "out_ln." in gguf_name:
        return True
    # 1-D tensors (safety net)
    if len(shape) <= 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Tokenizer / vocabulary construction
# ---------------------------------------------------------------------------


def build_vocab(model_dir: Path) -> tuple[list[str], list[float]]:
    """
    Build the full M2M-100 vocabulary token list and score list.

    Token ID layout:
      0 .. 127999   : tokens from vocab.json (IDs 0-127999), covering SPM + extras
      128000..128003: extra tokens from vocab.json not in SPM
      128004..128103: language tokens __XX__ (100 langs)
      128104..128111: padding tokens (empty string) to reach vocab_size=128112

    Scores:
      - IDs with corresponding SPM entry: use SPM score
      - Language tokens: score 0.0
      - Padding: score 0.0
      - Special tokens (pad, bos, eos, unk): score 0.0

    Returns:
      tokens: list of 128112 token strings
      scores: list of 128112 floats
    """
    vocab_json_path = model_dir / "vocab.json"
    spm_path = model_dir / "sentencepiece.bpe.model"

    if not vocab_json_path.exists():
        sys.exit(f"Missing vocab.json at {vocab_json_path}")
    if not spm_path.exists():
        sys.exit(f"Missing sentencepiece.bpe.model at {spm_path}")

    with open(vocab_json_path) as f:
        vj = json.load(f)

    # Build id-to-token from vocab.json (authoritative for IDs 0..128003)
    inv_vj: dict[int, str] = {v: k for k, v in vj.items()}

    # Load SPM for scores (128000 pieces, but IDs in SPM are offset from M2M IDs)
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    spm_size = sp.get_piece_size()  # 128000

    # M2M-100 ID layout vs SPM layout:
    #   M2M ID 0 = <s>  (bos)   → SPM ID 1 (<s>)
    #   M2M ID 1 = <pad>        → no SPM equivalent (score 0)
    #   M2M ID 2 = </s> (eos)   → SPM ID 2 (</s>)
    #   M2M ID 3 = <unk>        → SPM ID 0 (<unk>)
    #   M2M IDs 4..127999 → SPM IDs 3..127996 (offset -1 due to inserted <pad>)
    #   M2M IDs 128000..128003 → extra vocab.json tokens, score 0

    def get_spm_score(m2m_id: int) -> float:
        """Map M2M token ID to SPM score, or 0.0 if unavailable."""
        if m2m_id == 0:   # <s> → SPM 1
            return sp.get_score(1)
        if m2m_id == 1:   # <pad> → no SPM entry
            return 0.0
        if m2m_id == 2:   # </s> → SPM 2
            return sp.get_score(2)
        if m2m_id == 3:   # <unk> → SPM 0
            return sp.get_score(0)
        # Regular tokens: M2M ID K maps to SPM ID K-1
        spm_id = m2m_id - 1
        if 0 <= spm_id < spm_size:
            return sp.get_score(spm_id)
        return 0.0

    # Determine total vocab size
    vocab_size_base = len(inv_vj)   # 128004
    n_lang = len(LANG_CODES)        # 100
    # Target is 128112 (from config)
    vocab_size = 128112

    tokens: list[str] = []
    scores: list[float] = []

    # IDs 0 .. 128003 from vocab.json
    for i in range(vocab_size_base):
        tok = inv_vj.get(i, f"[UNK_{i}]")
        tokens.append(tok)
        scores.append(get_spm_score(i))

    # IDs 128004 .. 128103 — language tokens
    for code in LANG_CODES:
        tokens.append(f"__{code}__")
        scores.append(0.0)

    # Pad to vocab_size with empty strings
    while len(tokens) < vocab_size:
        tokens.append("")
        scores.append(0.0)

    assert len(tokens) == vocab_size, f"expected {vocab_size}, got {len(tokens)}"
    assert len(scores) == vocab_size

    print(f"  vocab:   {vocab_size} tokens ({vocab_size_base} from vocab.json, "
          f"{n_lang} lang codes, {vocab_size - vocab_size_base - n_lang} padding)")
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

    vocab_size    = cfg.get("vocab_size", 128112)
    d_model       = cfg.get("d_model", 1024)
    enc_n_layers  = cfg.get("encoder_layers", 12)
    dec_n_layers  = cfg.get("decoder_layers", 12)
    enc_n_heads   = cfg.get("encoder_attention_heads", 16)
    dec_n_heads   = cfg.get("decoder_attention_heads", 16)
    enc_ffn_dim   = cfg.get("encoder_ffn_dim", 4096)
    dec_ffn_dim   = cfg.get("decoder_ffn_dim", 4096)
    max_pos       = cfg.get("max_position_embeddings", 1024)
    scale_embed   = int(cfg.get("scale_embedding", True))
    bos_id        = cfg.get("bos_token_id", 0)
    eos_id        = cfg.get("eos_token_id", 2)
    pad_id        = cfg.get("pad_token_id", 1)
    dec_start_id  = cfg.get("decoder_start_token_id", 2)

    print(f"  arch:    M2M-100  d_model={d_model}  "
          f"enc={enc_n_layers}L×{enc_n_heads}H  "
          f"dec={dec_n_layers}L×{dec_n_heads}H  "
          f"vocab={vocab_size}")

    # ---- weights ----
    weights_path = input_dir / "pytorch_model.bin"
    if not weights_path.exists():
        sys.exit(f"Missing pytorch_model.bin at {weights_path}")
    print(f"  weights: {weights_path}")
    sd = torch.load(str(weights_path), map_location="cpu", weights_only=True, mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    print(f"  tensors: {len(sd)} in state dict")

    # ---- vocabulary ----
    tokens, scores = build_vocab(input_dir)

    # ---- sinusoidal positional embeddings ----
    # M2M-100 uses offset=2 (padding_idx=1); shape = (max_pos+2, d_model)
    pos_emb_rows = max_pos + 2
    print(f"  pos_emb: generating sinusoidal ({pos_emb_rows}, {d_model})  padding_idx=1")
    pos_emb = make_sinusoidal_embeddings(pos_emb_rows, d_model, padding_idx=1)

    # ---- write GGUF ----
    print(f"\nWriting: {out_path}")
    writer = gguf.GGUFWriter(str(out_path), arch="m2m100")

    # Metadata
    writer.add_uint32("m2m100.vocab_size",               vocab_size)
    writer.add_uint32("m2m100.d_model",                  d_model)
    writer.add_uint32("m2m100.encoder.n_layers",         enc_n_layers)
    writer.add_uint32("m2m100.encoder.n_heads",          enc_n_heads)
    writer.add_uint32("m2m100.encoder.ffn_dim",          enc_ffn_dim)
    writer.add_uint32("m2m100.decoder.n_layers",         dec_n_layers)
    writer.add_uint32("m2m100.decoder.n_heads",          dec_n_heads)
    writer.add_uint32("m2m100.decoder.ffn_dim",          dec_ffn_dim)
    writer.add_uint32("m2m100.max_position_embeddings",  max_pos)
    writer.add_uint32("m2m100.scale_embedding",          scale_embed)
    writer.add_uint32("m2m100.bos_token_id",             bos_id)
    writer.add_uint32("m2m100.eos_token_id",             eos_id)
    writer.add_uint32("m2m100.pad_token_id",             pad_id)
    writer.add_uint32("m2m100.decoder_start_token_id",   dec_start_id)

    # Tokenizer
    writer.add_string("tokenizer.ggml.model", "m2m100")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", scores)

    # Language codes and their token IDs
    lang_token_ids = [LANG_TOKEN_ID_START + i for i in range(len(LANG_CODES))]
    writer.add_array("m2m100.lang_codes",    LANG_CODES)
    writer.add_array("m2m100.lang_token_ids", lang_token_ids)

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

    # ---- positional embeddings (synthetic) ----
    write_tensor("enc.pos_emb", pos_emb)
    write_tensor("dec.pos_emb", pos_emb)

    # ---- model weights ----
    for pt_name in sorted(sd.keys()):
        gguf_name = remap_name(pt_name)
        if gguf_name is None:
            continue
        t = sd[pt_name].cpu().float().numpy()
        write_tensor(gguf_name, t)

    print(f"\n  total tensors written: {n_written}  (F16: {n_f16}, F32: {n_f32})")

    # Sanity check: expected tensor count
    # enc: N layers × (4 attn w + 4 attn b + 2 attn_ln + 2 ffn_up + 2 ffn_down + 2 ffn_ln) = N×16
    # dec: N layers × (4 attn w/b + 2 attn_ln + 4 cross w/b + 2 cross_ln + 2 ffn_up + 2 ffn_down + 2 ffn_ln) = N×26
    # global: shared.embed(1) + enc.out_ln(2) + dec.out_ln(2) + enc.pos_emb(1) + dec.pos_emb(1) = 7
    expected = (enc_n_layers * 16) + (dec_n_layers * 26) + 7
    if n_written == expected:
        print(f"  tensor count check: OK ({n_written} == {expected})")
    else:
        print(f"  [warn] tensor count: got {n_written}, expected ~{expected}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert facebook/m2m100_418M (HuggingFace) → GGUF F16",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="path to HuggingFace model directory (containing pytorch_model.bin, "
             "config.json, sentencepiece.bpe.model, vocab.json)",
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
