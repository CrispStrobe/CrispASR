#!/usr/bin/env python3
"""
Convert openbmb/VoxCPM2 (HF safetensors + audiovae.pth) → GGUF F16.

Architecture (from config.json):

  VoxCPM2 is a tokenizer-free diffusion autoregressive TTS system with
  4 sub-networks + AudioVAE. Pipeline: LocEnc → TSLM → RALM → LocDiT → VAE.

  Base LM (TSLM, 28 layers — MiniCPM-4 backbone):
    hidden_size          = 2048
    intermediate_size    = 6144
    num_attention_heads  = 16
    num_key_value_heads  = 2
    kv_channels          = 128
    vocab_size           = 73448
    max_position_embeddings = 32768
    rope_theta           = 10000
    rope_scaling         = LongRoPE (64 factors)
    rms_norm_eps         = 1e-5
    use_mup              = false (in HF release)
    scale_emb            = 12
    dim_model_base       = 256
    scale_depth          = 1.4

  Residual LM (RALM, 8 layers — same MiniCPM-4 arch, no RoPE, vocab=0):
    hidden_size          = 2048
    Same FFN/attention dims as base LM.

  Local Encoder (LocEnc, 12 layers — bidirectional MiniCPM-4):
    hidden_dim           = 1024
    ffn_dim              = 4096
    num_heads            = 16
    kv_channels          = 128
    Input: patches of shape [B, T, P=4, D=64]
    Output: [B, T, 1024] (CLS token from each local window)

  Local DiT (LocDiT, 12 layers — bidirectional MiniCPM-4 + CFM):
    hidden_dim           = 1024
    ffn_dim              = 4096
    num_heads            = 16
    kv_channels          = 128
    CFM solver: Euler, sigma_min=1e-6, t_scheduler=log-norm
    CFM cfg_rate         = 2.0

  AudioVAE V2 (separate .pth, 377 MB):
    Encoder: dim=128, rates=[2,5,8,8], latent_dim=64
    Decoder: dim=2048, rates=[8,6,5,2,2,2], SR-conditioned (scale_bias)
    Input SR: 16 kHz, Output SR: 48 kHz
    Snake1d activations, weight-norm causal convolutions

  Projection layers:
    enc_to_lm_proj:      Linear(1024, 2048)
    lm_to_dit_proj:      Linear(2048, 1024)
    res_to_dit_proj:     Linear(2048, 1024)
    fusion_concat_proj:  Linear(4096, 2048)
    FSQ in_proj:         Linear(2048, 512)
    FSQ out_proj:        Linear(512, 2048)
    stop_proj:           Linear(2048, 2048)
    stop_head:           Linear(2048, 2, no bias)

  Tokenizer:
    LlamaTokenizerFast-based (SentencePiece BPE), vocab=73448
    Special tokens: <unk>=0, <s>=1, </s>=2,
      <|audio_start|>=101, <|audio_end|>=102,
      <|audio_prompt_start|>=103, <|audio_prompt_end|>=104

  Tensor count: ~650+ tensors

Usage:
    python models/convert-voxcpm2-to-gguf.py \\
        --input openbmb/VoxCPM2 \\
        --output voxcpm2-f16.gguf

    # or from a local directory:
    python models/convert-voxcpm2-to-gguf.py \\
        --input /path/to/VoxCPM2 \\
        --output voxcpm2-f16.gguf
"""

from __future__ import annotations

import argparse
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

# Check for torch — needed for BF16 handling
_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Tensor name remapping (PyTorch state_dict → GGUF)
# ---------------------------------------------------------------------------

# Direct (non-patterned) mappings
DIRECT = {
    # Base LM top-level
    "base_lm.embed_tokens.weight": "tslm.token_embd.weight",
    "base_lm.norm.weight": "tslm.output_norm.weight",
    # Residual LM top-level (no embed_tokens since vocab=0)
    "residual_lm.norm.weight": "ralm.output_norm.weight",
    # Local Encoder
    "feat_encoder.special_token": "locenc.cls_token",
    "feat_encoder.in_proj.weight": "locenc.in_proj.weight",
    "feat_encoder.in_proj.bias": "locenc.in_proj.bias",
    "feat_encoder.encoder.norm.weight": "locenc.output_norm.weight",
    # Local DiT
    "feat_decoder.estimator.in_proj.weight": "locdit.in_proj.weight",
    "feat_decoder.estimator.in_proj.bias": "locdit.in_proj.bias",
    "feat_decoder.estimator.cond_proj.weight": "locdit.cond_proj.weight",
    "feat_decoder.estimator.cond_proj.bias": "locdit.cond_proj.bias",
    "feat_decoder.estimator.out_proj.weight": "locdit.out_proj.weight",
    "feat_decoder.estimator.out_proj.bias": "locdit.out_proj.bias",
    "feat_decoder.estimator.time_mlp.linear_1.weight": "locdit.time_mlp.0.weight",
    "feat_decoder.estimator.time_mlp.linear_1.bias": "locdit.time_mlp.0.bias",
    "feat_decoder.estimator.time_mlp.linear_2.weight": "locdit.time_mlp.1.weight",
    "feat_decoder.estimator.time_mlp.linear_2.bias": "locdit.time_mlp.1.bias",
    "feat_decoder.estimator.delta_time_mlp.linear_1.weight": "locdit.dt_mlp.0.weight",
    "feat_decoder.estimator.delta_time_mlp.linear_1.bias": "locdit.dt_mlp.0.bias",
    "feat_decoder.estimator.delta_time_mlp.linear_2.weight": "locdit.dt_mlp.1.weight",
    "feat_decoder.estimator.delta_time_mlp.linear_2.bias": "locdit.dt_mlp.1.bias",
    "feat_decoder.estimator.decoder.norm.weight": "locdit.output_norm.weight",
    # Projection layers
    "enc_to_lm_proj.weight": "proj.enc_to_lm.weight",
    "enc_to_lm_proj.bias": "proj.enc_to_lm.bias",
    "lm_to_dit_proj.weight": "proj.lm_to_dit.weight",
    "lm_to_dit_proj.bias": "proj.lm_to_dit.bias",
    "res_to_dit_proj.weight": "proj.res_to_dit.weight",
    "res_to_dit_proj.bias": "proj.res_to_dit.bias",
    "fusion_concat_proj.weight": "proj.fusion.weight",
    "fusion_concat_proj.bias": "proj.fusion.bias",
    # FSQ (Finite Scalar Quantization)
    "fsq_layer.in_proj.weight": "fsq.in_proj.weight",
    "fsq_layer.in_proj.bias": "fsq.in_proj.bias",
    "fsq_layer.out_proj.weight": "fsq.out_proj.weight",
    "fsq_layer.out_proj.bias": "fsq.out_proj.bias",
    # Stop predictor
    "stop_proj.weight": "stop.proj.weight",
    "stop_proj.bias": "stop.proj.bias",
    "stop_head.weight": "stop.head.weight",
}

# Per-layer patterns for transformer blocks
# Base LM (TSLM) — 28 layers
TSLM_LAYER_PATTERNS = [
    (r"base_lm\.layers\.(\d+)\.input_layernorm\.weight", "tslm.blk.{}.attn_norm.weight"),
    (r"base_lm\.layers\.(\d+)\.self_attn\.q_proj\.weight", "tslm.blk.{}.attn_q.weight"),
    (r"base_lm\.layers\.(\d+)\.self_attn\.k_proj\.weight", "tslm.blk.{}.attn_k.weight"),
    (r"base_lm\.layers\.(\d+)\.self_attn\.v_proj\.weight", "tslm.blk.{}.attn_v.weight"),
    (r"base_lm\.layers\.(\d+)\.self_attn\.o_proj\.weight", "tslm.blk.{}.attn_output.weight"),
    (r"base_lm\.layers\.(\d+)\.post_attention_layernorm\.weight", "tslm.blk.{}.ffn_norm.weight"),
    (r"base_lm\.layers\.(\d+)\.mlp\.gate_proj\.weight", "tslm.blk.{}.ffn_gate.weight"),
    (r"base_lm\.layers\.(\d+)\.mlp\.up_proj\.weight", "tslm.blk.{}.ffn_up.weight"),
    (r"base_lm\.layers\.(\d+)\.mlp\.down_proj\.weight", "tslm.blk.{}.ffn_down.weight"),
]

# Residual LM (RALM) — 8 layers
RALM_LAYER_PATTERNS = [
    (r"residual_lm\.layers\.(\d+)\.input_layernorm\.weight", "ralm.blk.{}.attn_norm.weight"),
    (r"residual_lm\.layers\.(\d+)\.self_attn\.q_proj\.weight", "ralm.blk.{}.attn_q.weight"),
    (r"residual_lm\.layers\.(\d+)\.self_attn\.k_proj\.weight", "ralm.blk.{}.attn_k.weight"),
    (r"residual_lm\.layers\.(\d+)\.self_attn\.v_proj\.weight", "ralm.blk.{}.attn_v.weight"),
    (r"residual_lm\.layers\.(\d+)\.self_attn\.o_proj\.weight", "ralm.blk.{}.attn_output.weight"),
    (r"residual_lm\.layers\.(\d+)\.post_attention_layernorm\.weight", "ralm.blk.{}.ffn_norm.weight"),
    (r"residual_lm\.layers\.(\d+)\.mlp\.gate_proj\.weight", "ralm.blk.{}.ffn_gate.weight"),
    (r"residual_lm\.layers\.(\d+)\.mlp\.up_proj\.weight", "ralm.blk.{}.ffn_up.weight"),
    (r"residual_lm\.layers\.(\d+)\.mlp\.down_proj\.weight", "ralm.blk.{}.ffn_down.weight"),
]

# Local Encoder (LocEnc) — 12 layers
LOCENC_LAYER_PATTERNS = [
    (r"feat_encoder\.encoder\.layers\.(\d+)\.input_layernorm\.weight", "locenc.blk.{}.attn_norm.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight", "locenc.blk.{}.attn_q.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight", "locenc.blk.{}.attn_k.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight", "locenc.blk.{}.attn_v.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.self_attn\.o_proj\.weight", "locenc.blk.{}.attn_output.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.post_attention_layernorm\.weight", "locenc.blk.{}.ffn_norm.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.mlp\.gate_proj\.weight", "locenc.blk.{}.ffn_gate.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.mlp\.up_proj\.weight", "locenc.blk.{}.ffn_up.weight"),
    (r"feat_encoder\.encoder\.layers\.(\d+)\.mlp\.down_proj\.weight", "locenc.blk.{}.ffn_down.weight"),
]

# Local DiT (LocDiT) — 12 layers
LOCDIT_LAYER_PATTERNS = [
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.input_layernorm\.weight", "locdit.blk.{}.attn_norm.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.weight", "locdit.blk.{}.attn_q.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.self_attn\.k_proj\.weight", "locdit.blk.{}.attn_k.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.weight", "locdit.blk.{}.attn_v.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.self_attn\.o_proj\.weight", "locdit.blk.{}.attn_output.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.post_attention_layernorm\.weight", "locdit.blk.{}.ffn_norm.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.mlp\.gate_proj\.weight", "locdit.blk.{}.ffn_gate.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.mlp\.up_proj\.weight", "locdit.blk.{}.ffn_up.weight"),
    (r"feat_decoder\.estimator\.decoder\.layers\.(\d+)\.mlp\.down_proj\.weight", "locdit.blk.{}.ffn_down.weight"),
]

# AudioVAE patterns — encoder
VAE_ENC_PATTERNS = [
    # First conv
    (r"encoder\.block\.0\.weight_g", "vae.enc.conv0.weight_g"),
    (r"encoder\.block\.0\.weight_v", "vae.enc.conv0.weight_v"),
    (r"encoder\.block\.0\.bias", "vae.enc.conv0.bias"),
    # Encoder blocks (4 blocks, each with 3 residual units + 1 strided conv)
    # ResUnit: block.N.block.{0,2}.{block.0, block.2} = Snake + WNConv
    (r"encoder\.block\.(\d+)\.block\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)",
     "vae.enc.block.{}.res.{}.{}"),
    # Strided downsample conv at end of each encoder block
    (r"encoder\.block\.(\d+)\.block\.3\.(weight_g|weight_v|bias|alpha)",
     "vae.enc.block.{}.down.{}"),
    # mu/logvar heads
    (r"encoder\.fc_mu\.(weight_g|weight_v|bias)", "vae.enc.fc_mu.{}"),
    (r"encoder\.fc_logvar\.(weight_g|weight_v|bias)", "vae.enc.fc_logvar.{}"),
]

# AudioVAE patterns — decoder (more complex due to SR conditioning)
VAE_DEC_PATTERNS = [
    # First conv(s) — may be 1 or 2 depending on depthwise
    (r"decoder\.model\.(\d+)\.(weight_g|weight_v|bias)", "vae.dec.conv.{}.{}"),
    # Decoder blocks (6 blocks)
    (r"decoder\.model\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)",
     "vae.dec.block.{}.{}.{}"),
    # SR conditioning layers
    (r"decoder\.sr_cond_model\.(\d+)\.(scale_embed|bias_embed)\.weight",
     "vae.dec.sr_cond.{}.{}.weight"),
]


def remap_vae_name(key: str) -> str | None:
    """Remap AudioVAE state_dict keys to GGUF names.

    The VAE has a complex nested structure with weight-norm parameters.
    We flatten it into a deterministic naming scheme.
    """
    # Strip "audio_vae." prefix if present
    if key.startswith("audio_vae."):
        key = key[len("audio_vae."):]

    # Encoder first conv
    m = re.match(r"encoder\.block\.0\.(weight_g|weight_v|bias)$", key)
    if m:
        return f"vae.enc.conv0.{m.group(1)}"

    # Encoder blocks: block.{1..4} = CausalEncoderBlock
    # Internal: block.{0,1,2}=ResUnits, block.3=Snake, block.4=strided conv
    # Each ResUnit: block.{0=Snake, 1=WNConv(dil), 2=Snake, 3=WNConv(1x1)}
    m = re.match(r"encoder\.block\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        blk_idx = int(m.group(1)) - 1  # block.1..4 → 0..3
        sub_idx = int(m.group(2))
        param = m.group(3)
        return f"vae.enc.blk.{blk_idx}.sub.{sub_idx}.{param}"

    # Encoder block internal nested (ResUnit internals)
    m = re.match(r"encoder\.block\.(\d+)\.block\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        blk_idx = int(m.group(1)) - 1
        res_idx = int(m.group(2))
        sub_idx = int(m.group(3))
        param = m.group(4)
        return f"vae.enc.blk.{blk_idx}.res.{res_idx}.{sub_idx}.{param}"

    # Encoder mu/logvar
    m = re.match(r"encoder\.fc_(mu|logvar)\.(weight_g|weight_v|bias)$", key)
    if m:
        return f"vae.enc.fc_{m.group(1)}.{m.group(2)}"

    # Decoder: model is a ModuleList with mixed layer types
    # We enumerate them with a flat index
    m = re.match(r"decoder\.model\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        return f"vae.dec.layer.{m.group(1)}.{m.group(2)}"

    # Decoder nested blocks (DecoderBlock internals)
    m = re.match(r"decoder\.model\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        return f"vae.dec.layer.{m.group(1)}.block.{m.group(2)}.{m.group(3)}"

    # Decoder deeper nesting (ResUnit inside DecoderBlock)
    m = re.match(r"decoder\.model\.(\d+)\.block\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        return f"vae.dec.layer.{m.group(1)}.block.{m.group(2)}.{m.group(3)}.{m.group(4)}"

    # Decoder even deeper (ResUnit → WNConv internals)
    m = re.match(r"decoder\.model\.(\d+)\.block\.(\d+)\.block\.(\d+)\.block\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        return f"vae.dec.layer.{m.group(1)}.block.{m.group(2)}.{m.group(3)}.{m.group(4)}.{m.group(5)}"

    # SR conditioning
    m = re.match(r"decoder\.sr_cond_model\.(\d+)\.(scale_embed|bias_embed)\.weight$", key)
    if m:
        return f"vae.dec.sr_cond.{m.group(1)}.{m.group(2)}"

    # SR conditioning out_layer
    m = re.match(r"decoder\.sr_cond_model\.(\d+)\.out_layer\.(\d+)\.(weight_g|weight_v|bias|alpha)$", key)
    if m:
        return f"vae.dec.sr_cond.{m.group(1)}.out.{m.group(2)}.{m.group(3)}"

    # Catch-all for any remaining VAE keys
    return f"vae.{key.replace('.', '_')}"


ALL_PATTERNS = (
    TSLM_LAYER_PATTERNS
    + RALM_LAYER_PATTERNS
    + LOCENC_LAYER_PATTERNS
    + LOCDIT_LAYER_PATTERNS
)


def remap_name(key: str) -> str | None:
    """Map a PyTorch state_dict key to a GGUF tensor name."""
    # Check direct mappings first
    if key in DIRECT:
        return DIRECT[key]

    # Check per-layer patterns
    for pat, tmpl in ALL_PATTERNS:
        m = re.match(pat, key)
        if m:
            return tmpl.format(m.group(1))

    # VAE keys
    if key.startswith("audio_vae.") or key.startswith("encoder.") or key.startswith("decoder."):
        return remap_vae_name(key)

    return None


def is_f32_tensor(gguf_name: str, shape: tuple[int, ...]) -> bool:
    """Determine if a tensor should stay F32 (norms, biases, small tensors)."""
    if gguf_name.endswith(".bias"):
        return True
    if "norm" in gguf_name:
        return True
    if "alpha" in gguf_name:  # Snake1d alpha parameters
        return True
    if "cls_token" in gguf_name:
        return True
    if "sr_cond" in gguf_name:  # SR conditioning embeddings — small
        return True
    if "weight_g" in gguf_name:  # weight-norm gain — 1D, small
        return True
    if len(shape) <= 1:
        return True
    # FSQ and stop layers are small
    if "fsq." in gguf_name or "stop." in gguf_name:
        return True
    return False


# ---------------------------------------------------------------------------
# Tokenizer serialization
# ---------------------------------------------------------------------------

def serialize_tokenizer(input_dir: Path) -> tuple[bytes, int]:
    """Serialize the tokenizer.json vocabulary for GGUF embedding.

    Returns (blob, n_vocab) where blob is packed token bytes.
    """
    tok_path = input_dir / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"missing tokenizer.json in {input_dir}")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = json.load(f)

    # Extract vocabulary from HF tokenizer.json
    vocab = tok.get("model", {}).get("vocab", {})
    if not vocab:
        sys.exit("could not find vocab in tokenizer.json")

    # Sort by token id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    n_vocab = len(sorted_vocab)

    out = bytearray()
    for token_str, _tid in sorted_vocab:
        b = token_str.encode("utf-8")
        if len(b) > 65535:
            b = b[:65535]
        out += struct.pack("<H", len(b))
        out += b

    return bytes(out), n_vocab


# ---------------------------------------------------------------------------
# LongRoPE factors serialization
# ---------------------------------------------------------------------------

def serialize_rope_factors(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract LongRoPE short/long factors from config."""
    lm_cfg = config.get("lm_config", {})
    rope_scaling = lm_cfg.get("rope_scaling", {})
    short_factors = np.array(rope_scaling.get("short_factor", []), dtype=np.float32)
    long_factors = np.array(rope_scaling.get("long_factor", []), dtype=np.float32)
    return short_factors, long_factors


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading config: {input_dir}")
    with open(input_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lm_cfg = cfg.get("lm_config", {})
    enc_cfg = cfg.get("encoder_config", {})
    dit_cfg = cfg.get("dit_config", {})
    vae_cfg = cfg.get("audio_vae_config", {})
    cfm_cfg = dit_cfg.get("cfm_config", {})

    # Find model files
    safetensor_path = input_dir / "model.safetensors"
    if not safetensor_path.exists():
        sys.exit(f"missing model.safetensors in {input_dir}")

    # AudioVAE — try safetensors first, then .pth
    vae_safetensors = input_dir / "audiovae.safetensors"
    vae_pth = input_dir / "audiovae.pth"
    if not vae_safetensors.exists() and not vae_pth.exists():
        sys.exit(f"missing audiovae.safetensors or audiovae.pth in {input_dir}")

    # Tokenizer
    tok_blob, n_vocab_from_tok = serialize_tokenizer(input_dir)
    print(f"  tokenizer: {n_vocab_from_tok} tokens, {len(tok_blob)/1024:.1f} KB blob")

    # LongRoPE factors
    short_factors, long_factors = serialize_rope_factors(cfg)
    print(f"  LongRoPE: {len(short_factors)} factors")

    # ----- Write GGUF -----
    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="voxcpm2")

    # --- Hyperparameters ---
    # Base LM (TSLM)
    writer.add_uint32("voxcpm2.tslm.n_layers", lm_cfg.get("num_hidden_layers", 28))
    writer.add_uint32("voxcpm2.tslm.d_model", lm_cfg.get("hidden_size", 2048))
    writer.add_uint32("voxcpm2.tslm.n_heads", lm_cfg.get("num_attention_heads", 16))
    writer.add_uint32("voxcpm2.tslm.n_kv_heads", lm_cfg.get("num_key_value_heads", 2))
    writer.add_uint32("voxcpm2.tslm.head_dim", lm_cfg.get("kv_channels", 128))
    writer.add_uint32("voxcpm2.tslm.ff_dim", lm_cfg.get("intermediate_size", 6144))
    writer.add_uint32("voxcpm2.tslm.vocab_size", lm_cfg.get("vocab_size", 73448))
    writer.add_uint32("voxcpm2.tslm.max_pos", lm_cfg.get("max_position_embeddings", 32768))
    writer.add_float32("voxcpm2.tslm.rope_theta", float(lm_cfg.get("rope_theta", 10000)))
    writer.add_float32("voxcpm2.tslm.rms_norm_eps", float(lm_cfg.get("rms_norm_eps", 1e-5)))
    writer.add_float32("voxcpm2.tslm.scale_emb", float(lm_cfg.get("scale_emb", 12)))
    writer.add_float32("voxcpm2.tslm.scale_depth", float(lm_cfg.get("scale_depth", 1.4)))
    writer.add_uint32("voxcpm2.tslm.dim_model_base", lm_cfg.get("dim_model_base", 256))
    writer.add_bool("voxcpm2.tslm.use_mup", lm_cfg.get("use_mup", False))
    rope_scaling = lm_cfg.get("rope_scaling", {})
    writer.add_uint32("voxcpm2.tslm.rope_orig_max_pos",
                      rope_scaling.get("original_max_position_embeddings", 32768))

    # Residual LM (RALM)
    writer.add_uint32("voxcpm2.ralm.n_layers", cfg.get("residual_lm_num_layers", 8))
    writer.add_bool("voxcpm2.ralm.no_rope", cfg.get("residual_lm_no_rope", True))

    # Local Encoder (LocEnc)
    writer.add_uint32("voxcpm2.locenc.n_layers", enc_cfg.get("num_layers", 12))
    writer.add_uint32("voxcpm2.locenc.d_model", enc_cfg.get("hidden_dim", 1024))
    writer.add_uint32("voxcpm2.locenc.ff_dim", enc_cfg.get("ffn_dim", 4096))
    writer.add_uint32("voxcpm2.locenc.n_heads", enc_cfg.get("num_heads", 16))
    writer.add_uint32("voxcpm2.locenc.n_kv_heads", enc_cfg.get("num_key_value_heads", 2))
    writer.add_uint32("voxcpm2.locenc.head_dim", enc_cfg.get("kv_channels", 128))

    # Local DiT (LocDiT)
    writer.add_uint32("voxcpm2.locdit.n_layers", dit_cfg.get("num_layers", 12))
    writer.add_uint32("voxcpm2.locdit.d_model", dit_cfg.get("hidden_dim", 1024))
    writer.add_uint32("voxcpm2.locdit.ff_dim", dit_cfg.get("ffn_dim", 4096))
    writer.add_uint32("voxcpm2.locdit.n_heads", dit_cfg.get("num_heads", 16))
    writer.add_uint32("voxcpm2.locdit.n_kv_heads", dit_cfg.get("num_key_value_heads", 2))
    writer.add_uint32("voxcpm2.locdit.head_dim", dit_cfg.get("kv_channels", 128))

    # CFM parameters
    writer.add_float32("voxcpm2.cfm.sigma_min", float(cfm_cfg.get("sigma_min", 1e-6)))
    writer.add_float32("voxcpm2.cfm.inference_cfg_rate", float(cfm_cfg.get("inference_cfg_rate", 2.0)))
    writer.add_string("voxcpm2.cfm.solver", cfm_cfg.get("solver", "euler"))

    # Shared params
    writer.add_uint32("voxcpm2.patch_size", cfg.get("patch_size", 4))
    writer.add_uint32("voxcpm2.feat_dim", cfg.get("feat_dim", 64))
    writer.add_uint32("voxcpm2.fsq.latent_dim", cfg.get("scalar_quantization_latent_dim", 512))
    writer.add_uint32("voxcpm2.fsq.scale", cfg.get("scalar_quantization_scale", 9))
    writer.add_uint32("voxcpm2.max_length", cfg.get("max_length", 8192))

    # AudioVAE params
    writer.add_uint32("voxcpm2.vae.latent_dim", vae_cfg.get("latent_dim", 64))
    writer.add_uint32("voxcpm2.vae.encoder_dim", vae_cfg.get("encoder_dim", 128))
    writer.add_uint32("voxcpm2.vae.decoder_dim", vae_cfg.get("decoder_dim", 2048))
    writer.add_uint32("voxcpm2.vae.sample_rate", vae_cfg.get("sample_rate", 16000))
    writer.add_uint32("voxcpm2.vae.out_sample_rate", vae_cfg.get("out_sample_rate", 48000))
    enc_rates = vae_cfg.get("encoder_rates", [2, 5, 8, 8])
    dec_rates = vae_cfg.get("decoder_rates", [8, 6, 5, 2, 2, 2])
    writer.add_array("voxcpm2.vae.encoder_rates", enc_rates)
    writer.add_array("voxcpm2.vae.decoder_rates", dec_rates)
    sr_boundaries = vae_cfg.get("sr_bin_boundaries", [20000, 30000, 40000])
    if sr_boundaries:
        writer.add_array("voxcpm2.vae.sr_bin_boundaries", sr_boundaries)

    # Special token IDs
    writer.add_uint32("voxcpm2.audio_start_token", 101)
    writer.add_uint32("voxcpm2.audio_end_token", 102)
    writer.add_uint32("voxcpm2.ref_audio_start_token", 103)
    writer.add_uint32("voxcpm2.ref_audio_end_token", 104)
    writer.add_uint32("voxcpm2.bos_token", lm_cfg.get("bos_token_id", 1))
    writer.add_uint32("voxcpm2.eos_token", lm_cfg.get("eos_token_id", 2))

    # Tokenizer blob
    tok_f32 = np.frombuffer(tok_blob, dtype=np.uint8).astype(np.float32)
    writer.add_tensor("tokenizer.vocab_tensor", tok_f32)
    writer.add_uint32("tokenizer.n_vocab", n_vocab_from_tok)

    # LongRoPE factors
    if len(short_factors) > 0:
        writer.add_tensor("tslm.rope_short_factors", short_factors)
        writer.add_tensor("tslm.rope_long_factors", long_factors)

    # ----- Convert model tensors (streaming to reduce memory) -----
    print(f"\nConverting model tensors from: {safetensor_path.name}")
    n_converted = 0
    n_skipped = 0

    # VoxCPM2 uses BF16 which numpy doesn't support natively.
    # Use torch framework for safe_open if available, else fall back to
    # raw numpy with manual BF16→F16 conversion.
    if _HAS_TORCH:
        framework = "pt"
    else:
        framework = "numpy"

    with safe_open(str(safetensor_path), framework=framework) as f:
        keys = list(f.keys())
        print(f"  {len(keys)} tensors in safetensors (framework={framework})")

        for key in keys:
            gguf_name = remap_name(key)
            if gguf_name is None:
                print(f"  SKIP: {key}")
                n_skipped += 1
                continue

            raw = f.get_tensor(key)

            # Convert to numpy
            if _HAS_TORCH and isinstance(raw, torch.Tensor):
                if raw.dtype == torch.bfloat16:
                    if is_f32_tensor(gguf_name, tuple(raw.shape)):
                        tensor = raw.float().numpy()
                    else:
                        tensor = raw.half().numpy()  # BF16 → F16
                else:
                    tensor = raw.numpy()
            else:
                tensor = raw

            shape = tensor.shape

            if is_f32_tensor(gguf_name, shape):
                tensor = tensor.astype(np.float32)
            elif tensor.dtype != np.float16:
                tensor = tensor.astype(np.float16)

            writer.add_tensor(gguf_name, tensor)
            n_converted += 1

            # Free memory explicitly
            del raw, tensor

            if n_converted % 50 == 0:
                print(f"  ... {n_converted} tensors converted")

    print(f"  Model: {n_converted} converted, {n_skipped} skipped")

    # ----- Convert AudioVAE tensors -----
    print(f"\nConverting AudioVAE tensors...")
    n_vae = 0

    if vae_safetensors.exists():
        with safe_open(str(vae_safetensors), framework="numpy") as f:
            vae_keys = list(f.keys())
            print(f"  {len(vae_keys)} tensors in audiovae.safetensors")
            for key in vae_keys:
                gguf_name = remap_vae_name(key)
                if gguf_name is None:
                    print(f"  SKIP VAE: {key}")
                    continue
                tensor = f.get_tensor(key)
                # VAE stays F32 for audio quality (following TrevorJS finding)
                tensor = tensor.astype(np.float32)
                writer.add_tensor(gguf_name, tensor)
                n_vae += 1
    else:
        # Load from .pth — need torch
        if not _HAS_TORCH:
            sys.exit("pip install torch (needed to load audiovae.pth)")

        print(f"  Loading {vae_pth.name} (this may use significant RAM)...")
        checkpoint = torch.load(str(vae_pth), map_location="cpu", weights_only=True)
        vae_state = checkpoint.get("state_dict", checkpoint)
        print(f"  {len(vae_state)} tensors in audiovae.pth")

        for key, tensor_pt in vae_state.items():
            gguf_name = remap_vae_name(key)
            if gguf_name is None:
                print(f"  SKIP VAE: {key}")
                continue
            tensor = tensor_pt.numpy().astype(np.float32)
            writer.add_tensor(gguf_name, tensor)
            n_vae += 1

        del vae_state, checkpoint

    print(f"  VAE: {n_vae} tensors converted")

    # ----- Finalize -----
    print(f"\nFinalizing GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total = n_converted + n_vae
    print(f"\nDone! {total} tensors total → {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024**3:.2f} GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert VoxCPM2 HuggingFace model to GGUF"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to local VoxCPM2 directory (or HF repo id for auto-download)"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output GGUF file path"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)

    # If input looks like a HF repo id, try to download
    if not input_dir.exists() and "/" in args.input and not args.input.startswith("/"):
        try:
            from huggingface_hub import snapshot_download
            print(f"Downloading from HuggingFace: {args.input}")
            input_dir = Path(snapshot_download(args.input))
        except ImportError:
            sys.exit("pip install huggingface_hub  (needed to download from HF)")
        except Exception as e:
            sys.exit(f"Failed to download {args.input}: {e}")

    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")

    convert(input_dir, Path(args.output))


if __name__ == "__main__":
    main()
