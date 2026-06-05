#!/usr/bin/env python3
"""
Convert Qwen/Qwen3-TTS-Tokenizer-12Hz (RVQ codec) → GGUF.

Only the decoder side (synthesis path) is converted.  The encoder
(voice-cloning from raw audio) is skipped — add a separate pass when
that feature lands.

Architecture recap (decoder only):
  codes (16, T)
  → SplitRVQ decode  [rvq_first × 1 + rvq_rest × 15 codebooks]
  → pre_conv         [causal conv 512→1024, k=3]
  → pre_transformer  [input_proj 1024→512 → 8L transformer → norm → output_proj 512→1024]
  → 2× ConvNeXt upsample [tconv k=2 stride=2 + ConvNeXt block, each 2×]
  → in_conv          [causal conv 1024→1536, k=7]
  → 4× DecoderBlock  [SnakeBeta + tconv(upsample_rates=[8,5,4,3]) + 3 ResUnits]
  → SnakeBeta(96) + causal conv 96→1 k=7 → clamp(-1,1)

GGUF tensor naming (all decoder tensors prefixed `codec.dec.`):

  RVQ:
    codec.dec.rvq_first.codebook          (2048, 256)   F32  precomputed
    codec.dec.rvq_first.out_proj_w        (512, 256, 1) F16
    codec.dec.rvq_rest.N.codebook         (2048, 256)   F32  N=0..14 precomputed
    codec.dec.rvq_rest.out_proj_w         (512, 256, 1) F16

  pre_conv:
    codec.dec.pre_conv_w                  (1024, 512, 3) F16
    codec.dec.pre_conv_b                  (1024,)        F32

  Transformer (pre_transformer):
    codec.dec.xfmr.in_proj_w             (512, 1024)    F16
    codec.dec.xfmr.in_proj_b             (512,)         F32
    codec.dec.xfmr.norm_w                (512,)         F32
    codec.dec.xfmr.out_proj_w            (1024, 512)    F16
    codec.dec.xfmr.out_proj_b            (1024,)        F32
    codec.dec.xfmr.blk.L.attn_norm_w    (512,)         F32  L=0..7
    codec.dec.xfmr.blk.L.ffn_norm_w     (512,)         F32
    codec.dec.xfmr.blk.L.attn_q_w       (1024, 512)    F16
    codec.dec.xfmr.blk.L.attn_k_w       (1024, 512)    F16
    codec.dec.xfmr.blk.L.attn_v_w       (1024, 512)    F16
    codec.dec.xfmr.blk.L.attn_o_w       (512, 1024)    F16
    codec.dec.xfmr.blk.L.attn_ls_w      (512,)         F32  layer_scale
    codec.dec.xfmr.blk.L.ffn_gate_w     (1024, 512)    F16
    codec.dec.xfmr.blk.L.ffn_up_w       (1024, 512)    F16
    codec.dec.xfmr.blk.L.ffn_down_w     (512, 1024)    F16
    codec.dec.xfmr.blk.L.ffn_ls_w       (512,)         F32  layer_scale

  ConvNeXt upsample (S=0,1):
    codec.dec.up.S.tconv_w               (1024,1024,2)  F16
    codec.dec.up.S.tconv_b               (1024,)        F32
    codec.dec.up.S.cnx.dw_w             (1024,1,7)     F16
    codec.dec.up.S.cnx.dw_b             (1024,)        F32
    codec.dec.up.S.cnx.norm_w           (1024,)        F32
    codec.dec.up.S.cnx.norm_b           (1024,)        F32
    codec.dec.up.S.cnx.pw1_w            (4096,1024)    F16
    codec.dec.up.S.cnx.pw1_b            (4096,)        F32
    codec.dec.up.S.cnx.pw2_w            (1024,4096)    F16
    codec.dec.up.S.cnx.pw2_b            (1024,)        F32
    codec.dec.up.S.cnx.gamma            (1024,)        F32

  Decoder:
    codec.dec.in_conv_w                  (1536,1024,7)  F16
    codec.dec.in_conv_b                  (1536,)        F32
    codec.dec.blk.B.snake_a              (C,)           F32  B=0..3
    codec.dec.blk.B.snake_b              (C,)           F32
    codec.dec.blk.B.tconv_w              (C_in,C_out,K) F16
    codec.dec.blk.B.tconv_b              (C_out,)       F32
    codec.dec.blk.B.res.U.act1_a         (C_out,)       F32  U=0..2
    codec.dec.blk.B.res.U.act1_b         (C_out,)       F32
    codec.dec.blk.B.res.U.act2_a         (C_out,)       F32
    codec.dec.blk.B.res.U.act2_b         (C_out,)       F32
    codec.dec.blk.B.res.U.conv1_w        (C,C,7)        F16
    codec.dec.blk.B.res.U.conv1_b        (C,)           F32
    codec.dec.blk.B.res.U.conv2_w        (C,C,1)        F16
    codec.dec.blk.B.res.U.conv2_b        (C,)           F32
    codec.dec.out_snake_a                (96,)          F32
    codec.dec.out_snake_b                (96,)          F32
    codec.dec.out_conv_w                 (1,96,7)       F16
    codec.dec.out_conv_b                 (1,)           F32

Usage:
    python models/convert-qwen3-tts-tokenizer-to-gguf.py \\
        --input /Volumes/backups/ai/huggingface-hub/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/<sha>/ \\
        --output /Volumes/backups/ai/crispasr-models/qwen3-tts-tokenizer-12hz.gguf
"""

from __future__ import annotations

import argparse
import json
import re
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
    return Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "*.txt"]))


def open_tensors(model_dir: Path):
    """Returns a dict {name: tensor_as_numpy_float32}."""
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        sys.exit(f"no *.safetensors in {model_dir}")
    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name2h = {}
    for h in handles:
        for k in h.keys():
            name2h[k] = h
    print(f"  Loaded {len(name2h)} safetensors tensors from {len(st_files)} file(s)")
    return name2h


def get_np(name2h: dict, name: str) -> np.ndarray:
    return name2h[name].get_tensor(name).to(torch.float32).numpy()


# ---------------------------------------------------------------------------
# Codebook precomputation
#   effective_codebook = embedding_sum / clamp(cluster_usage, 1e-5)
# ---------------------------------------------------------------------------

def precompute_codebooks(name2h: dict) -> dict[str, np.ndarray]:
    """Returns dict {gguf_name: codebook_array (F32)}."""
    out = {}

    # Decoder: rvq_first (1 semantic) + rvq_rest (15 acoustic)
    emb  = get_np(name2h, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
    use  = get_np(name2h, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
    out["codec.dec.rvq_first.codebook"] = emb / np.maximum(use[:, None], 1e-5)
    for i in range(15):
        emb = get_np(name2h, f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum")
        use = get_np(name2h, f"decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage")
        out[f"codec.dec.rvq_rest.{i}.codebook"] = emb / np.maximum(use[:, None], 1e-5)

    # Encoder: semantic (1 codebook) + acoustic (15 codebooks used for n_q=16)
    # encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.{embed_sum,cluster_usage}
    # encoder.quantizer.acoustic_residual_vector_quantizer.layers.{0..14}.codebook.*
    try:
        emb = get_np(name2h, "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum")
        use = get_np(name2h, "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage")
        out["codec.enc.rvq.sem.cb"] = emb / np.maximum(use[:, None], 1e-5)
        for i in range(15):
            emb = get_np(name2h, f"encoder.quantizer.acoustic_residual_vector_quantizer.layers.{i}.codebook.embed_sum")
            use = get_np(name2h, f"encoder.quantizer.acoustic_residual_vector_quantizer.layers.{i}.codebook.cluster_usage")
            out[f"codec.enc.rvq.ac.{i}.cb"] = emb / np.maximum(use[:, None], 1e-5)
    except KeyError as e:
        print(f"  [WARN] encoder codebook missing: {e}")

    return out


# ---------------------------------------------------------------------------
# Per-tensor name mapping
#   Returns (gguf_name, force_f32) or None to skip.
# ---------------------------------------------------------------------------

def map_encoder_tensor(name: str) -> tuple[str, bool] | None:
    """Map encoder.* safetensors → (gguf_name, force_f32). None = skip."""
    n = name

    # SEANet: layer index → (stage, role)
    # layers.0  → init conv
    # layers.{1,4,7,10}.block.{1,3} → resblocks stage 0..3
    # layers.{3,6,9,12} → stride convs stage 0..3
    # layers.14 → final conv
    layer_to_stage = {1: 0, 4: 1, 7: 2, 10: 3}
    stride_to_stage = {3: 0, 6: 1, 9: 2, 12: 3}

    m = re.match(r"encoder\.encoder\.layers\.(\d+)\.(.*)", n)
    if m:
        li, rest = int(m.group(1)), m.group(2)
        if li == 0:
            if rest == "conv.weight": return "codec.enc.seanet.init_w", False
            if rest == "conv.bias":   return "codec.enc.seanet.init_b", True
        elif li in layer_to_stage:
            s = layer_to_stage[li]
            bmap = {"block.1.conv.weight": (f"codec.enc.seanet.blk.{s}.short_w", False),
                    "block.1.conv.bias":   (f"codec.enc.seanet.blk.{s}.short_b", True),
                    "block.3.conv.weight": (f"codec.enc.seanet.blk.{s}.exp_w",   False),
                    "block.3.conv.bias":   (f"codec.enc.seanet.blk.{s}.exp_b",   True)}
            return bmap.get(rest)
        elif li in stride_to_stage:
            s = stride_to_stage[li]
            if rest == "conv.weight": return f"codec.enc.seanet.ds.{s}.w", False
            if rest == "conv.bias":   return f"codec.enc.seanet.ds.{s}.b", True
        elif li == 14:
            if rest == "conv.weight": return "codec.enc.seanet.final_w", False
            if rest == "conv.bias":   return "codec.enc.seanet.final_b", True
        return None

    # Encoder transformer
    m = re.match(r"encoder\.encoder_transformer\.layers\.(\d+)\.(.*)", n)
    if m:
        L, rest = m.group(1), m.group(2)
        tmap = {
            "input_layernorm.weight":          (f"codec.enc.xfmr.blk.{L}.norm1_w",  True),
            "input_layernorm.bias":            (f"codec.enc.xfmr.blk.{L}.norm1_b",  True),
            "post_attention_layernorm.weight": (f"codec.enc.xfmr.blk.{L}.norm2_w",  True),
            "post_attention_layernorm.bias":   (f"codec.enc.xfmr.blk.{L}.norm2_b",  True),
            "self_attn.q_proj.weight":         (f"codec.enc.xfmr.blk.{L}.attn_q_w", False),
            "self_attn.k_proj.weight":         (f"codec.enc.xfmr.blk.{L}.attn_k_w", False),
            "self_attn.v_proj.weight":         (f"codec.enc.xfmr.blk.{L}.attn_v_w", False),
            "self_attn.o_proj.weight":         (f"codec.enc.xfmr.blk.{L}.attn_o_w", False),
            "self_attn_layer_scale.scale":     (f"codec.enc.xfmr.blk.{L}.attn_ls",  True),
            "mlp.fc1.weight":                  (f"codec.enc.xfmr.blk.{L}.fc1_w",    False),
            "mlp.fc2.weight":                  (f"codec.enc.xfmr.blk.{L}.fc2_w",    False),
            "mlp_layer_scale.scale":           (f"codec.enc.xfmr.blk.{L}.ffn_ls",   True),
        }
        if rest in tmap: return tmap[rest]
        return None

    # Downsample
    if n == "encoder.downsample.conv.weight": return "codec.enc.ds_w", False

    # Encoder quantizer projections — force F32 since C++ side reads them
    # to a CPU buffer for nearest-neighbor search and residual subtraction.
    if n == "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight":
        return "codec.enc.rvq.sem.in_w", True
    if n == "encoder.quantizer.semantic_residual_vector_quantizer.output_proj.weight":
        return "codec.enc.rvq.sem.out_w", True
    if n == "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight":
        return "codec.enc.rvq.ac.in_w", True
    if n == "encoder.quantizer.acoustic_residual_vector_quantizer.output_proj.weight":
        return "codec.enc.rvq.ac.out_w", True

    return None  # other encoder tensors (codebook data, initialized flag) → skip


def map_decoder_tensor(name: str) -> tuple[str, bool] | None:
    """Map decoder.* safetensors → (gguf_name, force_f32). None = skip."""

    n = name

    # Encoder tensors are handled by map_encoder_tensor.
    if n.startswith("encoder."):
        return None

    # Skip cluster_usage and embedding_sum — handled by precompute_codebooks.
    if "_codebook.cluster_usage" in n or "_codebook.embedding_sum" in n or \
       ".codebook.embed_sum" in n or ".codebook.initialized" in n:
        return None

    # Skip quantizer input_proj — encoder-only path.
    if re.search(r"\.quantizer\.rvq_(first|rest)\.input_proj\.", n):
        return None

    # ---------- RVQ output_proj ----------
    if n == "decoder.quantizer.rvq_first.output_proj.weight":
        return "codec.dec.rvq_first.out_proj_w", False
    if n == "decoder.quantizer.rvq_rest.output_proj.weight":
        return "codec.dec.rvq_rest.out_proj_w", False

    # ---------- pre_conv ----------
    if n == "decoder.pre_conv.conv.weight":
        return "codec.dec.pre_conv_w", False
    if n == "decoder.pre_conv.conv.bias":
        return "codec.dec.pre_conv_b", True

    # ---------- transformer: top-level projections and norm ----------
    if n == "decoder.pre_transformer.input_proj.weight":
        return "codec.dec.xfmr.in_proj_w", False
    if n == "decoder.pre_transformer.input_proj.bias":
        return "codec.dec.xfmr.in_proj_b", True
    if n == "decoder.pre_transformer.norm.weight":
        return "codec.dec.xfmr.norm_w", True
    if n == "decoder.pre_transformer.output_proj.weight":
        return "codec.dec.xfmr.out_proj_w", False
    if n == "decoder.pre_transformer.output_proj.bias":
        return "codec.dec.xfmr.out_proj_b", True

    # ---------- transformer layers ----------
    m = re.match(r"decoder\.pre_transformer\.layers\.(\d+)\.(.*)", n)
    if m:
        L, rest = m.group(1), m.group(2)
        tmap = {
            "input_layernorm.weight":           ("attn_norm_w",  True),
            "post_attention_layernorm.weight":  ("ffn_norm_w",   True),
            "self_attn.q_proj.weight":          ("attn_q_w",     False),
            "self_attn.k_proj.weight":          ("attn_k_w",     False),
            "self_attn.v_proj.weight":          ("attn_v_w",     False),
            "self_attn.o_proj.weight":          ("attn_o_w",     False),
            "self_attn_layer_scale.scale":      ("attn_ls_w",    True),
            "mlp.gate_proj.weight":             ("ffn_gate_w",   False),
            "mlp.up_proj.weight":               ("ffn_up_w",     False),
            "mlp.down_proj.weight":             ("ffn_down_w",   False),
            "mlp_layer_scale.scale":            ("ffn_ls_w",     True),
        }
        if rest in tmap:
            suffix, f32 = tmap[rest]
            return f"codec.dec.xfmr.blk.{L}.{suffix}", f32
        return None  # unexpected sub-field → skip with warning

    # ---------- ConvNeXt upsample stages ----------
    m = re.match(r"decoder\.upsample\.(\d+)\.(.*)", n)
    if m:
        S, rest = m.group(1), m.group(2)
        umap = {
            "0.conv.weight":        ("tconv_w",    False),
            "0.conv.bias":          ("tconv_b",    True),
            "1.dwconv.conv.weight": ("cnx.dw_w",   False),
            "1.dwconv.conv.bias":   ("cnx.dw_b",   True),
            "1.norm.weight":        ("cnx.norm_w",  True),
            "1.norm.bias":          ("cnx.norm_b",  True),
            "1.pwconv1.weight":     ("cnx.pw1_w",   False),
            "1.pwconv1.bias":       ("cnx.pw1_b",   True),
            "1.pwconv2.weight":     ("cnx.pw2_w",   False),
            "1.pwconv2.bias":       ("cnx.pw2_b",   True),
            "1.gamma":              ("cnx.gamma",   True),
        }
        if rest in umap:
            suffix, f32 = umap[rest]
            return f"codec.dec.up.{S}.{suffix}", f32
        return None

    # ---------- decoder stack ----------
    m = re.match(r"decoder\.decoder\.(\d+)\.(.*)", n)
    if m:
        idx, rest = int(m.group(1)), m.group(2)

        # decoder.decoder.0 — causal in_conv 1024→1536
        if idx == 0:
            if rest == "conv.weight": return "codec.dec.in_conv_w", False
            if rest == "conv.bias":   return "codec.dec.in_conv_b", True
            return None

        # decoder.decoder.5 — final SnakeBeta
        if idx == 5:
            if rest == "alpha": return "codec.dec.out_snake_a", True
            if rest == "beta":  return "codec.dec.out_snake_b", True
            return None

        # decoder.decoder.6 — final causal conv 96→1
        if idx == 6:
            if rest == "conv.weight": return "codec.dec.out_conv_w", False
            if rest == "conv.bias":   return "codec.dec.out_conv_b", True
            return None

        # decoder.decoder.{1,2,3,4} → DecoderBlock B = idx-1
        if 1 <= idx <= 4:
            B = idx - 1

            # block.0 — SnakeBeta before the tconv
            if rest == "block.0.alpha": return f"codec.dec.blk.{B}.snake_a", True
            if rest == "block.0.beta":  return f"codec.dec.blk.{B}.snake_b", True

            # block.1 — transposed conv
            if rest == "block.1.conv.weight": return f"codec.dec.blk.{B}.tconv_w", False
            if rest == "block.1.conv.bias":   return f"codec.dec.blk.{B}.tconv_b", True

            # block.{2,3,4} — ResidualUnits U=0..2
            m2 = re.match(r"block\.([234])\.(.+)", rest)
            if m2:
                U = int(m2.group(1)) - 2   # 2→0, 3→1, 4→2
                rrest = m2.group(2)
                rmap = {
                    "act1.alpha":         ("act1_a",  True),
                    "act1.beta":          ("act1_b",  True),
                    "act2.alpha":         ("act2_a",  True),
                    "act2.beta":          ("act2_b",  True),
                    "conv1.conv.weight":  ("conv1_w", False),
                    "conv1.conv.bias":    ("conv1_b", True),
                    "conv2.conv.weight":  ("conv2_w", False),
                    "conv2.conv.bias":    ("conv2_b", True),
                }
                if rrest in rmap:
                    suffix, f32 = rmap[rrest]
                    return f"codec.dec.blk.{B}.res.{U}.{suffix}", f32
            return None

        return None  # unexpected idx

    # Anything left from the decoder.* namespace
    if n.startswith("decoder."):
        return None   # unmapped decoder tensor → fall through to warn

    return None  # non-decoder top-level


def main():
    ap = argparse.ArgumentParser(description="Convert Qwen3-TTS-Tokenizer-12Hz to GGUF (decoder only)")
    ap.add_argument("--input", required=True,
                    help="HF model ID or local snapshot directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    enc = cfg.get("encoder_config", cfg.get("encoder", {}))
    dec = cfg.get("decoder_config", cfg.get("decoder", {}))

    print(f"\nQwen3-TTS-Tokenizer-12Hz")
    print(f"  Encoder: {enc.get('num_hidden_layers')}L  hidden={enc.get('hidden_size')}  "
          f"heads={enc.get('num_attention_heads')}  ff={enc.get('intermediate_size')}  "
          f"rvq={enc.get('num_quantizers')}×{enc.get('codebook_size')}")
    print(f"  Decoder: {dec.get('num_hidden_layers')}L  hidden={dec.get('hidden_size')}  "
          f"heads={dec.get('num_attention_heads')}  ff={dec.get('intermediate_size')}  "
          f"rvq={dec.get('num_quantizers')}×{dec.get('codebook_size')}")
    print(f"  SR: {cfg.get('output_sample_rate', cfg.get('sampling_rate', 24000))} Hz  "
          f"frame={cfg.get('frame_rate', 12.5)} Hz")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    name2h = open_tensors(model_dir)

    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="qwen3tts_tokenizer", use_temp_file=True)
    w.add_name("qwen3-tts-tokenizer-12hz")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    # -----------------------------------------------------------------------
    # KV metadata — top-level audio params
    # -----------------------------------------------------------------------
    u32("qwen3tts_codec.input_sample_rate",
        cfg.get("input_sample_rate", cfg.get("sampling_rate", 24000)))
    u32("qwen3tts_codec.output_sample_rate",
        cfg.get("output_sample_rate", cfg.get("sampling_rate", 24000)))
    f32("qwen3tts_codec.frame_rate",
        enc.get("_frame_rate", cfg.get("frame_rate", 12.5)))
    u32("qwen3tts_codec.encode_downsample", cfg.get("encode_downsample_rate", 1920))
    u32("qwen3tts_codec.decode_upsample",   cfg.get("decode_upsample_rate", 1920))

    # decoder config
    u32("qwen3tts_codec.dec.n_layers",      dec.get("num_hidden_layers", 8))
    u32("qwen3tts_codec.dec.d_model",       dec.get("hidden_size", 512))
    u32("qwen3tts_codec.dec.n_heads",       dec.get("num_attention_heads", 16))
    u32("qwen3tts_codec.dec.ff_dim",        dec.get("intermediate_size", 1024))
    u32("qwen3tts_codec.dec.n_quantizers",  dec.get("num_quantizers", 16))
    u32("qwen3tts_codec.dec.codebook_size", dec.get("codebook_size", 2048))
    u32("qwen3tts_codec.dec.latent_dim",    dec.get("latent_dim", 1024))
    u32("qwen3tts_codec.dec.decoder_dim",   dec.get("decoder_dim", 1536))
    u32("qwen3tts_codec.dec.sliding_window",dec.get("sliding_window", 72))
    u32("qwen3tts_codec.dec.head_dim",      dec.get("head_dim", 64))
    u32("qwen3tts_codec.dec.max_pos",       dec.get("max_position_embeddings", 8000))
    f32("qwen3tts_codec.dec.rope_theta",    dec.get("rope_theta", 10000.0))
    f32("qwen3tts_codec.dec.rms_norm_eps",  dec.get("rms_norm_eps", 1e-5))
    f32("qwen3tts_codec.dec.layer_scale_initial", dec.get("layer_scale_initial_scale", 0.01))
    if "upsample_rates" in dec:
        w.add_array("qwen3tts_codec.dec.upsample_rates", list(dec["upsample_rates"]))
    if "upsampling_ratios" in dec:
        w.add_array("qwen3tts_codec.dec.upsampling_ratios", list(dec["upsampling_ratios"]))
    u32("qwen3tts_codec.dec.n_semantic_q",
        dec.get("num_semantic_quantizers", cfg.get("semantic_quantizers", 1)))
    u32("qwen3tts_codec.dec.semantic_codebook_size",
        dec.get("semantic_codebook_size", cfg.get("semantic_codebook_size", 4096)))

    # -----------------------------------------------------------------------
    # Pass 1: precomputed codebooks
    # -----------------------------------------------------------------------
    print("\n[Pass 1] Precomputing codebooks …")
    codebooks = precompute_codebooks(name2h)
    for gname, arr in sorted(codebooks.items()):
        arr_f32 = np.ascontiguousarray(arr.astype(np.float32))
        w.add_tensor(gname, arr_f32, raw_dtype=GGMLQuantizationType.F32)
        print(f"  [CB] {gname:55s} {list(arr.shape)}")

    # -----------------------------------------------------------------------
    # Pass 2: regular tensors
    # -----------------------------------------------------------------------
    print(f"\n[Pass 2] Mapping regular tensors …")
    n_mapped = 0
    n_skipped = 0
    n_unmapped = 0

    # Names that are consumed by the codebook precomputation pass — skip silently.
    codebook_consumed = set()
    for k in name2h:
        if "_codebook.embedding_sum" in k or "_codebook.cluster_usage" in k or \
           ".codebook.embed_sum" in k or ".codebook.initialized" in k:
            codebook_consumed.add(k)
        if re.search(r"\.quantizer\.rvq_(first|rest)\.input_proj\.", k):
            codebook_consumed.add(k)

    for hf_name in sorted(name2h.keys()):
        if hf_name in codebook_consumed:
            n_skipped += 1
            continue

        # Try encoder mapping first, then decoder
        if hf_name.startswith("encoder."):
            result = map_encoder_tensor(hf_name)
            if result is None:
                n_skipped += 1  # silently skip unmapped encoder tensors
                continue
        else:
            result = map_decoder_tensor(hf_name)
            if result is None:
                n_unmapped += 1
                print(f"  [WARN unmapped] {hf_name}", file=sys.stderr)
                continue

        gname, force_f32 = result
        arr = name2h[hf_name].get_tensor(hf_name).to(torch.float32).numpy()

        if arr.ndim <= 1 or force_f32:
            arr = np.ascontiguousarray(arr.astype(np.float32))
            w.add_tensor(gname, arr, raw_dtype=GGMLQuantizationType.F32)
        else:
            arr = np.ascontiguousarray(arr.astype(out_dtype))
            w.add_tensor(gname, arr, raw_dtype=out_qt)

        n_mapped += 1
        if n_mapped <= 40 or n_mapped % 50 == 0:
            print(f"  [{n_mapped:3d}] {gname:55s} {list(arr.shape)}  {arr.dtype}")

    total_tensors = len(codebooks) + n_mapped
    print(f"\nSummary: {len(codebooks)} codebooks + {n_mapped} tensors mapped, "
          f"{n_skipped} skipped (encoder/consumed), {n_unmapped} unmapped")
    if n_unmapped > 0:
        print(f"ERROR: {n_unmapped} tensors unmapped — extend map_decoder_tensor()", file=sys.stderr)
        sys.exit(1)

    print(f"\nWriting {out_path} …")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {total_tensors} tensors total)")


if __name__ == "__main__":
    main()
