#!/usr/bin/env python3
"""
Convert sesame/csm-1b HuggingFace safetensors -> GGUF F16/F32 for the
CrispASR `csm-tts` backend.

CSM (Conversational Speech Model) is a two-transformer TTS system:
  - Backbone: Llama-3.2 1B (16L, 32H, 8KVH, 2048d, 8192 ff)
    Produces first-codebook Mimi tokens autoregressively.
  - Depth decoder: Llama-3.2 100M (4L, 8H, 2KVH, 1024d, 8192 ff)
    Fills remaining 31 codebooks from backbone hidden state.
  - Mimi codec decoder: Kyutai Mimi (SEANet + 8L transformer + upsample)
    Converts 32-codebook RVQ tokens to 24 kHz PCM.

Architecture (from sesame/csm-1b/config.json):

  Backbone (model_type = "csm"):
    hidden_size          = 2048
    num_hidden_layers    = 16
    num_attention_heads  = 32
    num_key_value_heads  = 8
    head_dim             = 64
    intermediate_size    = 8192
    text_vocab_size      = 128256 (Llama-3.2 tokenizer)
    audio_vocab_size     = 2051 (Mimi codebook entries)
    audio_num_codebooks  = 32
    rope_theta           = 500000
    rms_norm_eps         = 1e-5
    max_position_embeddings = 2048

  Depth decoder (model_type = "csm_depth_decoder_model"):
    hidden_size          = 1024
    num_hidden_layers    = 4
    num_attention_heads  = 8
    num_key_value_heads  = 2
    head_dim             = 128
    intermediate_size    = 8192
    backbone_hidden_size = 2048
    rope_theta           = 500000
    rms_norm_eps         = 1e-5

  Mimi codec (same as kyutai/mimi):
    8 encoder + 8 decoder transformer layers (512d, 8H, GELU)
    SEANet encoder/decoder (64->128->256->512->1024->512 channels)
    32 codebooks (1 semantic + 31 acoustic), 2048 entries, dim=256
    Sample rate: 24000 Hz, frame rate: 12.5 Hz

Usage:
    python models/convert-csm-to-gguf.py \\
        --input sesame/csm-1b \\
        --output csm-1b-f16.gguf

    # or from local dir:
    python models/convert-csm-to-gguf.py \\
        --input /path/to/csm-1b \\
        --output csm-1b-f16.gguf
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
    print(f"Downloading {model_id}...", file=sys.stderr)
    return Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model",
        "*.safetensors.index.json",
    ]))


# ---------------------------------------------------------------------------
# Tensor name remapping: HF transformers -> GGUF
# ---------------------------------------------------------------------------

def map_tensor_name(hf_name: str) -> str | None:
    """Map a HuggingFace tensor name to our GGUF naming convention.

    Returns None for tensors that should be skipped (rotary_emb, etc.).
    """
    if ".rotary_emb." in hf_name:
        return None
    # Skip RVQ EMA stats (cluster_usage, initialized) -- not needed for inference
    if hf_name.endswith(".cluster_usage") or hf_name.endswith(".initialized"):
        return None

    n = hf_name

    # --- Backbone transformer ---
    # backbone_model.embed_tokens.embed_audio_tokens.weight -> backbone.audio_embd.weight
    if n == "backbone_model.embed_tokens.embed_audio_tokens.weight":
        return "backbone.audio_embd.weight"
    # embed_text_tokens.weight -> backbone.text_embd.weight
    if n == "embed_text_tokens.weight":
        return "backbone.text_embd.weight"
    # lm_head.weight -> backbone.codebook0_head.weight
    if n == "lm_head.weight":
        return "backbone.codebook0_head.weight"
    # backbone_model.norm.weight -> backbone.output_norm.weight
    if n == "backbone_model.norm.weight":
        return "backbone.output_norm.weight"

    # backbone_model.layers.N.xxx -> backbone.blk.N.xxx
    if n.startswith("backbone_model.layers."):
        n = n.replace("backbone_model.layers.", "backbone.blk.")
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

    # --- Depth decoder ---
    # depth_decoder.codebooks_head.weight -> depth.codebooks_head.weight
    if n == "depth_decoder.codebooks_head.weight":
        return "depth.codebooks_head.weight"
    # depth_decoder.model.embed_tokens.weight -> depth.token_embd.weight
    if n == "depth_decoder.model.embed_tokens.weight":
        return "depth.token_embd.weight"
    # depth_decoder.model.inputs_embeds_projector.weight -> depth.projection.weight
    if n == "depth_decoder.model.inputs_embeds_projector.weight":
        return "depth.projection.weight"
    # depth_decoder.model.norm.weight -> depth.output_norm.weight
    if n == "depth_decoder.model.norm.weight":
        return "depth.output_norm.weight"

    # depth_decoder.model.layers.N.xxx -> depth.blk.N.xxx
    if n.startswith("depth_decoder.model.layers."):
        n = n.replace("depth_decoder.model.layers.", "depth.blk.")
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

    # --- Mimi codec encoder ---
    # codec_model.encoder.layers.N.conv -> mimi.encoder.model.N.conv2
    # codec_model.encoder.layers.N.block.M.conv -> mimi.encoder.model.N.blkM
    if n.startswith("codec_model.encoder.layers."):
        rest = n[len("codec_model.encoder.layers."):]
        parts = rest.split(".")
        layer_idx = int(parts[0])
        if "block" in rest:
            # block.M.conv.weight/bias
            blk_idx = int(parts[2])
            suffix = parts[4]  # weight or bias
            return f"mimi.encoder.model.{layer_idx}.blk{blk_idx}.{suffix}"
        else:
            # conv.weight/bias
            suffix = parts[2]  # weight or bias
            return f"mimi.encoder.model.{layer_idx}.conv2.{suffix}"

    # --- Mimi encoder transformer ---
    # codec_model.encoder_transformer.layers.N.xxx
    if n.startswith("codec_model.encoder_transformer.layers."):
        rest = n[len("codec_model.encoder_transformer.layers."):]
        parts = rest.split(".", 1)
        layer_idx = int(parts[0])
        field = parts[1]
        prefix = f"mimi.enc_tfm.layers.{layer_idx}"

        if field == "input_layernorm.weight":
            return f"{prefix}.norm1.weight"
        if field == "input_layernorm.bias":
            return f"{prefix}.norm1.bias"
        if field == "post_attention_layernorm.weight":
            return f"{prefix}.norm2.weight"
        if field == "post_attention_layernorm.bias":
            return f"{prefix}.norm2.bias"
        if field == "self_attn_layer_scale.scale":
            return f"{prefix}.ls1"
        if field == "mlp_layer_scale.scale":
            return f"{prefix}.ls2"
        if field == "mlp.fc1.weight":
            return f"{prefix}.ffn_up_w"
        if field == "mlp.fc2.weight":
            return f"{prefix}.ffn_down_w"
        # Attention: q/k/v/o separate -> we'll fuse q+k+v at load time
        if field == "self_attn.q_proj.weight":
            return f"{prefix}.attn.q_w"
        if field == "self_attn.k_proj.weight":
            return f"{prefix}.attn.k_w"
        if field == "self_attn.v_proj.weight":
            return f"{prefix}.attn.v_w"
        if field == "self_attn.o_proj.weight":
            return f"{prefix}.attn.out_w"

        print(f"  WARN: unmapped encoder_transformer field: {field}", file=sys.stderr)
        return None

    # --- Mimi decoder transformer ---
    if n.startswith("codec_model.decoder_transformer.layers."):
        rest = n[len("codec_model.decoder_transformer.layers."):]
        parts = rest.split(".", 1)
        layer_idx = int(parts[0])
        field = parts[1]
        prefix = f"mimi.dec_tfm.layers.{layer_idx}"

        if field == "input_layernorm.weight":
            return f"{prefix}.norm1.weight"
        if field == "input_layernorm.bias":
            return f"{prefix}.norm1.bias"
        if field == "post_attention_layernorm.weight":
            return f"{prefix}.norm2.weight"
        if field == "post_attention_layernorm.bias":
            return f"{prefix}.norm2.bias"
        if field == "self_attn_layer_scale.scale":
            return f"{prefix}.ls1"
        if field == "mlp_layer_scale.scale":
            return f"{prefix}.ls2"
        if field == "mlp.fc1.weight":
            return f"{prefix}.ffn_up_w"
        if field == "mlp.fc2.weight":
            return f"{prefix}.ffn_down_w"
        if field == "self_attn.q_proj.weight":
            return f"{prefix}.attn.q_w"
        if field == "self_attn.k_proj.weight":
            return f"{prefix}.attn.k_w"
        if field == "self_attn.v_proj.weight":
            return f"{prefix}.attn.v_w"
        if field == "self_attn.o_proj.weight":
            return f"{prefix}.attn.out_w"

        print(f"  WARN: unmapped decoder_transformer field: {field}", file=sys.stderr)
        return None

    # --- Mimi codec decoder ---
    if n.startswith("codec_model.decoder.layers."):
        rest = n[len("codec_model.decoder.layers."):]
        parts = rest.split(".")
        layer_idx = int(parts[0])
        if "block" in rest:
            blk_idx = int(parts[2])
            suffix = parts[4]
            return f"mimi.decoder.model.{layer_idx}.blk{blk_idx}.{suffix}"
        else:
            suffix = parts[2]
            return f"mimi.decoder.model.{layer_idx}.conv2.{suffix}"

    # --- Mimi downsample/upsample ---
    if n == "codec_model.downsample.conv.weight":
        return "mimi.downsample.conv3.weight"
    if n == "codec_model.upsample.conv.weight":
        return "mimi.upsample.conv3.weight"

    # --- Mimi RVQ ---
    # Semantic quantizer
    if n == "codec_model.quantizer.semantic_residual_vector_quantizer.input_proj.weight":
        return "mimi.quantizer.rvq_first.input_proj.weight"
    if n == "codec_model.quantizer.semantic_residual_vector_quantizer.output_proj.weight":
        return "mimi.quantizer.rvq_first.output_proj.weight"
    if n.startswith("codec_model.quantizer.semantic_residual_vector_quantizer.layers."):
        rest = n[len("codec_model.quantizer.semantic_residual_vector_quantizer.layers."):]
        parts = rest.split(".")
        layer_idx = int(parts[0])
        if parts[1] == "codebook" and parts[2] == "embed_sum":
            return f"mimi.quantizer.rvq_first.vq.layers.{layer_idx}._codebook.embedding"
        return None  # skip cluster_usage, initialized

    # Acoustic quantizer
    if n == "codec_model.quantizer.acoustic_residual_vector_quantizer.input_proj.weight":
        return "mimi.quantizer.rvq_rest.input_proj.weight"
    if n == "codec_model.quantizer.acoustic_residual_vector_quantizer.output_proj.weight":
        return "mimi.quantizer.rvq_rest.output_proj.weight"
    if n.startswith("codec_model.quantizer.acoustic_residual_vector_quantizer.layers."):
        rest = n[len("codec_model.quantizer.acoustic_residual_vector_quantizer.layers."):]
        parts = rest.split(".")
        layer_idx = int(parts[0])
        if parts[1] == "codebook" and parts[2] == "embed_sum":
            return f"mimi.quantizer.rvq_rest.vq.layers.{layer_idx}._codebook.embedding"
        return None  # skip cluster_usage, initialized

    print(f"  WARN: completely unmapped tensor: {hf_name}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Fuse separate Q, K, V into single QKV tensor for Mimi transformer layers
# ---------------------------------------------------------------------------

def fuse_qkv_tensors(pending: dict, out_dtype, out_qt, writer: GGUFWriter, n_mapped: list):
    """Look for separate attn.q_w / attn.k_w / attn.v_w and fuse them
    into a single attn.qkv_w tensor, matching the kyutai_stt convention."""
    # Collect all prefixes that have q_w
    prefixes = set()
    for name in list(pending.keys()):
        if name.endswith(".attn.q_w"):
            prefixes.add(name[:-len(".attn.q_w")])

    fused_names = []
    for prefix in sorted(prefixes):
        q_name = f"{prefix}.attn.q_w"
        k_name = f"{prefix}.attn.k_w"
        v_name = f"{prefix}.attn.v_w"
        if q_name in pending and k_name in pending and v_name in pending:
            q = pending.pop(q_name)
            k = pending.pop(k_name)
            v = pending.pop(v_name)
            # Stack: [3*dim, dim] = cat([q, k, v], dim=0)
            # But Mimi uses 8 heads with dim=512, head_dim=64
            # q,k,v are all [512, 512] for full attention (no GQA in Mimi)
            qkv = np.concatenate([q, k, v], axis=0)
            qkv_name = f"{prefix}.attn.qkv_w"
            if qkv.ndim <= 1:
                writer.add_tensor(qkv_name, np.ascontiguousarray(qkv.astype(np.float32)),
                                  raw_dtype=GGMLQuantizationType.F32)
            else:
                writer.add_tensor(qkv_name, np.ascontiguousarray(qkv.astype(out_dtype)),
                                  raw_dtype=out_qt)
            n_mapped[0] += 1
            fused_names.append(qkv_name)
            print(f"  [fused] {qkv_name:55s} {qkv.shape}  {qkv.dtype}")

    return fused_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Sesame CSM-1B to GGUF")
    ap.add_argument("--input", required=True,
                    help="HF model ID (e.g. sesame/csm-1b) or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16"])
    args = ap.parse_args()

    model_dir = load_model_dir(args.input)

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    # --- Backbone hparams ---
    bb_n_layers = int(cfg["num_hidden_layers"])
    bb_d_model = int(cfg["hidden_size"])
    bb_n_heads = int(cfg["num_attention_heads"])
    bb_n_kv_heads = int(cfg.get("num_key_value_heads", bb_n_heads))
    bb_head_dim = int(cfg.get("head_dim", bb_d_model // bb_n_heads))
    bb_ff_dim = int(cfg["intermediate_size"])
    bb_max_pos = int(cfg.get("max_position_embeddings", 2048))
    rope_theta = float(cfg.get("rope_theta", 500000.0))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))
    text_vocab_size = int(cfg.get("text_vocab_size", 128256))
    audio_vocab_size = int(cfg.get("audio_vocab_size", 2051))
    audio_num_codebooks = int(cfg.get("audio_num_codebooks", 32))

    # --- Depth decoder hparams ---
    dd_cfg = cfg.get("depth_decoder_config", {})
    dd_n_layers = int(dd_cfg.get("num_hidden_layers", 4))
    dd_d_model = int(dd_cfg.get("hidden_size", 1024))
    dd_n_heads = int(dd_cfg.get("num_attention_heads", 8))
    dd_n_kv_heads = int(dd_cfg.get("num_key_value_heads", 2))
    dd_head_dim = int(dd_cfg.get("head_dim", dd_d_model // dd_n_heads))
    dd_ff_dim = int(dd_cfg.get("intermediate_size", 8192))
    dd_bb_hidden = int(dd_cfg.get("backbone_hidden_size", bb_d_model))

    # --- Mimi codec hparams ---
    mc_cfg = cfg.get("codec_config", {})
    mimi_dim = int(mc_cfg.get("hidden_size", 512))
    mimi_n_heads = int(mc_cfg.get("num_attention_heads", 8))
    mimi_n_layers = int(mc_cfg.get("num_hidden_layers", 8))
    mimi_codebook_dim = int(mc_cfg.get("codebook_dim", 256))
    mimi_codebook_size = int(mc_cfg.get("codebook_size", 2048))
    mimi_n_quantizers = int(mc_cfg.get("num_quantizers", 32))
    mimi_n_semantic = int(mc_cfg.get("num_semantic_quantizers", 1))
    mimi_sample_rate = int(mc_cfg.get("sampling_rate", 24000))
    mimi_frame_rate = float(mc_cfg.get("frame_rate", 12.5))

    # Special tokens
    audio_eos_token_id = int(cfg.get("audio_eos_token_id", 128003))
    audio_token_id = int(cfg.get("audio_token_id", 128002))
    bos_token_id = int(cfg.get("bos_token_id", 128000))
    codebook_eos_token_id = int(cfg.get("codebook_eos_token_id", 0))
    codebook_pad_token_id = int(cfg.get("codebook_pad_token_id", 2050))

    print(f"\nCSM-1B conversion")
    print(f"  Backbone:      {bb_n_layers}L  hidden={bb_d_model}  "
          f"heads={bb_n_heads}/{bb_n_kv_heads}  head_dim={bb_head_dim}  "
          f"ff={bb_ff_dim}  max_pos={bb_max_pos}")
    print(f"  Depth decoder: {dd_n_layers}L  hidden={dd_d_model}  "
          f"heads={dd_n_heads}/{dd_n_kv_heads}  head_dim={dd_head_dim}  "
          f"ff={dd_ff_dim}  backbone_hidden={dd_bb_hidden}")
    print(f"  Audio:         vocab={audio_vocab_size}  codebooks={audio_num_codebooks}  "
          f"text_vocab={text_vocab_size}")
    print(f"  Mimi:          dim={mimi_dim}  layers={mimi_n_layers}  heads={mimi_n_heads}  "
          f"cb_dim={mimi_codebook_dim}  cb_size={mimi_codebook_size}")
    print(f"  RoPE theta:    {rope_theta}  RMS eps: {rms_norm_eps}")

    out_dtype = np.float16 if args.outtype == "f16" else np.float32
    out_qt = GGMLQuantizationType.F16 if args.outtype == "f16" else GGMLQuantizationType.F32

    # --- Load safetensors ---
    # CSM uses an index file for sharded safetensors
    index_file = model_dir / "transformers.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        shard_files = sorted(set(weight_map.values()))
        st_files = [model_dir / f for f in shard_files]
    else:
        st_files = sorted(model_dir.glob("*.safetensors"))

    if not st_files:
        sys.exit(f"no safetensors in {model_dir}")

    handles = [safe_open(str(f), framework="pt") for f in st_files]
    name_to_idx = {}
    for i, h in enumerate(handles):
        for k in h.keys():
            name_to_idx[k] = i
    print(f"  Safetensors:   {len(name_to_idx)} tensors in {len(st_files)} file(s)")

    # --- Write GGUF ---
    out_path = Path(args.output)
    w = GGUFWriter(str(out_path), arch="csm-tts", use_temp_file=False)

    # Metadata
    w.add_name("csm-1b")

    def u32(k, v): w.add_uint32(k, int(v))
    def f32(k, v): w.add_float32(k, float(v))

    # Backbone
    u32("csm.backbone.n_layers", bb_n_layers)
    u32("csm.backbone.d_model", bb_d_model)
    u32("csm.backbone.n_heads", bb_n_heads)
    u32("csm.backbone.n_kv_heads", bb_n_kv_heads)
    u32("csm.backbone.head_dim", bb_head_dim)
    u32("csm.backbone.ff_dim", bb_ff_dim)
    u32("csm.backbone.max_pos", bb_max_pos)
    f32("csm.backbone.rope_theta", rope_theta)
    f32("csm.backbone.rms_norm_eps", rms_norm_eps)

    # Depth decoder
    u32("csm.depth.n_layers", dd_n_layers)
    u32("csm.depth.d_model", dd_d_model)
    u32("csm.depth.n_heads", dd_n_heads)
    u32("csm.depth.n_kv_heads", dd_n_kv_heads)
    u32("csm.depth.head_dim", dd_head_dim)
    u32("csm.depth.ff_dim", dd_ff_dim)
    u32("csm.depth.backbone_hidden", dd_bb_hidden)

    # Audio / text vocab
    u32("csm.text_vocab_size", text_vocab_size)
    u32("csm.audio_vocab_size", audio_vocab_size)
    u32("csm.audio_num_codebooks", audio_num_codebooks)

    # Special tokens
    u32("csm.audio_eos_token_id", audio_eos_token_id)
    u32("csm.audio_token_id", audio_token_id)
    u32("csm.bos_token_id", bos_token_id)
    u32("csm.codebook_eos_token_id", codebook_eos_token_id)
    u32("csm.codebook_pad_token_id", codebook_pad_token_id)

    # Mimi codec
    u32("csm.mimi.dim", mimi_dim)
    u32("csm.mimi.n_heads", mimi_n_heads)
    u32("csm.mimi.n_layers", mimi_n_layers)
    u32("csm.mimi.codebook_dim", mimi_codebook_dim)
    u32("csm.mimi.codebook_size", mimi_codebook_size)
    u32("csm.mimi.n_quantizers", mimi_n_quantizers)
    u32("csm.mimi.n_semantic", mimi_n_semantic)
    u32("csm.mimi.sample_rate", mimi_sample_rate)
    f32("csm.mimi.frame_rate", mimi_frame_rate)

    # --- Tokenizer (Llama-3.2 BPE) ---
    tj = model_dir / "tokenizer.json"
    if tj.exists():
        with open(tj) as f:
            tjd = json.load(f)

        model_block = tjd.get("model", {})
        base_vocab = model_block.get("vocab", {})
        added = tjd.get("added_tokens", [])

        max_id = 0
        if base_vocab:
            max_id = max(int(v) for v in base_vocab.values())
        if added:
            max_id = max(max_id, max(int(it["id"]) for it in added))

        toks = [""] * (max_id + 1)
        for tok, idx in base_vocab.items():
            idx = int(idx)
            if idx < len(toks):
                toks[idx] = tok
        for it in added:
            idx = int(it["id"])
            if idx < len(toks):
                toks[idx] = str(it["content"])
        w.add_token_list(toks)
        print(f"  Tokens:        {len(toks)} entries")

        merges = model_block.get("merges", [])
        if merges:
            if merges and isinstance(merges[0], list):
                merges = [" ".join(p) for p in merges]
            w.add_token_merges(merges)
            print(f"  Merges:        {len(merges)} entries")
    else:
        print("  WARN: no tokenizer.json found", file=sys.stderr)

    # --- Collect RVQ cluster_usage for embed_sum normalization ---
    # The HF transformers format stores embed_sum (unnormalized) + cluster_usage.
    # Actual embedding = embed_sum / cluster_usage. We need to normalize.
    rvq_cluster_usage = {}
    for hf_name in sorted(name_to_idx.keys()):
        if hf_name.endswith(".cluster_usage"):
            cu = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()
            # Map the cluster_usage key to the corresponding embed_sum key
            embed_sum_key = hf_name.replace(".cluster_usage", ".embed_sum")
            rvq_cluster_usage[embed_sum_key] = cu
            print(f"  [RVQ] cluster_usage for {hf_name}: shape={cu.shape}, "
                  f"mean={cu.mean():.2f}, min={cu.min():.2f}")

    # --- Map and write tensors ---
    n_mapped = [0]
    n_skipped = 0
    pending_qkv = {}  # for Mimi transformer QKV fusion

    for hf_name in sorted(name_to_idx.keys()):
        gn = map_tensor_name(hf_name)
        if gn is None:
            n_skipped += 1
            continue

        t = handles[name_to_idx[hf_name]].get_tensor(hf_name).to(torch.float32).numpy()

        # Normalize RVQ embed_sum by cluster_usage to get actual embeddings.
        # embed_sum has shape (codebook_dim, num_codes) = (256, 2048).
        # cluster_usage has shape (num_codes,) = (2048,).
        # actual_embedding = embed_sum / cluster_usage[None, :]
        if hf_name in rvq_cluster_usage:
            cu = rvq_cluster_usage[hf_name]
            # Clamp to moshi's EuclideanCodebook epsilon (1e-5), NOT 1.0. The
            # codebook used at decode time is the property
            #   embedding = embed_sum / cluster_usage.clamp(min=epsilon)
            # (moshi/quantization/core_vq.py). cluster_usage is a decayed EMA
            # whose values are mostly < 1 (typical mean ~0.6, min ~0.12), so
            # clamping at 1.0 left ~96% of codes effectively un-normalized and
            # produced buzzing audio (RVQ-dequant cos ~0.9). 1e-5 matches moshi.
            cu = np.maximum(cu, 1e-5)
            # embed_sum shape is (num_codes, codebook_dim) = (2048, 256)
            # cluster_usage shape is (num_codes,) = (2048,)
            # actual_embedding = embed_sum / cluster_usage[:, None]
            t = t / cu[:, None]
            print(f"  [RVQ] normalized {hf_name} by cluster_usage")

        # Separate Q/K/V for Mimi transformer -> collect for fusion
        if ".attn.q_w" in gn or ".attn.k_w" in gn or ".attn.v_w" in gn:
            pending_qkv[gn] = t
            continue

        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped[0] += 1
        if n_mapped[0] <= 30 or n_mapped[0] % 50 == 0:
            print(f"  [{n_mapped[0]}] {gn:55s} {t.shape}  {t.dtype}")

    # Fuse Mimi Q/K/V into QKV
    fused = fuse_qkv_tensors(pending_qkv, out_dtype, out_qt, w, n_mapped)

    # Any remaining unfused Q/K/V (shouldn't happen)
    for gn, t in pending_qkv.items():
        if t.ndim <= 1:
            t = np.ascontiguousarray(t.astype(np.float32))
            w.add_tensor(gn, t, raw_dtype=GGMLQuantizationType.F32)
        else:
            t = np.ascontiguousarray(t.astype(out_dtype))
            w.add_tensor(gn, t, raw_dtype=out_qt)
        n_mapped[0] += 1
        print(f"  [unfused] {gn:55s} {t.shape}  {t.dtype}")

    print(f"\nMapped: {n_mapped[0]}, skipped: {n_skipped}, fused QKV: {len(fused)}")
    print(f"Writing {out_path}...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    sz = out_path.stat().st_size / 1e9
    print(f"Done: {out_path}  ({sz:.2f} GB, {n_mapped[0]} tensors)")


if __name__ == "__main__":
    main()
