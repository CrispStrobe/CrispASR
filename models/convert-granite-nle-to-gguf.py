#!/usr/bin/env python3
"""
Convert ibm-granite/granite-speech-4.1-2b-nar (HF safetensors) → GGUF F16.

Architecture (`model_type: nle`, `architectures: ["NLENARDecoder"]`):

  Audio encoder (NLECTCEncoder): 16-layer Conformer with self-conditioning
    input_linear: (1024, 160)
    16 × NLEConformerBlock (Macaron: FFN + MHSA + Conv + FFN):
      ff1, attn (block-local, Shaw RPE), conv (Macaron), ff2, post_norm
    self_conditioning_layer = 8 → at the layer-8 boundary the running
      CTC logits (via `out`) are projected back into the hidden via `out_mid`
    out:     Linear(1024, 348)        -- char-level CTC head
    out_mid: Linear(348, 1024)        -- self-conditioning back-projection
    out_bpe: Linear(1024, 100353)     -- aux BPE CTC head, computed only on
                                         posterior-weighted-pooled hidden
                                         (window=4) at valid frames.

  Projector (EncoderProjectorQFormer): consumes 4 encoder hidden layers
    encoder_layer_indices = [4, 8, 12, -1]  -- 4 hidden states stacked along
                                                feature dim → 4 × 1024 = 4096
    layer_norms[0..3]:   per-layer LayerNorm
    layer_projector:     Linear(4096, 2048) + GELU
    Then block-split (block_size=15), pool by downsample_rate=5 → 3 query
    slots per 15 frames. Add `query` ([1, 3, 2048]) to mean-pooled context
    + `window_positions` ([1, 15, 2048]) on the K/V side.
    qformer.layers[0..1]: 2-layer simplified Q-Former (32-head SDPA cross-
      attention, MLP with mlp_ratio=2 and SiLU)
    out_norm + out_linear(2048 → 2048)  -- final projection to LLM dim

  LLM (Granite 4.0-1B, 40 layers): used as a NON-AUTOREGRESSIVE refiner.
    All self-attention layers patched to `is_causal=False` at runtime; the
    LLM does ONE forward pass over a flat sequence of
      [audio_embs_0, text_embs_0, audio_embs_1, text_embs_1, ...]
    where text_embs are the LLM-tokenised CTC-decoded text with `eos_id`
    insertion-slots between every output token. The slot positions absorb
    the editing logits; CTC-style argmax + unique_consecutive on those
    positions gives the final transcript.
    vocab_size = 100352 (vs 100353 for the AR variants); the BPE auxiliary
    head's vocab is 100353 (the extra slot is a CTC blank).
    NAR variant has tied embeddings (no separate `lm_head.weight`).

GGUF arch: `granite_nle`. Tensor name conventions:
  enc.input.{weight,bias}
  enc.blk.{N}.* (same naming as granite_speech)
  enc.ctc_out.{weight,bias}      (348 vocab)
  enc.ctc_mid.{weight,bias}      (348 → 1024 self-conditioning)
  enc.bpe_out.{weight,bias}      (100353 vocab — NEW)
  proj.layer_norm.{0..3}.{weight,bias}
  proj.layer_proj.{weight,bias}
  proj.query
  proj.window_positions
  proj.blk.{N}.attn_norm.{weight,bias}
  proj.blk.{N}.attn_{q,k,v,o}.{weight,bias}
  proj.blk.{N}.mlp_norm.{weight,bias}
  proj.blk.{N}.mlp_{fc1,fc2}.{weight,bias}
  proj.out_norm.{weight,bias}
  proj.out_linear.{weight,bias}
  llm.token_embd.weight
  llm.blk.{N}.{attn_norm,attn_q,attn_k,attn_v,attn_o,ffn_norm,ffn_gate,ffn_up,ffn_down}.weight
  llm.norm.weight
  (no lm_head — tied to token_embd)

Shaped weights are kept in F32 by default; pass --quant {q4_k,q5_k,q8_0}
to delegate to crispasr-quantize on the produced F16 GGUF.
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

DIRECT = {
    # Encoder input + CTC heads
    "encoder.input_linear.weight":   "enc.input.weight",
    "encoder.input_linear.bias":     "enc.input.bias",
    "encoder.out.weight":            "enc.ctc_out.weight",
    "encoder.out.bias":              "enc.ctc_out.bias",
    "encoder.out_mid.weight":        "enc.ctc_mid.weight",
    "encoder.out_mid.bias":          "enc.ctc_mid.bias",
    "encoder.out_bpe.weight":        "enc.bpe_out.weight",
    "encoder.out_bpe.bias":          "enc.bpe_out.bias",

    # Projector (NAR-specific structure)
    "projector.layer_projector.weight": "proj.layer_proj.weight",
    "projector.layer_projector.bias":   "proj.layer_proj.bias",
    "projector.query":                  "proj.query",
    "projector.window_positions":       "proj.window_positions",
    "projector.out_norm.weight":        "proj.out_norm.weight",
    "projector.out_norm.bias":          "proj.out_norm.bias",
    "projector.out_linear.weight":      "proj.out_linear.weight",
    "projector.out_linear.bias":        "proj.out_linear.bias",

    # LLM root tensors
    "llm.model.embed_tokens.weight":    "llm.token_embd.weight",
    "llm.model.norm.weight":            "llm.norm.weight",
}

REGEX_RULES = [
    # Encoder layers (same naming as granite_speech base)
    (r"encoder\.layers\.(\d+)\.attn\.pre_norm\.weight",        "enc.blk.{}.attn_norm.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.pre_norm\.bias",          "enc.blk.{}.attn_norm.bias"),
    (r"encoder\.layers\.(\d+)\.attn\.to_q\.weight",            "enc.blk.{}.attn_q.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.to_kv\.weight",           "enc.blk.{}.attn_kv.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.to_out\.weight",          "enc.blk.{}.attn_out.weight"),
    (r"encoder\.layers\.(\d+)\.attn\.to_out\.bias",            "enc.blk.{}.attn_out.bias"),
    (r"encoder\.layers\.(\d+)\.attn\.rel_pos_emb\.weight",     "enc.blk.{}.attn_rel_pos.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.up_conv\.weight",         "enc.blk.{}.conv_up.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.up_conv\.bias",           "enc.blk.{}.conv_up.bias"),
    (r"encoder\.layers\.(\d+)\.conv\.depth_conv\.conv\.weight","enc.blk.{}.conv_dw.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.weight",      "enc.blk.{}.conv_bn.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.bias",        "enc.blk.{}.conv_bn.bias"),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.running_mean","enc.blk.{}.conv_bn.running_mean"),
    (r"encoder\.layers\.(\d+)\.conv\.batch_norm\.running_var", "enc.blk.{}.conv_bn.running_var"),
    (r"encoder\.layers\.(\d+)\.conv\.down_conv\.weight",       "enc.blk.{}.conv_down.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.down_conv\.bias",         "enc.blk.{}.conv_down.bias"),
    (r"encoder\.layers\.(\d+)\.conv\.norm\.weight",            "enc.blk.{}.conv_norm.weight"),
    (r"encoder\.layers\.(\d+)\.conv\.norm\.bias",              "enc.blk.{}.conv_norm.bias"),
    (r"encoder\.layers\.(\d+)\.ff1\.pre_norm\.weight",         "enc.blk.{}.ff1_norm.weight"),
    (r"encoder\.layers\.(\d+)\.ff1\.pre_norm\.bias",           "enc.blk.{}.ff1_norm.bias"),
    (r"encoder\.layers\.(\d+)\.ff1\.up_proj\.weight",          "enc.blk.{}.ff1_up.weight"),
    (r"encoder\.layers\.(\d+)\.ff1\.up_proj\.bias",            "enc.blk.{}.ff1_up.bias"),
    (r"encoder\.layers\.(\d+)\.ff1\.down_proj\.weight",        "enc.blk.{}.ff1_down.weight"),
    (r"encoder\.layers\.(\d+)\.ff1\.down_proj\.bias",          "enc.blk.{}.ff1_down.bias"),
    (r"encoder\.layers\.(\d+)\.ff2\.pre_norm\.weight",         "enc.blk.{}.ff2_norm.weight"),
    (r"encoder\.layers\.(\d+)\.ff2\.pre_norm\.bias",           "enc.blk.{}.ff2_norm.bias"),
    (r"encoder\.layers\.(\d+)\.ff2\.up_proj\.weight",          "enc.blk.{}.ff2_up.weight"),
    (r"encoder\.layers\.(\d+)\.ff2\.up_proj\.bias",            "enc.blk.{}.ff2_up.bias"),
    (r"encoder\.layers\.(\d+)\.ff2\.down_proj\.weight",        "enc.blk.{}.ff2_down.weight"),
    (r"encoder\.layers\.(\d+)\.ff2\.down_proj\.bias",          "enc.blk.{}.ff2_down.bias"),
    (r"encoder\.layers\.(\d+)\.post_norm\.weight",             "enc.blk.{}.post_norm.weight"),
    (r"encoder\.layers\.(\d+)\.post_norm\.bias",               "enc.blk.{}.post_norm.bias"),

    # Projector layer-norms (one per encoder_layer_indices entry)
    (r"projector\.layer_norms\.(\d+)\.weight", "proj.layer_norm.{}.weight"),
    (r"projector\.layer_norms\.(\d+)\.bias",   "proj.layer_norm.{}.bias"),

    # NAR Q-Former: flat naming
    (r"projector\.qformer\.layers\.(\d+)\.attn_norm\.weight",                 "proj.blk.{}.attn_norm.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.attn_norm\.bias",                   "proj.blk.{}.attn_norm.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.q_proj\.weight",   "proj.blk.{}.attn_q.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.q_proj\.bias",     "proj.blk.{}.attn_q.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.k_proj\.weight",   "proj.blk.{}.attn_k.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.k_proj\.bias",     "proj.blk.{}.attn_k.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.v_proj\.weight",   "proj.blk.{}.attn_v.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.v_proj\.bias",     "proj.blk.{}.attn_v.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.o_proj\.weight",   "proj.blk.{}.attn_o.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.cross_attention\.o_proj\.bias",     "proj.blk.{}.attn_o.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.mlp_norm\.weight",                  "proj.blk.{}.mlp_norm.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.mlp_norm\.bias",                    "proj.blk.{}.mlp_norm.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.mlp\.fc1\.weight",                  "proj.blk.{}.mlp_fc1.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.mlp\.fc1\.bias",                    "proj.blk.{}.mlp_fc1.bias"),
    (r"projector\.qformer\.layers\.(\d+)\.mlp\.fc2\.weight",                  "proj.blk.{}.mlp_fc2.weight"),
    (r"projector\.qformer\.layers\.(\d+)\.mlp\.fc2\.bias",                    "proj.blk.{}.mlp_fc2.bias"),

    # LLM (Granite 4.0-1B). Same convention as granite_speech base.
    (r"llm\.model\.layers\.(\d+)\.input_layernorm\.weight",          "llm.blk.{}.attn_norm.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.q_proj\.weight",        "llm.blk.{}.attn_q.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.k_proj\.weight",        "llm.blk.{}.attn_k.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.v_proj\.weight",        "llm.blk.{}.attn_v.weight"),
    (r"llm\.model\.layers\.(\d+)\.self_attn\.o_proj\.weight",        "llm.blk.{}.attn_o.weight"),
    (r"llm\.model\.layers\.(\d+)\.post_attention_layernorm\.weight", "llm.blk.{}.ffn_norm.weight"),
    (r"llm\.model\.layers\.(\d+)\.mlp\.gate_proj\.weight",           "llm.blk.{}.ffn_gate.weight"),
    (r"llm\.model\.layers\.(\d+)\.mlp\.up_proj\.weight",             "llm.blk.{}.ffn_up.weight"),
    (r"llm\.model\.layers\.(\d+)\.mlp\.down_proj\.weight",           "llm.blk.{}.ffn_down.weight"),
]

SKIP_PATTERNS = [
    # BatchNorm batch counter — derived stats, not a learnable weight.
    re.compile(r"\.num_batches_tracked$"),
]


def remap(name: str):
    if name in DIRECT:
        return DIRECT[name]
    for pat, tmpl in REGEX_RULES:
        m = re.match(pat, name)
        if m:
            return tmpl.format(*m.groups())
    return None


def should_skip(name: str) -> bool:
    return any(p.search(name) for p in SKIP_PATTERNS)


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def to_numpy(t):
    """Cast a safetensors tensor to a numpy array, handling bfloat16."""
    import torch
    if isinstance(t, torch.Tensor):
        # numpy() doesn't support bfloat16; promote to float32 first.
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        return t.cpu().numpy()
    return np.asarray(t)


def is_f32_tensor(name: str, shape) -> bool:
    """Match the policy of convert-granite-speech-to-gguf.py: keep encoder /
    projector weights and small / norm tensors at F32; everything else
    (mainly LLM matmul weights) goes F16. Without this, an all-F32 NAR
    GGUF is ~10 GB instead of ~5 GB."""
    if name.endswith(".bias"):
        return True
    if "norm" in name:
        return True
    if "rel_pos" in name:
        return True
    if "running_mean" in name or "running_var" in name:
        return True
    if "window_pos" in name or "query" in name:
        return True
    if len(shape) <= 1:
        return True
    if name.startswith("enc."):
        return True
    if name.startswith("proj."):
        return True
    return False


def convert(input_dir: Path, out_path: Path) -> None:
    print(f"Loading: {input_dir}")
    with open(input_dir / "config.json") as f:
        cfg = json.load(f)

    enc_cfg = cfg.get("encoder_config", {})
    proj_cfg = cfg.get("projector_config", {})
    llm_cfg = cfg.get("llm_config", {})
    ctc_cfg = cfg.get("ctc_tokenizer_config", {}) or {}
    model_type = cfg.get("model_type", "")
    if model_type != "nle":
        sys.exit(
            f"refusing to convert: expected model_type='nle' for the NAR variant, got '{model_type}'. "
            "Use convert-granite-speech-to-gguf.py for the base / plus variants."
        )

    # Locate safetensor files (NAR ships a single file but we keep the
    # shard-index path for forward-compat with future quantised releases).
    idx_path = input_dir / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)
        wanted = sorted(set(idx.get("weight_map", {}).values()))
        files = [input_dir / n for n in wanted]
    else:
        files = sorted(input_dir.glob("model*.safetensors"))
    print(f"  shards: {[p.name for p in files]}")

    print(f"Writing: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out_path), arch="granite_nle", use_temp_file=True)

    # ----- Metadata -----
    writer.add_uint32("granite_nle.sample_rate", 16000)
    writer.add_uint32("granite_nle.n_mels", 80)

    # Encoder
    writer.add_uint32("granite_nle.enc.n_layers",     enc_cfg.get("num_layers", 16))
    writer.add_uint32("granite_nle.enc.d_model",      enc_cfg.get("hidden_dim", 1024))
    writer.add_uint32("granite_nle.enc.n_heads",      enc_cfg.get("num_heads", 8))
    writer.add_uint32("granite_nle.enc.head_dim",     enc_cfg.get("dim_head", 128))
    writer.add_uint32("granite_nle.enc.input_dim",    enc_cfg.get("input_dim", 160))
    writer.add_uint32("granite_nle.enc.conv_kernel",  enc_cfg.get("conv_kernel_size", 15))
    writer.add_uint32(
        "granite_nle.enc.ff_dim",
        enc_cfg.get("hidden_dim", 1024) * enc_cfg.get("feedforward_mult", 4),
    )
    writer.add_uint32("granite_nle.enc.context_size",            enc_cfg.get("context_size", 200))
    writer.add_uint32("granite_nle.enc.max_pos_emb",             enc_cfg.get("max_pos_emb", 512))
    writer.add_uint32("granite_nle.enc.ctc_vocab",               enc_cfg.get("output_dim", 348))
    writer.add_uint32("granite_nle.enc.bpe_vocab",               enc_cfg.get("bpe_output_dim", 100353))
    writer.add_uint32("granite_nle.enc.bpe_pooling_window",      enc_cfg.get("bpe_pooling_window", 4))
    writer.add_uint32("granite_nle.enc.self_conditioning_layer", enc_cfg.get("self_conditioning_layer", 8))

    # Projector
    enc_layer_indices = cfg.get("encoder_layer_indices", [4, 8, 12, -1])
    # gguf has no add_array_int32; pack as comma-separated string.
    writer.add_string(
        "granite_nle.proj.encoder_layer_indices",
        ",".join(str(i) for i in enc_layer_indices),
    )
    writer.add_uint32("granite_nle.proj.num_encoder_layers", proj_cfg.get("num_encoder_layers", len(enc_layer_indices)))
    writer.add_uint32("granite_nle.proj.encoder_dim",        proj_cfg.get("encoder_dim", 1024))
    writer.add_uint32("granite_nle.proj.hidden_size",        proj_cfg.get("hidden_size", 2048))
    writer.add_uint32("granite_nle.proj.llm_dim",            proj_cfg.get("llm_dim", 2048))
    writer.add_uint32("granite_nle.proj.n_layers",           proj_cfg.get("num_layers", 2))
    writer.add_uint32("granite_nle.proj.n_heads",            proj_cfg.get("num_heads", 32))
    writer.add_uint32("granite_nle.proj.mlp_ratio",          proj_cfg.get("mlp_ratio", 2))
    writer.add_uint32("granite_nle.proj.block_size",         proj_cfg.get("block_size", 15))
    writer.add_uint32("granite_nle.proj.downsample_rate",    proj_cfg.get("downsample_rate", 5))
    writer.add_float32("granite_nle.proj.layernorm_eps",     float(proj_cfg.get("layernorm_eps", 1e-6)))
    writer.add_bool("granite_nle.proj.attn_bias",            bool(proj_cfg.get("attn_bias", True)))
    writer.add_bool("granite_nle.proj.mlp_bias",             bool(proj_cfg.get("mlp_bias", True)))

    # LLM (Granite 4.0-1B used as NAR refiner)
    writer.add_uint32("granite_nle.llm.n_layers",      llm_cfg.get("num_hidden_layers", 40))
    writer.add_uint32("granite_nle.llm.d_model",       llm_cfg.get("hidden_size", 2048))
    writer.add_uint32("granite_nle.llm.n_heads",       llm_cfg.get("num_attention_heads", 16))
    writer.add_uint32("granite_nle.llm.n_kv_heads",    llm_cfg.get("num_key_value_heads", 4))
    writer.add_uint32(
        "granite_nle.llm.head_dim",
        llm_cfg.get("hidden_size", 2048) // llm_cfg.get("num_attention_heads", 16),
    )
    writer.add_uint32("granite_nle.llm.ff_dim",        llm_cfg.get("intermediate_size", 4096))
    writer.add_float32("granite_nle.llm.rope_theta",   float(llm_cfg.get("rope_theta", 10000)))
    writer.add_float32("granite_nle.llm.rms_norm_eps", float(llm_cfg.get("rms_norm_eps", 1e-5)))
    writer.add_uint32("granite_nle.llm.vocab_size",    llm_cfg.get("vocab_size", 100352))
    writer.add_float32("granite_nle.llm.embedding_multiplier", float(llm_cfg.get("embedding_multiplier", 12.0)))
    writer.add_float32("granite_nle.llm.attention_multiplier", float(llm_cfg.get("attention_multiplier", 0.0078125)))
    writer.add_float32("granite_nle.llm.residual_multiplier",  float(llm_cfg.get("residual_multiplier", 0.22)))
    writer.add_float32("granite_nle.llm.logits_scaling",       float(llm_cfg.get("logits_scaling", 8.0)))
    writer.add_uint32("granite_nle.llm.eos_token_id",          llm_cfg.get("eos_token_id", 100257))
    writer.add_uint32("granite_nle.llm.bos_token_id",          llm_cfg.get("bos_token_id", 100257))
    writer.add_uint32("granite_nle.llm.pad_token_id",          llm_cfg.get("pad_token_id", 100256))
    writer.add_bool("granite_nle.llm.tie_word_embeddings",     bool(llm_cfg.get("tie_word_embeddings", True)))
    writer.add_bool("granite_nle.scale_projected_embeddings",  bool(cfg.get("scale_projected_embeddings", True)))

    # CTC tokenizer (char2idx). Pack as parallel arrays so the runtime can
    # rebuild the table without parsing JSON. Sort by index for stability.
    char2idx = ctc_cfg.get("char2idx", {})
    if char2idx:
        items = sorted(char2idx.items(), key=lambda kv: kv[1])
        # Verify monotonic indices (no gaps, no duplicates).
        for i, (_, idx) in enumerate(items):
            if i != idx:
                # Some checkpoints leave idx 0..31 unmapped (printable-only ranges).
                # We pad with empty placeholders so the index→char lookup stays
                # O(1) at runtime.
                break
        max_idx = items[-1][1]
        chars = [""] * (max_idx + 1)
        for ch, idx in items:
            chars[idx] = ch
        writer.add_uint32("granite_nle.ctc.vocab_size", len(chars))
        # add_array doesn't take strings; emit one key per slot.
        # That's wasteful for 348 entries — pack into a single newline-joined
        # string instead. The runtime splits on '\n'.
        writer.add_string("granite_nle.ctc.vocab", "\n".join(chars))

    # LLM tokenizer — same names as base granite_speech for consistency
    # (the runtime reads these via core_gguf::open_metadata).
    n_vocab_llm = llm_cfg.get("vocab_size", 100352)
    vocab_path = input_dir / "vocab.json"
    merges_path = input_dir / "merges.txt"
    tokenizer_json_path = input_dir / "tokenizer.json"
    vocab_dict = None
    merges_list = None
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab_dict = json.load(f)
        if merges_path.exists():
            with open(merges_path, encoding="utf-8") as f:
                merges_list = [
                    line.rstrip("\n") for line in f
                    if line.strip() and not line.startswith("#")
                ]
    elif tokenizer_json_path.exists():
        with open(tokenizer_json_path, encoding="utf-8") as f:
            tj = json.load(f)
        model = tj.get("model", {})
        if model.get("type") == "BPE":
            vocab_dict = model.get("vocab", {})
            raw_merges = model.get("merges", [])
            merges_list = []
            for m in raw_merges:
                if isinstance(m, list):
                    merges_list.append(" ".join(m))
                elif isinstance(m, str):
                    merges_list.append(m)
            for at in tj.get("added_tokens", []):
                tid = at.get("id")
                content = at.get("content")
                if tid is not None and content is not None:
                    vocab_dict[content] = tid
    if vocab_dict is not None:
        tokens = [""] * n_vocab_llm
        for tok_str, tok_id in vocab_dict.items():
            if 0 <= tok_id < n_vocab_llm:
                tokens[tok_id] = tok_str
        writer.add_array("tokenizer.ggml.tokens", tokens)
        print(f"  tokenizer: {len(tokens)} tokens")
    if merges_list is not None:
        writer.add_array("tokenizer.ggml.merges", merges_list)
        print(f"  merges:    {len(merges_list)} BPE merges")

    # ----- Tensor data -----
    skipped = []
    unmapped = []
    written = 0
    print()
    for fp in files:
        print(f"  shard: {fp.name}")
        with safe_open(fp, framework="pt") as f:
            for name in f.keys():
                if should_skip(name):
                    skipped.append(name)
                    continue
                new_name = remap(name)
                if new_name is None:
                    unmapped.append(name)
                    continue
                t = f.get_tensor(name)
                arr = to_numpy(t)
                # Promote everything to F32 first, then downcast LLM matmul
                # weights to F16 to keep the on-disk size near 5 GB rather
                # than 10 GB. is_f32_tensor mirrors the policy of the base
                # granite_speech converter.
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                if not is_f32_tensor(new_name, arr.shape):
                    arr = arr.astype(np.float16)
                writer.add_tensor(new_name, arr)
                written += 1
                if written <= 6 or written % 50 == 0:
                    print(f"    [{written}] {name:60s} -> {new_name:50s} {list(arr.shape)} {arr.dtype}")

    print()
    print(f"  wrote {written} tensors")
    if skipped:
        print(f"  skipped {len(skipped)} (e.g. {skipped[0]})")
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped tensors:")
        for n in unmapped[:10]:
            print(f"    {n}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    print(f"Finalising GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path}  ({out_path.stat().st_size / 2**30:.2f} GiB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="HF snapshot dir for ibm-granite/granite-speech-4.1-2b-nar")
    p.add_argument("--output", required=True, type=Path,
                   help="output GGUF path")
    args = p.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
